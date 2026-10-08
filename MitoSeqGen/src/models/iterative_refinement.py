"""MitoSeqGen-IR: constrained iterative-refinement (non-autoregressive) codon design.

Motivation
----------
Every published codon optimizer treats codon design as autoregressive sequence
generation. It is not. Because the target protein is given:

  * output length is known exactly (len(protein)), so there is no length
    prediction problem;
  * each position's label set is the synonym class of its amino acid under the
    mitochondrial code -- 2 to 6 codons, never more;

so the output space is the product lattice PROD_i Syn(aa_i). The two standard
objections to non-autoregressive generation (unknown length, multimodality over
a huge vocabulary) therefore do not apply here, which makes this one of the few
sequence tasks where NAR decoding is better-posed than AR decoding rather than a
speed/quality compromise.

This matters beyond elegance. GC content and folding free energy are GLOBAL
properties of the whole sequence. Left-to-right AR decoding cannot see
downstream positions, which is the most likely reason three separate
training-time GC losses failed to move GC deviation in the AR model. An
iterative refiner conditions on the entire current hypothesis at every step, so
global properties are visible to the model and steerable at decode time.

Method
------
Training is the conditional masked language model (CMLM) objective of
Ghazvininejad et al.'s Mask-Predict, adapted to the synonym lattice: a random
subset of codon positions is replaced by <MASK>, and the model predicts them
from the protein plus the unmasked codons, with loss on masked positions only.

Inference is mask-predict: start fully masked, predict all positions in
parallel, then for T-1 rounds re-mask the lowest-confidence positions and
re-predict them with full bidirectional context.

Every prediction, in training and inference, is masked to the synonym class of
the required amino acid, so protein-exactness and the absence of internal stop
codons remain guaranteed by construction -- the same guarantee the AR model
gives, preserved under a completely different decoding scheme.

Retrieval conditioning (optional, n_retrieved > 0) adds aligned homolog codons
as a per-position side input; see scripts/retrieval_index.py.
"""

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import AA_VOCAB, VOCAB  # noqa: E402
from src.models.constraints import build_synonym_mask_matrix  # noqa: E402

PAD_ID = VOCAB["<PAD>"]
N_CODON_TOKENS = len(VOCAB)          # 68 (4 special + 64 codons)
MASK_ID = N_CODON_TOKENS             # 68 -> appended as the <MASK> codon token
N_AA_TOKENS = len(AA_VOCAB)          # 25


def build_full_constraint_mask(device=torch.device("cpu")) -> torch.Tensor:
    """(N_AA_TOKENS, N_CODON_TOKENS) bool. Row aa_id = codon ids allowed there.

    build_synonym_mask_matrix covers real amino acids and '*'. Its rows for the
    four special tokens are all-False, which would make every logit -inf at
    those positions. AA_VOCAB and VOCAB share the same four leading special
    tokens at the same indices, so a special amino-acid token maps to the codon
    token of the same id.
    """
    mask = build_synonym_mask_matrix(device=device).clone()
    for tok in ("<PAD>", "<BOS>", "<EOS>", "<UNK>"):
        mask[AA_VOCAB[tok], :] = False
        mask[AA_VOCAB[tok], VOCAB[tok]] = True
    return mask


class MitoSeqGenIR(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 12,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_position_embeddings: int = 768,
        n_retrieved: int = 0,
    ):
        super().__init__()
        self.n_retrieved = n_retrieved
        self.mask_token_id = MASK_ID

        self.codon_embedding = nn.Embedding(N_CODON_TOKENS + 1, d_model, padding_idx=PAD_ID)
        self.aa_embedding = nn.Embedding(N_AA_TOKENS, d_model, padding_idx=AA_VOCAB["<PAD>"])
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        if n_retrieved > 0:
            # retrieved homolog codon at this position, and whether that homolog's
            # amino acid actually matches the target's (i.e. whether its codon is
            # even a legal suggestion here)
            self.retrieved_embedding = nn.Embedding(N_CODON_TOKENS + 1, d_model, padding_idx=PAD_ID)
            self.identity_embedding = nn.Embedding(2, d_model)
            self.retrieved_scale = nn.Parameter(torch.zeros(1))  # start at 0: ignore retrieval, learn to use it

        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, N_CODON_TOKENS)

        self.register_buffer("constraint_mask", build_full_constraint_mask(), persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        aa_tokens: torch.Tensor,            # (B, L) long
        codon_inputs: torch.Tensor,         # (B, L) long, MASK_ID where unknown
        retrieved_codons: Optional[torch.Tensor] = None,   # (B, k, L)
        retrieved_identity: Optional[torch.Tensor] = None,  # (B, k, L) in {0,1}
        apply_constraints: bool = True,
    ) -> torch.Tensor:
        B, L = aa_tokens.shape
        pos = torch.arange(L, device=aa_tokens.device).unsqueeze(0).expand(B, L)
        x = self.aa_embedding(aa_tokens) + self.codon_embedding(codon_inputs) + self.position_embedding(pos)

        if self.n_retrieved > 0 and retrieved_codons is not None:
            r = self.retrieved_embedding(retrieved_codons) + self.identity_embedding(retrieved_identity)
            # only positions where the homolog's amino acid matches carry usable signal
            r = r * retrieved_identity.unsqueeze(-1).to(r.dtype)
            x = x + self.retrieved_scale * r.mean(dim=1)

        x = self.dropout(self.input_norm(x))
        pad_mask = aa_tokens.eq(AA_VOCAB["<PAD>"])
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        logits = self.output_projection(self.output_norm(h))

        if apply_constraints:
            allowed = self.constraint_mask[aa_tokens]          # (B, L, N_CODON_TOKENS)
            logits = logits.masked_fill(~allowed, float("-inf"))
        return logits


# ---------------------------------------------------------------------------
# CMLM training-time masking
# ---------------------------------------------------------------------------

def cmlm_mask(
    codon_targets: torch.Tensor,
    aa_tokens: torch.Tensor,
    generator: Optional[torch.Generator] = None,
):
    """Replace a random subset of real codon positions with <MASK>.

    Per sequence a ratio is drawn uniformly from (0, 1] and that fraction of the
    non-special positions is masked (at least one). Uniform sampling is what
    lets one model serve every round of mask-predict, where the number of masked
    positions falls from L to 0.

    Returns (codon_inputs, loss_mask).
    """
    B, L = codon_targets.shape
    special = (
        aa_tokens.eq(AA_VOCAB["<PAD>"]) | aa_tokens.eq(AA_VOCAB["<BOS>"]) | aa_tokens.eq(AA_VOCAB["<EOS>"])
    )
    maskable = ~special

    ratio = torch.rand(B, 1, device=codon_targets.device, generator=generator)
    score = torch.rand(B, L, device=codon_targets.device, generator=generator)
    score = score.masked_fill(~maskable, 2.0)          # never select special positions

    n_maskable = maskable.sum(dim=1, keepdim=True)
    n_mask = (ratio * n_maskable.float()).ceil().clamp(min=1).long()

    order = score.argsort(dim=1)
    rank = order.argsort(dim=1)
    selected = (rank < n_mask) & maskable

    codon_inputs = codon_targets.masked_fill(selected, MASK_ID)
    return codon_inputs, selected


# ---------------------------------------------------------------------------
# Mask-predict inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def mask_predict_generate(
    model: MitoSeqGenIR,
    aa_tokens: torch.Tensor,
    n_iterations: int = 10,
    retrieved_codons: Optional[torch.Tensor] = None,
    retrieved_identity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Constrained mask-predict decoding. Returns (B, L) codon token ids.

    Round 0 predicts every position from an all-<MASK> hypothesis; each later
    round re-masks the lowest-confidence positions and re-predicts them with the
    rest of the sequence visible. Because every round is masked to the synonym
    class, the hypothesis is protein-exact at every intermediate step, not only
    at the end.
    """
    model.eval()
    B, L = aa_tokens.shape
    special = (
        aa_tokens.eq(AA_VOCAB["<PAD>"]) | aa_tokens.eq(AA_VOCAB["<BOS>"]) | aa_tokens.eq(AA_VOCAB["<EOS>"])
    )
    maskable = ~special
    n_maskable = maskable.sum(dim=1, keepdim=True).float()

    codon_inputs = torch.where(maskable, torch.full_like(aa_tokens, MASK_ID), aa_tokens)
    scores = torch.zeros(B, L, device=aa_tokens.device)

    for t in range(n_iterations):
        logits = model(aa_tokens, codon_inputs, retrieved_codons, retrieved_identity)
        probs = logits.softmax(dim=-1)
        conf, pred = probs.max(dim=-1)

        if t == 0:
            codon_inputs = torch.where(maskable, pred, codon_inputs)
            scores = torch.where(maskable, conf, torch.ones_like(conf))
        else:
            update = codon_inputs.eq(MASK_ID)
            codon_inputs = torch.where(update, pred, codon_inputs)
            scores = torch.where(update, conf, scores)

        if t == n_iterations - 1:
            break

        # linear decay of the number of positions re-masked for the next round
        n_mask = ((n_maskable * (n_iterations - 1 - t) / n_iterations).long()).clamp(min=0)
        if int(n_mask.max()) == 0:
            break
        masked_scores = scores.masked_fill(~maskable, 2.0)
        order = masked_scores.argsort(dim=1)
        rank = order.argsort(dim=1)
        to_mask = (rank < n_mask) & maskable
        codon_inputs = codon_inputs.masked_fill(to_mask, MASK_ID)

    return codon_inputs
