"""Constrained CDS generation for MitoSeqGen.

Given a protein sequence, autoregressively decodes a codon-optimized mRNA
CDS one position at a time, masking each step's logits down to the
synonymous-codon set for that position's amino acid (see constraints.py).
Because the target protein is fixed and known in advance, this task does
not need free-form generation with an unknown stop point — the sequence
length is exactly len(protein) + 1 (one codon per residue, plus the final
stop codon), so there is no risk of a runaway or truncated generation.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import AA_VOCAB, CODON_TOKENS, INV_VOCAB, VOCAB
from src.models.constraints import STOP_CODON_IDS, SYNONYM_TABLE
from src.models.transformer import MitoSeqTransformer

_CODON_GC_BASES = {VOCAB[c]: sum(1 for b in c if b in "GC") for c in CODON_TOKENS}


@torch.no_grad()
def generate_cds(
    model: MitoSeqTransformer,
    protein_sequence: str,
    device: torch.device,
    strategy: str = "greedy",
    temperature: float = 1.0,
    rng: Optional[torch.Generator] = None,
    cond: Optional[int] = None,
) -> str:
    """protein_sequence: raw amino-acid string (no BOS/EOS/stop marker).
    strategy: "greedy" (argmax, deterministic) or "sample" (temperature-scaled
    multinomial sampling among the allowed synonymous codons)."""
    model.eval()

    src_ids = [AA_VOCAB["<BOS>"]] + [AA_VOCAB.get(aa, AA_VOCAB["<UNK>"]) for aa in protein_sequence] + [AA_VOCAB["<EOS>"]]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)  # (seq_len, batch=1)
    cond_t = None
    if cond is not None and getattr(model, "n_conditions", 0) > 0:
        cond_t = torch.tensor([cond], dtype=torch.long, device=device)
    memory = model.encode(src, cond=cond_t)

    tgt_ids: List[int] = [VOCAB["<BOS>"]]
    for aa in protein_sequence:
        allowed = SYNONYM_TABLE.get(aa)
        if not allowed:
            allowed = [VOCAB["<UNK>"]]
        tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(1)
        decoded = model.decode(tgt, memory)
        logits = model.output_projection(decoded[-1, 0, :])  # last position, batch 0
        next_id = _pick(logits, allowed, strategy, temperature, rng)
        tgt_ids.append(next_id)

    # Final position: must be a valid mitochondrial stop codon.
    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(1)
    decoded = model.decode(tgt, memory)
    logits = model.output_projection(decoded[-1, 0, :])
    stop_id = _pick(logits, STOP_CODON_IDS, strategy, temperature, rng)
    tgt_ids.append(stop_id)

    codons = [INV_VOCAB[t] for t in tgt_ids[1:] if INV_VOCAB[t] not in ("<PAD>", "<BOS>", "<EOS>", "<UNK>")]
    return "".join(codons)


def _pick(logits: torch.Tensor, allowed_ids: List[int], strategy: str, temperature: float, rng) -> int:
    allowed_logits = logits[allowed_ids]
    if strategy == "greedy":
        idx = int(torch.argmax(allowed_logits).item())
    elif strategy == "sample":
        probs = torch.softmax(allowed_logits / max(temperature, 1e-6), dim=-1)
        idx = int(torch.multinomial(probs, 1, generator=rng).item())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return allowed_ids[idx]


@torch.no_grad()
def generate_cds_gc_guided(
    model: MitoSeqTransformer,
    protein_sequence: str,
    device: torch.device,
    target_gc: float = 0.44,
    beta: float = 1.0,
) -> str:
    """Same constrained decoding as generate_cds, but at each position scores
    each biologically-valid candidate codon by
        model_logprob(candidate) - beta * (running_gc_fraction_if_chosen - target_gc)**2
    instead of pure argmax on model log-probability alone, then picks the max.

    This exists because baking GC-content matching into the *training* loss
    was tried three times and measurably failed or backfired each time (see
    src/training/losses.py docstring) -- the core problem is that GC content
    is a *compositional* property of the whole finished sequence, which
    teacher-forced training can't faithfully represent (it only ever sees the
    true prefix, never what the model will actually generate). Two published
    codon-optimization systems (CodonRL; an evolutionary/GA-based mRNA
    designer) independently handle GC the same way this function does:
    as a scoring term applied during decoding/search on the actual sequence
    being built, not as a training-time gradient. beta=0 reduces exactly to
    generate_cds's greedy behavior; target_gc should be a population-level
    statistic (e.g. mean GC of natural training sequences), not a specific
    test example's true value, to avoid any test-time leakage."""
    model.eval()

    src_ids = [AA_VOCAB["<BOS>"]] + [AA_VOCAB.get(aa, AA_VOCAB["<UNK>"]) for aa in protein_sequence] + [AA_VOCAB["<EOS>"]]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)
    cond_t = None
    if cond is not None and getattr(model, "n_conditions", 0) > 0:
        cond_t = torch.tensor([cond], dtype=torch.long, device=device)
    memory = model.encode(src, cond=cond_t)

    tgt_ids: List[int] = [VOCAB["<BOS>"]]
    gc_bases_so_far = 0
    codons_so_far = 0

    def score_and_pick(logits: torch.Tensor, allowed_ids: List[int]) -> int:
        nonlocal gc_bases_so_far, codons_so_far
        log_probs = torch.log_softmax(logits[allowed_ids], dim=-1)
        best_score, best_id = None, allowed_ids[0]
        for i, cid in enumerate(allowed_ids):
            new_gc_bases = gc_bases_so_far + _CODON_GC_BASES.get(cid, 0)
            new_codons = codons_so_far + 1
            new_gc_frac = new_gc_bases / (3.0 * new_codons)
            score = log_probs[i].item() - beta * (new_gc_frac - target_gc) ** 2
            if best_score is None or score > best_score:
                best_score, best_id = score, cid
        gc_bases_so_far += _CODON_GC_BASES.get(best_id, 0)
        codons_so_far += 1
        return best_id

    for aa in protein_sequence:
        allowed = SYNONYM_TABLE.get(aa) or [VOCAB["<UNK>"]]
        tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(1)
        decoded = model.decode(tgt, memory)
        logits = model.output_projection(decoded[-1, 0, :])
        tgt_ids.append(score_and_pick(logits, allowed))

    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(1)
    decoded = model.decode(tgt, memory)
    logits = model.output_projection(decoded[-1, 0, :])
    tgt_ids.append(score_and_pick(logits, STOP_CODON_IDS))

    codons = [INV_VOCAB[t] for t in tgt_ids[1:] if INV_VOCAB[t] not in ("<PAD>", "<BOS>", "<EOS>", "<UNK>")]
    return "".join(codons)


def generate_batch(
    model: MitoSeqTransformer, protein_sequences: List[str], device: torch.device, strategy: str = "greedy", **kwargs
) -> List[str]:
    return [generate_cds(model, seq, device, strategy=strategy, **kwargs) for seq in protein_sequences]


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> MitoSeqTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]["model"]
    model = MitoSeqTransformer(
        src_vocab_size=len(AA_VOCAB),
        tgt_vocab_size=len(VOCAB),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_position_embeddings=config["max_position_embeddings"],
        n_conditions=config.get("n_conditions", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
