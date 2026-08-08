"""Multi-objective loss functions for MitoSeqGen (Phase 5).

Primary loss (cross-entropy) is defined in train.py directly. This module
holds the two auxiliary losses that can be made properly differentiable:

  - GC-deviation loss: per-sequence, regresses the model's expected GC
    fraction toward that specific training example's true GC fraction (not
    a fixed population-level band — see gc_deviation_loss docstring for why
    the band-based version was replaced after producing near-zero gradient
    for an entire training run).
  - Soft mt-CAI loss: within the *true* target amino acid's synonym class
    only, nudges probability mass toward higher-usage codons. Evaluation
    showed this works exactly as designed (mt-CAI significantly increases,
    p=0.0008) but moves *away* from natural sequences, which average a
    lower mt-CAI (0.74) than any codon-optimization baseline (0.87-1.0) —
    real mitochondrial genes are not simply optimizing translational
    codon-usage bias. Since this project's evaluation defines success as
    closeness to natural sequences, lambda_cai is set to 0.0 in config for
    later runs; the loss itself is left implemented/available rather than
    deleted in case a future experiment wants classic CAI-maximizing
    optimization (e.g. for heterologous expression) as a distinct objective.

A third auxiliary term the original spec calls for — an MFE/secondary-
structure penalty via ViennaRNA — is NOT included here. MFE is computed by
a discrete dynamic-programming algorithm with no closed-form gradient
w.r.t. codon choice, so using it as a training signal would require a
REINFORCE/policy-gradient setup (sampling discrete sequences, scoring them,
variance-reduced gradient estimation) rather than a simple differentiable
term. That is real, separate scope — see train.py / project notes for the
decision on whether to add it.

Both losses here are computed under teacher forcing: they read the
*predicted* distribution at each position but use the *ground-truth*
target codon only to know which amino acid (and therefore which synonym
class) that position is supposed to encode — they do not need an actual
decoded sequence.
"""

import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import VOCAB, CODON_TOKENS
from src.models.constraints import build_synonym_mask_matrix, SYNONYM_TABLE
from src.genetic_codes import MITOCHONDRIAL_GENETIC_CODE

def _build_codon_gc_counts(device: torch.device) -> torch.Tensor:
    """(vocab_size,) tensor: number of G/C bases in each vocab token's codon
    (0 for special/non-codon tokens)."""
    counts = torch.zeros(len(VOCAB), device=device)
    for codon in CODON_TOKENS:
        counts[VOCAB[codon]] = sum(1 for b in codon if b in "GC")
    return counts


def _build_target_to_synonym_class(device: torch.device) -> torch.Tensor:
    """(vocab_size, vocab_size) boolean: row t gives the synonym-class mask
    to use when the ground-truth target at that position is token t (i.e.
    "which tokens are valid alternatives given the true amino acid here").
    Non-codon tokens (PAD/BOS/EOS/UNK) get an all-False row — those
    positions are excluded from the soft-CAI loss entirely."""
    from Bio.Seq import Seq
    mask = torch.zeros((len(VOCAB), len(VOCAB)), dtype=torch.bool, device=device)
    for codon in CODON_TOKENS:
        codon_id = VOCAB[codon]
        aa = str(Seq(codon).translate(table=MITOCHONDRIAL_GENETIC_CODE))
        if aa in SYNONYM_TABLE:
            for syn_id in SYNONYM_TABLE[aa]:
                mask[codon_id, syn_id] = True
    return mask


class AuxiliaryLosses:
    """Precomputes the constant lookup tensors once per device, then exposes
    the two differentiable auxiliary loss terms for use in the training loop."""

    def __init__(self, train_records, device: torch.device):
        self.device = device
        self.codon_gc_counts = _build_codon_gc_counts(device)
        self.target_to_synonym_class = _build_target_to_synonym_class(device)
        self.rscu_weights = self._build_rscu_weight_tensor(train_records, device)

    def _build_rscu_weight_tensor(self, train_records, device: torch.device) -> torch.Tensor:
        from src.evaluation.metrics import compute_rscu_weights
        weights_by_codon = compute_rscu_weights(train_records)
        weights = torch.full((len(VOCAB),), 1e-6, device=device)
        for codon, w in weights_by_codon.items():
            weights[VOCAB[codon]] = max(w, 1e-6)
        return weights

    def gc_deviation_loss(self, logits: torch.Tensor, targets: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """logits: (seq_len, batch, vocab_size). targets: (seq_len, batch) ground-truth
        token ids. non_pad_mask: (seq_len, batch) bool, True at real (non-pad) positions.

        Per-position, restricted to the true target's synonym class (same masking
        pattern as soft_mt_cai_loss below): computes the model's expected GC-base
        count for that position's codon choice *among only the biologically valid
        alternatives for the true amino acid*, and penalizes its (L1) distance from
        the true codon's own GC-base count.

        Two earlier designs were tried and measurably failed before this one:
          1. A batch-pooled version checked against a fixed [0.40, 0.47] band —
             logged mean_gc_loss ~= 1e-5 (effectively zero) for 28 straight epochs
             because the batch-average landed inside the band almost every step
             even though individual sequences varied well outside it.
          2. A per-sequence version (aggregate expected vs. true GC fraction over
             the whole sequence) fixed #1's near-zero-gradient problem — the loss
             genuinely dropped over training (0.0148->0.0027 across 26 epochs) —
             but computed its softmax over the *full* vocabulary, unrestricted by
             synonym class. That let the loss reduce itself by shifting probability
             onto wrong-amino-acid codons with favorable GC content — tokens the
             constrained decoder can never actually select — while the resulting
             competing gradient degraded the real, in-class distribution CE depends
             on. Measured result: GC-deviation-from-natural got *worse*
             (0.0419->0.0564, p=3e-10 vs. the CE-only baseline), BLEU got *worse*
             (0.3132->0.2809, p=1e-11), codon diversity got worse (p=1e-9) — the
             "fix" actively hurt every metric it touched despite the loss curve
             looking healthy during training.

        Restricting to the synonym class first (this version) guarantees any
        gradient this loss produces can only shift preference *among* codons the
        decoder could actually choose at that position — it cannot "cheat" by
        moving probability mass onto biologically invalid alternatives."""
        seq_len, batch = targets.shape
        flat_targets = targets.reshape(-1)
        class_mask = self.target_to_synonym_class[flat_targets]  # (seq_len*batch, vocab_size)
        valid_position = class_mask.any(dim=-1)  # False for special-token positions

        flat_logits = logits.reshape(-1, logits.shape[-1])
        masked_logits = flat_logits.masked_fill(~class_mask, float("-inf"))
        safe_logits = torch.where(class_mask, masked_logits, torch.zeros_like(masked_logits))
        probs = torch.softmax(safe_logits, dim=-1)

        expected_gc = (probs * self.codon_gc_counts.unsqueeze(0)).sum(dim=-1)  # (seq_len*batch,)
        true_gc = self.codon_gc_counts[flat_targets]  # (seq_len*batch,)

        mask = (non_pad_mask.reshape(-1).float()) * valid_position.float()
        denom = mask.sum().clamp_min(1.0)
        return ((expected_gc - true_gc).abs() * mask).sum() / denom

    def soft_mt_cai_loss(self, logits: torch.Tensor, targets: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """logits: (seq_len, batch, vocab_size). targets: (seq_len, batch) —
        ground-truth token ids (teacher forcing), used only to select each
        position's synonym class. Encourages probability mass toward
        higher-RSCU-weight codons *within* that class, never away from the
        correct amino acid (the class itself is defined by the true target)."""
        seq_len, batch = targets.shape
        flat_targets = targets.reshape(-1)
        class_mask = self.target_to_synonym_class[flat_targets]  # (seq_len*batch, vocab_size)
        valid_position = class_mask.any(dim=-1)  # False for special-token positions

        flat_logits = logits.reshape(-1, logits.shape[-1])
        masked_logits = flat_logits.masked_fill(~class_mask, float("-inf"))
        # Positions with no valid class (all -inf) would NaN in softmax; guard them.
        safe_logits = torch.where(class_mask, masked_logits, torch.zeros_like(masked_logits))
        probs = torch.softmax(safe_logits, dim=-1)

        expected_weight = (probs * self.rscu_weights.unsqueeze(0)).sum(dim=-1)  # (seq_len*batch,)
        loss_per_position = -torch.log(expected_weight.clamp_min(1e-8))

        mask = (non_pad_mask.reshape(-1).float()) * valid_position.float()
        denom = mask.sum().clamp_min(1.0)
        return (loss_per_position * mask).sum() / denom


def combined_loss(
    ce_loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    non_pad_mask: torch.Tensor,
    aux: AuxiliaryLosses,
    lambda_gc: float,
    lambda_cai: float,
) -> Dict[str, torch.Tensor]:
    gc_loss = aux.gc_deviation_loss(logits, targets, non_pad_mask)
    cai_loss = aux.soft_mt_cai_loss(logits, targets, non_pad_mask)
    total = ce_loss + lambda_gc * gc_loss + lambda_cai * cai_loss
    return {"total": total, "ce": ce_loss, "gc": gc_loss, "cai": cai_loss}
