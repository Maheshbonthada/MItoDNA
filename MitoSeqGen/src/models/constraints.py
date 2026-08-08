"""Decode-time constraint enforcement for MitoSeqGen (Absolute Rule #9:
never generate a sequence with an internal mt-stop codon).

This task is not free-form text generation — the target protein is fixed
(given as input), so the model's real job at each position is choosing the
best *synonymous* codon for that position's amino acid, not choosing from
the full 68-token vocabulary. Restricting the softmax at each step to
exactly the codons that translate to the required amino acid under the
mitochondrial genetic code guarantees, by construction:
  1. The generated CDS translates to precisely the requested protein.
  2. No internal stop codon can ever be emitted (stop codons are never in
     any amino acid's synonym set).
  3. The final codon is drawn only from the 4 valid mitochondrial stop
     codons (TAA, TAG, AGA, AGG).
This is a stronger, more direct guarantee than post-hoc filtering/rejection
sampling on unconstrained model output.
"""

import sys
from pathlib import Path
from typing import Dict, List

import torch
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import AA_VOCAB, CODON_TOKENS, VOCAB
from src.genetic_codes import MITOCHONDRIAL_GENETIC_CODE

STOP_CODONS = {"TAA", "TAG", "AGA", "AGG"}
STOP_CODON_IDS = [VOCAB[c] for c in STOP_CODONS]


def build_synonym_table() -> Dict[str, List[int]]:
    """amino-acid character -> list of codon token ids that translate to it
    under the mitochondrial genetic code. Stop codons are never included in
    any amino acid's set, which is exactly what makes internal stops
    structurally impossible under this constraint scheme."""
    table: Dict[str, List[int]] = {aa: [] for aa in AA_VOCAB if len(aa) == 1}
    for codon in CODON_TOKENS:
        if codon in STOP_CODONS:
            continue
        aa = str(Seq(codon).translate(table=MITOCHONDRIAL_GENETIC_CODE))
        if aa in table:
            table[aa].append(VOCAB[codon])
    return table


SYNONYM_TABLE = build_synonym_table()


def build_synonym_mask_matrix(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """(len(AA_VOCAB), len(VOCAB)) boolean matrix — row aa_token_id gives the
    allowed codon token ids for that amino acid. Row for '*' (AA_VOCAB's stop
    marker) is set to the 4 valid mt stop codons."""
    mask = torch.zeros((len(AA_VOCAB), len(VOCAB)), dtype=torch.bool, device=device)
    for aa, aa_id in AA_VOCAB.items():
        if aa in SYNONYM_TABLE:
            for tok_id in SYNONYM_TABLE[aa]:
                mask[aa_id, tok_id] = True
    stop_aa_id = AA_VOCAB["*"]
    mask[stop_aa_id, :] = False
    for tok_id in STOP_CODON_IDS:
        mask[stop_aa_id, tok_id] = True
    return mask


def apply_constraint_mask(logits: torch.Tensor, allowed_token_ids: List[int]) -> torch.Tensor:
    """Sets logits of every token NOT in allowed_token_ids to -inf in place."""
    mask = torch.full_like(logits, float("-inf"))
    mask[..., allowed_token_ids] = 0.0
    return logits + mask
