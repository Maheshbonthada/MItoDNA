"""Required comparison baselines for MitoSeqGen (Phase 5/Absolute Rule #3:
never report a metric without its baseline). All four are built from the
same training corpus the model was trained on, so the comparison is fair
(no baseline gets to see more data than the model did).

Baseline 1 vs Baseline 3 are intentionally similar in mechanism (both are
deterministic single-best-codon lookup tables) — that similarity is itself
a fact about how classical codon optimization tools work, not an artifact
of this implementation. The literature distinction is normally in which
reference sequences the frequency table is built from: high-expression
genes (a real, published proxy for translational efficiency) vs a naive
global average. That's how they're differentiated here.

Baseline 4 (CodonTransformer) is a documented **approximation**, not the
real pretrained CodonTransformer model — running the actual model was out
of scope for this pass. It approximates "what a nuclear-genetic-code
optimizer would produce if naively applied to a mitochondrial protein":
a standard-genetic-code most-frequent-codon table, with the mitochondrial
code's 4 reassigned codons (UGA, AGA, AGG, AUA) remapped to their correct
mitochondrial meanings post-hoc, per the reviewer defense in Phase 7 Q4.
This limitation should be stated plainly in any results this baseline
appears in, not glossed over.
"""

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Bio.Seq import Seq

from src.genetic_codes import MITOCHONDRIAL_GENETIC_CODE, STANDARD_GENETIC_CODE
from src.models.constraints import STOP_CODONS, SYNONYM_TABLE

HIGH_EXPRESSION_GENES = {"COX1", "COX2", "COX3", "CYTB"}


def _codon_to_aa_map(table: int) -> Dict[str, str]:
    """All 64 codons translated once (cheap) rather than per-occurrence
    (there can be tens of millions of codon occurrences across the corpus —
    calling Bio.Seq.translate() per occurrence instead of per unique codon
    was the original, much slower approach)."""
    bases = "TCAG"
    codons = [a + b + c for a in bases for b in bases for c in bases]
    return {codon: str(Seq(codon).translate(table=table)) for codon in codons}


def _codon_counts(records: List[Dict], gene_filter=None, table: int = MITOCHONDRIAL_GENETIC_CODE) -> Dict[str, Counter]:
    codon_to_aa = _codon_to_aa_map(table)
    counts: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        if gene_filter is not None and rec.get("gene_name") not in gene_filter:
            continue
        seq = rec["sequence"].upper()
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            if codon in STOP_CODONS:
                continue
            aa = codon_to_aa.get(codon)
            if aa and aa not in ("*", "X"):
                counts[aa][codon] += 1
    return counts


def build_lookup_table(records: List[Dict], gene_filter=None, table: int = MITOCHONDRIAL_GENETIC_CODE) -> Dict[str, str]:
    """amino acid -> single most frequent codon, from the given record subset."""
    counts = _codon_counts(records, gene_filter=gene_filter, table=table)
    lookup = {}
    for aa, aa_codon_counts in counts.items():
        if aa_codon_counts:
            lookup[aa] = aa_codon_counts.most_common(1)[0][0]
    # Fallback for any amino acid unseen in this subset: use the first synonymous codon.
    for aa, codons in SYNONYM_TABLE.items():
        if aa not in lookup and codons:
            from src.data.preprocess import INV_VOCAB
            lookup[aa] = INV_VOCAB[codons[0]]
    return lookup


# NCBI standard genetic code reassignments vs mitochondrial: UGA (stop->Trp),
# AGA/AGG (Arg->stop), AUA (Ile->Met). A nuclear-code-trained model would
# never emit AGA/AGG for Arg-omitting-stop or use UGA for Trp; the "remap"
# baseline patches these specific positions after building the table.
_STANDARD_CODE_STOP_TRIPLETS = {"TAA", "TAG", "TGA"}


def build_standard_code_remap_table(records: List[Dict]) -> Dict[str, str]:
    """Approximates CodonTransformer-style nuclear-code optimization, then
    remaps the 4 codons whose meaning differs in the mitochondrial code."""
    counts = _codon_counts(records, gene_filter=None, table=STANDARD_GENETIC_CODE)
    lookup = {}
    for aa, aa_codon_counts in counts.items():
        if aa in ("*", "X"):
            continue
        # Exclude codons that are mitochondrial stops even if valid under the standard code.
        filtered = Counter({c: n for c, n in aa_codon_counts.items() if c not in STOP_CODONS})
        if filtered:
            lookup[aa] = filtered.most_common(1)[0][0]
    for aa, codons in SYNONYM_TABLE.items():
        if aa not in lookup and codons:
            from src.data.preprocess import INV_VOCAB
            lookup[aa] = INV_VOCAB[codons[0]]
    # AUA is standard-code Ile but mitochondrial Met; if the standard-code table
    # picked AUA for Ile, it would be biologically wrong under the mt code.
    if lookup.get("I") == "ATA":
        alt = [c for c in SYNONYM_TABLE.get("I", []) if c]
        from src.data.preprocess import INV_VOCAB
        alternatives = [INV_VOCAB[c] for c in SYNONYM_TABLE["I"] if INV_VOCAB[c] != "ATA"]
        if alternatives:
            lookup["I"] = alternatives[0]
    return lookup


def generate_from_lookup(protein_sequence: str, lookup: Dict[str, str], stop_codon: str = "TAA") -> str:
    codons = [lookup.get(aa, "UNK") for aa in protein_sequence]
    return "".join(codons) + stop_codon


def generate_random_synonymous(protein_sequence: str, rng: random.Random, stop_codon: str = None) -> str:
    from src.data.preprocess import INV_VOCAB
    codons = []
    for aa in protein_sequence:
        allowed = SYNONYM_TABLE.get(aa)
        if not allowed:
            codons.append("NNN")
            continue
        codons.append(INV_VOCAB[rng.choice(allowed)])
    stop = stop_codon or rng.choice(sorted(STOP_CODONS))
    return "".join(codons) + stop


class Baselines:
    """Builds all 4 baselines from a training-record set once, then generates."""

    def __init__(self, train_records: List[Dict], seed: int = 42):
        self.mt_cai_lookup = build_lookup_table(train_records, gene_filter=HIGH_EXPRESSION_GENES)
        self.most_frequent = build_lookup_table(train_records, gene_filter=None)
        self.codon_transformer_remap = build_standard_code_remap_table(train_records)
        self.rng = random.Random(seed)

    def generate(self, protein_sequence: str, baseline_name: str) -> str:
        if baseline_name == "mt_cai_lookup":
            return generate_from_lookup(protein_sequence, self.mt_cai_lookup)
        if baseline_name == "random_synonymous":
            return generate_random_synonymous(protein_sequence, self.rng)
        if baseline_name == "most_frequent_codon":
            return generate_from_lookup(protein_sequence, self.most_frequent)
        if baseline_name == "codontransformer_remap":
            return generate_from_lookup(protein_sequence, self.codon_transformer_remap)
        raise ValueError(f"Unknown baseline: {baseline_name}")

    def generate_all(self, protein_sequence: str) -> Dict[str, str]:
        return {name: self.generate(protein_sequence, name) for name in
                ["mt_cai_lookup", "random_synonymous", "most_frequent_codon", "codontransformer_remap"]}
