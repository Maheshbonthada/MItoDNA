"""Replication-strand identity for vertebrate mitochondrial protein-coding genes.

In the vertebrate mitochondrial genome, 12 of the 13 protein-coding genes are
encoded on the heavy strand; ND6 alone is encoded on the light strand. The two
strands have opposite nucleotide composition, so ND6 shows the opposite GC skew
and the opposite codon bias to every other gene -- a long-established result in
mitochondrial genomics (Asakawa et al. 1991; Faith & Pollock 2003; Wei et al.
2010), not something this work discovers.

Measured on our own training corpus, mean GC skew (G-C)/(G+C):

    ATP8 -0.498  ND2 -0.451  ATP6 -0.400  ND5 -0.387  ND4L -0.361
    ND4  -0.353  ND1 -0.339  CYTB -0.335  ND3  -0.330  COX3 -0.270
    COX2 -0.250  COX1 -0.193                           ND6  +0.442

Every heavy-strand gene is negative; ND6 is positive. GC *content* does not
separate them (ND6 is 0.456, squarely inside the others' range), so metrics
based on GC content cannot detect this at all.

A model trained on the pooled 13 genes learns a single averaged strand prior and
applies it to ND6 as well, generating near-zero skew where nature gives +0.44.
These helpers supply strand identity as a one-bit conditioning signal so the
model does not have to infer it from the amino-acid sequence alone.
"""

from typing import Dict, List

# The sole light-strand protein-coding gene in the vertebrate mitochondrial genome.
LIGHT_STRAND_GENES = {"ND6"}

ALL_GENES: List[str] = [
    "ND1", "ND2", "COX1", "COX2", "ATP8", "ATP6", "COX3",
    "ND3", "ND4L", "ND4", "ND5", "ND6", "CYTB",
]

STRAND_HEAVY = 0
STRAND_LIGHT = 1
N_STRANDS = 2

GENE_INDEX: Dict[str, int] = {g: i for i, g in enumerate(ALL_GENES)}
N_GENES = len(ALL_GENES)


def strand_of(gene_name: str) -> int:
    """Gene symbol -> 0 (heavy) or 1 (light). Unknown symbols default to heavy,
    which is correct for 12 of the 13 genes."""
    return STRAND_LIGHT if gene_name in LIGHT_STRAND_GENES else STRAND_HEAVY


def gene_of(gene_name: str) -> int:
    """Gene symbol -> index in ALL_GENES; unknown symbols map to 0."""
    return GENE_INDEX.get(gene_name, 0)


def condition_of(gene_name: str, mode: str) -> int:
    if mode == "strand":
        return strand_of(gene_name)
    if mode == "gene":
        return gene_of(gene_name)
    return 0


def n_conditions(mode: str) -> int:
    return {"none": 0, "strand": N_STRANDS, "gene": N_GENES}[mode]
