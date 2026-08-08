"""Evaluation metrics for MitoSeqGen."""

import hashlib
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Set

import numpy as np
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.genetic_codes import MITOCHONDRIAL_GENETIC_CODE as MITO_GENETIC_CODE


def compute_mfe(cds_sequence: str) -> float:
    """Minimum Free Energy (kcal/mol) of the mRNA secondary structure via
    ViennaRNA. More negative = more stable secondary structure."""
    import RNA
    rna_seq = cds_sequence.upper().replace("T", "U")
    _, mfe = RNA.fold(rna_seq)
    return float(mfe)


def gc_content(cds_sequence: str) -> float:
    seq = cds_sequence.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def sequence_hash(cds_sequence: str) -> str:
    return hashlib.sha256(cds_sequence.upper().encode()).hexdigest()


def novel_sequence_rate(generated_sequences: Iterable[str], training_sequence_hashes: Set[str]) -> float:
    """% of generated sequences that are not exact copies of any training
    sequence — evidence of generalisation rather than memorisation."""
    generated = list(generated_sequences)
    if not generated:
        return 0.0
    novel = sum(1 for seq in generated if sequence_hash(seq) not in training_sequence_hashes)
    return novel / len(generated)


def compute_rscu_weights(records: list) -> dict:
    """Sharp & Li (1987)-style CAI weights: for each amino acid, the most-used
    synonymous codon gets weight 1.0, others get weight relative to it
    (count(codon) / max count among that amino acid's synonyms). This is
    distinct from raw global codon frequency (compute_codon_usage), which
    sums to 1.0 *across all 61 codons* and is not comparable across amino
    acids of different degeneracy — using it as CAI weights structurally
    caps the score near ~1/61 regardless of how good the codon choices are."""
    from src.models.constraints import SYNONYM_TABLE
    from src.data.preprocess import INV_VOCAB

    counts = Counter()
    for rec in records:
        seq = rec.get("sequence", "").upper()
        for i in range(0, len(seq) - 2, 3):
            counts[seq[i:i + 3]] += 1

    weights = {}
    for aa, codon_ids in SYNONYM_TABLE.items():
        codons = [INV_VOCAB[c] for c in codon_ids]
        aa_counts = {c: counts.get(c, 0) for c in codons}
        max_count = max(aa_counts.values()) if aa_counts else 0
        for c, n in aa_counts.items():
            weights[c] = (n / max_count) if max_count > 0 else (1.0 / len(codons))
    return weights


def _codon_to_aa_map() -> dict:
    """Precompute all 64 codon->amino-acid translations once. Calling
    Bio.Seq.translate() per-codon-occurrence inside a hot loop (as an
    earlier version of this function, and an earlier version of
    baselines.py, both did) is a real, measured performance bug --
    Seq.translate() has enough per-call overhead that doing it once per
    codon *occurrence* across thousands of sequences turns a sub-second
    computation into a multi-minute one. Only 64 codons exist, so
    precomputing is trivial and this map is reused across all calls."""
    return {
        codon: str(Seq(codon).translate(table=MITO_GENETIC_CODE))
        for codon in _CODON_LIST
    }


_CODON_TO_AA = None


def compute_mt_cai(cds_sequence: str, rscu_weights: dict) -> float:
    """rscu_weights: codon -> weight in (0, 1], from compute_rscu_weights()
    (NOT raw global frequency — see that function's docstring)."""
    global _CODON_TO_AA
    if _CODON_TO_AA is None:
        _CODON_TO_AA = _codon_to_aa_map()

    seq_upper = cds_sequence.upper()
    codons = [seq_upper[i:i+3] for i in range(0, len(seq_upper) - 3, 3)]
    log_sum = 0.0
    n = 0
    for codon in codons:
        aa = _CODON_TO_AA.get(codon)
        if aa is None or aa in ("*", "X") or codon not in rscu_weights:
            continue
        w = rscu_weights.get(codon, 0.0)
        if w > 0:
            log_sum += np.log(w)
            n += 1
    return float(np.exp(log_sum / n)) if n > 0 else 0.0


def translation_complies_with_mt_code(cds_sequence: str) -> bool:
    protein = str(Seq(cds_sequence.upper()).translate(table=MITO_GENETIC_CODE, to_stop=False))
    return "*" not in protein[:-1]


def codon_diversity(cds_sequence: str) -> float:
    seq_upper = cds_sequence.upper()
    codons = [seq_upper[i:i+3] for i in range(0, len(seq_upper), 3)]
    counts = Counter(codons)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return float(entropy)


def simple_bleu(reference: str, hypothesis: str, n: int = 4) -> float:
    ref_codons = [reference[i:i+3] for i in range(0, len(reference), 3)]
    hyp_codons = [hypothesis[i:i+3] for i in range(0, len(hypothesis), 3)]
    if not ref_codons or not hyp_codons:
        return 0.0
    precisions = []
    for k in range(1, n + 1):
        ref_ngrams = Counter(tuple(ref_codons[i:i + k]) for i in range(len(ref_codons) - k + 1))
        hyp_ngrams = Counter(tuple(hyp_codons[i:i + k]) for i in range(len(hyp_codons) - k + 1))
        matches = sum(min(count, hyp_ngrams[ng]) for ng, count in ref_ngrams.items())
        possible = max(len(hyp_codons) - k + 1, 0)
        precisions.append(matches / possible if possible > 0 else 0.0)
    log_precision = sum(math.log(max(p, 1e-8)) for p in precisions) / n
    geo_mean = float(math.exp(log_precision))
    bp = math.exp(1 - len(ref_codons) / len(hyp_codons)) if len(hyp_codons) < len(ref_codons) else 1.0
    return float(bp * geo_mean)


_CODON_LIST = [
    a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
]
_CODON_INDEX = {c: i for i, c in enumerate(_CODON_LIST)}


def _codons_of(seq: str) -> list:
    seq = seq.upper()
    return [seq[i:i + 3] for i in range(0, len(seq) - 2, 3) if len(seq[i:i + 3]) == 3]


def build_markov_transition_counts(sequences: Iterable[str]) -> np.ndarray:
    """First-order codon-to-codon transition COUNT matrix (64x64) pooled across
    all given sequences: counts[i, j] = number of times codon i was
    immediately followed by codon j. This captures local sequential/dicodon
    structure (codon-pair bias) that a purely compositional metric like GC
    content or overall codon-usage frequency cannot see -- two sequences with
    identical codon usage can still have very different transition
    structure. Codon-pair/dicodon bias is an established concept in
    molecular sequence analysis (e.g. codon-pair deoptimization in
    attenuated-vaccine design), which motivates using it here as an
    additional, independent similarity axis."""
    counts = np.zeros((64, 64), dtype=np.float64)
    for seq in sequences:
        codons = _codons_of(seq)
        for i in range(len(codons) - 1):
            a, b = codons[i], codons[i + 1]
            if a in _CODON_INDEX and b in _CODON_INDEX:
                counts[_CODON_INDEX[a], _CODON_INDEX[b]] += 1
    return counts


def markov_transition_matrix(counts: np.ndarray, smoothing: float = 1e-3) -> np.ndarray:
    """Row-normalize a transition COUNT matrix into a transition PROBABILITY
    matrix, with Laplace smoothing so no transition has exactly zero
    probability (needed for KL divergence to stay finite)."""
    smoothed = counts + smoothing
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def markov_transition_kl_divergence(generated_counts: np.ndarray, reference_probs: np.ndarray) -> float:
    """Stationary-frequency-weighted KL divergence D(generated || reference)
    between two first-order codon Markov chains: for each "from" codon state,
    weight that state's row-wise KL divergence by how often the *generated*
    sequences actually visit that state, then sum. This is the standard way
    to collapse a full transition-matrix comparison into a single scalar
    while still respecting that some codons are visited far more often than
    others."""
    row_totals = generated_counts.sum(axis=1)
    total = row_totals.sum()
    if total == 0:
        return 0.0
    state_weights = row_totals / total
    generated_probs = markov_transition_matrix(generated_counts)

    kl_per_row = np.sum(
        generated_probs * (np.log(generated_probs) - np.log(reference_probs)), axis=1
    )
    return float(np.sum(state_weights * kl_per_row))


def markov_transition_chi2_test(generated_counts: np.ndarray, reference_probs: np.ndarray):
    """Chi-squared goodness-of-fit test: are the *observed* codon-to-codon
    transition counts in the generated sequences consistent with the
    transition probabilities estimated from natural sequences? Rows (from-
    codon states) with fewer than 5 total observed transitions are excluded
    per the standard chi-squared minimum-expected-count convention. Returns
    (chi2_statistic, p_value, degrees_of_freedom, n_observations).

    CAVEAT (verified empirically, not theoretical): at the sample sizes here
    (hundreds of thousands of pooled codon-pair transitions across ~5,000
    sequences), this test's p-value is saturated and uninformative --
    chi2_p_value == 0.0 for *every* method we evaluated, including
    natural_reference itself (real, held-out natural sequences tested against
    a transition matrix built from other natural sequences). A p-value that
    rejects the null even for genuinely natural data carries no discriminating
    information at this N; it does not mean "no method is close to natural,"
    it means the test has enough power to detect biologically negligible
    deviations as significant. Cohen's w (see markov_transition_cohens_w),
    which is normalized by n and therefore comparable across methods with
    different total transition counts, is the metric that actually
    discriminates here and should be treated as primary; the raw chi2 p-value
    is reported for completeness only and should not be interpreted as
    evidence of anything on its own."""
    from scipy.stats import chisquare

    chi2_total, dof_total, n_total = 0.0, 0, 0
    for i in range(64):
        observed = generated_counts[i]
        row_total = observed.sum()
        if row_total < 5:
            continue
        expected = reference_probs[i] * row_total
        # drop reference-impossible columns (expected ~0) to avoid divide-by-near-zero blowup
        keep = expected > 1e-6
        if keep.sum() < 2:
            continue
        stat, _ = chisquare(observed[keep], f_exp=expected[keep])
        chi2_total += stat
        dof_total += keep.sum() - 1
        n_total += row_total

    if dof_total <= 0:
        return 0.0, 1.0, 0, 0
    from scipy.stats import chi2 as chi2_dist
    p_value = float(chi2_dist.sf(chi2_total, dof_total))
    return float(chi2_total), p_value, int(dof_total), float(n_total)


def markov_transition_cohens_w(chi2_statistic: float, n_observations: float) -> float:
    """Cohen's w = sqrt(chi2 / n): the standard sample-size-normalized effect
    size for a chi-squared goodness-of-fit test (the correct companion
    statistic here, analogous to how rank-biserial correlation accompanies
    the Wilcoxon p-value elsewhere in this codebase -- see
    significance.rank_biserial_effect_size). Unlike the raw chi-squared
    statistic or its p-value, w does not mechanically grow with sample size,
    so it is the metric that should actually be compared across methods.
    Conventional (Cohen 1988) interpretation bands: ~0.1 small, ~0.3 medium,
    ~0.5 large -- though with 64 possible from-states here the natural
    ceiling is higher than in a simple 2-category test, so these bands are a
    rough guide, not a hard threshold."""
    if n_observations <= 0:
        return 0.0
    return float(np.sqrt(chi2_statistic / n_observations))


def compute_codon_usage(records: list) -> dict:
    counts = Counter()
    for rec in records:
        seq = rec.get("sequence", "").upper()
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        counts.update(codons)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {codon: count / total for codon, count in counts.items()}


def aggregate_metrics(results: list) -> dict:
    if not results:
        return {
            "mean_mt_cai": 0.0,
            "compliance_rate": 0.0,
            "mean_codon_diversity": 0.0,
            "mean_bleu": 0.0,
            "mean_mfe": 0.0,
            "mean_gc_content": 0.0,
            "novel_sequence_rate": 0.0,
            "num_records": 0,
        }

    def _mean(key, default=0.0):
        vals = [res[key] for res in results if key in res and res[key] is not None]
        return float(np.mean(vals)) if vals else default

    return {
        "mean_mt_cai": _mean("mt_cai"),
        "compliance_rate": float(np.mean([1.0 if res.get("code_compliant") else 0.0 for res in results])),
        "mean_codon_diversity": _mean("codon_diversity"),
        "mean_bleu": _mean("bleu"),
        "mean_mfe": _mean("mfe"),
        "std_mfe": float(np.std([res["mfe"] for res in results if "mfe" in res])) if any("mfe" in r for r in results) else 0.0,
        "mean_mfe_delta_from_natural": _mean("mfe_delta_from_natural"),
        "mean_gc_content": _mean("gc_content"),
        "mean_gc_delta_from_natural": _mean("gc_delta_from_natural"),
        "novel_sequence_rate": _mean("is_novel"),
        "num_records": len(results),
    }
