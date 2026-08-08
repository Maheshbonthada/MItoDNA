"""Statistical significance testing for MitoSeqGen evaluation results
(Phase 7 Q5): paired Wilcoxon signed-rank test per (baseline, metric) pair
— appropriate here because every baseline is evaluated on the exact same
test-set proteins as the model, making each comparison naturally paired and
non-parametric (CAI/MFE scores are not normally distributed).

Two multiple-comparison corrections are reported side by side, since they
answer different questions and reviewers may expect either:
  - Bonferroni: controls family-wise error rate (probability of *any* false
    positive across all comparisons) — conservative, appropriate when a
    single false positive would be costly to the paper's claims.
  - Benjamini-Hochberg (FDR): controls the expected *proportion* of false
    positives among comparisons called significant — standard practice in
    genomics/bioinformatics multiple-testing contexts (e.g. differential
    expression), and less conservative than Bonferroni when many
    comparisons are run together, as here (5 metrics x 4-5 baselines).
Both corrections are applied to the same family: all (baseline, metric)
comparisons in a single report, not just within one baseline.
"""

from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

COMPARISON_METRICS = ["mt_cai", "bleu", "codon_diversity", "mfe_delta_from_natural", "gc_delta_from_natural"]


def rank_biserial_effect_size(model_vals: np.ndarray, baseline_vals: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the standard effect-size
    companion to the Wilcoxon signed-rank test (a p-value alone says
    whether a difference is unlikely to be noise, not how large it is).
    Ranges [-1, 1]: computed as (W+ - W-) / (W+ + W-), the normalized
    difference between the summed positive-difference ranks and summed
    negative-difference ranks. +1 means every paired difference favored the
    model, -1 means every one favored the baseline."""
    diff = model_vals - baseline_vals
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    pos_sum = ranks[nonzero > 0].sum()
    neg_sum = ranks[nonzero < 0].sum()
    return float((pos_sum - neg_sum) / (pos_sum + neg_sum))

# For each metric, whether a *higher* value favors the model (True) or a
# *lower* value does (False) — needed to report direction, not just p-value.
HIGHER_IS_BETTER = {
    "mt_cai": None,  # ambiguous on its own (see mt-CAI caveat in report) — report direction without a "better" label
    "bleu": True,
    "codon_diversity": None,  # ambiguous — closeness to natural's diversity matters more than raw magnitude
    "mfe_delta_from_natural": False,
    "gc_delta_from_natural": False,
}


def run_significance_tests(raw_results: Dict[str, List[dict]], model_name: str = "mitoseqgen") -> Dict:
    baseline_names = [name for name in raw_results if name not in (model_name, "natural_reference")]
    model_results = raw_results[model_name]

    comparisons = []
    for baseline in baseline_names:
        baseline_results = raw_results[baseline]
        for metric in COMPARISON_METRICS:
            model_vals = np.array([r[metric] for r in model_results if metric in r])
            baseline_vals = np.array([r[metric] for r in baseline_results if metric in r])
            n = min(len(model_vals), len(baseline_vals))
            model_vals, baseline_vals = model_vals[:n], baseline_vals[:n]

            diff = model_vals - baseline_vals
            if np.allclose(diff, 0):
                comparisons.append({
                    "baseline": baseline, "metric": metric, "n": n,
                    "model_mean": float(np.mean(model_vals)), "baseline_mean": float(np.mean(baseline_vals)),
                    "p_value": 1.0, "statistic": 0.0, "identical": True, "rank_biserial_effect_size": 0.0,
                })
                continue
            try:
                stat, p = wilcoxon(model_vals, baseline_vals)
            except ValueError:
                # all-zero differences after ties correction, etc.
                stat, p = 0.0, 1.0
            comparisons.append({
                "baseline": baseline, "metric": metric, "n": n,
                "model_mean": float(np.mean(model_vals)), "baseline_mean": float(np.mean(baseline_vals)),
                "p_value": float(p), "statistic": float(stat), "identical": False,
                "rank_biserial_effect_size": rank_biserial_effect_size(model_vals, baseline_vals),
            })

    m = len(comparisons)
    alpha = 0.05
    bonferroni_alpha = alpha / m if m > 0 else alpha

    p_values = [c["p_value"] for c in comparisons]
    if m > 0:
        fdr_reject, fdr_qvalues, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    else:
        fdr_reject, fdr_qvalues = [], []

    for c, q, rej in zip(comparisons, fdr_qvalues, fdr_reject):
        c["bonferroni_alpha"] = bonferroni_alpha
        c["significant_after_bonferroni"] = c["p_value"] < bonferroni_alpha
        c["fdr_qvalue"] = float(q)
        c["significant_after_fdr"] = bool(rej)
        # kept for backward compatibility with existing callers/UI that read this key
        c["significant_after_correction"] = c["significant_after_bonferroni"]

    return {
        "num_comparisons": m,
        "family_wise_alpha": alpha,
        "bonferroni_corrected_alpha": bonferroni_alpha,
        "fdr_method": "benjamini_hochberg",
        "comparisons": comparisons,
    }


def print_significance_summary(sig_results: Dict, model_name: str = "mitoseqgen"):
    print(f"\nWilcoxon signed-rank tests: {model_name} vs. each baseline")
    print(f"Bonferroni-corrected alpha = {sig_results['bonferroni_corrected_alpha']:.5f} "
          f"(family-wise 0.05 / {sig_results['num_comparisons']} comparisons); "
          f"Benjamini-Hochberg FDR also reported (q-value, alpha=0.05)\n")
    header = (f"{'baseline':24} {'metric':24} {'model_mean':>12} {'baseline_mean':>14} "
              f"{'p_value':>10} {'bonferroni':>11} {'fdr_qval':>9} {'fdr_sig':>8}")
    print(header)
    print("-" * len(header))
    for c in sig_results["comparisons"]:
        bonf = "yes" if c["significant_after_bonferroni"] else "no"
        fdr = "yes" if c["significant_after_fdr"] else "no"
        print(f"{c['baseline']:24} {c['metric']:24} {c['model_mean']:12.4f} {c['baseline_mean']:14.4f} "
              f"{c['p_value']:10.2e} {bonf:>11} {c['fdr_qvalue']:9.2e} {fdr:>8}")
