"""Phase 1: identity-stratified evaluation with species-level bootstrap CIs.

Replaces the old single-number comparison (and its pseudo-replicated
Wilcoxon p<1e-300) with:
  - per-protein results for EVERY method, joined on record index
  - stratification by max protein identity to the nearest same-gene
    TRAINING protein, so we can see where learning helps and where the task
    is just retrieval
  - species-level (cluster) bootstrap CIs, since 13 genes per species are
    not independent samples

Every table in the paper should be generated from the JSON this emits.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import random  # noqa: E402

N_BOOT = 2000
# Coarse strata from the workplan, plus finer high-identity bins because
# that is where essentially all the mass actually is.
STRATA = [
    ("<60%", 0.0, 0.60),
    ("60-80%", 0.60, 0.80),
    ("80-90%", 0.80, 0.90),
    ("90-95%", 0.90, 0.95),
    (">=95%", 0.95, 1.01),
]
METRICS = ["bleu", "gc_delta_from_natural", "mfe_delta_from_natural", "mt_cai", "codon_diversity"]


def species_bootstrap_ci(values, species, n_boot=N_BOOT, seed=0):
    """Mean + 95% CI resampling SPECIES (clusters), not sequences."""
    values = np.asarray(values, dtype=float)
    species = np.asarray(species)
    uniq = np.unique(species)
    groups = {s: np.where(species == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, len(uniq))
        idx = np.concatenate([groups[s] for s in pick])
        boots[b] = values[idx].mean()
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_species_bootstrap(a, b, species, n_boot=N_BOOT, seed=0):
    """CI on mean(a-b), resampling species. Significant iff CI excludes 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m, lo, hi = species_bootstrap_ci(d, species, n_boot, seed)
    return {"mean_diff": m, "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0),
            "frac_a_better": float((d > 0).mean())}


def main():
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    test_idx = json.load(open(ROOT / "data/splits/test.json"))["indices"]
    eval_order = random.Random(42).sample(test_idx, len(test_idx))  # order used by evaluate.py

    # ---- per-method, per-record metric dicts ----
    per_method = defaultdict(dict)  # method -> {record_idx: {metric: value}}

    rep = json.load(open(ROOT / "data/qc_reports/evaluation_report_full.json"))
    for method, rows in rep["raw_results"].items():
        assert len(rows) == len(eval_order), f"{method}: {len(rows)} != {len(eval_order)}"
        for ridx, row in zip(eval_order, rows):
            per_method[method][ridx] = {
                "bleu": row["bleu"],
                "gc_delta_from_natural": row.get("gc_delta_from_natural", 0.0),
                "mfe_delta_from_natural": row.get("mfe_delta_from_natural", 0.0),
                "mt_cai": row["mt_cai"],
                "codon_diversity": row["codon_diversity"],
            }

    # ---- homology-copy baseline (indexed by record idx already) ----
    retr = json.load(open(ROOT / "data/qc_reports/retrieval_test.json"))
    identity = {}
    for o in retr["per_sequence"]:
        identity[o["idx"]] = o["max_identity_to_train"]
        per_method["homology_copy"][o["idx"]] = {
            "bleu": o["bleu_homology_copy"],
            "gc_delta_from_natural": np.nan,
            "mfe_delta_from_natural": np.nan,
            "mt_cai": np.nan,
            "codon_diversity": np.nan,
        }

    # ---- CodonTransformer pretrained / fine-tuned (join on record id) ----
    id_to_idx = {recs[i]["id"]: i for i in test_idx}
    for name, fname in [("codontransformer_pretrained", "codontransformer_real_results_full.json"),
                        ("codontransformer_finetuned", "codontransformer_finetuned_results.json")]:
        rows = json.load(open(ROOT / f"data/qc_reports/{fname}"))
        for row in rows:
            ridx = id_to_idx.get(row["id"])
            if ridx is None:
                continue
            per_method[name][ridx] = {
                "bleu": row["bleu"],
                "gc_delta_from_natural": np.nan,
                "mfe_delta_from_natural": np.nan,
                "mt_cai": row["mt_cai"],
                "codon_diversity": row["codon_diversity"],
            }
    # fine-tuned deltas live in a separate file, in the same order as its results file
    try:
        deltas = json.load(open(ROOT / "data/qc_reports/codontransformer_finetuned_deltas.json"))
        ft_rows = json.load(open(ROOT / "data/qc_reports/codontransformer_finetuned_results.json"))
        for row, gcd, mfed in zip(ft_rows, deltas["gc_deltas"], deltas["mfe_deltas"]):
            ridx = id_to_idx.get(row["id"])
            if ridx is not None:
                per_method["codontransformer_finetuned"][ridx]["gc_delta_from_natural"] = gcd
                per_method["codontransformer_finetuned"][ridx]["mfe_delta_from_natural"] = mfed
    except (FileNotFoundError, KeyError) as e:
        print(f"WARN: fine-tuned deltas unavailable ({e})")

    methods = list(per_method.keys())
    common = set.intersection(*[set(per_method[m].keys()) for m in methods])
    common = sorted(common)
    print(f"methods: {methods}")
    print(f"records with results for every method: {len(common)} / {len(test_idx)}")

    species = np.array([recs[i]["species"] for i in common])
    ident = np.array([identity[i] for i in common])

    # ---- identity distribution ----
    out = {
        "n_records": len(common),
        "n_species": int(len(np.unique(species))),
        "identity_distribution": {
            "mean": float(ident.mean()), "median": float(np.median(ident)),
            "p05": float(np.percentile(ident, 5)), "p95": float(np.percentile(ident, 95)),
        },
        "strata": {}, "overall": {}, "headline_comparisons": {},
    }

    # ---- overall per-method, per-metric with species bootstrap ----
    for m in methods:
        out["overall"][m] = {}
        for metric in METRICS:
            v = np.array([per_method[m][i][metric] for i in common], dtype=float)
            if np.isnan(v).all():
                continue
            mask = ~np.isnan(v)
            mean, lo, hi = species_bootstrap_ci(v[mask], species[mask])
            out["overall"][m][metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(mask.sum())}

    # ---- stratified ----
    for label, lo_b, hi_b in STRATA:
        sel = np.where((ident >= lo_b) & (ident < hi_b))[0]
        entry = {"n": int(len(sel)), "n_species": int(len(np.unique(species[sel]))) if len(sel) else 0,
                 "identity_range": [lo_b, hi_b], "methods": {}}
        if len(sel) >= 2 and len(np.unique(species[sel])) >= 2:
            for m in methods:
                entry["methods"][m] = {}
                for metric in METRICS:
                    v = np.array([per_method[m][common[i]][metric] for i in sel], dtype=float)
                    mask = ~np.isnan(v)
                    if mask.sum() < 2:
                        continue
                    mean, clo, chi = species_bootstrap_ci(v[mask], species[sel][mask])
                    entry["methods"][m][metric] = {"mean": mean, "ci_lo": clo, "ci_hi": chi, "n": int(mask.sum())}
        else:
            entry["note"] = "too few records/species for bootstrap"
        out["strata"][label] = entry

    # ---- headline paired comparisons on BLEU ----
    def bleu_vec(m, sub=None):
        ix = common if sub is None else [common[i] for i in sub]
        return np.array([per_method[m][i]["bleu"] for i in ix], dtype=float)

    pairs = [("mitoseqgen", "homology_copy"),
             ("mitoseqgen", "codontransformer_finetuned"),
             ("mitoseqgen", "codontransformer_remap"),
             ("homology_copy", "codontransformer_finetuned")]
    for a, b in pairs:
        out["headline_comparisons"][f"{a}_vs_{b}__overall"] = paired_species_bootstrap(
            bleu_vec(a), bleu_vec(b), species)
        for label, lo_b, hi_b in STRATA:
            sel = np.where((ident >= lo_b) & (ident < hi_b))[0]
            if len(sel) < 2 or len(np.unique(species[sel])) < 2:
                continue
            out["headline_comparisons"][f"{a}_vs_{b}__{label}"] = paired_species_bootstrap(
                bleu_vec(a, sel), bleu_vec(b, sel), species[sel])

    json.dump(out, open(ROOT / "data/qc_reports/stratified_evaluation.json", "w"), indent=2)

    # ---- console report ----
    print(f"\nidentity: mean {ident.mean():.3f} median {np.median(ident):.3f}")
    print("\n=== BLEU-4 by identity stratum (mean [95% CI], species bootstrap) ===")
    show = ["homology_copy", "mitoseqgen", "codontransformer_finetuned", "codontransformer_remap",
            "mt_cai_lookup", "most_frequent_codon", "random_synonymous"]
    hdr = f"{'stratum':<10}{'n':>6}{'nsp':>5}  " + "".join(f"{m[:18]:>20}" for m in show)
    print(hdr)
    for label, _, _ in STRATA:
        e = out["strata"][label]
        if not e["methods"]:
            print(f"{label:<10}{e['n']:>6}{e['n_species']:>5}  (too few)")
            continue
        row = f"{label:<10}{e['n']:>6}{e['n_species']:>5}  "
        for m in show:
            d = e["methods"].get(m, {}).get("bleu")
            row += f"{d['mean']:>20.3f}" if d else f"{'-':>20}"
        print(row)
    row = f"{'OVERALL':<10}{len(common):>6}{out['n_species']:>5}  "
    for m in show:
        d = out["overall"].get(m, {}).get("bleu")
        row += f"{d['mean']:>20.3f}" if d else f"{'-':>20}"
    print(row)

    print("\n=== headline paired comparisons (BLEU, species bootstrap) ===")
    for k, v in out["headline_comparisons"].items():
        flag = "SIG" if v["significant"] else "ns "
        print(f"  {flag} {k:<55} diff {v['mean_diff']:+.4f}  CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]  A better on {v['frac_a_better']*100:.0f}%")

    print("\nwrote data/qc_reports/stratified_evaluation.json")


if __name__ == "__main__":
    main()
