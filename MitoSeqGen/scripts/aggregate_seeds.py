"""Aggregate the seed-replication sweep into data/qc_reports/seed_replication.json.

Reviewer concern (TCBB review, major point 2): the species-level bootstrap
quantifies data-sampling variance only. A single run per condition cannot
support the claim that the strand bit reverses the pooled comparison, because
training variance is a separate and unmeasured source of uncertainty.

This script reads every available (condition, seed) evaluation and reports:

  - per-seed BLEU and paired difference against ortholog-copy;
  - the across-seed mean and standard deviation of that difference, which is
    the training-variance quantity the bootstrap does not capture;
  - a paired-by-seed comparison of strand against control, which is the
    cleanest statement of the effect if the seeds pair up.

It runs on whatever has finished, so it can be used to watch the sweep.

Usage:  python scripts/aggregate_seeds.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data/qc_reports"
OUT = QC / "seed_replication.json"

# seed 42 runs predate the sweep and use the original filenames
SEED42 = {
    "none": QC / "model_eval_ident70.json",
    "strand": QC / "model_eval_ident70_strand.json",
}


def load_runs():
    runs = {"none": {}, "strand": {}}
    for cond, path in SEED42.items():
        if path.exists():
            runs[cond][42] = path
    for p in sorted(QC.glob("model_eval_ident70_ident70_*_s*.json")):
        m = re.search(r"ident70_(none|strand)_s(\d+)\.json$", p.name)
        if m:
            runs[m.group(1)][int(m.group(2))] = p
    return runs


def summarise(path):
    d = json.load(open(path))
    comp = d["comparisons"]["mitoseqgen_vs_homology_copy"]
    return {
        "bleu": d["summary"]["mitoseqgen"]["bleu"]["mean"],
        "paired_diff": comp["mean_diff"],
        "ci_lo": comp["ci_lo"],
        "ci_hi": comp["ci_hi"],
        "win_rate": comp["frac_model_better"],
        "significant": comp["significant"],
    }


def main():
    runs = load_runs()
    out = {"per_run": {"none": {}, "strand": {}}, "summary": {}}

    for cond in ("none", "strand"):
        for seed, path in sorted(runs[cond].items()):
            out["per_run"][cond][str(seed)] = summarise(path)

    for cond in ("none", "strand"):
        rows = out["per_run"][cond]
        if not rows:
            print(f"[{cond}] no runs yet")
            continue
        diffs = np.array([r["paired_diff"] for r in rows.values()])
        bleus = np.array([r["bleu"] for r in rows.values()])
        out["summary"][cond] = {
            "n_seeds": len(diffs),
            "seeds": sorted(int(k) for k in rows),
            "bleu_mean": float(bleus.mean()),
            "bleu_sd": float(bleus.std(ddof=1)) if len(bleus) > 1 else None,
            "diff_mean": float(diffs.mean()),
            "diff_sd": float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
            "diff_min": float(diffs.min()),
            "diff_max": float(diffs.max()),
            "n_seeds_beating_copy": int((diffs > 0).sum()),
        }
        sd = out["summary"][cond]["diff_sd"]
        sd_s = f"{sd:.4f}" if sd is not None else "n/a"
        print(f"[{cond}] seeds={sorted(int(k) for k in rows)} "
              f"BLEU {bleus.mean():.4f} | diff vs copy {diffs.mean():+.4f} "
              f"(sd {sd_s}, range {diffs.min():+.4f}..{diffs.max():+.4f}) "
              f"| beats copy in {int((diffs > 0).sum())}/{len(diffs)} runs")

    # paired across the seeds that exist in both conditions
    shared = sorted(set(runs["none"]) & set(runs["strand"]))
    if len(shared) >= 2:
        c = np.array([out["per_run"]["none"][str(s)]["paired_diff"] for s in shared])
        t = np.array([out["per_run"]["strand"][str(s)]["paired_diff"] for s in shared])
        delta = t - c
        n = len(delta)
        sd = float(delta.std(ddof=1))
        se = sd / np.sqrt(n)
        out["summary"]["strand_minus_control"] = {
            "paired_seeds": shared,
            "n": n,
            "mean": float(delta.mean()),
            "sd": sd,
            "sem": float(se),
            "t": float(delta.mean() / se) if se > 0 else None,
            "all_positive": bool((delta > 0).all()),
        }
        print(f"\n[strand - control] paired over seeds {shared}: "
              f"{delta.mean():+.4f} (sd {sd:.4f}, sem {se:.4f}), "
              f"all positive: {(delta > 0).all()}")
    else:
        print(f"\n[strand - control] only {len(shared)} paired seed(s); "
              f"need >=2 for a across-seed statement")

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
