"""Evaluate any seed-sweep run whose training finished but whose evaluation is missing.

Why this exists: the eval step inside scripts/run_seed_sweep.sh passed
"data/splits/<file>" to scripts/evaluate_model_split.py, which prepends
"data/splits/" itself, so every eval in the sweep fails with FileNotFoundError
on a doubled path. The training half of the sweep is unaffected, so rather than
edit a script bash is still reading -- which risks corrupting a long-running
job -- this script does the evaluation pass separately and correctly.

Run it after the sweep finishes, or at any time: runs whose training is still
in progress are skipped automatically.

A run counts as "still training" if its metrics.json was modified within
--min-idle-min minutes. One epoch takes about 17.5 minutes on this hardware, so
the default of 25 is comfortably longer than an epoch.

Usage:
  python scripts/evaluate_pending_seeds.py                 # evaluate what is ready
  python scripts/evaluate_pending_seeds.py --dry-run       # just report
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data/qc_reports"
EXP = ROOT / "experiments"

TEST_SPLIT = "test_ident70_per_gene.json"
TRAIN_SPLIT = "train_ident70_per_gene.json"


def find_runs():
    out = []
    for d in sorted(EXP.glob("*_ident70_*_s[0-9]")):
        m = re.search(r"_ident70_(none|strand)_s(\d+)$", d.name)
        if not m:
            continue
        out.append((d, m.group(1), int(m.group(2))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-idle-min", type=float, default=25.0,
                    help="skip runs whose metrics.json changed more recently than this")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = find_runs()
    if not runs:
        print("no seed-sweep runs found")
        return

    now = time.time()
    done = pending = skipped = failed = 0

    for d, cond, seed in runs:
        tag = f"ident70_{cond}_s{seed}"
        out_json = QC / f"model_eval_ident70_{tag}.json"
        ckpt = d / "checkpoints" / "best.pt"
        metrics = d / "metrics.json"

        if out_json.exists():
            print(f"[done ] {tag}")
            done += 1
            continue
        if not ckpt.exists():
            print(f"[wait ] {tag}: no checkpoint yet")
            skipped += 1
            continue
        if metrics.exists():
            idle_min = (now - metrics.stat().st_mtime) / 60.0
            if idle_min < args.min_idle_min:
                n_ep = len(json.load(open(metrics)))
                print(f"[train] {tag}: still training "
                      f"(epoch {n_ep}, last write {idle_min:.0f} min ago)")
                skipped += 1
                continue

        if args.dry_run:
            print(f"[ready] {tag}: would evaluate")
            pending += 1
            continue

        print(f"[eval ] {tag} ...", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts/evaluate_model_split.py"),
               "--checkpoint", str(ckpt),
               "--tag", "ident70",
               "--test-split", TEST_SPLIT,
               "--train-split", TRAIN_SPLIT,
               "--condition", cond,
               "--workers", str(args.workers),
               "--out-suffix", f"_{tag}"]
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode == 0 and out_json.exists():
            res = json.load(open(out_json))
            c = res["comparisons"]["mitoseqgen_vs_homology_copy"]
            print(f"[ok   ] {tag}: BLEU {res['summary']['mitoseqgen']['bleu']['mean']:.4f}, "
                  f"vs copy {c['mean_diff']:+.4f}")
            pending += 1
        else:
            print(f"[FAIL ] {tag}: exit {r.returncode}")
            failed += 1

    print(f"\nalready evaluated: {done} | evaluated now: {pending} | "
          f"not ready: {skipped} | failed: {failed}")
    if done + pending:
        print("\nrun `python scripts/aggregate_seeds.py` to summarise")


if __name__ == "__main__":
    main()
