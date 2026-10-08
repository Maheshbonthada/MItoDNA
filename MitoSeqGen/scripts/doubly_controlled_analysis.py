"""Species-disjoint AND identity-controlled evaluation.

Reviewer concern (TCBB review, secondary point 3): the identity-controlled
split is constructed per gene, so a test species can appear in training under
a different gene. Within a species, genes share strand regime, mutational bias
and codon pool, so per-gene identity control does not close that channel.

scripts/species_overlap_analysis.py measures the channel directly and finds it
is almost total: 971 of 980 test species (99.1%) appear in training for some
other gene, leaving only 9 sequences in the "wholly absent species" subset.
The reviewer's suggested subset test is therefore not estimable on this corpus,
and we say so rather than reporting a 9-point comparison.

This script runs the construction that *is* available. The original
species-level split is species-disjoint by construction. Intersecting it with
an identity bound gives a test set that is simultaneously:

  (a) species-disjoint  -- no test species appears anywhere in training, for
      any gene, so the cross-gene channel is closed completely; and
  (b) identity-bounded  -- no test protein exceeds the bound against any
      same-gene training protein.

The cost is sample size: n=92 at a 0.70 bound, n=246 at 0.80. We report both
with species-level bootstrap intervals, and treat them as a consistency check
on the headline claim rather than as a primary result.

Usage:  python scripts/doubly_controlled_analysis.py
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QC = ROOT / "data/qc_reports"
N_BOOT = 10000
RNG_SEED = 20261008
BOUNDS = [0.70, 0.80, 0.90]


def paired_species_bootstrap(diffs, species, n_boot=N_BOOT, seed=RNG_SEED):
    by_sp = defaultdict(list)
    for d, s in zip(diffs, species):
        by_sp[s].append(d)
    keys = list(by_sp)
    arrs = [np.array(by_sp[k]) for k in keys]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        means[b] = np.concatenate([arrs[i] for i in pick]).mean()
    return float(np.mean(diffs)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    train_idx = json.load(open(ROOT / "data/splits/train.json"))["indices"]
    test_idx = json.load(open(ROOT / "data/splits/test.json"))["indices"]
    eval_order = random.Random(42).sample(test_idx, len(test_idx))  # order used by evaluate.py

    train_species = {recs[i]["species"] for i in train_idx if i < len(recs)}
    test_species = {recs[i]["species"] for i in test_idx}
    overlap = test_species & train_species
    print(f"original split: {len(test_species)} test species, "
          f"{len(overlap)} also in training  -> species-disjoint: {not overlap}")

    # ---- model BLEU per record ----
    rep = json.load(open(QC / "evaluation_report_full.json"))
    model_bleu = {}
    for ridx, row in zip(eval_order, rep["raw_results"]["mitoseqgen"]):
        model_bleu[ridx] = row["bleu"]

    # ---- copy baseline + identity per record ----
    retr = json.load(open(QC / "retrieval_orig_test.json"))["per_sequence"]
    copy_bleu, identity = {}, {}
    for o in retr:
        copy_bleu[o["idx"]] = o["bleu_homology_copy"]
        identity[o["idx"]] = o["max_identity_to_train"]

    usable = [i for i in test_idx if i in model_bleu and i in copy_bleu]
    print(f"records with both model and copy scores: {len(usable)}")

    out = {
        "original_split_species_disjoint": not overlap,
        "n_test_species": len(test_species),
        "n_species_overlap_with_train": len(overlap),
        "n_usable": len(usable),
        "bounds": {},
    }

    for bound in BOUNDS:
        sub = [i for i in usable if identity[i] <= bound]
        if len(sub) < 5:
            print(f"\nbound <={bound:.2f}: n={len(sub)} -- too few to estimate")
            continue
        sp = [recs[i]["species"] for i in sub]
        genes = Counter(recs[i]["gene_name"] for i in sub)
        m = np.array([model_bleu[i] for i in sub])
        c = np.array([copy_bleu[i] for i in sub])
        diff, lo, hi = paired_species_bootstrap(m - c, sp)
        sig = lo > 0 or hi < 0

        print(f"\nbound <={bound:.2f}: n={len(sub)}, species={len(set(sp))}, genes={len(genes)}")
        print(f"  ortholog-copy BLEU {c.mean():.4f}")
        print(f"  model         BLEU {m.mean():.4f}")
        print(f"  model - copy  {diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"({'excludes' if sig else 'includes'} 0), model wins {(m > c).mean():.1%}")

        out["bounds"][f"{bound:.2f}"] = {
            "n": len(sub), "n_species": len(set(sp)), "n_genes": len(genes),
            "genes": dict(sorted(genes.items())),
            "bleu_homology_copy": float(c.mean()),
            "bleu_model": float(m.mean()),
            "paired_diff": diff, "ci_lo": lo, "ci_hi": hi,
            "significant": bool(sig),
            "model_win_rate": float((m > c).mean()),
        }

    json.dump(out, open(QC / "doubly_controlled_analysis.json", "w"), indent=2)
    print(f"\nwrote {QC / 'doubly_controlled_analysis.json'}")


if __name__ == "__main__":
    main()
