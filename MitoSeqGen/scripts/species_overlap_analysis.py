"""Quantify cross-gene species leakage in the identity-controlled split.

Reviewer concern (TCBB review, secondary point 3): the identity-controlled
split is built per gene, so a species may appear in train for gene A and in
test for gene B. Within a species, genes share strand regime, mutational bias
and codon pool, so this is a real leakage channel that the per-gene identity
bound does not close.

This script measures it directly:

  1. How many test species also appear in training (for any other gene)?
  2. Re-evaluate every method on the "clean" subset of test sequences whose
     species appears nowhere in training. If the model-vs-copy conclusion is
     an artefact of cross-gene species leakage, it should move on that subset.

Both the ortholog-copy baseline and the strand-conditioned model are scored on
the clean subset, with the same paired species-level bootstrap used elsewhere.

Usage:  python scripts/species_overlap_analysis.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QC = ROOT / "data/qc_reports"
N_BOOT = 10000
RNG_SEED = 20261008


def paired_species_bootstrap(diffs, species, n_boot=N_BOOT, seed=RNG_SEED):
    """Cluster bootstrap over species: resample species, not sequences."""
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
    train_idx = json.load(open(ROOT / "data/splits/train_ident70_per_gene.json"))["indices"]
    test_idx = json.load(open(ROOT / "data/splits/test_ident70_per_gene.json"))["indices"]

    # augmented records live past the end of the original record list
    train_idx = [i for i in train_idx if i < len(recs)]

    train_species = {recs[i]["species"] for i in train_idx}
    test_species = {recs[i]["species"] for i in test_idx}

    shared = test_species & train_species
    clean_species = test_species - train_species
    clean_test = [i for i in test_idx if recs[i]["species"] in clean_species]

    print(f"test sequences                : {len(test_idx)}")
    print(f"test species                  : {len(test_species)}")
    print(f"  also in training (any gene) : {len(shared)} "
          f"({len(shared)/len(test_species):.1%})")
    print(f"  absent from training        : {len(clean_species)}")
    print(f"clean test sequences          : {len(clean_test)} "
          f"({len(clean_test)/len(test_idx):.1%})")

    per_gene_clean = Counter(recs[i]["gene_name"] for i in clean_test)
    print(f"clean subset genes            : {dict(sorted(per_gene_clean.items()))}")

    # ---- re-score on the clean subset ------------------------------------
    base = json.load(open(QC / "baseline_eval_ident70.json"))["per_sequence"]
    model_ctrl = json.load(open(QC / "model_eval_ident70.json"))["per_sequence_model"]
    model_strand = json.load(open(QC / "model_eval_ident70_strand.json"))["per_sequence_model"]

    clean_set = set(clean_test)
    rows = []
    for key in base:
        idx = int(key)
        if idx not in clean_set:
            continue
        if key not in model_ctrl or key not in model_strand:
            continue
        rows.append((idx, recs[idx]["species"], recs[idx]["gene_name"],
                     base[key], model_ctrl[key], model_strand[key]))

    print(f"\nscored on clean subset        : {len(rows)}")
    if not rows:
        print("NOTHING TO SCORE -- clean subset is empty")
        return

    species = [r[1] for r in rows]
    copy_b = np.array([r[3]["homology_copy"]["bleu"] for r in rows])
    ctrl_b = np.array([r[4]["bleu"] for r in rows])
    strand_b = np.array([r[5]["bleu"] for r in rows])

    out = {
        "n_test": len(test_idx),
        "n_test_species": len(test_species),
        "n_species_shared_with_train": len(shared),
        "frac_species_shared": len(shared) / len(test_species),
        "n_clean_species": len(clean_species),
        "n_clean_sequences": len(clean_test),
        "frac_clean_sequences": len(clean_test) / len(test_idx),
        "clean_subset_genes": dict(sorted(per_gene_clean.items())),
        "n_scored": len(rows),
        "clean_subset": {},
    }

    for name, arr in (("homology_copy", copy_b), ("model_control", ctrl_b),
                      ("model_strand", strand_b)):
        out["clean_subset"][name] = {"bleu_mean": float(arr.mean())}
        print(f"  {name:16s} BLEU {arr.mean():.4f}")

    for name, arr in (("model_control", ctrl_b), ("model_strand", strand_b)):
        d = arr - copy_b
        m, lo, hi = paired_species_bootstrap(d, species)
        out["clean_subset"][name]["paired_vs_copy"] = {
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "win_rate": float((d > 0).mean()),
        }
        sig = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"  {name:16s} - copy = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({sig})")

    json.dump(out, open(QC / "species_overlap_analysis.json", "w"), indent=2)
    print(f"\nwrote {QC / 'species_overlap_analysis.json'}")


if __name__ == "__main__":
    main()
