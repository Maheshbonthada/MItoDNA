"""Evaluate a trained MitoSeqGen checkpoint on a split and compare, paired, to
the baselines already scored by scripts/evaluate_baselines_split.py.

Generation runs on the GPU; the metric suite (ViennaRNA folding dominates) is
parallelised across CPU workers. The headline comparison is against the
nearest-ortholog homology-copy baseline, with species-level bootstrap CIs --
the sequences are not independent samples (13 genes per species).

Usage:
  python scripts/evaluate_model_split.py \
      --checkpoint experiments/20261007_094622_ident70/checkpoints/best.pt \
      --tag ident70 --test-split test_ident70_per_gene.json
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from src.models.generate import generate_cds, load_model_from_checkpoint  # noqa: E402
from src.strand import condition_of  # noqa: E402

_W = {}


def _init_worker(cache_path):
    with open(cache_path, "rb") as f:
        _W.update(pickle.load(f))


def _metrics_worker(item):
    from src.evaluation.metrics import (codon_diversity, compute_mfe, compute_mt_cai,
                                        gc_content, simple_bleu,
                                        translation_complies_with_mt_code)
    idx, gen, reference = item
    mfe, gc = compute_mfe(gen), gc_content(gen)
    nat_mfe, nat_gc = compute_mfe(reference), gc_content(reference)
    return idx, {
        "mt_cai": compute_mt_cai(gen, _W["rscu"]),
        "code_compliant": bool(translation_complies_with_mt_code(gen)),
        "codon_diversity": codon_diversity(gen),
        "bleu": simple_bleu(reference, gen),
        "mfe": mfe, "gc_content": gc,
        "mfe_delta_from_natural": abs(mfe - nat_mfe),
        "gc_delta_from_natural": abs(gc - nat_gc),
    }


def species_bootstrap(values, species, n_boot=2000, seed=0):
    values, species = np.asarray(values, float), np.asarray(species)
    uniq = np.unique(species)
    groups = {s: np.where(species == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    boots = np.array([values[np.concatenate([groups[s] for s in rng.choice(uniq, len(uniq))])].mean()
                      for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--train-split", required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--condition", default="none", choices=["none", "strand", "gene"],
                    help="Must match the conditioning the checkpoint was trained with.")
    ap.add_argument("--out-suffix", default="", help="suffix for the output filename")
    args = ap.parse_args()

    t0 = time.time()
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    test_idx = json.load(open(ROOT / f"data/splits/{args.test_split}"))["indices"]
    train_idx = json.load(open(ROOT / f"data/splits/{args.train_split}"))["indices"]

    base = json.load(open(ROOT / f"data/qc_reports/baseline_eval_{args.tag}.json"))
    base_per = {int(k): v for k, v in base["per_sequence"].items()}
    test_idx = [i for i in test_idx if i in base_per]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(Path(args.checkpoint), device)
    print(f"generating {len(test_idx)} sequences on {device} "
          f"(condition={args.condition}, model n_conditions={model.n_conditions})...", flush=True)
    if args.condition != "none" and model.n_conditions == 0:
        raise SystemExit("ERROR: --condition given but checkpoint has no conditioning embedding")
    if args.condition == "none" and model.n_conditions > 0:
        raise SystemExit("ERROR: checkpoint is conditioned but --condition=none")
    gens = {}
    for n, i in enumerate(test_idx, 1):
        protein = recs[i]["protein_sequence"].rstrip("*")
        cond = condition_of(recs[i]["gene_name"], args.condition) if args.condition != "none" else None
        gens[i] = generate_cds(model, protein, device, strategy="greedy", cond=cond)
        if n % 250 == 0:
            print(f"    {n}/{len(test_idx)} ({time.time()-t0:.0f}s)", flush=True)

    from src.evaluation.metrics import compute_rscu_weights
    rscu = compute_rscu_weights([recs[i] for i in train_idx])
    cache = Path(tempfile.gettempdir()) / f"mitoseqgen_modeleval_{args.tag}.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"rscu": rscu}, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    items = [(i, gens[i], recs[i]["sequence"]) for i in test_idx]
    print(f"scoring on {n_workers} workers...", flush=True)
    model_res = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(str(cache),)) as ex:
        for idx, m in ex.map(_metrics_worker, items, chunksize=4):
            model_res[idx] = m
    cache.unlink(missing_ok=True)

    order = test_idx
    species = np.array([recs[i]["species"] for i in order])
    methods = list(base["summary"].keys()) + ["mitoseqgen"]

    def vec(method, metric):
        if method == "mitoseqgen":
            return np.array([model_res[i][metric] for i in order], float)
        return np.array([base_per[i][method][metric] for i in order], float)

    out = {"tag": args.tag, "checkpoint": args.checkpoint, "n": len(order),
           "n_species": int(len(np.unique(species))), "summary": {}, "comparisons": {}}
    for m in methods:
        out["summary"][m] = {}
        for metric in ["bleu", "mt_cai", "codon_diversity", "gc_content",
                       "gc_delta_from_natural", "mfe_delta_from_natural"]:
            mean, lo, hi = species_bootstrap(vec(m, metric), species)
            out["summary"][m][metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        out["summary"][m]["compliance_rate"] = float(np.mean(
            [model_res[i]["code_compliant"] if m == "mitoseqgen" else base_per[i][m]["code_compliant"]
             for i in order]))

    for rival in ["homology_copy", "mt_cai_lookup", "most_frequent_codon",
                  "codontransformer_remap", "random_synonymous"]:
        d = vec("mitoseqgen", "bleu") - vec(rival, "bleu")
        mean, lo, hi = species_bootstrap(d, species)
        out["comparisons"][f"mitoseqgen_vs_{rival}"] = {
            "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0),
            "frac_model_better": float((d > 0).mean())}

    out["condition"] = args.condition
    json.dump({**out, "per_sequence_model": {str(k): v for k, v in model_res.items()},
               "generated_sequences": {str(k): v for k, v in gens.items()}},
              open(ROOT / f"data/qc_reports/model_eval_{args.tag}{args.out_suffix}.json", "w"))

    print(f"\n=== {args.tag} (n={len(order)}, {out['n_species']} species) ===")
    print(f"{'method':<26}{'BLEU [95% CI]':>28}{'mt-CAI':>9}{'GCdev':>8}{'MFEdev':>9}")
    for m in sorted(methods, key=lambda x: -out["summary"][x]["bleu"]["mean"]):
        s = out["summary"][m]
        b = s["bleu"]
        print(f"{m:<26}{b['mean']:>10.3f} [{b['ci_lo']:.3f},{b['ci_hi']:.3f}]"
              f"{s['mt_cai']['mean']:>9.3f}{s['gc_delta_from_natural']['mean']:>8.3f}"
              f"{s['mfe_delta_from_natural']['mean']:>9.2f}")
    print("\n=== paired comparisons (BLEU, species bootstrap) ===")
    for k, v in out["comparisons"].items():
        print(f"  {'SIG' if v['significant'] else 'ns '} {k:<45} "
              f"diff {v['mean_diff']:+.4f} CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] "
              f"model better on {v['frac_model_better']*100:.0f}%")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
