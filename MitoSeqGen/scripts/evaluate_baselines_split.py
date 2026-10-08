"""Evaluate all non-model methods on an arbitrary split, in parallel.

Covers the four heuristic baselines, the natural reference, and the
nearest-ortholog homology-copy baseline, so the model's own numbers can be
merged in later without re-running the expensive parts (ViennaRNA folding
dominates: ~6 methods x N sequences).

Per-sequence results are written out, not just means, so every downstream
comparison can be paired and species-bootstrapped.

Usage:
  python scripts/evaluate_baselines_split.py --tag ident70 \
      --test-split test_ident70_per_gene.json \
      --train-split train_ident70_per_gene.json \
      --retrieval retrieval_ident70_test.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

_W = {}

METHODS = ["mt_cai_lookup", "random_synonymous", "most_frequent_codon",
           "codontransformer_remap", "homology_copy", "natural_reference"]


def _init_worker(cache_path):
    with open(cache_path, "rb") as f:
        _W.update(pickle.load(f))
    from src.evaluation.baselines import Baselines
    recs = _W["train_records"]
    _W["baselines"] = Baselines(recs, seed=42)


def _eval_worker(item):
    """(record_idx, protein, reference_dna, homology_copy_dna) -> {method: metrics}"""
    from src.evaluation.metrics import (codon_diversity, compute_mfe, compute_mt_cai,
                                        gc_content, simple_bleu,
                                        translation_complies_with_mt_code)
    idx, protein, reference, copy_seq = item
    rscu = _W["rscu"]
    b = _W["baselines"]

    seqs = {
        "mt_cai_lookup": b.generate(protein, "mt_cai_lookup"),
        "random_synonymous": b.generate(protein, "random_synonymous"),
        "most_frequent_codon": b.generate(protein, "most_frequent_codon"),
        "codontransformer_remap": b.generate(protein, "codontransformer_remap"),
        "homology_copy": copy_seq,
        "natural_reference": reference,
    }
    nat_mfe = compute_mfe(reference)
    nat_gc = gc_content(reference)

    out = {}
    for name, s in seqs.items():
        mfe, gc = compute_mfe(s), gc_content(s)
        out[name] = {
            "mt_cai": compute_mt_cai(s, rscu),
            "code_compliant": bool(translation_complies_with_mt_code(s)),
            "codon_diversity": codon_diversity(s),
            "bleu": simple_bleu(reference, s),
            "mfe": mfe, "gc_content": gc,
            "mfe_delta_from_natural": abs(mfe - nat_mfe),
            "gc_delta_from_natural": abs(gc - nat_gc),
        }
    return idx, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--train-split", required=True)
    ap.add_argument("--retrieval", required=True, help="retrieval_*.json with homology_copy_sequence")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    train_idx = json.load(open(ROOT / f"data/splits/{args.train_split}"))["indices"]
    test_idx = json.load(open(ROOT / f"data/splits/{args.test_split}"))["indices"]
    retr = json.load(open(ROOT / f"data/qc_reports/{args.retrieval}"))["per_sequence"]
    copy_of = {o["idx"]: o["homology_copy_sequence"] for o in retr}

    from src.evaluation.metrics import compute_rscu_weights
    train_records = [recs[i] for i in train_idx]
    rscu = compute_rscu_weights(train_records)

    cache = Path(tempfile.gettempdir()) / f"mitoseqgen_eval_{args.tag}.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"train_records": train_records, "rscu": rscu}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    items = [(i, recs[i]["protein_sequence"].rstrip("*"), recs[i]["sequence"], copy_of[i])
             for i in test_idx if i in copy_of]
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(items)} test sequences x {len(METHODS)} methods on {n_workers} workers "
          f"(ViennaRNA folding dominates)", flush=True)

    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(str(cache),)) as ex:
        for idx, res in ex.map(_eval_worker, items, chunksize=4):
            results[idx] = res
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    cache.unlink(missing_ok=True)

    species = np.array([recs[i]["species"] for i in results])
    order = list(results.keys())
    summary = {}
    for m in METHODS:
        summary[m] = {}
        for metric in ["bleu", "mt_cai", "codon_diversity", "gc_content",
                       "gc_delta_from_natural", "mfe", "mfe_delta_from_natural"]:
            v = np.array([results[i][m][metric] for i in order], dtype=float)
            summary[m][metric] = float(v.mean())
        summary[m]["compliance_rate"] = float(np.mean([results[i][m]["code_compliant"] for i in order]))

    json.dump({"tag": args.tag, "n": len(order), "summary": summary,
               "per_sequence": {str(k): v for k, v in results.items()},
               "runtime_sec": time.time() - t0},
              open(ROOT / f"data/qc_reports/baseline_eval_{args.tag}.json", "w"))

    print(f"\n{'method':<26}{'BLEU':>8}{'mt-CAI':>9}{'GCdev':>8}{'MFEdev':>9}{'divers':>8}")
    for m in METHODS:
        s = summary[m]
        print(f"{m:<26}{s['bleu']:>8.3f}{s['mt_cai']:>9.3f}{s['gc_delta_from_natural']:>8.3f}"
              f"{s['mfe_delta_from_natural']:>9.2f}{s['codon_diversity']:>8.3f}")
    print(f"\ntotal {time.time()-t0:.0f}s -> data/qc_reports/baseline_eval_{args.tag}.json")


if __name__ == "__main__":
    main()
