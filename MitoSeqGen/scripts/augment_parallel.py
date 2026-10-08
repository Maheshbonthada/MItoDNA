"""Parallel RSCU synonymous-variant augmentation for an arbitrary train split.

Same algorithm as src/data/augment.py (identical synonym groups, identical
RSCU weighting, protein/start/stop preserved), but:
  - variant generation is spread across worker processes
  - per-record RNG is seeded from the record index, so output is deterministic
    and independent of worker scheduling/order
  - output JSON is written without indent (the original wrote a ~1.1 GB file
    with indent=2, which dominated runtime for no benefit)

Usage:
  python scripts/augment_parallel.py \
      --train-split train_ident70_per_gene.json \
      --out-tokenized mito_cds_tokenized_augmented_ident70.json \
      --out-split train_augmented_ident70.json
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

_W = {}


def _init_worker(cache_path):
    from src.data.augment import build_synonym_groups  # noqa: F401
    with open(cache_path, "rb") as f:
        _W.update(pickle.load(f))


def _variants_worker(item):
    """(orig_idx, sequence, n_variants, seed) -> (orig_idx, [new_seq, ...])"""
    import random
    from src.data.augment import generate_synonymous_variant
    orig_idx, seq, n_variants, seed = item
    rng = random.Random(seed * 1000003 + orig_idx)
    return orig_idx, [generate_synonymous_variant(seq, _W["groups"], _W["weights"], rng)
                      for _ in range(n_variants)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized", default="mito_cds_tokenized.json")
    ap.add_argument("--train-split", required=True)
    ap.add_argument("--out-tokenized", required=True)
    ap.add_argument("--out-split", required=True)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    from src.data.augment import build_synonym_groups, compute_rscu_weights
    from src.data.preprocess import tokenize_cds

    t0 = time.time()
    data = ROOT / "data"
    print("loading records...", flush=True)
    payload = json.load(open(data / "processed" / args.tokenized))
    records = payload["records"]
    train_indices = json.load(open(data / "splits" / args.train_split))["indices"]
    train_records = [records[i] for i in train_indices]
    print(f"  {len(records)} records, {len(train_indices)} train ({time.time()-t0:.0f}s)", flush=True)

    groups = build_synonym_groups()
    weights = compute_rscu_weights(train_records, groups)

    cache = Path(tempfile.gettempdir()) / "mitoseqgen_aug_cache.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"groups": groups, "weights": weights}, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    items = [(i, records[i]["sequence"], args.variants, args.seed) for i in train_indices]
    print(f"generating {len(items) * args.variants} variants on {n_workers} workers...", flush=True)

    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(str(cache),)) as ex:
        for orig_idx, seqs in ex.map(_variants_worker, items, chunksize=64):
            results[orig_idx] = seqs
            done += 1
            if done % 5000 == 0:
                print(f"    {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    cache.unlink(missing_ok=True)
    print(f"  generation done ({time.time()-t0:.0f}s), tokenizing + assembling...", flush=True)

    augmented = list(records)
    new_train_indices = list(train_indices)
    for orig_idx in train_indices:  # deterministic append order
        rec = records[orig_idx]
        for v, new_seq in enumerate(results[orig_idx]):
            new_rec = dict(rec)
            new_rec["id"] = f"{rec['id']}_aug{v + 1}"
            new_rec["sequence"] = new_seq
            new_rec["codon_tokens"] = tokenize_cds(new_seq)
            new_rec["is_augmented"] = True
            new_rec["source_id"] = rec["id"]
            augmented.append(new_rec)
            new_train_indices.append(len(augmented) - 1)

    print(f"writing {len(augmented)} records ({time.time()-t0:.0f}s)...", flush=True)
    with open(data / "processed" / args.out_tokenized, "w") as f:
        json.dump({"codon_vocab": payload["codon_vocab"], "aa_vocab": payload["aa_vocab"],
                   "records": augmented}, f)
    with open(data / "splits" / args.out_split, "w") as f:
        json.dump({"indices": new_train_indices}, f)

    print(f"\n{len(train_indices)} -> {len(new_train_indices)} train records "
          f"({args.variants} variants each)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
