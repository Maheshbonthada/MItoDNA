"""Generate sequences from a checkpoint on CPU, parallelised across workers.

Autoregressive decoding is sequential within a sequence but embarrassingly
parallel across sequences. Running it on CPU across many workers lets sequence
generation proceed while the GPU is occupied by a training run, and for a model
this small it is competitive with single-stream GPU decoding.

Only sequences are produced (not the metric suite), because the metrics for a
given checkpoint may already exist; the sequences are what downstream
composition analyses such as GC skew require.

Usage:
  python scripts/generate_sequences_cpu.py \
      --checkpoint experiments/20261007_094622_ident70/checkpoints/best.pt \
      --test-split test_ident70_per_gene.json --out sequences_ident70.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

_W = {}


def _init_worker(ckpt_path, condition):
    torch.set_num_threads(1)  # workers must not oversubscribe each other
    from src.models.generate import load_model_from_checkpoint
    _W["model"] = load_model_from_checkpoint(Path(ckpt_path), torch.device("cpu"))
    _W["condition"] = condition


def _gen_worker(item):
    from src.models.generate import generate_cds
    from src.strand import condition_of
    idx, protein, gene = item
    cond = condition_of(gene, _W["condition"]) if _W["condition"] != "none" else None
    return idx, generate_cds(_W["model"], protein, torch.device("cpu"),
                             strategy="greedy", cond=cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="none", choices=["none", "strand", "gene"])
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    test_idx = json.load(open(ROOT / f"data/splits/{args.test_split}"))["indices"]

    # fail fast on a conditioning mismatch rather than silently generating garbage
    from src.models.generate import load_model_from_checkpoint
    probe = load_model_from_checkpoint(Path(args.checkpoint), torch.device("cpu"))
    if (args.condition == "none") != (probe.n_conditions == 0):
        raise SystemExit(f"ERROR: --condition={args.condition} but checkpoint has "
                         f"n_conditions={probe.n_conditions}")
    del probe

    items = [(i, recs[i]["protein_sequence"].rstrip("*"), recs[i]["gene_name"]) for i in test_idx]
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 4)
    print(f"generating {len(items)} sequences on CPU, {n_workers} workers "
          f"(condition={args.condition})", flush=True)

    out, done = {}, 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(args.checkpoint, args.condition)) as ex:
        for idx, seq in ex.map(_gen_worker, items, chunksize=4):
            out[idx] = seq
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    path = ROOT / "data/qc_reports" / args.out
    json.dump({"checkpoint": args.checkpoint, "condition": args.condition,
               "n": len(out), "sequences": {str(k): v for k, v in out.items()}},
              open(path, "w"))
    print(f"wrote {path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
