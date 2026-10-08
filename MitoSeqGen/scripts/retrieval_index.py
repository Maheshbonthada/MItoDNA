"""Parallel retrieval index + homology-copy baseline, for any split pair.

Generalises scripts/build_retrieval_and_verify.py:
  - arbitrary --query-split / --train-split files
  - exhaustive same-gene neighbour search parallelised over worker processes
  - emits top-K neighbours (for the retrieval-conditioned model, Phase 2/3)
    and the top-1 homology-copy baseline + its BLEU (for Phase 1)

Neighbour search is the expensive part and runs in workers; the CIGAR
alignment and copy construction run in the parent for the top-K only.

Leakage guards: a query never retrieves itself, and the neighbour pool is
built from the given train split only (pass the NON-augmented train split so
synthetic RSCU variants of a record can never be retrieved).

Usage:
  python scripts/retrieval_index.py --tag ident70 \
      --query-split test_ident70_per_gene.json \
      --train-split train_ident70_per_gene.json
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import edlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from src.evaluation.metrics import simple_bleu  # noqa: E402

_W = {}


def _init_worker(cache_path):
    with open(cache_path, "rb") as f:
        _W.update(pickle.load(f))


def _topk_worker(item):
    """(query_idx, protein, gene, k, max_identity) -> (query_idx, [(edit_distance, train_idx), ...])

    max_identity caps how similar a retrieved neighbour may be. This matters for
    TRAINING queries: their true nearest neighbours sit in the same identity
    cluster (often >95% identical), whereas test queries are capped at the split
    threshold. Without the cap the model would learn "the retrieved codon is
    almost always right" and then meet a far weaker signal at test time. Capping
    training retrieval at the test threshold matches the two distributions.
    """
    qi, p, gene, k, max_identity = item
    pool = _W["pool"].get(gene, ())
    lp = len(p)
    out = []
    for tj, q in pool:
        if tj == qi:
            continue
        ed = edlib.align(p, q, mode="NW", task="distance")["editDistance"]
        if max_identity is not None:
            if 1 - ed / (lp if lp > len(q) else len(q)) > max_identity:
                continue
        out.append((ed, tj))
    out.sort()
    return qi, out[:k]


def codons_of(seq):
    seq = seq.upper()
    return [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]


def parse_cigar(cigar):
    ops, num = [], ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            ops.append((ch, int(num)))
            num = ""
    return ops


def align_map(query, target):
    """(n_identical, {query_pos: target_pos}, edit_distance) for a global alignment.
    edlib CIGAR semantics: '='/'X' consume both, 'I' consumes query, 'D' consumes target."""
    r = edlib.align(query, target, mode="NW", task="path")
    qi = ti = 0
    ident = 0
    pos_map = {}
    for op, ln in parse_cigar(r["cigar"]):
        if op in ("=", "X"):
            for _ in range(ln):
                pos_map[qi] = ti
                if query[qi] == target[ti]:
                    ident += 1
                qi += 1
                ti += 1
        elif op == "I":
            qi += ln
        elif op == "D":
            ti += ln
    return ident, pos_map, r["editDistance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--query-split", required=True)
    ap.add_argument("--train-split", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-identity", type=float, default=None,
                    help="Cap retrieved-neighbour identity (use the split threshold for TRAIN/VAL "
                         "queries so the retrieval signal matches test-time difficulty).")
    args = ap.parse_args()

    t0 = time.time()
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    train_idx = json.load(open(ROOT / f"data/splits/{args.train_split}"))["indices"]
    query_idx = json.load(open(ROOT / f"data/splits/{args.query_split}"))["indices"]
    print(f"{len(query_idx)} queries vs {len(train_idx)} train records")

    proteins = {i: recs[i]["protein_sequence"].rstrip("*") for i in range(len(recs))}
    pool = defaultdict(list)
    for i in train_idx:
        pool[recs[i]["gene_name"]].append((i, proteins[i]))

    # gene-specific most-frequent codon per amino acid, for non-identical positions
    fallback = {}
    for gene, idxs in pool.items():
        cnt = defaultdict(Counter)
        for i, _ in idxs:
            p = proteins[i]
            for a, c in zip(p, codons_of(recs[i]["sequence"])):
                cnt[a][c] += 1
        fallback[gene] = {a: c.most_common(1)[0][0] for a, c in cnt.items()}

    cache = Path(tempfile.gettempdir()) / f"mitoseqgen_retr_{args.tag}.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"pool": dict(pool)}, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    items = [(i, proteins[i], recs[i]["gene_name"], args.k, args.max_identity) for i in query_idx]
    print(f"exhaustive neighbour search on {n_workers} workers...", flush=True)
    topk = {}
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(str(cache),)) as ex:
        for qi, res in ex.map(_topk_worker, items, chunksize=8):
            topk[qi] = res
            done += 1
            if done % 1000 == 0:
                print(f"    {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    cache.unlink(missing_ok=True)
    print(f"  search done ({time.time()-t0:.0f}s); aligning top-{args.k} + building copy baseline...", flush=True)

    out = []
    for qi in query_idx:
        q = recs[qi]
        gene = q["gene_name"]
        prot = proteins[qi]
        ref = q["sequence"]
        ref_cod = codons_of(ref)
        if not topk[qi]:
            continue

        neighbours, maps = [], []
        for rank, (_, j) in enumerate(topk[qi]):
            hp = proteins[j]
            ident, pos_map, ed = align_map(prot, hp)
            maps.append(pos_map)
            neighbours.append({
                "rank": rank, "train_idx": j, "species": recs[j]["species"], "edit_distance": ed,
                # standard sequence identity -- same definition used to enforce the
                # identity-controlled split, so the two are directly comparable
                "identity": 1 - ed / max(len(prot), len(hp)),
                "identity_over_query": ident / max(len(prot), 1),
                "identity_over_max": ident / max(len(prot), len(hp)),
            })

        j = neighbours[0]["train_idx"]
        hp, hc = proteins[j], codons_of(recs[j]["sequence"])
        fb = fallback[gene]
        gen, copied = [], 0
        for p, aa in enumerate(prot):
            t = maps[0].get(p)
            if t is not None and t < len(hp) and hp[t] == aa and t < len(hc):
                gen.append(hc[t])
                copied += 1
            else:
                gen.append(fb.get(aa, ref_cod[p] if p < len(ref_cod) else "TAA"))
        # Append a terminal stop only if the reference actually has one. 9/46,264
        # QC-passed records are truncated CDS whose protein carries no trailing '*';
        # appending unconditionally would make the generated sequence one codon
        # longer than its reference and corrupt that record's BLEU.
        if len(ref_cod) > len(prot):
            gen.append(hc[len(hp)] if len(hc) > len(hp) else "TAA")
        gen_seq = "".join(gen)

        out.append({
            "idx": qi, "gene": gene, "species": q["species"], "protein_len": len(prot),
            "max_identity_to_train": neighbours[0]["identity"],
            "max_identity_over_query": neighbours[0]["identity_over_query"],
            "neighbor_species": neighbours[0]["species"],
            "copied_fraction": copied / max(len(prot), 1),
            "len_ok": len(gen) == len(ref_cod),
            "bleu_homology_copy": simple_bleu(ref, gen_seq),
            "homology_copy_sequence": gen_seq,
            "neighbors": neighbours,
        })

    bleu = np.array([o["bleu_homology_copy"] for o in out])
    pid = np.array([o["max_identity_to_train"] for o in out])
    summary = {
        "tag": args.tag, "query_split": args.query_split, "train_split": args.train_split,
        "n": len(out), "k": args.k, "max_identity_cap": args.max_identity,
        "n_queries_with_no_neighbor": int(len(query_idx) - len(out)),
        "all_lengths_ok": bool(all(o["len_ok"] for o in out)),
        "bleu_homology_copy_mean": float(bleu.mean()),
        "bleu_homology_copy_sem": float(bleu.std(ddof=1) / np.sqrt(len(bleu))),
        "identity_to_train": {
            "mean": float(pid.mean()), "median": float(np.median(pid)),
            "p10": float(np.percentile(pid, 10)), "p90": float(np.percentile(pid, 90)),
            "max": float(pid.max()),
        },
        "runtime_sec": time.time() - t0,
    }
    outpath = ROOT / f"data/qc_reports/retrieval_{args.tag}.json"
    json.dump({"summary": summary, "per_sequence": out}, open(outpath, "w"))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
