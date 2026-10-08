"""Phase 1.6: an identity-controlled split.

Why this exists
---------------
The published species-level split leaves test proteins a median 96.8%
identical to their nearest same-gene training protein, so a trivial
nearest-ortholog copy baseline beats every learned model. Random same-gene
pairs, by contrast, span ~0.49-0.92 identity, so the conservation ceiling is
NOT intrinsic -- it is an artifact of having ~2,800 training proteins per
gene. A harder split is therefore constructible.

Construction
------------
1. Per gene, greedy CD-HIT-style clustering of protein sequences at
   THRESHOLD identity (longest-first, representative-based).
2. Union-find over (gene, cluster) and species: a species is linked to every
   cluster it has a sequence in, so assigning a connected component to a
   split keeps BOTH species-level and identity-level separation. This avoids
   the weaker per-gene-only split, where a species could be train for ND1 and
   test for COX1.
3. Assign whole components to train/val/test, largest-first, to hit target
   proportions.

Guarantee: no test protein shares >THRESHOLD identity with any training
protein of the same gene (up to greedy-clustering approximation, which is
verified empirically afterwards and reported).

Usage: python scripts/build_identity_controlled_split.py --threshold 0.7
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import edlib
import numpy as np

# ---- multiprocessing worker state (Windows uses spawn, so workers load from a
# pickle via the initializer rather than inheriting memory) ----
_W = {}


def _init_worker(cache_path):
    with open(cache_path, "rb") as f:
        _W.update(pickle.load(f))


def _max_identity_worker(item):
    """(record_idx, protein, gene) -> (record_idx, max identity to same-gene train)."""
    i, p, gene = item
    lp = len(p)
    best = 0.0
    for q in _W["train_prots"].get(gene, ()):
        ed = edlib.align(p, q, mode="NW", task="distance")["editDistance"]
        v = 1 - ed / (lp if lp > len(q) else len(q))
        if v > best:
            best = v
    return i, best

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_gene(indices, proteins, threshold):
    """Greedy longest-first clustering. Returns {record_idx: cluster_id}."""
    order = sorted(indices, key=lambda i: -len(proteins[i]))
    reps = []  # (rep_idx, cluster_id)
    assign = {}
    for i in order:
        p = proteins[i]
        placed = False
        # compare against representatives; edlib k-limit prunes hard
        max_ed = int(len(p) * (1.0 - threshold))
        for rep_idx, cid in reps:
            rp = proteins[rep_idx]
            if abs(len(rp) - len(p)) > max_ed:
                continue
            r = edlib.align(p, rp, mode="NW", task="distance", k=max_ed)
            if r["editDistance"] != -1:
                assign[i] = cid
                placed = True
                break
        if not placed:
            cid = len(reps)
            reps.append((i, cid))
            assign[i] = cid
    return assign, len(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--test-frac", type=float, default=0.12)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count()-2")
    ap.add_argument("--mode", default="component", choices=["component", "per_gene"],
                    help="component: union-find over species+clusters (strict, but collapses "
                         "because COX1 is near-invariant across vertebrates). per_gene: assign "
                         "clusters within each gene independently -- gives genuine low-identity "
                         "test cases for divergent genes, at the cost of allowing a species to be "
                         "train for one gene and test for another (different proteins; disclosed).")
    args = ap.parse_args()

    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    proteins = {i: r["protein_sequence"].rstrip("*") for i, r in enumerate(recs)}
    by_gene = defaultdict(list)
    for i, r in enumerate(recs):
        by_gene[r["gene_name"]].append(i)

    t0 = time.time()
    uf = UnionFind()
    cluster_of = {}
    gene_clusters = {}
    cluster_counts = {}
    print(f"clustering at {args.threshold:.0%} identity")
    for gene, idxs in sorted(by_gene.items()):
        assign, nrep = cluster_gene(idxs, proteins, args.threshold)
        gene_clusters[gene] = assign
        cluster_counts[gene] = nrep
        for i, cid in assign.items():
            key = ("cluster", gene, cid)
            cluster_of[i] = key
            uf.union(key, ("species", recs[i]["species"]))
        print(f"  {gene:6} {len(idxs):6} seqs -> {nrep:5} clusters   ({time.time()-t0:.0f}s)")

    print("\nconservation ranking (fewer clusters = more conserved):")
    for g, c in sorted(cluster_counts.items(), key=lambda kv: kv[1]):
        print(f"  {g:6} {c:5} clusters / {len(by_gene[g])} seqs")

    assigned = {"train": [], "val": [], "test": []}
    if args.mode == "component":
        comp = defaultdict(list)
        for i in range(len(recs)):
            comp[uf.find(cluster_of[i])].append(i)
        comps = sorted(comp.values(), key=len, reverse=True)
        print(f"\n{len(comps)} connected components; largest {len(comps[0])} "
              f"({len(comps[0])/len(recs):.1%} of data)")
        n = len(recs)
        targets = {"test": args.test_frac * n, "val": args.val_frac * n,
                   "train": (1 - args.test_frac - args.val_frac) * n}
        for c in comps:
            deficit = {k: targets[k] - len(assigned[k]) for k in targets}
            pick = max(deficit, key=deficit.get)
            assigned[pick].extend(c)
    else:  # per_gene
        print("\nper-gene cluster assignment:")
        for gene, idxs in sorted(by_gene.items()):
            assign = gene_clusters[gene]
            members = defaultdict(list)
            for i, cid in assign.items():
                members[cid].append(i)
            clusters = sorted(members.values(), key=len, reverse=True)
            ng = len(idxs)
            tg = {"test": args.test_frac * ng, "val": args.val_frac * ng,
                  "train": (1 - args.test_frac - args.val_frac) * ng}
            local = {"train": [], "val": [], "test": []}
            for c in clusters:
                deficit = {k: tg[k] - len(local[k]) for k in tg}
                local[max(deficit, key=deficit.get)].extend(c)
            for k in assigned:
                assigned[k].extend(local[k])
            print(f"  {gene:6} train={len(local['train']):5} val={len(local['val']):5} test={len(local['test']):5}")

    print("\nsplit sizes:", {k: len(v) for k, v in assigned.items()})
    for k, v in assigned.items():
        print(f"  {k:5} species={len({recs[i]['species'] for i in v}):5}")

    # ---- HARD ENFORCEMENT ----
    # Greedy representative-based clustering is leaky: two sequences assigned to
    # different clusters can still exceed the threshold, because each is only
    # ever compared to cluster representatives. So we compute exact max identity
    # to train for EVERY val/test sequence and drop the violators. This turns an
    # approximate guarantee into an exact one, at the cost of a smaller test set.
    train_prots = defaultdict(list)
    for i in assigned["train"]:
        train_prots[recs[i]["gene_name"]].append(proteins[i])

    cache = Path(tempfile.gettempdir()) / "mitoseqgen_split_cache.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"train_prots": dict(train_prots)}, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    identity_map = {}
    for split in ("val", "test"):
        items = [(i, proteins[i], recs[i]["gene_name"]) for i in assigned[split]]
        print(f"\nenforcing threshold on {split} ({len(items)} seqs, {n_workers} workers)...", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(str(cache),)) as ex:
            for i, mi in ex.map(_max_identity_worker, items, chunksize=8):
                identity_map[i] = mi
                done += 1
                if done % 2000 == 0:
                    print(f"    {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        keep = [i for i in assigned[split] if identity_map[i] <= args.threshold]
        print(f"  kept {len(keep)}, dropped {len(assigned[split]) - len(keep)}")
        assigned[split] = keep
    cache.unlink(missing_ok=True)

    idents = np.array([identity_map[i] for i in assigned["test"]]) if assigned["test"] else np.array([0.0])
    print(f"\nFINAL test max-identity: n={len(idents)} mean {idents.mean():.3f} "
          f"median {np.median(idents):.3f} max {idents.max():.3f}")
    print("per-gene surviving test counts:")
    gc_ = defaultdict(int)
    for i in assigned["test"]:
        gc_[recs[i]["gene_name"]] += 1
    for g in sorted(by_gene):
        print(f"  {g:6} {gc_.get(g,0):5}")
    print("\nfinal split sizes:", {k: len(v) for k, v in assigned.items()})

    tag = f"ident{int(args.threshold*100)}_{args.mode}"
    outdir = ROOT / "data/splits"
    for k, v in assigned.items():
        json.dump({"indices": sorted(v),
                   "construction": {"type": "identity_controlled", "threshold": args.threshold, "mode": args.mode,
                                    "seed": args.seed}},
                  open(outdir / f"{k}_{tag}.json", "w"))
    json.dump({"threshold": args.threshold, "mode": args.mode, "cluster_counts": cluster_counts,
               "sizes": {k: len(v) for k, v in assigned.items()},
               "per_gene_test_counts": {g: gc_.get(g, 0) for g in sorted(by_gene)},
               "achieved_max_identity": {
                   "n_sampled": int(len(idents)), "mean": float(idents.mean()),
                   "median": float(np.median(idents)), "p95": float(np.percentile(idents, 95)),
                   "max": float(idents.max()),
                   "frac_above_threshold": float((idents > args.threshold).mean())},
               "runtime_sec": time.time() - t0},
              open(ROOT / f"data/qc_reports/split_{tag}_report.json", "w"), indent=2)
    print(f"\nwrote data/splits/{{train,val,test}}_{tag}.json")


if __name__ == "__main__":
    main()
