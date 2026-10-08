"""Phase 0 verification + Phase 1 strata + Phase 2 retrieval index, in one pass.

INDEPENDENT reimplementation of scripts/homology_copy_baseline.py:
  - neighbour search is EXHAUSTIVE over all same-gene training proteins
    (no 3-mer prefilter, no TOP_CANDIDATES cutoff -- removes v1's main risk)
  - alignment is edlib (bit-parallel Myers), not Bio.Align.PairwiseAligner
  - stop codon is handled explicitly: records carry len(protein)+1 codons,
    the last being a real mt stop; the copy takes the neighbour's own stop.

If v1's BLEU 0.410 does not reproduce here, the Phase 0 gate fails.

Emits, for every sequence in a split:
  - top-K nearest same-gene TRAINING neighbours (K=5) with identity + CIGAR
  - the top-1 homology-copy sequence and its BLEU vs the real sequence
  - max-identity-to-training (-> Phase 1 identity strata)

Leakage guards (Phase 2.2): a training query never retrieves itself, and
never retrieves an augmented variant of itself (augmented records live at
index >= len(original records) and are excluded from the neighbour pool
entirely -- this script indexes only the 46,264 original records).

Usage:  python scripts/build_retrieval_and_verify.py --split test [--k 5]
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import edlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import simple_bleu  # noqa: E402

K_DEFAULT = 5


def codons_of(seq):
    seq = seq.upper()
    return [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]


def parse_cigar(cigar):
    """edlib CIGAR -> list of (op, length). Ops: = X I D.
    Semantics (verified): '='/'X' consume query+target, 'I' consumes query
    only, 'D' consumes target only."""
    ops, num = [], ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            ops.append((ch, int(num)))
            num = ""
    return ops


def align_map(query, target):
    """Return (n_identical, {query_pos: target_pos}) for a global alignment."""
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
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N queries")
    args = ap.parse_args()

    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    train_idx = json.load(open(ROOT / "data/splits/train.json"))["indices"]
    query_idx = json.load(open(ROOT / f"data/splits/{args.split}.json"))["indices"]
    if args.limit:
        query_idx = query_idx[:args.limit]

    # neighbour pool: ORIGINAL training records only (no augmented variants)
    by_gene = defaultdict(list)
    for i in train_idx:
        by_gene[recs[i]["gene_name"]].append(i)

    # gene-specific most-frequent codon per amino acid, for non-identical positions
    fallback = {}
    for gene, idxs in by_gene.items():
        cnt = defaultdict(Counter)
        for i in idxs:
            p = recs[i]["protein_sequence"].rstrip("*")
            cs = codons_of(recs[i]["sequence"])
            for a, c in zip(p, cs):
                cnt[a][c] += 1
        fallback[gene] = {a: c.most_common(1)[0][0] for a, c in cnt.items()}

    prot_cache = {i: recs[i]["protein_sequence"].rstrip("*") for i in train_idx}
    cod_cache = {i: codons_of(recs[i]["sequence"]) for i in train_idx}

    out = []
    t0 = time.time()
    for n, qi_idx in enumerate(query_idx, 1):
        q = recs[qi_idx]
        gene = q["gene_name"]
        prot = q["protein_sequence"].rstrip("*")
        ref = q["sequence"]
        ref_cod = codons_of(ref)

        # --- exhaustive distance pass (cheap) ---
        pool = [j for j in by_gene[gene] if j != qi_idx]
        dists = [(edlib.align(prot, prot_cache[j], mode="NW", task="distance")["editDistance"], j)
                 for j in pool]
        dists.sort()
        topk = dists[:args.k]

        neighbours = []
        for rank, (_, j) in enumerate(topk):
            hp = prot_cache[j]
            ident, pos_map, ed = align_map(prot, hp)
            neighbours.append({
                "rank": rank, "train_idx": j, "species": recs[j]["species"],
                "edit_distance": ed,
                "identity_over_query": ident / max(len(prot), 1),
                "identity_over_max": ident / max(len(prot), len(hp)),
                "pos_map": pos_map if rank < args.k else None,
            })

        # --- top-1 homology-copy baseline ---
        best = neighbours[0]
        j = best["train_idx"]
        hp, hc = prot_cache[j], cod_cache[j]
        fb = fallback[gene]
        gen = []
        copied = 0
        for p, aa in enumerate(prot):
            t = best["pos_map"].get(p)
            if t is not None and t < len(hp) and hp[t] == aa and t < len(hc):
                gen.append(hc[t])
                copied += 1
            else:
                gen.append(fb.get(aa, ref_cod[p] if p < len(ref_cod) else "TAA"))
        # terminal stop: records carry len(protein)+1 codons; copy the neighbour's own stop
        gen.append(hc[len(hp)] if len(hc) > len(hp) else "TAA")
        gen_seq = "".join(gen)

        out.append({
            "idx": qi_idx, "gene": gene, "species": q["species"], "protein_len": len(prot),
            "max_identity_to_train": best["identity_over_query"],
            "max_identity_over_max": best["identity_over_max"],
            "neighbor_species": best["species"],
            "copied_fraction": copied / max(len(prot), 1),
            "codon_identity_to_reference": sum(1 for a, b in zip(ref_cod, gen) if a == b) / max(len(ref_cod), 1),
            "len_ok": len(gen) == len(ref_cod),
            "bleu_homology_copy": simple_bleu(ref, gen_seq),
            "neighbors": [{kk: vv for kk, vv in nb.items() if kk != "pos_map"} for nb in neighbours],
        })
        if n % 500 == 0:
            el = time.time() - t0
            print(f"{n}/{len(query_idx)}  {el:.0f}s  mean BLEU {np.mean([o['bleu_homology_copy'] for o in out]):.4f}", flush=True)

    bleu = np.array([o["bleu_homology_copy"] for o in out])
    pid = np.array([o["max_identity_to_train"] for o in out])
    summary = {
        "split": args.split, "n": len(out),
        "all_lengths_ok": bool(all(o["len_ok"] for o in out)),
        "bleu_homology_copy_mean": float(bleu.mean()),
        "bleu_homology_copy_sem": float(bleu.std(ddof=1) / np.sqrt(len(bleu))),
        "mean_copied_fraction": float(np.mean([o["copied_fraction"] for o in out])),
        "identity_to_train": {
            "mean": float(pid.mean()), "median": float(np.median(pid)),
            "p10": float(np.percentile(pid, 10)), "p90": float(np.percentile(pid, 90)),
            "frac_ge_0.9": float((pid >= 0.9).mean()), "frac_ge_0.8": float((pid >= 0.8).mean()),
            "frac_lt_0.6": float((pid < 0.6).mean()),
        },
        "runtime_sec": time.time() - t0,
    }
    outpath = ROOT / f"data/qc_reports/retrieval_{args.split}.json"
    json.dump({"summary": summary, "per_sequence": out}, open(outpath, "w"))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
