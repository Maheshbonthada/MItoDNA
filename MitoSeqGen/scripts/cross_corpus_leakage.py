"""Cross-corpus replication of the homology-leakage measurement.

Reviewer concern (TCBB review, major point 1): every number in the paper comes
from vertebrate mitochondria -- a closed 13-gene set with extraordinarily dense
ortholog sampling, which is close to the worst case for homology leakage rather
than a representative one. If the median nearest-training-neighbour identity
elsewhere is 60% rather than 96.7%, the headline claim weakens considerably.

This script runs the same two measurements on the CodonTransformer corpus
(Fallahpour et al., Nat. Commun. 16:3205, 2025), the training data of the
leading published multispecies codon optimizer, obtained from its own
HuggingFace release:

  1. Nearest-training-neighbour protein identity for held-out sequences, under
     the random holdout that corpus is conventionally split with.
  2. The ortholog-copy baseline itself: copy codons from that nearest training
     neighbour and score BLEU-4 against the natural sequence, versus the
     organism's most-frequent-codon table.

Scope and honesty notes:
  - Neighbour search is restricted to the SAME ORGANISM. CodonTransformer
    conditions generation on the target organism, so the same-organism pool is
    what a retrieval baseline for that task would have access to. This is the
    direct analogue of our same-gene ortholog pool.
  - Search is EXHAUSTIVE within that pool, with an exact length-ratio bound
    (identity over the longer sequence cannot exceed min(l1,l2)/max(l1,l2)),
    so pruning never changes the answer, only the runtime.
  - We do not run the CodonTransformer model itself here, so this script makes
    no claim about that model's accuracy. It measures the corpus, not the model.

Usage:
  python scripts/cross_corpus_leakage.py --queries 300 --workers 12
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import edlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import simple_bleu  # noqa: E402

CSV_PATH = ROOT / "data/external/codontransformer_dataset.csv"
OUT_PATH = ROOT / "data/qc_reports/cross_corpus_leakage.json"

# Organisms spanning the corpus's taxonomic range, each with enough sequences
# to form a meaningful training pool. Chosen before any identity was measured.
ORGANISMS = [
    "Homo sapiens",
    "Escherichia coli general",
    "Saccharomyces cerevisiae",
    "Arabidopsis thaliana",
    "Danio rerio",
    "Drosophila melanogaster",
    "Caenorhabditis elegans",
    "Bacillus subtilis",
]

TEST_FRAC = 0.10
SPLIT_SEED = 42

_W = {}


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
    """(n_identical, {query_pos: target_pos}) for a global alignment."""
    r = edlib.align(query, target, mode="NW", task="path")
    qi = ti = ident = 0
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
    return ident, pos_map


def _init(prots, order):
    _W["prots"] = prots
    _W["order"] = order   # training indices sorted by protein length


def _nearest(qprot):
    """Exact nearest neighbour by identity-over-longer, with a length bound.

    identity_over_max <= min(lq, lt) / max(lq, lt), so once the candidates'
    length ratio falls below the best identity found so far, no remaining
    candidate can win and the scan stops. Exact, not heuristic.
    """
    prots = _W["prots"]
    order = _W["order"]
    lq = len(qprot)

    lens = [len(prots[j]) for j in order]
    # start at the insertion point for lq and walk outwards
    lo = 0
    hi = len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        if lens[mid] < lq:
            lo = mid + 1
        else:
            hi = mid
    left, right = lo - 1, lo

    best_ident, best_j = -1.0, None
    while left >= 0 or right < len(order):
        # pick the side whose length is closer to lq
        take_right = False
        if left < 0:
            take_right = True
        elif right < len(order) and abs(lens[right] - lq) <= abs(lq - lens[left]):
            take_right = True

        j = order[right] if take_right else order[left]
        lt = lens[right] if take_right else lens[left]
        if take_right:
            right += 1
        else:
            left -= 1

        bound = min(lq, lt) / max(lq, lt)
        if bound <= best_ident:
            # both sides only get worse from here
            if (left < 0 or min(lq, lens[left]) / max(lq, lens[left]) <= best_ident) and \
               (right >= len(order) or min(lq, lens[right]) / max(lq, lens[right]) <= best_ident):
                break
            continue

        t = prots[j]
        ed = edlib.align(qprot, t, mode="NW", task="distance")["editDistance"]
        ident = 1.0 - ed / max(lq, lt)
        if ident > best_ident:
            best_ident, best_j = ident, j

    return best_ident, best_j


def _worker(task):
    qi, qprot, qdna = task
    ident, j = _nearest(qprot)
    return qi, ident, j


def load_organism_rows(wanted):
    """Stream the CSV once, keeping only the organisms we need."""
    rows = defaultdict(list)
    csv.field_size_limit(10 ** 9)
    n = 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            org = row.get("organism")
            if org in wanted:
                p = (row.get("protein") or "").rstrip("_*")
                d = (row.get("dna") or "").upper()
                if p and d and len(d) == 3 * (len(p) + 1):
                    rows[org].append((p, d))
    print(f"scanned {n} rows; kept " +
          ", ".join(f"{o}:{len(v)}" for o, v in rows.items()), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=300,
                    help="held-out queries sampled per organism")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--organisms", nargs="*", default=None)
    args = ap.parse_args()

    wanted = set(args.organisms or ORGANISMS)
    t0 = time.time()
    data = load_organism_rows(wanted)

    results = {}
    for org in sorted(data):
        recs = data[org]
        if len(recs) < 200:
            print(f"[skip] {org}: only {len(recs)} sequences", flush=True)
            continue

        rng = random.Random(SPLIT_SEED)
        idx = list(range(len(recs)))
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * TEST_FRAC))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        prots = {j: recs[j][0] for j in train_idx}
        order = sorted(train_idx, key=lambda j: len(prots[j]))

        # organism-specific most-frequent codon table (the usual heuristic baseline)
        cnt = defaultdict(Counter)
        for j in train_idx:
            p, d = recs[j]
            for aa, c in zip(p, codons_of(d)):
                cnt[aa][c] += 1
        mfc = {aa: c.most_common(1)[0][0] for aa, c in cnt.items()}

        qsample = test_idx[:args.queries]
        tasks = [(j, recs[j][0], recs[j][1]) for j in qsample]

        t1 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init, initargs=(prots, order)) as ex:
            found = list(ex.map(_worker, tasks, chunksize=4))

        idents, copy_bleu, mfc_bleu, copied_frac = [], [], [], []
        for qi, ident, j in found:
            qp, qd = recs[qi]
            ref_cod = codons_of(qd)
            idents.append(ident)
            if j is None:
                continue
            tp, td = recs[j]
            tc = codons_of(td)
            _, pos_map = align_map(qp, tp)
            gen, copied = [], 0
            for pos, aa in enumerate(qp):
                t = pos_map.get(pos)
                if t is not None and t < len(tp) and tp[t] == aa and t < len(tc):
                    gen.append(tc[t]); copied += 1
                else:
                    gen.append(mfc.get(aa, ref_cod[pos] if pos < len(ref_cod) else "TAA"))
            gen.append(tc[len(tp)] if len(tc) > len(tp) else ref_cod[-1])
            copy_bleu.append(simple_bleu(qd, "".join(gen)))
            copied_frac.append(copied / max(len(qp), 1))

            heur = [mfc.get(aa, "TAA") for aa in qp] + [ref_cod[-1]]
            mfc_bleu.append(simple_bleu(qd, "".join(heur)))

        a = np.array(idents)
        cb, mb = np.array(copy_bleu), np.array(mfc_bleu)
        d = cb - mb
        res = {
            "n_total": len(recs), "n_train": len(train_idx), "n_test": len(test_idx),
            "n_queries": len(qsample),
            "identity": {
                "median": float(np.median(a)), "mean": float(a.mean()),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                "frac_eq_1.0": float((a >= 0.9999).mean()),
                "frac_ge_0.9": float((a >= 0.9).mean()),
                "frac_ge_0.8": float((a >= 0.8).mean()),
                "frac_lt_0.6": float((a < 0.6).mean()),
            },
            "bleu_ortholog_copy": float(cb.mean()),
            "bleu_most_frequent_codon": float(mb.mean()),
            "copy_minus_mfc": float(d.mean()),
            "copy_win_rate": float((d > 0).mean()),
            "mean_copied_fraction": float(np.mean(copied_frac)),
            "runtime_sec": time.time() - t1,
        }
        results[org] = res
        print(f"[{org}] n={len(recs)} median identity {res['identity']['median']:.3f} "
              f"| exact {res['identity']['frac_eq_1.0']:.1%} "
              f"| copy BLEU {res['bleu_ortholog_copy']:.3f} vs mfc "
              f"{res['bleu_most_frequent_codon']:.3f} "
              f"({res['runtime_sec']:.0f}s)", flush=True)

    out = {
        "source": "CodonTransformer dataset.csv (HuggingFace adibvafa/CodonTransformer)",
        "accessed": "2026-10-08",
        "split": {"scheme": "random holdout", "test_frac": TEST_FRAC, "seed": SPLIT_SEED},
        "neighbour_pool": "same organism, exhaustive, exact length-ratio pruning",
        "queries_per_organism": args.queries,
        "organisms": results,
        "total_runtime_sec": time.time() - t0,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
