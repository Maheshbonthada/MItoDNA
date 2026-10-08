"""Homology-copy baseline + test-to-train identity audit.

Question this answers: does MitoSeqGen's BLEU-4 advantage reflect learned
codon-choice, or could a model that merely copies codons from the nearest
training ortholog (same gene, different species) do just as well?

For every held-out test protein we
  1. find the most similar same-gene TRAINING protein (3-mer prefilter, then
     global alignment),
  2. report protein identity and codon (nucleotide) identity to that neighbour,
  3. build a "copy" sequence: at each aligned position with an identical amino
     acid, copy the neighbour's codon; elsewhere use the gene-specific most
     frequent mitochondrial-code codon for that amino acid,
  4. score BLEU-4 against the real sequence with the paper's own simple_bleu.

Outputs data/qc_reports/homology_copy_baseline.json
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from Bio import Align

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import simple_bleu  # noqa: E402
from src.models.constraints import SYNONYM_TABLE  # noqa: E402

K = 3
TOP_CANDIDATES = 8


def codons(seq):
    seq = seq.upper()
    return [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]


def main():
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    train_idx = json.load(open(ROOT / "data/splits/train.json"))["indices"]
    test_idx = json.load(open(ROOT / "data/splits/test.json"))["indices"]
    # same order MitoSeqGen's evaluation used (rng.sample of the full test set, seed 42)
    order = random.Random(42).sample(test_idx, len(test_idx))

    by_gene = defaultdict(list)
    for i in train_idx:
        by_gene[recs[i]["gene_name"]].append(i)

    # gene-specific most frequent codon per amino acid (fallback for non-identical positions)
    fallback = {}
    inv_index = {}
    for gene, idxs in by_gene.items():
        cnt = defaultdict(Counter)
        inv = defaultdict(list)
        for i in idxs:
            p = recs[i]["protein_sequence"].rstrip("*")
            cs = codons(recs[i]["sequence"])
            for a, c in zip(p, cs):
                cnt[a][c] += 1
            for kmer in {p[j:j + K] for j in range(len(p) - K + 1)}:
                inv[kmer].append(i)
        fallback[gene] = {a: c.most_common(1)[0][0] for a, c in cnt.items()}
        inv_index[gene] = inv

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 2, -1
    aligner.open_gap_score, aligner.extend_gap_score = -4, -0.5

    out = []
    for n, ti in enumerate(order, 1):
        t = recs[ti]
        gene = t["gene_name"]
        prot = t["protein_sequence"].rstrip("*")
        ref = t["sequence"]
        votes = Counter()
        for kmer in {prot[j:j + K] for j in range(len(prot) - K + 1)}:
            for c in inv_index[gene].get(kmer, ()):
                votes[c] += 1
        cands = [c for c, _ in votes.most_common(TOP_CANDIDATES)] or by_gene[gene][:TOP_CANDIDATES]

        best = None
        for c in cands:
            hp = recs[c]["protein_sequence"].rstrip("*")
            aln = aligner.align(prot, hp)[0]
            blocks_q, blocks_t = aln.aligned
            ident = sum(1 for (qs, qe), (ts, te) in zip(blocks_q, blocks_t)
                        for a, b in zip(prot[qs:qe], hp[ts:te]) if a == b)
            if best is None or ident > best[0]:
                best = (ident, c, aln)
        ident, c, aln = best
        hp_cod = codons(recs[c]["sequence"])
        pos_map = {}
        for (qs, qe), (ts, te) in zip(*aln.aligned):
            for q, h in zip(range(qs, qe), range(ts, te)):
                pos_map[q] = h
        hp = recs[c]["protein_sequence"].rstrip("*")
        fb = fallback[gene]
        syn_fb = lambda a: fb.get(a) or SYNONYM_TABLE[a][0]
        gen = []
        copied = 0
        for q, a in enumerate(prot):
            h = pos_map.get(q)
            if h is not None and h < len(hp_cod) and hp[h] == a:
                gen.append(hp_cod[h])
                copied += 1
            else:
                gen.append(syn_fb(a))
        ref_cod = codons(ref)
        if len(ref_cod) > len(prot):  # reference carries a terminal stop codon
            gen.append(hp_cod[-1] if hp_cod and len(hp_cod) > len(hp) else ref_cod[-1])
        gen_seq = "".join(gen)
        nt_match = sum(1 for a, b in zip(ref_cod, gen) if a == b) / max(len(ref_cod), 1)
        out.append({
            "idx": ti, "gene": gene, "species": t["species"], "protein_len": len(prot),
            "neighbor_species": recs[c]["species"],
            "protein_identity_to_nearest_train": ident / max(len(prot), len(hp)),
            "codon_identity_of_copy_to_reference": nt_match,
            "bleu_homology_copy": simple_bleu(ref, gen_seq),
        })
        if n % 250 == 0:
            print(f"{n}/{len(order)}  mean BLEU so far {np.mean([o['bleu_homology_copy'] for o in out]):.3f}", flush=True)

    pid = np.array([o["protein_identity_to_nearest_train"] for o in out])
    bleu = np.array([o["bleu_homology_copy"] for o in out])
    summary = {
        "n": len(out),
        "protein_identity_to_nearest_train": {
            "mean": float(pid.mean()), "median": float(np.median(pid)),
            "p10": float(np.percentile(pid, 10)), "p90": float(np.percentile(pid, 90)),
            "frac_ge_0.9": float((pid >= 0.9).mean()), "frac_ge_0.8": float((pid >= 0.8).mean()),
        },
        "bleu_homology_copy_mean": float(bleu.mean()),
        "bleu_homology_copy_sem": float(bleu.std(ddof=1) / np.sqrt(len(bleu))),
    }

    # paired comparison with MitoSeqGen (same sample order as evaluate.py, seed 42)
    rep = json.load(open(ROOT / "data/qc_reports/evaluation_report_full.json"))
    ms = np.array([r["bleu"] for r in rep["raw_results"]["mitoseqgen"]])
    if len(ms) == len(bleu):
        d = ms - bleu
        summary["mitoseqgen_bleu_mean"] = float(ms.mean())
        summary["mean_diff_mitoseqgen_minus_copy"] = float(d.mean())
        summary["frac_proteins_mitoseqgen_better"] = float((d > 0).mean())
        # species-level bootstrap (resample species, not sequences)
        sp = np.array([o["species"] for o in out])
        uniq = np.unique(sp)
        groups = {s: np.where(sp == s)[0] for s in uniq}
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            pick = rng.choice(uniq, len(uniq))
            idx = np.concatenate([groups[s] for s in pick])
            boots.append(d[idx].mean())
        summary["species_bootstrap_95CI_diff"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        summary["n_test_species"] = int(len(uniq))
    else:
        summary["warning"] = f"MitoSeqGen raw results length {len(ms)} != {len(bleu)}; no paired comparison"

    json.dump({"summary": summary, "per_protein": out},
              open(ROOT / "data/qc_reports/homology_copy_baseline.json", "w"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
