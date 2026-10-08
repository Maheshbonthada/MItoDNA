"""Strand / per-gene analysis feeding Figures 4 and 5.

Produces two artefacts:

  gc_skew_by_gene.json
    corpus    : mean GC skew (G-C)/(G+C) per gene over the training split
    generated : per gene, mean skew of natural / model-generated / ortholog-copy
                sequences (needs generated sequences, so only genes present in
                the model_eval_* files appear)

  per_gene_delta.json
    for each run (control, strand, ...): per-gene mean BLEU difference
    model - ortholog-copy, the win rate, and n

Both are read directly by paper/figures/generate_tcbb_figures.py, so every
number in a figure traces back to a released JSON.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data/qc_reports"
sys.path.insert(0, str(ROOT))


def gc_skew(seq):
    s = seq.upper()
    g, c = s.count("G"), s.count("C")
    return (g - c) / (g + c) if (g + c) else 0.0


def main():
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]
    train_idx = json.load(open(ROOT / "data/splits/train_ident70_per_gene.json"))["indices"]

    # ---- corpus GC skew per gene ----
    by_gene = defaultdict(list)
    for i in train_idx:
        by_gene[recs[i]["gene_name"]].append(gc_skew(recs[i]["sequence"]))
    corpus = {g: float(np.mean(v)) for g, v in by_gene.items()}

    # ---- which model_eval files exist? ----
    runs = {}
    for key, fname in [("control", "model_eval_ident70.json"),
                       ("strand", "model_eval_ident70_strand.json")]:
        p = QC / fname
        if p.exists():
            runs[key] = json.load(open(p))
        else:
            print(f"  (skipping {key}: {fname} not present yet)")

    base = json.load(open(QC / "baseline_eval_ident70.json"))["per_sequence"]
    copy_seq = {o["idx"]: o["homology_copy_sequence"]
                for o in json.load(open(QC / "retrieval_ident70_test.json"))["per_sequence"]}

    # ---- generated skew, per gene ----
    # Sequences may come either from an evaluation run that saved them, or from a
    # standalone generation pass (scripts/generate_sequences_cpu.py), which is how
    # they were produced while the GPU was occupied by training.
    def load_sequences(run_key, standalone):
        p = QC / standalone
        if p.exists():
            return {int(k): v for k, v in json.load(open(p))["sequences"].items()}
        r = runs.get(run_key)
        if r and "generated_sequences" in r:
            return {int(k): v for k, v in r["generated_sequences"].items()}
        return None

    generated = {}
    gseq = load_sequences("control", "sequences_ident70_control.json")
    if gseq:
        g_by = defaultdict(lambda: defaultdict(list))
        for i, seq in gseq.items():
            gene = recs[i]["gene_name"]
            g_by[gene]["model"].append(gc_skew(seq))
            g_by[gene]["natural"].append(gc_skew(recs[i]["sequence"]))
            if i in copy_seq:
                g_by[gene]["copy"].append(gc_skew(copy_seq[i]))
        sseq = load_sequences("strand", "sequences_ident70_strand.json")
        if sseq:
            for i, seq in sseq.items():
                g_by[recs[i]["gene_name"]]["model_strand"].append(gc_skew(seq))
        # keep the four genes used in the figure, if available, else the largest
        wanted = [g for g in ["ND6", "ND4", "ATP8", "ND5"] if g in g_by]
        for g in wanted:
            generated[g] = {k: float(np.mean(v)) for k, v in g_by[g].items()}
    else:
        print("  (no generated sequences available yet -- rerun after model evaluation)")

    json.dump({"corpus": corpus, "generated": generated},
              open(QC / "gc_skew_by_gene.json", "w"), indent=2)

    # ---- per-gene BLEU delta vs ortholog-copy ----
    out = {}
    for key, run in runs.items():
        per = {int(k): v for k, v in run["per_sequence_model"].items()}
        g_by = defaultdict(list)
        for i, m in per.items():
            g_by[recs[i]["gene_name"]].append(m["bleu"] - base[str(i)]["homology_copy"]["bleu"])
        out[key] = {g: {"mean_diff": float(np.mean(v)),
                        "win_rate": float(np.mean(np.array(v) > 0)),
                        "n": len(v)} for g, v in g_by.items()}
        allv = np.concatenate([np.array(v) for v in g_by.values()])
        no_nd6 = np.concatenate([np.array(v) for g, v in g_by.items() if g != "ND6"])
        out[key]["__pooled__"] = {"mean_diff": float(allv.mean()),
                                  "win_rate": float((allv > 0).mean()), "n": int(allv.size)}
        out[key]["__pooled_excl_ND6__"] = {"mean_diff": float(no_nd6.mean()),
                                           "win_rate": float((no_nd6 > 0).mean()),
                                           "n": int(no_nd6.size)}
    json.dump(out, open(QC / "per_gene_delta.json", "w"), indent=2)

    # ---- console report ----
    print("\nGC skew per gene (training corpus):")
    for g in sorted(corpus, key=corpus.get):
        flag = "   <-- LIGHT STRAND" if g == "ND6" else ""
        print(f"  {g:6}{corpus[g]:+.3f}{flag}")
    if generated:
        print("\ngenerated vs natural GC skew:")
        for g, d in generated.items():
            parts = "  ".join(f"{k}={v:+.3f}" for k, v in d.items())
            print(f"  {g:6}{parts}")
    for key, d in out.items():
        print(f"\nper-gene BLEU delta (model - copy), run={key}:")
        for g in sorted([g for g in d if not g.startswith("__")],
                        key=lambda g: -d[g]["mean_diff"]):
            print(f"  {g:6}n={d[g]['n']:4d}  {d[g]['mean_diff']:+.4f}  wins {d[g]['win_rate']*100:5.1f}%")
        print(f"  POOLED          {d['__pooled__']['mean_diff']:+.4f}  "
              f"wins {d['__pooled__']['win_rate']*100:.1f}%")
        print(f"  POOLED excl ND6 {d['__pooled_excl_ND6__']['mean_diff']:+.4f}  "
              f"wins {d['__pooled_excl_ND6__']['win_rate']*100:.1f}%")
    print(f"\nwrote {QC/'gc_skew_by_gene.json'} and {QC/'per_gene_delta.json'}")


if __name__ == "__main__":
    main()
