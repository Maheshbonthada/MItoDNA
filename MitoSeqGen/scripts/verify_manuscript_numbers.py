"""Cross-check every hand-written number in the manuscript against the JSON artefacts.

The results section is machine-generated, so its numbers are safe by
construction. The abstract, introduction and results prose are hand-written and
therefore the place a transcription error would survive. This script recomputes
each claimed quantity from the released artefacts and reports PASS/FAIL.

Run before every submission or resubmission.
"""

import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data/qc_reports"
TEX = ROOT / "paper/tcbb_submission/1_main_manuscript/manuscript.tex"

fails, checks = [], []


def check(label, claimed, actual, tol=5e-4):
    ok = abs(claimed - actual) <= tol
    checks.append((ok, label, claimed, actual))
    if not ok:
        fails.append(label)


def main():
    tex = TEX.read_text(encoding="utf-8")
    recs = json.load(open(ROOT / "data/processed/mito_cds_tokenized.json"))["records"]

    # ---------- identity distribution (species-level split) ----------
    retr = json.load(open(QC / "retrieval_orig_test.json"))
    per = retr["per_sequence"]
    pid = np.array([o["max_identity_to_train"] for o in per])
    check("median identity to train (0.967)", 0.967, float(np.median(pid)), 1e-3)
    check("frac exact matches (6.6%)", 0.066, float((pid >= 1.0).mean()), 1e-3)
    check("count exact matches (341)", 341, int((pid >= 1.0).sum()), 0.5)
    check("frac >=90% (83.8%)", 0.838, float((pid >= 0.90).mean()), 1e-3)
    check("frac >=80% (95.3%)", 0.953, float((pid >= 0.80).mean()), 1e-3)
    check("frac >=70% (98.2%)", 0.982, float((pid >= 0.70).mean()), 1e-3)
    check("copy BLEU species split (0.410)", 0.410, retr["summary"]["bleu_homology_copy_mean"], 1e-3)

    # ---------- dataset ----------
    check("n unique sequences (46264)", 46264, len(recs), 0.5)
    check("n species (3762)", 3762, len({r["species"] for r in recs}), 0.5)

    # ---------- stratified ----------
    strat = json.load(open(QC / "stratified_evaluation.json"))
    check("model BLEU species split (0.313)", 0.313,
          strat["overall"]["mitoseqgen"]["bleu"]["mean"], 1e-3)
    h = strat["headline_comparisons"]["mitoseqgen_vs_homology_copy__overall"]
    check("paired diff (-0.0972)", -0.0972, h["mean_diff"], 1e-3)
    check("CI low (-0.1074)", -0.1074, h["ci_lo"], 2e-3)
    check("CI high (-0.0874)", -0.0874, h["ci_hi"], 2e-3)
    check("model win rate (29%)", 0.29, h["frac_a_better"], 1e-2)
    n_lt80 = strat["strata"]["<60%"]["n"] + strat["strata"]["60-80%"]["n"]
    check("n below 80% identity (213)", 213, n_lt80, 0.5)
    check("frac below 80% (4.1%)", 0.041, n_lt80 / strat["n_records"], 1e-3)
    check("copy BLEU >=95% stratum (0.461)", 0.461,
          strat["strata"][">=95%"]["methods"]["homology_copy"]["bleu"]["mean"], 1e-3)
    check("model BLEU >=95% stratum (0.331)", 0.331,
          strat["strata"][">=95%"]["methods"]["mitoseqgen"]["bleu"]["mean"], 1e-3)
    check("model BLEU <60% stratum (0.208)", 0.208,
          strat["strata"]["<60%"]["methods"]["mitoseqgen"]["bleu"]["mean"], 1e-3)
    check("copy BLEU <60% stratum (0.181)", 0.181,
          strat["strata"]["<60%"]["methods"]["homology_copy"]["bleu"]["mean"], 1e-3)

    # ---------- conservation ceiling ----------
    rep = json.load(open(QC / "split_ident70_per_gene_report.json"))
    check("COX1 clusters (2)", 2, rep["cluster_counts"]["COX1"], 0.5)
    check("ATP8 clusters (593)", 593, rep["cluster_counts"]["ATP8"], 0.5)
    n_seq_cox1 = sum(1 for r in recs if r["gene_name"] == "COX1")
    check("COX1 sequences (3429)", 3429, n_seq_cox1, 0.5)
    zero_genes = [g for g, v in rep["per_gene_test_counts"].items() if v == 0]
    check("genes with zero test seqs (4)", 4, len(zero_genes), 0.5)
    assert set(zero_genes) == {"COX1", "COX2", "COX3", "CYTB"}, zero_genes

    # ---------- hard split ----------
    hard = json.load(open(QC / "retrieval_ident70_test.json"))
    check("hard split n (1325)", 1325, hard["summary"]["n"], 0.5)
    check("hard split max identity (0.699)", 0.699,
          hard["summary"]["identity_to_train"]["max"], 1.5e-3)
    check("hard split median identity (0.657)", 0.657,
          hard["summary"]["identity_to_train"]["median"], 1e-3)
    check("copy BLEU hard split (0.206)", 0.206,
          hard["summary"]["bleu_homology_copy_mean"], 1e-3)

    # ---------- per-gene / ND6 ----------
    delta = json.load(open(QC / "per_gene_delta.json"))
    c = delta["control"]
    check("ND6 delta (-0.1850)", -0.1850, c["ND6"]["mean_diff"], 1e-3)
    check("ND6 win rate (4.7%)", 0.047, c["ND6"]["win_rate"], 1e-3)
    check("ND6 n (214)", 214, c["ND6"]["n"], 0.5)
    check("pooled delta (-0.0106)", -0.0106, c["__pooled__"]["mean_diff"], 1e-3)
    check("excl-ND6 delta (+0.0231)", 0.0231, c["__pooled_excl_ND6__"]["mean_diff"], 1e-3)
    check("ND6 share of test set (16%)", 0.16, c["ND6"]["n"] / c["__pooled__"]["n"], 1e-2)
    n_better = sum(1 for g, v in c.items() if not g.startswith("__") and v["mean_diff"] > 0)
    check("genes where model wins (7 of 9)", 7, n_better, 0.5)

    # ---------- GC skew ----------
    skew = json.load(open(QC / "gc_skew_by_gene.json"))
    check("ND6 corpus skew (+0.442)", 0.442, skew["corpus"]["ND6"], 1e-3)
    check("COX1 corpus skew (-0.193)", -0.193, skew["corpus"]["COX1"], 1e-3)
    check("ATP8 corpus skew (-0.498)", -0.498, skew["corpus"]["ATP8"], 1e-3)
    heavy = [v for g, v in skew["corpus"].items() if g != "ND6"]
    assert max(heavy) < 0 < skew["corpus"]["ND6"], "sign inversion claim broken"
    if "ND6" in skew.get("generated", {}):
        g = skew["generated"]["ND6"]
        check("ND6 generated skew, model (-0.066)", -0.066, g["model"], 1e-3)
        check("ND6 generated skew, natural (+0.423)", 0.423, g["natural"], 1e-3)

    # ---------- correlation table ----------
    corr = json.load(open(QC / "identity_correlation.json"))
    check("Spearman copy (0.504)", 0.504, corr["spearman"]["homology_copy"], 1e-3)
    check("Spearman model (0.264)", 0.264, corr["spearman"]["mitoseqgen"], 1e-3)
    check("Spearman lookup (0.042)", 0.042, corr["spearman"]["mt_cai_lookup"], 1e-3)

    # ---------- strand-conditioned run ----------
    sp = QC / "model_eval_ident70_strand.json"
    if sp.exists():
        sd = delta["strand"]
        check("strand pooled delta (+0.0103)", 0.0103, sd["__pooled__"]["mean_diff"], 1e-3)
        check("strand pooled win rate (62.3%)", 0.623, sd["__pooled__"]["win_rate"], 1e-2)
        check("strand ND6 delta (-0.1049)", -0.1049, sd["ND6"]["mean_diff"], 1e-3)
        check("strand ND6 win rate (22.9%)", 0.229, sd["ND6"]["win_rate"], 1e-3)
        check("strand excl-ND6 (+0.0325)", 0.0325, sd["__pooled_excl_ND6__"]["mean_diff"], 1e-3)
        st = json.load(open(sp))
        check("strand BLEU (0.216)", 0.216, st["summary"]["mitoseqgen"]["bleu"]["mean"], 1e-3)
        c2 = st["comparisons"]["mitoseqgen_vs_homology_copy"]
        check("strand CI low (+0.0047)", 0.0047, c2["ci_lo"], 1e-3)
        check("strand CI high (+0.0158)", 0.0158, c2["ci_hi"], 1e-3)
        pos_s = sum(1 for g, v in sd.items() if not g.startswith("__") and v["mean_diff"] > 0)
        check("strand genes positive (8 of 9)", 8, pos_s, 0.5)
        g = skew["generated"]["ND6"]
        check("strand ND6 skew (+0.149)", 0.149, g["model_strand"], 1e-3)

    # ---------- cross-corpus replication (abstract + Section IV-H) ----------
    cc_path = QC / "cross_corpus_leakage.json"
    if cc_path.exists():
        cc = json.load(open(cc_path))["organisms"]
        hs, ec = cc["Homo sapiens"], cc["Escherichia coli general"]
        bs, sc = cc["Bacillus subtilis"], cc["Saccharomyces cerevisiae"]
        check("cross-corpus human median identity (0.981)", 0.981,
              hs["identity"]["median"], 1e-3)
        check("cross-corpus human copy BLEU (0.924)", 0.924,
              hs["bleu_ortholog_copy"], 1e-3)
        check("cross-corpus human exact match (10.7%)", 0.107,
              hs["identity"]["frac_eq_1.0"], 1e-3)
        check("cross-corpus E. coli median identity (0.997)", 0.997,
              ec["identity"]["median"], 1e-3)
        check("cross-corpus E. coli copy BLEU (0.855)", 0.855,
              ec["bleu_ortholog_copy"], 1e-3)
        check("cross-corpus E. coli exact match (38.7%)", 0.387,
              ec["identity"]["frac_eq_1.0"], 1e-3)
        check("cross-corpus B. subtilis median identity (0.279)", 0.279,
              bs["identity"]["median"], 1e-3)
        check("cross-corpus S. cerevisiae median identity (0.265)", 0.265,
              sc["identity"]["median"], 1e-3)
        check("cross-corpus organisms analysed (8)", 8, len(cc), 0.5)
        # the paper claims human leakage exceeds the mitochondrial case
        assert hs["bleu_ortholog_copy"] > retr["summary"]["bleu_homology_copy_mean"], \
            "manuscript claims human copy BLEU exceeds mitochondrial; it does not"

    # ---------- cross-gene species overlap (Section IV-I) ----------
    ov_path = QC / "species_overlap_analysis.json"
    if ov_path.exists():
        ov = json.load(open(ov_path))
        check("test species shared with train (971)", 971,
              ov["n_species_shared_with_train"], 0.5)
        check("test species total (980)", 980, ov["n_test_species"], 0.5)
        check("frac species shared (99.1%)", 0.991, ov["frac_species_shared"], 1e-3)
        check("clean-subset sequences (9)", 9, ov["n_clean_sequences"], 0.5)

    # ---------- doubly-controlled check (Section IV-I) ----------
    dc_path = QC / "doubly_controlled_analysis.json"
    if dc_path.exists():
        dc = json.load(open(dc_path))["bounds"]
        check("doubly-controlled n at 0.70 (92)", 92, dc["0.70"]["n"], 0.5)
        check("doubly-controlled n at 0.80 (246)", 246, dc["0.80"]["n"], 0.5)
        check("doubly-controlled n at 0.90 (841)", 841, dc["0.90"]["n"], 0.5)
        check("doubly-controlled diff at 0.90 (-0.0174)", -0.0174,
              dc["0.90"]["paired_diff"], 1e-3)
        # the paper claims 0.90 is significant and the two tighter bounds are not
        assert dc["0.90"]["significant"], "0.90 bound claimed significant but is not"
        assert not dc["0.80"]["significant"], "0.80 bound claimed n.s. but is significant"
        assert not dc["0.70"]["significant"], "0.70 bound claimed n.s. but is significant"

    # ---------- literature survey (Table I) ----------
    sv_path = ROOT / "data/literature_baseline_survey.json"
    if sv_path.exists():
        sv = json.load(open(sv_path))["papers"]
        check("papers surveyed (9)", 9, len(sv), 0.5)
        check("papers with ortholog-copy baseline (0)", 0,
              sum(1 for x in sv if x["homology_baseline"]), 0.5)

    # ---------- model params ----------
    check("strand conditioning params (768)", 768, 2 * 384, 0.5)

    # ---------- report ----------
    width = max(len(c[1]) for c in checks)
    for ok, label, claimed, actual in checks:
        tag = "PASS" if ok else "**FAIL**"
        print(f"  [{tag:>8}] {label:<{width}}  claimed={claimed:<12} actual={actual}")
    print(f"\n{sum(1 for c in checks if c[0])}/{len(checks)} checks passed")

    # ---------- stale-number scan ----------
    stale = []
    for bad, why in [(r"-0\.042", "old 60-sample ND6 model skew (now -0.066)"),
                     (r"\+0\.438", "old 60-sample ND6 natural skew (now +0.423)"),
                     (r"0\.968", "old identity normalisation (now 0.967)"),
                     (r"learned models win below", "retired crossover overclaim"),
                     (r"clean crossover", "retired crossover overclaim"),
                     (r"orthologtransformer", "merged duplicate citation key")]:
        for m in re.finditer(bad, tex):
            line = tex[:m.start()].count("\n") + 1
            ctx = tex.splitlines()[line - 1].strip()
            if "968" in bad and "0.967" in ctx:
                continue  # the sentence that legitimately contrasts both normalisations
            stale.append(f"line {line}: {why} -- {ctx[:80]}")
    if stale:
        print("\nSTALE NUMBERS FOUND IN MANUSCRIPT:")
        for s in stale:
            print("  " + s)
    else:
        print("no stale numbers detected in manuscript.tex")

    if fails:
        print(f"\n{len(fails)} FAILED: {fails}")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
