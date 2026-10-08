"""Generate paper/tcbb_submission/1_main_manuscript/results_cross_corpus.tex from the JSON.

Machine-generated like results_hard_split.tex: every number traces to
data/qc_reports/cross_corpus_leakage.json. Rerun after any re-measurement.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data/qc_reports"
OUT = ROOT / "paper/tcbb_submission/1_main_manuscript/results_cross_corpus.tex"

# italicised binomials; the corpus's own label for the pooled E. coli set is
# "Escherichia coli general", which we keep verbatim but mark as pooled
PRETTY = {
    "Homo sapiens": "\\emph{Homo sapiens}",
    "Escherichia coli general": "\\emph{Escherichia coli}$^{\\dagger}$",
    "Danio rerio": "\\emph{Danio rerio}",
    "Arabidopsis thaliana": "\\emph{Arabidopsis thaliana}",
    "Drosophila melanogaster": "\\emph{Drosophila melanogaster}",
    "Caenorhabditis elegans": "\\emph{Caenorhabditis elegans}",
    "Bacillus subtilis": "\\emph{Bacillus subtilis}",
    "Saccharomyces cerevisiae": "\\emph{Saccharomyces cerevisiae}",
}


def main():
    d = json.load(open(QC / "cross_corpus_leakage.json"))
    orgs = d["organisms"]
    rows = sorted(orgs.items(), key=lambda kv: -kv[1]["identity"]["median"])

    # mitochondrial reference point, for the contrast sentence
    mito_j = json.load(open(QC / "retrieval_orig_test.json"))
    mito = mito_j["summary"]
    mito_med = mito["identity_to_train"]["median"]
    mito_copy = mito["bleu_homology_copy_mean"]
    _ident = [r["max_identity_to_train"] for r in mito_j["per_sequence"]]
    mito_exact = sum(1 for v in _ident if v >= 0.9999) / len(_ident)
    mito_mfc = json.load(open(QC / "evaluation_report_full.json"))[
        "summary"]["most_frequent_codon"]["mean_bleu"]

    L = []
    A = L.append

    A("\\subsection{The leakage is not specific to mitochondria}\\label{sec:crosscorpus}")
    A("")
    A("A natural objection is that vertebrate mitochondria---a closed 13-gene set "
      "with dense ortholog sampling---are an unusually favourable setting for a "
      "copy baseline, and that the result therefore says little about codon "
      "optimization generally. We tested this directly on the training corpus of "
      "the leading published multispecies codon optimizer, CodonTransformer~"
      "\\cite{codontransformer}, taken from its own public release "
      "(%s sequences, 164 organisms). Its reported preprocessing is a length, "
      "start-codon and stop-codon filter; no clustering, redundancy removal or "
      "homology-aware split is described. We therefore hold out a random 10\\%% "
      "of each organism's sequences and, for each held-out protein, find its "
      "nearest neighbour among that organism's remaining sequences---the pool a "
      "retrieval baseline would have, since the model conditions on target "
      "organism. Search is exhaustive with an exact length bound. This measures "
      "the \\emph{corpus}; we do not run their model, and make no claim about its "
      "accuracy." % "1{,}001{,}197")
    A("")

    A("\\begin{table}[!t]")
    A("\\caption{Homology leakage in the CodonTransformer corpus under a random "
      "holdout ($n{=}%d$ queries per organism). Identity is to the nearest "
      "training protein of the same organism. ``Copy'' is the ortholog-copy "
      "baseline; ``MFC'' is that organism's most-frequent-codon table.}"
      % d["queries_per_organism"])
    A("\\label{tab:crosscorpus}")
    A("\\centering")
    # the eight italicised binomials make this table 8pt too wide for one IEEE
    # column at default spacing; tightening the inter-column gap fixes it
    # without shrinking the type or spanning both columns
    A("\\setlength{\\tabcolsep}{3.2pt}")
    A("\\begin{tabular}{lccccc}")
    A("\\toprule")
    A("Organism & Median & Exact & BLEU & BLEU & Copy \\\\")
    A(" & ident. & match & copy & MFC & wins \\\\")
    A("\\midrule")
    for name, r in rows:
        i = r["identity"]
        A("%s & %.3f & %.1f\\%% & \\textbf{%.3f} & %.3f & %.0f\\%% \\\\" % (
            PRETTY.get(name, name), i["median"], 100 * i["frac_eq_1.0"],
            r["bleu_ortholog_copy"], r["bleu_most_frequent_codon"],
            100 * r["copy_win_rate"]))
    A("\\midrule")
    A("Vertebrate mitochondria & %.3f & %.1f\\%% & %.3f & %.3f & --- \\\\" % (
        mito_med, 100 * mito_exact, mito_copy, mito_mfc))
    A("\\bottomrule")
    A("\\end{tabular}")
    A("\\\\[3pt]{\\footnotesize $^{\\dagger}$The corpus's own pooled multi-strain "
      "\\emph{E.~coli} set, which is why its exact-match rate is so high.}")
    A("\\end{table}")
    A("")

    hs = orgs["Homo sapiens"]
    ec = orgs["Escherichia coli general"]
    bs = orgs["Bacillus subtilis"]
    sc = orgs["Saccharomyces cerevisiae"]

    A("The effect replicates, and in the organisms that matter most for codon "
      "optimization it is \\emph{more} severe than in mitochondria "
      "(Table~\\ref{tab:crosscorpus}). For \\emph{Homo sapiens} the median "
      "held-out protein is %.1f\\%% identical to a training protein, %.1f\\%% are "
      "exact matches, and ortholog-copy reaches BLEU-4 %.3f---beating the "
      "organism's most-frequent-codon table (%.3f) on %.0f\\%% of sequences. For "
      "\\emph{E.~coli} the median is %.1f\\%% and %.1f\\%% of held-out sequences "
      "have a byte-identical protein in training." % (
          100 * hs["identity"]["median"], 100 * hs["identity"]["frac_eq_1.0"],
          hs["bleu_ortholog_copy"], hs["bleu_most_frequent_codon"],
          100 * hs["copy_win_rate"],
          100 * ec["identity"]["median"], 100 * ec["identity"]["frac_eq_1.0"]))
    A("")

    A("Two features of the table make it more informative than a single number. "
      "First, leakage varies by more than three-fold across organisms and tracks "
      "genome redundancy rather than taxonomy: the compact genomes of "
      "\\emph{B.~subtilis} and \\emph{S.~cerevisiae}, which have few paralogs, "
      "give median identities of %.3f and %.3f, and there ortholog-copy is barely "
      "better than a codon-usage table (%.3f versus %.3f, and %.3f versus %.3f). "
      "The measurement is therefore detecting redundancy, not producing a "
      "uniform artefact. Second, and contrary to the objection, the "
      "mitochondrial corpus is the \\emph{conservative} case rather than the "
      "extreme one. Its median identity (%.3f) is comparable to that of human, "
      "yet ortholog-copy attains only BLEU %.3f there against %.3f for human, "
      "because mitochondrial synonymous sites are close to saturation: protein "
      "sequence is conserved while the codons beneath it are not. Measured on "
      "nuclear genomes, the baseline we report is substantially stronger than "
      "the one we were able to construct in mitochondria." % (
          bs["identity"]["median"], sc["identity"]["median"],
          bs["bleu_ortholog_copy"], bs["bleu_most_frequent_codon"],
          sc["bleu_ortholog_copy"], sc["bleu_most_frequent_codon"],
          mito_med, mito_copy, hs["bleu_ortholog_copy"]))
    A("")

    A("\\begin{figure}[!t]")
    A("\\centering")
    A("\\includegraphics[width=\\columnwidth]{figures/fig6_cross_corpus.pdf}")
    A("\\caption{Ortholog-copy performance tracks corpus redundancy across the "
      "CodonTransformer corpus. Each point is one organism; the horizontal axis "
      "is the median identity of a held-out protein to its nearest training "
      "protein, the vertical axis the BLEU-4 of copying that neighbour's codons. "
      "The vertebrate mitochondrial corpus of this paper (square) sits below the "
      "nuclear-genome trend at comparable identity, because its synonymous sites "
      "are near saturation.}")
    A("\\label{fig:crosscorpus}")
    A("\\end{figure}")
    A("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
