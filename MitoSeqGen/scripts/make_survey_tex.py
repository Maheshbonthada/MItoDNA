"""Generate paper/tcbb_submission/1_main_manuscript/survey_table.tex from the literature survey JSON.

Reviewer concern (TCBB review, secondary point 1): the claim that no published
study reports an ortholog-copy control is asserted in prose. A table of papers
surveyed against baselines reported converts that into something checkable.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/tcbb_submission/1_main_manuscript/survey_table.tex"
SRC = ROOT / "data/literature_baseline_survey.json"

# the baseline categories we tabulate, in column order
CATS = [
    ("freq", "Frequency / CAI"),
    ("algo", "Algorithmic"),
    ("neural", "Neural"),
    ("commercial", "Commercial"),
    ("wt", "Wild-type"),
    ("homology", "\\textbf{Ortholog-copy}"),
]


def categorise(paper):
    """Map a paper's reported baselines onto the tabulated categories."""
    names = " ; ".join(paper["baselines"]).lower()
    got = set()
    if any(k in names for k in ("frequency", "cai", "codon usage", "preferred-codon",
                                "most-preferred", "conventional codon")):
        got.add("freq")
    if any(k in names for k in ("lineardesign", "brute-force", "random")):
        got.add("algo")
    if any(k in names for k in ("codontransformer", "icor", "gemorna", "codongpt",
                                "bert", "calm", "mistral", "esm2", "language model")):
        got.add("neural")
    if any(k in names for k in ("twist", "genewiz", "idt", "genscript", "gensmart")):
        got.add("commercial")
    if any(k in names for k in ("wild-type", "natural", "original")):
        got.add("wt")
    if paper["homology_baseline"]:
        got.add("homology")
    return got


def main():
    d = json.load(open(SRC))
    papers = d["papers"]

    L = []
    A = L.append

    A("\\begin{table}[!t]")
    A("\\caption{Baselines reported in the evaluation sections of published "
      "deep-learning codon-optimization and codon-language-model papers. "
      "Compiled by reading each paper's own baseline section. No paper "
      "we could access reports a baseline that copies codons from a retrieved "
      "homolog.}")
    A("\\label{tab:survey}")
    A("\\centering")
    A("\\setlength{\\tabcolsep}{3pt}")
    A("\\begin{tabular}{ll" + "c" * len(CATS) + "}")
    A("\\toprule")
    A("Method & Venue & " + " & ".join(
        "\\rotatebox{90}{%s}" % lab for _, lab in CATS) + " \\\\")
    A("\\midrule")
    n_homology = 0
    for p in papers:
        got = categorise(p)
        if "homology" in got:
            n_homology += 1
        marks = " & ".join("\\checkmark" if k in got else "---" for k, _ in CATS)
        A("%s & %s & %s \\\\" % (p["short"], p["venue"], marks))
    A("\\midrule")
    # we evaluate no commercial tool, so that column is honestly blank for us
    OURS = {"freq", "algo", "neural", "wt", "homology"}
    A("\\textbf{This paper} & --- & " + " & ".join(
        "\\checkmark" if k in OURS else "---" for k, _ in CATS) + " \\\\")
    A("\\bottomrule")
    A("\\end{tabular}")
    A("\\end{table}")
    A("")

    A("Table~\\ref{tab:survey} records what each of %d accessible published "
      "evaluations actually compares against. Frequency and algorithmic "
      "baselines are near-universal and several papers compare against other "
      "neural models, but the ortholog-copy column is empty: %d of %d report "
      "such a control. The closest instance is the mBART codon model, which "
      "conditions on an ortholog and reports a frequency-rank "
      "\\emph{mimicking} model---but not what copying that ortholog's codons "
      "alone achieves, which is the quantity that decides whether the "
      "conditioning is doing any work. We could not obtain the full baseline "
      "tables of two further cited papers, which are therefore counted neither "
      "for nor against this claim. The survey is released as "
      "\\texttt{data/literature\\_baseline\\_survey.json} so that it can be "
      "checked and extended."
      % (len(papers), n_homology, len(papers)))
    A("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines); papers={len(papers)}, "
          f"with ortholog-copy baseline={n_homology}")


if __name__ == "__main__":
    main()
