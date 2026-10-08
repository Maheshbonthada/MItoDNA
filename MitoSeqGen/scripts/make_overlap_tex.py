"""Generate paper/tcbb_submission/1_main_manuscript/results_overlap.tex from the JSON artefacts.

Covers the cross-gene species-overlap channel in the identity-controlled split
and the doubly-controlled (species-disjoint AND identity-bounded) check.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data/qc_reports"
OUT = ROOT / "paper/tcbb_submission/1_main_manuscript/results_overlap.tex"


def main():
    ov = json.load(open(QC / "species_overlap_analysis.json"))
    dc = json.load(open(QC / "doubly_controlled_analysis.json"))

    L = []
    A = L.append

    A("\\subsection{The cross-gene species channel}\\label{sec:overlap}")
    A("")
    A("Because the identity bound is enforced per gene, a species may be in "
      "training for one gene and in test for another. Genes within a species "
      "share strand regime, mutational bias and codon pool, so this is a real "
      "leakage channel that the per-gene bound does not close. We measured it. "
      "It is close to total: %d of %d test species (%.1f\\%%) appear in "
      "training under some other gene, leaving only %d test sequences "
      "(%.1f\\%%) whose species is absent from training entirely. The obvious "
      "check---re-scoring on that clean subset---is therefore not estimable "
      "here, and we decline to report a %d-point comparison as though it were."
      % (ov["n_species_shared_with_train"], ov["n_test_species"],
         100 * ov["frac_species_shared"], ov["n_clean_sequences"],
         100 * ov["frac_clean_sequences"], ov["n_clean_sequences"]))
    A("")

    b70 = dc["bounds"].get("0.70")
    b80 = dc["bounds"].get("0.80")
    b90 = dc["bounds"].get("0.90")

    A("A construction that \\emph{is} available closes the channel completely at "
      "the cost of sample size. The original species-level split is "
      "species-disjoint by construction---no test species appears anywhere in "
      "training, for any gene. Intersecting it with an identity bound gives a "
      "test set that is simultaneously species-disjoint and identity-bounded. "
      "At a 0.90 bound ($n{=}%d$, %d species) ortholog-copy still wins, by "
      "$%+.4f$ (95\\%% CI $[%+.4f, %+.4f]$). At 0.80 ($n{=}%d$) and 0.70 "
      "($n{=}%d$) the difference is no longer distinguishable from zero "
      "($%+.4f$, CI $[%+.4f, %+.4f]$ and $%+.4f$, CI $[%+.4f, %+.4f]$)."
      % (b90["n"], b90["n_species"], b90["paired_diff"], b90["ci_lo"], b90["ci_hi"],
         b80["n"], b70["n"],
         b80["paired_diff"], b80["ci_lo"], b80["ci_hi"],
         b70["paired_diff"], b70["ci_lo"], b70["ci_hi"]))
    A("")
    A("This is the same pattern as Section~\\ref{sec:strata}, reproduced under "
      "complete species disjointness: retrieval dominates where homology is "
      "available and nothing separates where it is not. The cross-gene channel "
      "is therefore not what produces the headline result. We nonetheless "
      "report the channel's size, because an identity-controlled split for a "
      "multi-gene organelle cannot close it and remain usable---union-find over "
      "(species, cluster) pairs collapses to a single component through COX1, "
      "as Section~\\ref{sec:ceiling} shows.")
    A("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
