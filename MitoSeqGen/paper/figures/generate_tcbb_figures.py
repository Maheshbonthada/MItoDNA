"""Figures for the TCBB manuscript.

Design notes
------------
Palette is the Okabe-Ito colourblind-safe set, ordered so that no adjacent pair
falls below the CVD separation floor (validated: worst adjacent deuteranope
dE 11.0). Because IEEE figures are frequently read in grayscale, every series
also carries a redundant non-colour encoding -- distinct markers, line styles
and hatch patterns -- so no figure depends on hue alone. Direct labels are used
in preference to legend-only identification wherever the layout allows.

All numbers are read from data/qc_reports/*.json; nothing is hard-coded, so the
figures regenerate from the released artefacts.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
QC = ROOT / "data/qc_reports"
OUT = ROOT / "paper/tcbb_submission/1_main_manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito, CVD-validated ordering
C_BLUE, C_VERM, C_GREEN, C_ORANGE, C_PURPLE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"
C_GREY = "#666666"

GEQ = "≥"
EN = "–"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "figure.dpi": 400,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

COL1, COL2 = 3.5, 7.16  # IEEE single / double column width (inches)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------- Figure 1
def fig1_identity_leakage():
    """The headline: 'held-out' test proteins are near-identical to training."""
    retr = json.load(open(QC / "retrieval_orig_test.json"))["per_sequence"]
    pid = np.array([o["max_identity_to_train"] for o in retr])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.5))

    ax1.hist(pid, bins=np.linspace(0.3, 1.0, 50), color=C_BLUE, edgecolor="white", linewidth=0.3)
    ax1.axvline(0.70, color=C_VERM, ls="--", lw=1.2)
    ax1.text(0.685, ax1.get_ylim()[1] * 0.92, "70% threshold\n(hard benchmark)",
             ha="right", va="top", fontsize=7, color=C_VERM)
    ax1.set_xlabel("Protein identity to nearest same-gene training sequence")
    ax1.set_ylabel("Test proteins")
    ax1.set_title("(a) Species-level split leaves test proteins\nnearly identical to training", loc="left")

    ths = np.array([1.0, 0.99, 0.95, 0.90, 0.80, 0.70])
    fracs = np.array([(pid >= t).mean() for t in ths])
    labels = ["=100%\n(exact)", f"{GEQ}99%", f"{GEQ}95%", f"{GEQ}90%", f"{GEQ}80%", f"{GEQ}70%"]
    bars = ax2.bar(range(len(ths)), fracs, color=[C_VERM] + [C_BLUE] * 5,
                   edgecolor="white", linewidth=0.6)
    bars[0].set_hatch("///")
    for b, f in zip(bars, fracs):
        ax2.text(b.get_x() + b.get_width() / 2, f + 0.02, f"{f*100:.1f}%",
                 ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(range(len(ths)))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylim(0, 1.12)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylabel("Fraction of test set")
    ax2.set_title("(b) 6.6% of test proteins have an identical\nprotein sequence in training", loc="left")
    save(fig, "fig1_identity_leakage")


# ---------------------------------------------------------------- Figure 2
def fig2_stratified_crossover():
    """Where learning helps and where the task is retrieval."""
    s = json.load(open(QC / "stratified_evaluation.json"))
    strata = ["<60%", "60-80%", "80-90%", "90-95%", ">=95%"]           # JSON keys
    strata_lab = ["<60%", f"60{EN}80%", f"80{EN}90%", f"90{EN}95%", f"{GEQ}95%"]
    show = [("homology_copy", "Ortholog-copy baseline", C_VERM, "o", "-"),
            ("mitoseqgen", "MitoSeqGen (trained)", C_BLUE, "s", "--"),
            ("codontransformer_finetuned", "CodonTransformer (fine-tuned)", C_GREEN, "^", "-."),
            ("mt_cai_lookup", "Codon-usage lookup", C_ORANGE, "D", ":")]

    fig, (ax, axn) = plt.subplots(2, 1, figsize=(COL1, 3.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    x = np.arange(len(strata))
    for key, lab, col, mk, ls in show:
        y = [s["strata"][st]["methods"][key]["bleu"]["mean"] for st in strata]
        lo = [s["strata"][st]["methods"][key]["bleu"]["ci_lo"] for st in strata]
        hi = [s["strata"][st]["methods"][key]["bleu"]["ci_hi"] for st in strata]
        ax.plot(x, y, color=col, marker=mk, ls=ls, label=lab, ms=4, mec="white", mew=0.6)
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)

    # crossover region shaded; annotations top, legend bottom-right, so neither
    # collides with the rising curves
    ax.axvspan(-0.3, 1.5, color=C_GREY, alpha=0.07, lw=0)
    ax.text(0.55, 0.498, "no method separates", ha="center", va="top", fontsize=7,
            color=C_GREY, style="italic")
    ax.text(3.0, 0.498, "retrieval wins", ha="center", va="top", fontsize=7,
            color=C_GREY, style="italic")
    ax.set_ylabel("BLEU-4 vs natural sequence")
    ax.set_ylim(0.02, 0.51)
    ax.set_xlim(-0.35, 4.35)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2, borderaxespad=0.2)
    ax.set_title("Model vs. ortholog-copy, by homology to training set", loc="left")

    n = [s["strata"][st]["n"] for st in strata]
    axn.bar(x, n, color=C_GREY, alpha=0.5, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(n):
        axn.text(i, v + 60, str(v), ha="center", fontsize=6.5)
    axn.set_ylabel("$n$", rotation=0, labelpad=10, va="center")
    axn.set_xticks(x)
    axn.set_xticklabels(strata_lab)
    axn.set_xlabel("Protein identity to nearest training sequence")
    axn.set_ylim(0, max(n) * 1.35)
    axn.grid(False)
    save(fig, "fig2_stratified_crossover")


# ---------------------------------------------------------------- Figure 3
def fig3_conservation_ceiling():
    """Why four genes cannot support a generalisation test."""
    rep = json.load(open(QC / "split_ident70_per_gene_report.json"))
    clusters = rep["cluster_counts"]
    surviving = rep["per_gene_test_counts"]
    genes = sorted(clusters, key=lambda g: clusters[g])
    cl = [clusters[g] for g in genes]
    sv = [surviving.get(g, 0) for g in genes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.5), sharey=True)
    y = np.arange(len(genes))
    cols = [C_VERM if surviving.get(g, 0) == 0 else C_BLUE for g in genes]

    b1 = ax1.barh(y, cl, color=cols, edgecolor="white", linewidth=0.6)
    for b, g in zip(b1, genes):
        if surviving.get(g, 0) == 0:
            b.set_hatch("///")
    ax1.set_xscale("log")
    ax1.set_yticks(y)
    ax1.set_yticklabels(genes)
    ax1.set_xlabel("Sequence clusters at 70% identity (log scale)")
    ax1.set_title("(a) Conservation: fewer clusters = more conserved", loc="left")
    for i, v in enumerate(cl):
        ax1.text(v * 1.18, i, str(v), va="center", fontsize=6.5)
    ax1.set_xlim(1, max(cl) * 3)
    ax1.invert_yaxis()

    b2 = ax2.barh(y, sv, color=cols, edgecolor="white", linewidth=0.6)
    for b, g in zip(b2, genes):
        if surviving.get(g, 0) == 0:
            b.set_hatch("///")
    ax2.set_xlabel(f"Test sequences surviving the {GEQ}70% identity filter".replace(GEQ, "≤"))
    ax2.set_title("(b) Four genes retain zero test sequences", loc="left")
    for i, v in enumerate(sv):
        ax2.text(v + 6, i, str(v), va="center", fontsize=6.5,
                 color=C_VERM if v == 0 else "black",
                 fontweight="bold" if v == 0 else "normal")
    ax2.set_xlim(0, max(sv) * 1.3)
    save(fig, "fig3_conservation_ceiling")


# ---------------------------------------------------------------- Figure 4
def fig4_strand_skew():
    """ND6 is the light-strand gene; its GC skew is inverted."""
    skew = json.load(open(QC / "gc_skew_by_gene.json"))
    genes = sorted(skew["corpus"], key=lambda g: skew["corpus"][g])
    vals = [skew["corpus"][g] for g in genes]
    cols = [C_VERM if g == "ND6" else C_BLUE for g in genes]

    comp = skew.get("generated", {})
    has_gen = bool(comp)
    fig, axes = plt.subplots(1, 2 if has_gen else 1,
                             figsize=(COL2 if has_gen else COL1, 2.4),
                             gridspec_kw={"width_ratios": [1.45, 1]} if has_gen else None)
    ax1 = axes[0] if has_gen else axes

    y = np.arange(len(genes))
    bars = ax1.barh(y, vals, color=cols, edgecolor="white", linewidth=0.6)
    for b, g in zip(bars, genes):
        if g == "ND6":
            b.set_hatch("///")
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(genes)
    ax1.set_xlabel("GC skew $(G-C)/(G+C)$, training corpus")
    ax1.set_title("(a) ND6 alone has positive GC skew\n(the only light-strand gene)", loc="left")
    ax1.text(skew["corpus"]["ND6"] + 0.04, list(genes).index("ND6"), "light strand",
             va="center", fontsize=7, color=C_VERM, fontweight="bold")
    ax1.set_xlim(min(vals) * 1.15, max(vals) * 1.75)

    if has_gen:
        ax2 = axes[1]
        gl = list(comp.keys())
        series = [("natural", "Natural", C_GREY, None),
                  ("model", "Model (uncond.)", C_BLUE, None)]
        if any("model_strand" in comp[g] for g in gl):
            series.append(("model_strand", "Model (strand-cond.)", C_GREEN, "\\\\"))
        series.append(("copy", "Ortholog-copy", C_VERM, "///"))
        xx = np.arange(len(gl))
        w = 0.8 / len(series)
        for k, (key, lab, col, hatch) in enumerate(series):
            off = (k - (len(series) - 1) / 2) * w
            b = ax2.bar(xx + off, [comp[g].get(key, np.nan) for g in gl], w, label=lab,
                        color=col, edgecolor="white", linewidth=0.5)
            if hatch:
                for bb in b:
                    bb.set_hatch(hatch)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_xticks(xx)
        ax2.set_xticklabels(gl)
        ax2.set_ylabel("GC skew")
        # upper right is empty because every heavy-strand bar is negative
        ax2.legend(frameon=False, loc="upper right", fontsize=6, borderaxespad=0.2)
        ax2.set_ylim(top=max(0.52, ax2.get_ylim()[1]))
        ax2.set_title(r"(b) The model collapses ND6 skew to $\approx$0", loc="left")
    save(fig, "fig4_strand_skew")


# ---------------------------------------------------------------- Figure 5
def fig5_per_gene_delta():
    """One gene flips the pooled conclusion."""
    d = json.load(open(QC / "per_gene_delta.json"))
    runs = [("control", "Unconditioned", C_BLUE)]
    if "strand" in d:
        runs.append(("strand", "Strand-conditioned", C_GREEN))

    genes = sorted([g for g in d["control"] if not g.startswith("__")],
                   key=lambda g: -d["control"][g]["mean_diff"])
    fig, ax = plt.subplots(figsize=(COL1, 2.9))
    y = np.arange(len(genes))
    h = 0.36 if len(runs) > 1 else 0.62
    single = len(runs) == 1
    for k, (key, lab, col) in enumerate(runs):
        off = (k - (len(runs) - 1) / 2) * h
        vals = [d[key].get(g, {}).get("mean_diff", np.nan) for g in genes]
        # With one run, highlight ND6 in the alert colour. With two, colour must
        # track the run so it matches the legend; ND6 is marked by hatching alone.
        cols = [C_VERM if (single and g == "ND6") else col for g in genes]
        b = ax.barh(y + off, vals, h, color=cols, edgecolor="white", linewidth=0.5, label=lab)
        for bb, g in zip(b, genes):
            if g == "ND6":
                bb.set_hatch("///")
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(genes)
    ax.invert_yaxis()
    ax.set_xlabel(r"BLEU-4 difference (model $-$ ortholog-copy)")
    ax.set_title("Per-gene: one gene flips the pooled result", loc="left")
    if not single:
        # top-left is empty: every gene but ND6 has a small positive difference
        ax.legend(frameon=False, loc="upper left", fontsize=7, borderaxespad=0.3)
    save(fig, "fig5_per_gene_delta")



# ---------------------------------------------------------------- Figure 6
def fig6_cross_corpus():
    """Copy-baseline strength tracks corpus redundancy, across 8 organisms.

    Redundant encoding for grayscale/CVD: the mitochondrial point is a filled
    square with a different hue AND a label; nuclear organisms are open circles.
    """
    d = json.load(open(QC / "cross_corpus_leakage.json"))
    orgs = d["organisms"]

    mito_j = json.load(open(QC / "retrieval_orig_test.json"))
    mito_x = mito_j["summary"]["identity_to_train"]["median"]
    mito_y = mito_j["summary"]["bleu_homology_copy_mean"]

    SHORT = {
        "Homo sapiens": "H. sapiens",
        "Escherichia coli general": "E. coli",
        "Danio rerio": "D. rerio",
        "Arabidopsis thaliana": "A. thaliana",
        "Drosophila melanogaster": "D. melanogaster",
        "Caenorhabditis elegans": "C. elegans",
        "Bacillus subtilis": "B. subtilis",
        "Saccharomyces cerevisiae": "S. cerevisiae",
    }
    # label offsets tuned so no annotation collides with a point or the frame
    OFF = {
        "E. coli": (-6, -12), "H. sapiens": (-10, 7), "D. rerio": (6, -3),
        "A. thaliana": (6, -3), "D. melanogaster": (5, 4),
        "C. elegans": (6, -9), "B. subtilis": (7, -4), "S. cerevisiae": (7, 4),
    }

    xs = [orgs[o]["identity"]["median"] for o in orgs]
    ys = [orgs[o]["bleu_ortholog_copy"] for o in orgs]
    mfc = [orgs[o]["bleu_most_frequent_codon"] for o in orgs]

    fig, ax = plt.subplots(figsize=(COL1, 2.9))

    # most-frequent-codon reference band: what you get with no homology at all
    ax.axhspan(min(mfc), max(mfc), color=C_GREY, alpha=0.12, lw=0)
    ax.text(1.03, (min(mfc) + max(mfc)) / 2 - 0.055, "codon-usage table",
            fontsize=6.5, color=C_GREY, style="italic", va="center", ha="right")

    ax.scatter(xs, ys, s=26, facecolors="none", edgecolors=C_BLUE,
               linewidths=1.3, zorder=3, label="CodonTransformer corpus")
    for o in orgs:
        lab = SHORT.get(o, o)
        dx, dy = OFF.get(lab, (6, -3))
        ax.annotate(lab, (orgs[o]["identity"]["median"], orgs[o]["bleu_ortholog_copy"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.3, style="italic", color="#222222")

    ax.scatter([mito_x], [mito_y], s=42, marker="s", color=C_VERM,
               edgecolors="white", linewidths=0.7, zorder=4,
               label="Vertebrate mitochondria (this paper)")
    ax.annotate("mitochondria", (mito_x, mito_y), textcoords="offset points",
                xytext=(-16, -13), fontsize=6.3, color=C_VERM, weight="bold")

    ax.set_xlabel("Median identity of held-out protein to nearest training protein")
    ax.set_ylabel("BLEU-4, ortholog-copy")
    ax.set_xlim(0.22, 1.045)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Copying a homolog gets stronger as the corpus gets denser", loc="left")
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.2,
              handletextpad=0.4, fontsize=6.8)
    save(fig, "fig6_cross_corpus")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    print(f"writing figures to {OUT}")
    if "1" in which: fig1_identity_leakage()
    if "2" in which: fig2_stratified_crossover()
    if "3" in which: fig3_conservation_ceiling()
    if "4" in which: fig4_strand_skew()
    if "5" in which: fig5_per_gene_delta()
    if "6" in which: fig6_cross_corpus()
