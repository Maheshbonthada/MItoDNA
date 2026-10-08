"""Build the single LaTeX archive IEEE Research Exchange (ReX) asks for.

ReX's upload page states: "You may bundle LaTeX manuscript files in a single
archive including all LaTeX files, BibTeX files, figures, tables, all LaTeX
classes and packages, and any other material that belongs to your main
manuscript", and allows a maximum of one Main Manuscript file.

So the archive must compile on a machine that is not this one. The only
non-standard dependency is IEEEtran.cls, which is bundled. Everything else the
preamble loads (graphicx, amsmath, amssymb, booktabs, url, hyperref, xcolor) is
part of any normal TeX distribution.

The archive deliberately excludes:
  - supplemental material (ReX: the main manuscript "should not include any
    supplementary materials" -- it is uploaded separately)
  - PNG previews of the figures (the PDFs are what LaTeX embeds)
  - aux/log/out build artefacts

After writing the archive this script unpacks it into a scratch directory and
compiles it there, so a broken bundle cannot reach the portal unnoticed.

Usage:  python scripts/build_submission_archive.py
"""

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper/tcbb_submission/1_main_manuscript"
OUT = SRC / "manuscript_latex_bundle.zip"

TEX_FILES = [
    "manuscript.tex",
    "survey_table.tex",
    "results_hard_split.tex",
    "results_cross_corpus.tex",
    "results_overlap.tex",
]


def find_cls():
    """Locate IEEEtran.cls via kpsewhich, falling back to a MiKTeX tree walk."""
    try:
        r = subprocess.run(["kpsewhich", "IEEEtran.cls"], capture_output=True,
                           text=True, timeout=60)
        p = Path(r.stdout.strip())
        if r.returncode == 0 and p.is_file():
            return p
    except Exception:
        pass
    for base in [Path.home() / "AppData/Local/Programs/MiKTeX",
                 Path("C:/Program Files/MiKTeX")]:
        if base.exists():
            for p in base.rglob("IEEEtran.cls"):
                return p
    return None


def page_count(pdf: Path):
    d = pdf.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", d))


def main():
    cls = find_cls()
    if cls is None:
        sys.exit("ERROR: IEEEtran.cls not found; cannot bundle a self-contained archive")
    print(f"IEEEtran.cls: {cls}")

    figs = sorted((SRC / "figures").glob("*.pdf"))
    if not figs:
        sys.exit("ERROR: no figure PDFs found")

    missing = [f for f in TEX_FILES if not (SRC / f).is_file()]
    if missing:
        sys.exit(f"ERROR: missing source files: {missing}")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in TEX_FILES:
            z.write(SRC / f, f)
        z.write(cls, "IEEEtran.cls")
        for f in figs:
            z.write(f, f"figures/{f.name}")

    names = zipfile.ZipFile(OUT).namelist()
    print(f"\nwrote {OUT.name}  ({OUT.stat().st_size/1024:.0f} KB, {len(names)} files)")
    for n in names:
        print("   ", n)

    # ---- compile the archive somewhere else, exactly as a stranger would ----
    print("\nverifying the archive compiles from a clean directory ...")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(OUT) as z:
            z.extractall(tmp)
        for i in range(2):
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
                cwd=tmp, capture_output=True, text=True, timeout=600)
        pdf = tmp / "manuscript.pdf"
        if not pdf.is_file():
            print(r.stdout[-3000:])
            sys.exit("ERROR: archive did not produce a PDF")

        log = (tmp / "manuscript.log").read_text(errors="ignore")
        errors = [l for l in log.splitlines() if l.startswith("!")]
        overfull = [l for l in log.splitlines() if "Overfull" in l]
        undef = [l for l in log.splitlines()
                 if "undefined" in l.lower() and "Warning" in l]
        pages = page_count(pdf)

        print(f"  pages         : {pages}  (IEEE CS limit for a regular paper: 12)")
        print(f"  LaTeX errors  : {len(errors)}")
        print(f"  overfull hbox : {len(overfull)}")
        print(f"  undefined refs: {len(undef)}")
        for l in errors[:5]:
            print("   !", l)

        if errors:
            sys.exit("ERROR: archive compiles with errors")
        if pages > 12:
            sys.exit(f"ERROR: {pages} pages exceeds the 12-page limit "
                     f"($220 per overlength page)")

        # keep the independently built PDF beside the archive as proof
        shutil.copy(pdf, SRC / "manuscript.pdf")
        print(f"  refreshed manuscript.pdf from the archive build")

    print("\nARCHIVE OK -- this is the single file to upload as Main Manuscript")


if __name__ == "__main__":
    main()
