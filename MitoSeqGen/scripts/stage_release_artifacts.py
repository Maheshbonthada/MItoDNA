"""Assemble exactly the artefacts the manuscript claims are public.

Section VII of the manuscript states that the code, the curated dataset, the
trained weights, the identity-controlled split files, the retrieval indices and
every evaluation artefact behind the tables and figures are publicly available.
This script collects the data-side half of that claim into one staging folder
ready to upload to the Hugging Face dataset repo, and reports what is missing
rather than silently skipping it.

The code-side half goes to GitHub and needs no staging: .gitignore already
excludes data/, *.pt and experiment checkpoints, so `git add` picks up only
source.

Usage:
  python scripts/stage_release_artifacts.py
  python scripts/stage_release_artifacts.py --out D:/hf_upload
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every artefact the manuscript or reproduce.md names, grouped by what it backs.
ARTEFACTS = {
    "homology leakage + copy baseline (Sec. IV-A, IV-B)": [
        "data/qc_reports/retrieval_orig_test.json",
        "data/qc_reports/retrieval_test.json",
        "data/qc_reports/homology_copy_baseline.json",
    ],
    "identity stratification (Sec. IV-C, Table II)": [
        "data/qc_reports/stratified_evaluation.json",
        "data/qc_reports/evaluation_report_full.json",
    ],
    "conservation ceiling + the released split (Sec. IV-D, IV-E)": [
        "data/qc_reports/split_ident70_per_gene_report.json",
        "data/qc_reports/retrieval_ident70_test.json",
        "data/splits/train_ident70_per_gene.json",
        "data/splits/val_ident70_per_gene.json",
        "data/splits/test_ident70_per_gene.json",
        "data/splits/train.json",
        "data/splits/val.json",
        "data/splits/test.json",
    ],
    "identity-controlled results (Table IV)": [
        "data/qc_reports/baseline_eval_ident70.json",
        "data/qc_reports/model_eval_ident70.json",
        "data/qc_reports/model_eval_ident70_strand.json",
    ],
    "ND6 / strand analysis (Sec. IV-F, IV-G, Figs. 4-5)": [
        "data/qc_reports/per_gene_delta.json",
        "data/qc_reports/gc_skew_by_gene.json",
    ],
    "cross-corpus replication (Sec. IV-H, Table V, Fig. 6)": [
        "data/qc_reports/cross_corpus_leakage.json",
    ],
    "species overlap + doubly-controlled check (Sec. IV-I)": [
        "data/qc_reports/species_overlap_analysis.json",
        "data/qc_reports/doubly_controlled_analysis.json",
    ],
    "literature survey (Table I)": [
        "data/literature_baseline_survey.json",
    ],
    "seed replication (pending)": [
        "data/qc_reports/seed_replication.json",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "release_staging"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest, total, missing = [], 0, []
    for group, files in ARTEFACTS.items():
        for rel in files:
            src = ROOT / rel
            if not src.is_file():
                missing.append((group, rel))
                continue
            dst = out / Path(rel).name
            shutil.copy2(src, dst)
            size = src.stat().st_size
            total += size
            manifest.append({"file": Path(rel).name, "source": rel,
                             "backs": group, "bytes": size})

    # also stage the strand-conditioned checkpoint, the one weight the paper needs
    ckpt = ROOT / "experiments/20261007_150614_ident70_strand/checkpoints/best.pt"
    if ckpt.is_file():
        dst = out / "mitoseqgen_strand_best.pt"
        shutil.copy2(ckpt, dst)
        total += ckpt.stat().st_size
        manifest.append({"file": dst.name, "source": str(ckpt.relative_to(ROOT)),
                         "backs": "strand-conditioned weights (Sec. IV-G)",
                         "bytes": ckpt.stat().st_size})
    else:
        missing.append(("strand weights", str(ckpt.relative_to(ROOT))))

    (out / "MANIFEST.json").write_text(
        json.dumps({"staged": manifest,
                    "total_bytes": total,
                    "missing": [{"backs": g, "file": f} for g, f in missing]},
                   indent=2), encoding="utf-8")

    print(f"staged {len(manifest)} files, {total/1048576:.1f} MB -> {out}")
    for m in manifest:
        print(f"  {m['bytes']/1048576:7.2f} MB  {m['file']}")
    if missing:
        print(f"\nMISSING ({len(missing)}) -- generate these before release:")
        for g, f in missing:
            print(f"  {f}   [{g}]")
    else:
        print("\nnothing missing")
    print(f"\nmanifest: {out / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
