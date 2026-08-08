"""Full evaluation harness for MitoSeqGen.

Generates sequences with the trained model AND all 4 required baselines for
the same held-out test-set proteins (species never seen during training or
validation), computes the full metric suite on each, and reports a
side-by-side comparison — because per Absolute Rule #3, no metric is
reported without its baseline.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.evaluation.baselines import Baselines
from src.evaluation.metrics import (
    aggregate_metrics,
    build_markov_transition_counts,
    codon_diversity,
    compute_mfe,
    compute_mt_cai,
    compute_rscu_weights,
    gc_content,
    markov_transition_chi2_test,
    markov_transition_cohens_w,
    markov_transition_kl_divergence,
    markov_transition_matrix,
    novel_sequence_rate,
    sequence_hash,
    simple_bleu,
    translation_complies_with_mt_code,
)
from src.evaluation.significance import print_significance_summary, run_significance_tests
from src.models.generate import generate_cds, load_model_from_checkpoint

DATA_DIR = ROOT / "data"
DEFAULT_CHECKPOINT = ROOT / "experiments" / "20260722_140803" / "checkpoints" / "best.pt"


def load_split(name: str) -> List[int]:
    with open(DATA_DIR / "splits" / f"{name}.json") as f:
        return json.load(f)["indices"]


def load_records() -> List[Dict]:
    with open(DATA_DIR / "processed" / "mito_cds_tokenized.json") as f:
        return json.load(f)["records"]


def evaluate_sequence(generated: str, reference: str, rscu_weights: dict, train_hashes: set) -> dict:
    return {
        "mt_cai": compute_mt_cai(generated, rscu_weights),
        "code_compliant": translation_complies_with_mt_code(generated),
        "codon_diversity": codon_diversity(generated),
        "bleu": simple_bleu(reference, generated),
        "mfe": compute_mfe(generated),
        "gc_content": gc_content(generated),
        "is_novel": 1.0 if sequence_hash(generated) not in train_hashes else 0.0,
    }


def run_evaluation(checkpoint_path: Path, sample_size: int, seed: int = 42, device: torch.device = None) -> Dict:
    logging.info(f"Loading records and splits...")
    records = load_records()
    train_idx = load_split("train")
    test_idx = load_split("test")
    train_records = [records[i] for i in train_idx]

    logging.info(f"Train records: {len(train_records)} | Test records: {len(test_idx)}")

    rng = random.Random(seed)
    sample_idx = rng.sample(test_idx, min(sample_size, len(test_idx)))
    logging.info(f"Evaluating on {len(sample_idx)} held-out test-set proteins (species never seen in train/val)")

    logging.info("Building baselines from training corpus...")
    baselines = Baselines(train_records, seed=seed)

    logging.info("Computing training-corpus RSCU weight table for mt-CAI scoring...")
    rscu_weights = compute_rscu_weights(train_records)

    logging.info("Hashing training sequences for novel-sequence-rate check...")
    train_hashes = {sequence_hash(r["sequence"]) for r in train_records}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model from {checkpoint_path} on {device}...")
    model = load_model_from_checkpoint(checkpoint_path, device)

    method_names = ["mitoseqgen", "mt_cai_lookup", "random_synonymous", "most_frequent_codon", "codontransformer_remap", "natural_reference"]
    results: Dict[str, List[dict]] = {name: [] for name in method_names}
    generated_sequences: Dict[str, List[str]] = {name: [] for name in method_names}

    t0 = time.time()
    for n, idx in enumerate(sample_idx, 1):
        rec = records[idx]
        protein = rec["protein_sequence"].rstrip("*")
        reference = rec["sequence"]

        # The real, evolved sequence for this held-out species — not a baseline to
        # "beat," but the ground truth other methods should be compared against,
        # especially for MFE where "more negative" isn't unambiguously "better"
        # biologically (excessive 5'-end stability can impede translation
        # initiation); what matters is how close a method lands to nature's own value.
        natural_result = evaluate_sequence(reference, reference, rscu_weights, train_hashes)
        results["natural_reference"].append(natural_result)
        generated_sequences["natural_reference"].append(reference)

        generated_by_method = {"mitoseqgen": generate_cds(model, protein, device, strategy="greedy")}
        for name in ["mt_cai_lookup", "random_synonymous", "most_frequent_codon", "codontransformer_remap"]:
            generated_by_method[name] = baselines.generate(protein, name)

        for name, seq in generated_by_method.items():
            r = evaluate_sequence(seq, reference, rscu_weights, train_hashes)
            r["mfe_delta_from_natural"] = abs(r["mfe"] - natural_result["mfe"])
            r["gc_delta_from_natural"] = abs(r["gc_content"] - natural_result["gc_content"])
            results[name].append(r)
            generated_sequences[name].append(seq)

        if n % 25 == 0:
            elapsed = time.time() - t0
            logging.info(f"  {n}/{len(sample_idx)} evaluated ({elapsed:.0f}s elapsed, ~{elapsed / n:.1f}s/sequence)")

    summary = {name: aggregate_metrics(results[name]) for name in method_names}

    logging.info("Building reference codon Markov transition matrix from training corpus...")
    reference_counts = build_markov_transition_counts(r["sequence"] for r in train_records)
    reference_probs = markov_transition_matrix(reference_counts)

    markov_results = {}
    for name in method_names:
        gen_counts = build_markov_transition_counts(generated_sequences[name])
        kl = markov_transition_kl_divergence(gen_counts, reference_probs)
        chi2_stat, chi2_p, dof, n_obs = markov_transition_chi2_test(gen_counts, reference_probs)
        cohens_w = markov_transition_cohens_w(chi2_stat, n_obs)
        markov_results[name] = {
            "kl_divergence_from_natural_transitions": kl,
            "chi2_statistic": chi2_stat,
            "chi2_p_value": chi2_p,
            "chi2_degrees_of_freedom": dof,
            "chi2_n_observations": n_obs,
            "cohens_w": cohens_w,
            "note": "chi2_p_value is ~0 for every method incl. natural_reference at this sample size "
                    "(saturated, uninformative -- see markov_transition_chi2_test docstring); "
                    "cohens_w is the sample-size-normalized effect size and is the primary comparison statistic.",
        }
    logging.info("Markov transition analysis (lower KL / lower Cohen's w = closer to natural training-set transition structure):")
    logging.info("  (chi2 p-values omitted from this summary -- saturated at ~0 for all methods incl. natural_reference at this N; see report JSON 'note' field)")
    for name, m in markov_results.items():
        logging.info(f"  {name:24} KL={m['kl_divergence_from_natural_transitions']:.4f}  cohens_w={m['cohens_w']:.4f}")

    return {
        "summary": summary, "raw_results": results, "sample_size": len(sample_idx),
        "checkpoint": str(checkpoint_path), "markov_transition_analysis": markov_results,
    }


def print_comparison_table(summary: Dict[str, dict]):
    metrics = [
        "mean_mt_cai", "compliance_rate", "mean_codon_diversity", "mean_bleu",
        "mean_mfe", "mean_mfe_delta_from_natural", "mean_gc_content", "mean_gc_delta_from_natural",
        "novel_sequence_rate",
    ]
    header = f"{'metric':22} " + " ".join(f"{name:22}" for name in summary.keys())
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:22} " + " ".join(f"{summary[name].get(m, 0.0):22.4f}" for name in summary.keys())
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="Force 'cpu' or 'cuda'; default auto-detects. "
                         "Use 'cpu' to sanity-check a checkpoint without contending with a concurrent GPU training run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(args.device) if args.device else None
    result = run_evaluation(Path(args.checkpoint), args.sample_size, args.seed, device=device)

    print()
    print_comparison_table(result["summary"])

    sig_results = run_significance_tests(result["raw_results"])
    print_significance_summary(sig_results)

    output_path = Path(args.output) if args.output else ROOT / "data" / "qc_reports" / f"evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": result["summary"],
            "sample_size": result["sample_size"],
            "checkpoint": result["checkpoint"],
            "significance_tests": sig_results,
            "markov_transition_analysis": result["markov_transition_analysis"],
            "raw_results": result["raw_results"],
        }, f, indent=2)
    logging.info(f"Saved evaluation report (incl. raw results + significance tests) to {output_path}")


if __name__ == "__main__":
    main()
