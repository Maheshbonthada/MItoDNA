"""One-off: recompute the Markov transition analysis with the Cohen's w fix
(see metrics.markov_transition_chi2_test docstring) against the exact same
generated sequences as the already-completed full evaluation run, without
re-running the expensive per-sequence metric suite (MFE folding was the
bottleneck there, ~4s/sequence; this script skips it entirely since Markov
analysis only needs the generated DNA strings). Sequence generation here is
deterministic (same seed=42, same greedy decoding) so results are identical
to what evaluate.py's main run actually evaluated.
"""


import json
import logging
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.baselines import Baselines
from src.evaluation.metrics import (
    build_markov_transition_counts, markov_transition_chi2_test,
    markov_transition_cohens_w, markov_transition_kl_divergence, markov_transition_matrix,
)
from src.evaluation.evaluate import load_records, load_split
from src.models.generate import generate_cds, load_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

CHECKPOINT = ROOT / "experiments" / "20260722_140803" / "checkpoints" / "best.pt"
SEED = 42
SAMPLE_SIZE = 10000  # matches the original run; capped to actual test-set size below


def main():
    records = load_records()
    train_idx = load_split("train")
    test_idx = load_split("test")
    train_records = [records[i] for i in train_idx]

    rng = random.Random(SEED)
    sample_idx = rng.sample(test_idx, min(SAMPLE_SIZE, len(test_idx)))
    logging.info(f"regenerating sequences for {len(sample_idx)} test proteins (must match original run's sample)")

    baselines = Baselines(train_records, seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(CHECKPOINT, device)

    method_names = ["mitoseqgen", "mt_cai_lookup", "random_synonymous", "most_frequent_codon", "codontransformer_remap", "natural_reference"]
    generated_sequences = {name: [] for name in method_names}

    for n, idx in enumerate(sample_idx, 1):
        rec = records[idx]
        protein = rec["protein_sequence"].rstrip("*")
        reference = rec["sequence"]
        generated_sequences["natural_reference"].append(reference)
        generated_sequences["mitoseqgen"].append(generate_cds(model, protein, device, strategy="greedy"))
        for name in ["mt_cai_lookup", "random_synonymous", "most_frequent_codon", "codontransformer_remap"]:
            generated_sequences[name].append(baselines.generate(protein, name))
        if n % 500 == 0:
            logging.info(f"  {n}/{len(sample_idx)} regenerated")

    logging.info("building reference transition matrix from training corpus...")
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
        }
        logging.info(f"  {name:24} KL={kl:.4f}  chi2_p={chi2_p:.2e}  n_obs={n_obs:.0f}  cohens_w={cohens_w:.4f}")

    out_path = ROOT / "data" / "qc_reports" / "markov_transition_analysis_with_cohens_w.json"
    with open(out_path, "w") as f:
        json.dump(markov_results, f, indent=2)
    logging.info(f"saved to {out_path}")
    logging.info("MARKOV RECOMPUTE COMPLETE")

if __name__ == "__main__":
    main()
