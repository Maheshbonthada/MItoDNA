"""Evaluate the CodonTransformer-fine-tuned-on-mitochondrial-data ablation
(src/training/finetune_codontransformer_baseline.py) on the full held-out
test set, using the exact same metric suite and match_protein=True fairness
setting as the pretrained (non-fine-tuned) CodonTransformer benchmark, so the
two are directly comparable in the manuscript's results table:
"does fine-tuning the existing tool close the gap to a from-scratch model?"
"""

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Bio.Seq import Seq
from CodonTransformer.CodonPrediction import predict_dna_sequence, load_model
from CodonTransformer.CodonUtils import ORGANISM2ID
from transformers import PreTrainedTokenizerFast

from src.genetic_codes import MITOCHONDRIAL_GENETIC_CODE
from src.evaluation.metrics import (
    gc_content, compute_mt_cai, compute_rscu_weights, simple_bleu,
    sequence_hash, codon_diversity, compute_mfe,
)

CHECKPOINT_DIR = ROOT / "experiments" / "codontransformer_finetuned"
OUT_PATH = ROOT / "data" / "qc_reports" / "codontransformer_finetuned_results.json"
ERRORS_PATH = ROOT / "data" / "qc_reports" / "codontransformer_finetuned_errors.json"


def load_finetuned_model_and_tokenizer(device: torch.device):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(CHECKPOINT_DIR / "tokenizer"))

    # Same load-on-CPU-then-resize-then-move sequence used (and required -- see
    # finetune_codontransformer_baseline.py's comment) during training, so the
    # embedding matrix shape matches the checkpoint's state dict before loading it.
    model = load_model(device=torch.device("cpu"))
    model.resize_token_embeddings(len(tokenizer))

    checkpoint = torch.load(CHECKPOINT_DIR / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"loaded fine-tuned checkpoint from epoch {checkpoint['epoch']}", flush=True)

    model.to(device)
    model.eval()
    return model, tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    model, tokenizer = load_finetuned_model_and_tokenizer(device)
    print("model loaded", flush=True)

    with open(ROOT / "data" / "processed" / "mito_cds_tokenized.json") as f:
        records = json.load(f)["records"]
    with open(ROOT / "data" / "splits" / "train.json") as f:
        train_idx = json.load(f)["indices"]
    with open(ROOT / "data" / "splits" / "test.json") as f:
        test_idx = json.load(f)["indices"]

    train_records = [records[i] for i in train_idx]
    rscu = compute_rscu_weights(train_records)
    train_hashes = {sequence_hash(r["sequence"]) for r in train_records}

    organism_id = ORGANISM2ID["Homo sapiens"]

    results, errors = [], []
    t0 = time.time()
    for n, idx in enumerate(test_idx, 1):
        rec = records[idx]
        protein = rec["protein_sequence"].rstrip("*")
        ref = rec["sequence"]
        try:
            out = predict_dna_sequence(
                protein=protein, organism=organism_id, device=device,
                tokenizer=tokenizer, model=model, deterministic=True,
                match_protein=True,
            )
            dna = out.predicted_dna
            mito_translation = str(Seq(dna).translate(table=MITOCHONDRIAL_GENETIC_CODE, to_stop=False))
            standard_translation = str(Seq(dna).translate(to_stop=False))
            results.append({
                "id": rec["id"], "gene": rec["gene_name"], "species": rec["species"],
                "dna": dna,
                "has_premature_stop_mito": "*" in mito_translation[:-1],
                "has_premature_stop_std": "*" in standard_translation[:-1],
                "protein_correct_under_mito": mito_translation.rstrip("*") == protein,
                "protein_correct_under_std": standard_translation.rstrip("*") == protein,
                "gc_content": gc_content(dna),
                "mt_cai": compute_mt_cai(dna, rscu),
                "codon_diversity": codon_diversity(dna),
                "bleu": simple_bleu(ref, dna),
                "mfe": compute_mfe(dna),
                "is_novel": sequence_hash(dna) not in train_hashes,
            })
        except Exception as e:
            errors.append({"idx": n, "gene": rec["gene_name"], "protein_len": len(protein), "error": str(e)})
            print(f"  [ERROR] idx={n} gene={rec['gene_name']}: {e}", flush=True)
            continue

        if n % 100 == 0:
            elapsed = time.time() - t0
            print(f"{n}/{len(test_idx)} done ({elapsed:.0f}s, {elapsed/n:.2f}s/seq, {len(errors)} errors)", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    with open(ERRORS_PATH, "w") as f:
        json.dump(errors, f, indent=2)

    n_ok = len(results)
    print(f"--- FINE-TUNED CODONTRANSFORMER, FULL TEST SET (n={n_ok} succeeded, {len(errors)} errors), match_protein=True ---", flush=True)
    print(f"premature stop under STANDARD code: {sum(r['has_premature_stop_std'] for r in results)}/{n_ok} = "
          f"{sum(r['has_premature_stop_std'] for r in results)/n_ok*100:.1f}%", flush=True)
    print(f"premature stop under MITOCHONDRIAL code: {sum(r['has_premature_stop_mito'] for r in results)}/{n_ok} = "
          f"{sum(r['has_premature_stop_mito'] for r in results)/n_ok*100:.1f}%", flush=True)
    print(f"protein-correct under STANDARD code: {sum(r['protein_correct_under_std'] for r in results)}/{n_ok} = "
          f"{sum(r['protein_correct_under_std'] for r in results)/n_ok*100:.1f}%", flush=True)
    print(f"protein-correct under MITOCHONDRIAL code: {sum(r['protein_correct_under_mito'] for r in results)}/{n_ok} = "
          f"{sum(r['protein_correct_under_mito'] for r in results)/n_ok*100:.1f}%", flush=True)
    mean_gc = sum(r["gc_content"] for r in results) / n_ok
    mean_cai = sum(r["mt_cai"] for r in results) / n_ok
    mean_bleu = sum(r["bleu"] for r in results) / n_ok
    mean_div = sum(r["codon_diversity"] for r in results) / n_ok
    mean_mfe = sum(r["mfe"] for r in results) / n_ok
    print(f"mean_gc={mean_gc:.4f} mean_mt_cai={mean_cai:.4f} mean_bleu={mean_bleu:.4f} "
          f"mean_diversity={mean_div:.4f} mean_mfe={mean_mfe:.2f}", flush=True)
    print("FINE-TUNED EVAL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
