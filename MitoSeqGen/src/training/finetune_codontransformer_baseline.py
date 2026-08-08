"""Fine-tune the real, pretrained CodonTransformer model on our mitochondrial
training corpus, as the key missing ablation for the manuscript: "why not
just fine-tune the existing general-purpose tool instead of training a new
model from scratch?"

CodonTransformer's tokenizer vocabulary is built purely from standard-
genetic-code-valid amino-acid/codon pairings (verified empirically: 90
tokens total, none of "w_tga", "m_ata", "__aga", "__agg" exist). This means
naive fine-tuning cannot even represent the correct mitochondrial-specific
codon choices for the four reassigned codons -- the tokenizer must be
extended first. This script does that: adds the 4 missing tokens, resizes
the model's embedding/LM-head layers, then fine-tunes end-to-end via the
same masked-language-modelling objective the model was originally trained
with (input = protein with fully-masked codons, i.e. exactly the
distribution predict_dna_sequence() queries at inference time; label = the
true codon at every position).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CodonTransformer.CodonData import get_merged_seq
from CodonTransformer.CodonPrediction import load_model, load_tokenizer, tokenize

NEW_TOKENS = ["w_tga", "m_ata", "__aga", "__agg"]
DEFAULT_OUT_DIR = ROOT / "experiments" / "codontransformer_finetuned"


def build_label_merged_seq(protein: str, dna: str) -> str:
    """Replicates CodonTransformer.CodonData.get_merged_seq's *token format*
    exactly (space-joined "{aa}_{codon}" tokens, terminal position uses the
    stop symbol "_"), but without its stop-codon detection, which is hardcoded
    to the standard-code stop set {TAA, TAG, TGA}
    (CodonTransformer.CodonUtils.STOP_CODONS). 5.2% of our training sequences
    end in AGA or AGG -- the mitochondrial-code-specific stop reassignment --
    which that set does not recognise; get_merged_seq's preprocess_dna_sequence
    then silently appends a spurious extra "UNK" codon to such sequences,
    desynchronising the protein/DNA codon counts and raising a ValueError for
    ~1 in 5 batches (P(>=1 affected record in a batch of 4) given a 5.2%
    per-record rate). This is the same class of mitochondrial-code blindness
    the rest of this project characterises in CodonTransformer generally --
    here it surfaces in the library's own preprocessing utility. We instead
    derive the token boundaries directly from our already-known-correct
    protein and DNA lengths, so any terminal codon (including the two new
    __aga/__agg tokens added to the tokenizer) is handled uniformly."""
    protein = protein.upper().strip()
    if protein.endswith("*"):
        protein = protein[:-1]
    protein = protein.rstrip("_") + "_"  # exactly one trailing stop symbol

    dna = dna.upper().strip()
    n_codons = len(dna) // 3
    if len(dna) % 3 != 0 or n_codons != len(protein):
        raise ValueError(
            f"protein/DNA codon count mismatch after stop-symbol normalisation: "
            f"protein={len(protein)} tokens, dna={n_codons} codons"
        )
    return " ".join(f"{protein[i]}_{dna[i * 3:i * 3 + 3]}" for i in range(len(protein)))


def build_extended_tokenizer():
    tokenizer = load_tokenizer()
    added = tokenizer.add_tokens(NEW_TOKENS)
    logging.info(f"Added {added} new tokens to CodonTransformer vocab: {NEW_TOKENS}")
    logging.info(f"New vocab size: {len(tokenizer)}")
    return tokenizer


def load_training_examples(split_name: str = "train"):
    with open(ROOT / "data" / "processed" / "mito_cds_tokenized.json") as f:
        records = json.load(f)["records"]
    with open(ROOT / "data" / "splits" / f"{split_name}.json") as f:
        idx = json.load(f)["indices"]
    return [records[i] for i in idx]


def make_batch(records, tokenizer, organism_id: int, device):
    """Returns (input_ids, attention_mask, token_type_ids, label_ids), all
    (batch, seq_len). Input is fully masked (AA_UNK at every codon
    position, matching predict_dna_sequence's inference-time input
    distribution exactly); labels are the true merged tokens."""
    input_dicts, label_dicts = [], []
    for r in records:
        protein = r["protein_sequence"].rstrip("*")
        dna = r["sequence"]
        input_dicts.append({"idx": 0, "codons": get_merged_seq(protein=protein, dna=""), "organism": organism_id})
        # NOT get_merged_seq here -- see build_label_merged_seq's docstring for why
        # (its stop-codon detection is blind to the mitochondrial AGA/AGG reassignment).
        label_dicts.append({"idx": 0, "codons": build_label_merged_seq(protein, dna), "organism": organism_id})

    input_batch = tokenize(input_dicts, tokenizer=tokenizer)
    label_batch = tokenize(label_dicts, tokenizer=tokenizer)

    # Pad both to the same width (protein length determines codon count for
    # both, so they only differ if the tokenizer's own padding picked a
    # different max-in-batch length by chance; make them agree explicitly).
    max_len = max(input_batch["input_ids"].shape[1], label_batch["input_ids"].shape[1])

    def pad_to(t, width, pad_value):
        if t.shape[1] >= width:
            return t
        pad = torch.full((t.shape[0], width - t.shape[1]), pad_value, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    pad_id = tokenizer.pad_token_id
    input_ids = pad_to(input_batch["input_ids"], max_len, pad_id)
    attention_mask = pad_to(input_batch["attention_mask"], max_len, 0)
    token_type_ids = pad_to(input_batch["token_type_ids"], max_len, organism_id)
    label_ids = pad_to(label_batch["input_ids"], max_len, -100)  # -100 = ignored by cross-entropy

    # Never compute loss on PAD/CLS/SEP label positions.
    label_ids = label_ids.masked_fill(attention_mask == 0, -100)

    return (
        input_ids.to(device), attention_mask.to(device),
        token_type_ids.to(device), label_ids.to(device),
    )


def train(
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-5,
    max_protein_len: int = 500,
    out_dir: Path = DEFAULT_OUT_DIR,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device={device}")

    tokenizer = build_extended_tokenizer()
    # resize_token_embeddings()'s mean/covariance-based smart init for the new tokens
    # is numerically unsafe when run directly on GPU (verified: produces all-NaN rows
    # for every new-token embedding, apparently a CUDA-vs-CPU linear-algebra precision
    # issue in the covariance sampling step) -- load and resize on CPU first, then move
    # to the target device. This was the root cause of the 100%-NaN-loss failure in the
    # first two launch attempts of this script.
    model = load_model(device=torch.device("cpu"))
    model.resize_token_embeddings(len(tokenizer))
    assert not torch.isnan(model.get_input_embeddings().weight).any(), \
        "resized embeddings contain NaN before device transfer -- do not proceed"
    model.to(device)
    model.train()

    organism_id = 51  # "Homo sapiens" per ORGANISM2ID -- see manuscript limitations note
    from CodonTransformer.CodonUtils import ORGANISM2ID
    organism_id = ORGANISM2ID["Homo sapiens"]

    train_records = [r for r in load_training_examples("train") if len(r["protein_sequence"].rstrip("*")) <= max_protein_len]
    logging.info(f"Fine-tuning on {len(train_records)} mitochondrial training sequences "
                 f"(protein length <= {max_protein_len}, organism fixed to 'Homo sapiens' as the "
                 f"closest available proxy -- CodonTransformer has no mitochondrial/vertebrate-non-human option)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    out_dir.mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(42)
    n = len(train_records)

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        order = list(range(n))
        rng.shuffle(order)
        total_loss, n_batches = 0.0, 0
        optimizer.zero_grad()
        accum_count = 0  # successful micro-batches accumulated since the last optimizer.step()

        n_skipped, n_nonfinite = 0, 0
        for step, start in enumerate(range(0, n, batch_size)):
            batch_idx = order[start:start + batch_size]
            batch_records = [train_records[i] for i in batch_idx]
            t_step = time.time()
            try:
                input_ids, attention_mask, token_type_ids, label_ids = make_batch(batch_records, tokenizer, organism_id, device)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
                    logits = out.logits  # (batch, seq_len, vocab)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), label_ids.reshape(-1), ignore_index=-100)

                if not torch.isfinite(loss):
                    # Do NOT backward() a non-finite loss -- it would poison gradients (and,
                    # via clip_grad_norm_'s total-norm computation, the *entire* parameter set
                    # on the next optimizer.step(), not just this micro-batch). Also do not
                    # zero_grad() here: any gradients already accumulated from earlier
                    # successful micro-batches in this window are still valid and must survive.
                    logging.warning(f"epoch {epoch} step {step}: non-finite loss ({loss.item()}), skipping batch")
                    n_nonfinite += 1
                    continue

                (loss / grad_accum).backward()
                accum_count += 1
                total_loss += loss.item()
                n_batches += 1

                if accum_count == grad_accum:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        logging.warning(f"epoch {epoch} step {step}: non-finite grad norm ({grad_norm}), discarding accumulated gradients")
                    optimizer.zero_grad()
                    accum_count = 0
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"epoch {epoch} step {step}: CUDA OOM on batch (protein lengths "
                                 f"{[len(r['protein_sequence'].rstrip('*')) for r in batch_records]}) -- skipping, clearing cache")
                torch.cuda.empty_cache()
                n_skipped += 1
                continue
            except Exception as e:
                logging.warning(f"epoch {epoch} step {step}: unexpected error, skipping batch: {e}")
                n_skipped += 1
                continue

            step_time = time.time() - t_step
            if step_time > 10:
                max_len = max(len(r["protein_sequence"].rstrip("*")) for r in batch_records)
                logging.info(f"  [slow step] epoch {epoch} step {step} took {step_time:.1f}s (max protein len in batch: {max_len})")

            if n_batches % 50 == 0 and n_batches > 0:
                elapsed = time.time() - t_start
                logging.info(f"epoch {epoch} step {n_batches}/{(n + batch_size - 1)//batch_size} "
                             f"loss={total_loss/n_batches:.4f} elapsed={elapsed/60:.1f}min skipped={n_skipped} nonfinite={n_nonfinite}")

        if accum_count > 0:
            # Flush a final partial accumulation window at epoch end rather than dropping it.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            optimizer.zero_grad()

        logging.info(f"Epoch {epoch}/{epochs} done. mean_loss={total_loss/max(n_batches,1):.4f} "
                     f"elapsed={(time.time()-t_start)/60:.1f}min skipped={n_skipped} nonfinite={n_nonfinite}")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, out_dir / "checkpoint.pt")
        tokenizer.save_pretrained(str(out_dir / "tokenizer"))
        logging.info(f"Saved checkpoint to {out_dir / 'checkpoint.pt'}")

    logging.info("Fine-tuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_protein_len", type=int, default=500)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr, max_protein_len=args.max_protein_len)
