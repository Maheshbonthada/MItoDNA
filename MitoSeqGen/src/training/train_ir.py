"""CMLM training for MitoSeqGen-IR (constrained iterative refinement).

Same data, splits, optimizer family, precision and hardware budget as
src/training/train.py, so an IR run is directly comparable to the
autoregressive run -- the decoding scheme and training objective are the only
intended differences.

Objective: conditional masked language modelling over the synonym lattice. Per
batch a random fraction of codon positions is replaced by <MASK> and the model
predicts them from the protein plus the surviving codons; cross-entropy is
taken over masked positions only. Logits are constrained to each position's
synonym class, so the loss is a choice among 2-6 legal codons rather than over
the full vocabulary.

Usage:
  python -m src.training.train_ir --config config/config_ident70.yaml --tag ir_ident70
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BucketBatchSampler, MitoCDSDataset  # noqa: E402
from src.data.preprocess import AA_VOCAB, VOCAB  # noqa: E402
from src.models.iterative_refinement import MitoSeqGenIR, cmlm_mask  # noqa: E402

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PAD_ID = VOCAB["<PAD>"]
AA_PAD_ID = AA_VOCAB["<PAD>"]


def collate(batch):
    L = max(len(r["codon_tokens"]) for r in batch)
    B = len(batch)
    aa = torch.full((B, L), AA_PAD_ID, dtype=torch.long)
    cd = torch.full((B, L), PAD_ID, dtype=torch.long)
    for i, r in enumerate(batch):
        pt, ct = r["protein_tokens"], r["codon_tokens"]
        n = min(len(pt), len(ct))
        aa[i, :n] = torch.tensor(pt[:n], dtype=torch.long)
        cd[i, :n] = torch.tensor(ct[:n], dtype=torch.long)
    return aa, cd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, amp_dtype, optimizer=None,
              scheduler=None, accum_steps=1, grad_clip=1.0, seed=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    gen = None
    if seed is not None:  # deterministic masking for validation comparability
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else torch.enable_grad()
    for step, (aa, cd) in enumerate(loader):
        aa, cd = aa.to(device, non_blocking=True), cd.to(device, non_blocking=True)
        codon_inputs, loss_mask = cmlm_mask(cd, aa, generator=gen)

        with torch.set_grad_enabled(train):
            with ctx:
                logits = model(aa, codon_inputs)
                # constrained logits contain -inf; compute loss in fp32 for stability
                sel = loss_mask.view(-1)
                flat = logits.view(-1, logits.size(-1)).float()[sel]
                tgt = cd.view(-1)[sel]
                loss = criterion(flat, tgt)

            if train:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        n = int(sel.sum())
        total_loss += float(loss) * n
        total_tokens += n
        total_correct += int((flat.argmax(-1) == tgt).sum())
    return total_loss / max(total_tokens, 1), total_correct / max(total_tokens, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config" / "base_config.yaml"))
    ap.add_argument("--tag", default="ir")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    set_seed(config["project"]["seed"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.tag}"
    run_dir = EXPERIMENTS_DIR / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "logs" / "train.log")],
    )
    json.dump(config, open(run_dir / "config.json", "w"), indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[
        config["hardware"]["mixed_precision"]]
    if device.type != "cuda":
        amp_dtype = None

    data_dir = PROJECT_ROOT / "data"
    tok = data_dir / "processed" / config["data"]["tokenized_file"]
    train_ds = MitoCDSDataset(tok, data_dir / "splits" / config["data"]["train_split"])
    val_ds = MitoCDSDataset(tok, data_dir / "splits" / config["data"]["val_split"])
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Train records={len(train_ds)} | Val records={len(val_ds)} | device={device}")

    bs = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_sampler=BucketBatchSampler(train_ds.lengths(), bs, shuffle=True,
                                                   seed=config["project"]["seed"]),
        collate_fn=collate, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(
        val_ds, batch_sampler=BucketBatchSampler(val_ds.lengths(), bs, shuffle=False),
        collate_fn=collate, num_workers=0, pin_memory=device.type == "cuda")

    mc = config["model"]
    model = MitoSeqGenIR(
        d_model=mc["d_model"], nhead=mc["nhead"], num_layers=args.layers,
        dim_feedforward=mc["dim_feedforward"], dropout=mc["dropout"],
        max_position_embeddings=mc["max_position_embeddings"],
    ).to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} "
                 f"(encoder-only, {args.layers} layers)")

    tc = config["training"]
    epochs = args.epochs or tc["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["learning_rate"]),
                                  weight_decay=float(tc["weight_decay"]))
    criterion = torch.nn.CrossEntropyLoss()
    accum = tc["gradient_accumulation_steps"]
    total_steps = max(1, (len(train_loader) // accum) * epochs)
    warmup = min(tc["warmup_steps"], max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.1, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val, no_improve, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loader.batch_sampler.set_epoch(epoch)
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, amp_dtype,
                                    optimizer, scheduler, accum, tc["grad_clip_norm"])
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, amp_dtype, seed=1234)
        dt = time.time() - t0
        logging.info(f"epoch {epoch:3d}/{epochs}  train_loss {tr_loss:.4f} acc {tr_acc:.4f}  "
                     f"val_loss {va_loss:.4f} acc {va_acc:.4f}  ppl {math.exp(min(va_loss,20)):.3f}  {dt:.0f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": dt})
        json.dump({"history": history}, open(run_dir / "metrics.json", "w"), indent=2)

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "config": config, "layers": args.layers},
                   run_dir / "checkpoints" / "latest.pt")
        if va_loss < best_val:
            best_val, no_improve = va_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "config": config, "layers": args.layers, "val_loss": va_loss},
                       run_dir / "checkpoints" / "best.pt")
            logging.info(f"  new best (val_loss {va_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= tc["early_stopping_patience"]:
                logging.info(f"early stopping at epoch {epoch}")
                break
    logging.info(f"done. best val_loss {best_val:.4f}")


if __name__ == "__main__":
    main()
