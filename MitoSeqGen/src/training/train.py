"""Training entrypoint for MitoSeqGen.

Reads config/base_config.yaml (model size, batch size, hardware settings)
rather than hardcoding hyperparameters, and logs everything Phase 5 of the
project spec requires for reproducibility: exact config used, environment
(Python/torch/CUDA/GPU versions), and per-epoch metrics, all under a unique
experiments/<run_id>/ directory.
"""

import argparse
import json
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import BucketBatchSampler, MitoCDSDataset, collate_fn
from src.data.preprocess import AA_VOCAB, VOCAB
from src.models.transformer import MitoSeqTransformer
from src.training.losses import AuxiliaryLosses, combined_loss

PROJECT_ROOT = ROOT
CONFIG_PATH = PROJECT_ROOT / "config" / "base_config.yaml"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str) -> torch.device:
    if pref == "cuda" and not torch.cuda.is_available():
        logging.warning("cuda requested but not available, falling back to cpu")
        return torch.device("cpu")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def resolve_num_workers(pref, cpu_count: int) -> int:
    if pref == "auto":
        return max(1, min(cpu_count - 2, 16))
    return int(pref)


def make_run_dir(config: dict) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENTS_DIR / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return run_dir


def log_environment_info(run_dir: Path):
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
    with open(run_dir / "environment.json", "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"Environment: {info}")


def train_collate(batch):
    return collate_fn(batch, pad_token_id=VOCAB["<PAD>"])


def make_autocast(device: torch.device, amp_dtype):
    if amp_dtype is None:
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    n = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        with make_autocast(device, amp_dtype):
            logits = model(src, tgt[:-1, :])
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt[1:, :].reshape(-1))
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def build_model(config: dict, device: torch.device) -> MitoSeqTransformer:
    m = config["model"]
    model = MitoSeqTransformer(
        src_vocab_size=len(AA_VOCAB),
        tgt_vocab_size=len(VOCAB),
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        max_position_embeddings=m["max_position_embeddings"],
    ).to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint (e.g. experiments/<run_id>/checkpoints/latest.pt) to resume training from. "
             "Continues in the same run directory and appends to its existing metrics history.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    set_seed(config["project"]["seed"])

    cpu_count = os.cpu_count() or 1
    device = resolve_device(config["hardware"]["device"])
    num_workers = resolve_num_workers(config["hardware"]["num_workers"], cpu_count)
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(max(1, cpu_count // 2))

    precision = config["hardware"]["mixed_precision"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[precision]
    if amp_dtype is not None and device.type != "cuda":
        logging.warning("Mixed precision requested but device is not cuda; disabling autocast")
        amp_dtype = None

    resuming = args.resume is not None
    if resuming:
        run_dir = Path(args.resume).resolve().parents[1]  # .../checkpoints/x.pt -> run_dir
    else:
        run_dir = make_run_dir(config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "logs" / "train.log")],
    )
    logging.info(f"Run directory: {run_dir}" + (" (resuming)" if resuming else ""))
    if not resuming:
        log_environment_info(run_dir)

    data_dir = PROJECT_ROOT / "data"
    tokenized_path = data_dir / "processed" / config["data"]["tokenized_file"]
    train_dataset = MitoCDSDataset(tokenized_path, data_dir / "splits" / config["data"]["train_split"])
    val_dataset = MitoCDSDataset(tokenized_path, data_dir / "splits" / config["data"]["val_split"])

    logging.info(f"device={device} precision={precision} cpu_count={cpu_count} num_workers={num_workers}")
    logging.info(f"Train records={len(train_dataset)} | Val records={len(val_dataset)}")

    batch_size = config["training"]["batch_size"]
    pin_memory = config["hardware"]["pin_memory"] and device.type == "cuda"

    train_sampler = BucketBatchSampler(train_dataset.lengths(), batch_size, shuffle=True, seed=config["project"]["seed"])
    val_sampler = BucketBatchSampler(val_dataset.lengths(), batch_size, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=train_collate, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=train_collate, persistent_workers=num_workers > 0,
    )

    model = build_model(config, device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,}")

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"])
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=VOCAB["<PAD>"])

    use_aux_losses = train_cfg.get("lambda_gc", 0.0) > 0 or train_cfg.get("lambda_cai", 0.0) > 0
    aux_losses = None
    if use_aux_losses:
        logging.info("Building auxiliary loss tensors (GC-deviation, soft mt-CAI) from training corpus...")
        aux_losses = AuxiliaryLosses(train_dataset.records, device)
        logging.info(f"Auxiliary losses active: lambda_gc={train_cfg.get('lambda_gc', 0.0)} lambda_cai={train_cfg.get('lambda_cai', 0.0)}")

    accum_steps = train_cfg["gradient_accumulation_steps"]
    total_steps = max(1, (len(train_loader) // accum_steps) * train_cfg["epochs"])
    warmup_steps = min(train_cfg["warmup_steps"], max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    metrics_history = []
    start_epoch = 1
    grad_clip = train_cfg["grad_clip_norm"]

    if resuming:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            logging.warning("Checkpoint has no scheduler state (saved before resume support was added) — LR schedule restarts from this point.")
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        start_epoch = ckpt["epoch"] + 1
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics_history = json.load(f)
        logging.info(f"Resumed from epoch {ckpt['epoch']} (val_loss={ckpt.get('val_loss'):.4f}); continuing at epoch {start_epoch}")

    t_start = time.time()
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss, total_ce, total_gc, total_cai = 0.0, 0.0, 0.0, 0.0
        optimizer.zero_grad()
        for step, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device, non_blocking=pin_memory), tgt.to(device, non_blocking=pin_memory)
            targets = tgt[1:, :]
            with make_autocast(device, amp_dtype):
                logits = model(src, tgt[:-1, :])
                ce = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                if aux_losses is not None:
                    non_pad_mask = targets != VOCAB["<PAD>"]
                    parts = combined_loss(
                        ce, logits, targets, non_pad_mask, aux_losses,
                        lambda_gc=train_cfg.get("lambda_gc", 0.0), lambda_cai=train_cfg.get("lambda_cai", 0.0),
                    )
                    step_loss = parts["total"]
                    total_gc += parts["gc"].item()
                    total_cai += parts["cai"].item()
                else:
                    step_loss = ce
                loss = step_loss / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            total_loss += step_loss.item()
            total_ce += ce.item()

        train_loss = total_loss / len(train_loader)
        train_ce = total_ce / len(train_loader)
        # Validation loss stays pure cross-entropy (not the multi-objective total) so
        # early-stopping/checkpoint-selection remains comparable across runs regardless
        # of whether auxiliary losses are enabled, and comparable to the CE-only baseline run.
        val_loss = evaluate(model, val_loader, criterion, device, amp_dtype)
        elapsed = time.time() - t_start
        aux_log = ""
        if aux_losses is not None:
            aux_log = f" | train_ce={train_ce:.4f} | mean_gc_loss={total_gc / len(train_loader):.4f} | mean_cai_loss={total_cai / len(train_loader):.4f}"
        logging.info(
            f"Epoch {epoch}/{train_cfg['epochs']} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | elapsed={elapsed / 60:.1f}min{aux_log}"
        )
        metrics_history.append({
            "epoch": epoch, "train_loss": train_loss, "train_ce": train_ce, "val_loss": val_loss, "elapsed_sec": elapsed,
            **({"mean_gc_loss": total_gc / len(train_loader), "mean_cai_loss": total_cai / len(train_loader)} if aux_losses is not None else {}),
        })
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_history, f, indent=2)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        # latest.pt every epoch so stopping mid-run never loses progress, regardless
        # of whether this epoch improved on the best validation loss seen so far.
        torch.save(checkpoint_payload, run_dir / "checkpoints" / "latest.pt")
        if is_best:
            torch.save(checkpoint_payload, run_dir / "checkpoints" / "best.pt")
            logging.info(f"Saved best checkpoint (val_loss={val_loss:.4f})")

        if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
            logging.info(f"Early stopping at epoch {epoch} (no improvement for {epochs_without_improvement} epochs)")
            break

    total_min = (time.time() - t_start) / 60
    logging.info(f"Training complete. Best val_loss={best_val_loss:.4f}. Total time={total_min:.1f}min")


if __name__ == "__main__":
    main()
