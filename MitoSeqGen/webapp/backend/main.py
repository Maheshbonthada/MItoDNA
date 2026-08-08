"""FastAPI backend for the MitoSeqGen public showcase UI.

Serves dataset/training/evaluation stats already produced by the research
pipeline in src/, plus a live protein -> codon-optimized-CDS generation demo.
Inference always runs on CPU by explicit device choice, regardless of CUDA
availability, so the demo never contends with a GPU training run happening
in the background.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import AA_VOCAB
from src.evaluation.baselines import Baselines
from src.evaluation.metrics import (
    codon_diversity,
    compute_mfe,
    compute_mt_cai,
    compute_rscu_weights,
    gc_content,
    simple_bleu,
    translation_complies_with_mt_code,
)
from src.models.generate import generate_cds, load_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("mitoseqgen-api")

DEVICE = torch.device("cpu")  # deliberate: never touch the GPU that may be training
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"
QC_DIR = DATA_DIR / "qc_reports"

MODEL_REGISTRY = {
    "model1": {
        "label": "Model 1 (cross-entropy baseline)",
        "run_dir": EXPERIMENTS_DIR / "20260722_140803",
        "evaluation_report": QC_DIR / "evaluation_report.json",
    },
    "model2": {
        "label": "Model 2 (multi-objective: CE + GC[batch-pooled, inert] + mt-CAI)",
        "run_dir": EXPERIMENTS_DIR / "20260805_091030",
        "evaluation_report": QC_DIR / "evaluation_report_model2.json",
    },
    "model3": {
        "label": "Model 3 (multi-objective: CE + GC[per-sequence, unrestricted-vocab — regressed on eval])",
        "run_dir": EXPERIMENTS_DIR / "20260805_174421",
        "evaluation_report": QC_DIR / "evaluation_report_model3.json",
    },
    "model4": {
        "label": "Model 4 (multi-objective: CE + GC[per-position, synonym-class-restricted])",
        "run_dir": EXPERIMENTS_DIR / "20260806_103249",
        "evaluation_report": QC_DIR / "evaluation_report_model4.json",
    },
}

MAX_PROTEIN_LEN = 700  # bounds CPU-inference latency for the live demo

app = FastAPI(title="MitoSeqGen API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: Dict[str, tuple] = {}  # model_id -> (mtime, model)
_baselines: Optional[Baselines] = None
_rscu_weights: Optional[dict] = None
_train_hashes: Optional[set] = None
_sample_records: Optional[List[dict]] = None


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _load_processed_records() -> List[dict]:
    return _read_json(DATA_DIR / "processed" / "mito_cds_tokenized.json")["records"]


def _load_split(name: str) -> List[int]:
    return _read_json(DATA_DIR / "splits" / f"{name}.json")["indices"]


@app.on_event("startup")
def _startup():
    global _baselines, _rscu_weights, _train_hashes, _sample_records
    from src.evaluation.metrics import sequence_hash

    t0 = time.time()
    log.info("Loading processed records and building baselines/RSCU table (one-time startup cost)...")
    records = _load_processed_records()
    train_idx = _load_split("train")
    test_idx = _load_split("test")
    train_records = [records[i] for i in train_idx]

    _baselines = Baselines(train_records, seed=42)
    _rscu_weights = compute_rscu_weights(train_records)
    _train_hashes = {sequence_hash(r["sequence"]) for r in train_records}

    rng_idx = test_idx[:12]
    _sample_records = [
        {
            "id": records[i]["id"],
            "gene_name": records[i]["gene_name"],
            "species": records[i]["species"],
            "protein_sequence": records[i]["protein_sequence"].rstrip("*"),
            "reference_cds": records[i]["sequence"],
        }
        for i in rng_idx
    ]
    log.info(f"Startup ready in {time.time() - t0:.1f}s ({len(train_records)} train records)")


def _get_model(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model_id '{model_id}'")
    ckpt_path = MODEL_REGISTRY[model_id]["run_dir"] / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise HTTPException(status_code=503, detail=f"{model_id} has no checkpoint yet")

    mtime = ckpt_path.stat().st_mtime
    cached = _model_cache.get(model_id)
    if cached and cached[0] == mtime:
        return cached[1]

    log.info(f"(Re)loading {model_id} from {ckpt_path} onto CPU...")
    model = load_model_from_checkpoint(ckpt_path, DEVICE)
    _model_cache[model_id] = (mtime, model)
    return model


def _run_status(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "logs" / "train.log"
    config_path = run_dir / "config.json"

    epochs = _read_json(metrics_path) if metrics_path.exists() else []
    configured_epochs = None
    early_stopping_patience = None
    if config_path.exists():
        cfg = _read_json(config_path)
        configured_epochs = cfg.get("training", {}).get("epochs")
        early_stopping_patience = cfg.get("training", {}).get("early_stopping_patience")

    last_line = ""
    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        last_line = lines[-1].strip() if lines else ""

    is_complete = configured_epochs is not None and len(epochs) >= configured_epochs
    # Heuristic early-stopping detection: no improvement in val_loss for
    # `patience` consecutive trailing epochs and the run stopped short.
    if not is_complete and early_stopping_patience and len(epochs) > early_stopping_patience:
        best = min(e["val_loss"] for e in epochs)
        best_idx = next(i for i, e in enumerate(epochs) if e["val_loss"] == best)
        if len(epochs) - 1 - best_idx >= early_stopping_patience:
            is_complete = True

    best_epoch = min(epochs, key=lambda e: e["val_loss"]) if epochs else None

    return {
        "epochs_completed": len(epochs),
        "configured_epochs": configured_epochs,
        "status": "complete" if is_complete else ("training" if epochs else "starting"),
        "best_val_loss": best_epoch["val_loss"] if best_epoch else None,
        "best_epoch": best_epoch["epoch"] if best_epoch else None,
        "latest_log_line": last_line,
        "history": epochs,
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/models")
def list_models():
    out = []
    for model_id, meta in MODEL_REGISTRY.items():
        status = _run_status(meta["run_dir"])
        out.append({
            "model_id": model_id,
            "label": meta["label"],
            "has_evaluation": meta["evaluation_report"] is not None and meta["evaluation_report"].exists(),
            **status,
        })
    return out


@app.get("/api/training/{model_id}")
def training_curve(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model_id '{model_id}'")
    return _run_status(MODEL_REGISTRY[model_id]["run_dir"])


@app.get("/api/evaluation/{model_id}")
def evaluation(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model_id '{model_id}'")
    report_path = MODEL_REGISTRY[model_id]["evaluation_report"]
    if report_path is None or not report_path.exists():
        raise HTTPException(status_code=404, detail=f"No evaluation report yet for {model_id}")
    report = _read_json(report_path)
    # raw_results can be large (n=200 x 6 methods); the dashboard only needs
    # summary + significance tests, so drop it rather than ship it over the wire.
    report.pop("raw_results", None)
    return report


def _qc_headline(qc: dict) -> dict:
    # "warnings" can be 100k+ entries (one per borderline record) and
    # "rejection_breakdown"/"taxonomic_breakdown" are internal audit detail —
    # neither belongs in a dashboard payload, so whitelist what the UI needs.
    top_rejections = dict(
        sorted(qc.get("rejection_breakdown", {}).items(), key=lambda kv: -kv[1])[:10]
    )
    return {
        "run_timestamp": qc.get("run_timestamp"),
        "total_downloaded": qc.get("total_downloaded"),
        "total_passed_qc": qc.get("total_passed_qc"),
        "total_rejected": qc.get("total_rejected"),
        "species_count": qc.get("species_count"),
        "gene_distribution": qc.get("gene_distribution"),
        "top_rejection_reasons": top_rejections,
        "reviewer_readiness": qc.get("reviewer_readiness"),
    }


@app.get("/api/dataset")
def dataset_summary():
    qc_path = QC_DIR / "qc_report_2026-07-22.json"
    trna_path = QC_DIR / "trna_report_2026-07-22.json"
    out = {}
    if qc_path.exists():
        out["cds_qc"] = _qc_headline(_read_json(qc_path))
    if trna_path.exists():
        out["trna_qc"] = _read_json(trna_path)  # already small/flat, no whitelisting needed
    train_idx = _load_split("train")
    val_idx = _load_split("val")
    test_idx = _load_split("test")
    out["splits"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    return out


@app.get("/api/samples")
def sample_proteins():
    return _sample_records


class GenerateRequest(BaseModel):
    model_id: str = Field(default="model1")
    protein_sequence: Optional[str] = None
    sample_id: Optional[str] = None
    include_baselines: bool = True


def _metrics_for(seq: str, reference: Optional[str]) -> dict:
    m = {
        "mt_cai": compute_mt_cai(seq, _rscu_weights),
        "code_compliant": translation_complies_with_mt_code(seq),
        "codon_diversity": codon_diversity(seq),
        "gc_content": gc_content(seq),
        "mfe": compute_mfe(seq),
    }
    if reference:
        m["bleu"] = simple_bleu(reference, seq)
    return m


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.sample_id:
        rec = next((r for r in _sample_records if r["id"] == req.sample_id), None)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Unknown sample_id '{req.sample_id}'")
        protein = rec["protein_sequence"]
        reference = rec["reference_cds"]
        label = f"{rec['gene_name']} ({rec['species']}) — held-out test set"
    elif req.protein_sequence:
        protein = req.protein_sequence.strip().upper().rstrip("*")
        reference = None
        label = "custom input"
    else:
        raise HTTPException(status_code=400, detail="Provide either protein_sequence or sample_id")

    if not protein:
        raise HTTPException(status_code=400, detail="Empty protein sequence")
    if len(protein) > MAX_PROTEIN_LEN:
        raise HTTPException(status_code=400, detail=f"Protein too long for the live demo (max {MAX_PROTEIN_LEN} aa)")
    bad_chars = sorted(set(protein) - set(AA_VOCAB.keys()))
    if bad_chars:
        raise HTTPException(status_code=400, detail=f"Invalid amino acid character(s): {bad_chars}")

    model = _get_model(req.model_id)
    t0 = time.time()
    generated = generate_cds(model, protein, DEVICE, strategy="greedy")
    elapsed = time.time() - t0

    result = {
        "input_label": label,
        "protein_sequence": protein,
        "generated_cds": generated,
        "reference_cds": reference,
        "generation_seconds": elapsed,
        "metrics": _metrics_for(generated, reference),
    }
    if reference:
        result["reference_metrics"] = _metrics_for(reference, reference)

    if req.include_baselines:
        baseline_out = {}
        for name in ["mt_cai_lookup", "random_synonymous", "most_frequent_codon", "codontransformer_remap"]:
            seq = _baselines.generate(protein, name)
            baseline_out[name] = {"generated_cds": seq, "metrics": _metrics_for(seq, reference)}
        result["baselines"] = baseline_out

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
