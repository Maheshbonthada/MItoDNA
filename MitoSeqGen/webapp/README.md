# MitoSeqGen dashboard

Full-stack showcase for the MitoSeqGen research pipeline: dataset/training/evaluation
stats plus a live protein -> codon-optimized-CDS generation demo.

- `backend/` — FastAPI, reuses `src/` directly (no duplicated logic). Serves dataset QC
  summaries, per-model training curves (polls `experiments/*/metrics.json` + logs), full
  evaluation reports with Wilcoxon significance tests, and a `/api/generate` endpoint.
- `frontend/` — React + Vite + Recharts.

## Why generation runs on CPU

`/api/generate` forces `device="cpu"` regardless of CUDA availability, specifically so the
demo never contends with a GPU model-training run happening in the background. Expect
roughly 2-10s per request depending on protein length (see `MAX_PROTEIN_LEN` in
`backend/main.py`).

## Run locally

Backend (from the repo root, so `src/` imports resolve):

```
pip install -r requirements.txt
python -m uvicorn webapp.backend.main:app --reload --port 8000
```

Frontend:

```
cd webapp/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend talks to `http://127.0.0.1:8000` by default;
override with a `VITE_API_BASE` env var if the backend runs elsewhere.

## Notes

- `GET /api/models` reflects live training state — a model still training shows
  `status: "training"` and its epoch count updates as `experiments/<run>/metrics.json`
  grows, so the Training tab in the UI updates automatically (polls every 15s).
- `GET /api/evaluation/{model_id}` 404s until `src/evaluation/evaluate.py` has been run
  for that model and its report saved to `data/qc_reports/`.
