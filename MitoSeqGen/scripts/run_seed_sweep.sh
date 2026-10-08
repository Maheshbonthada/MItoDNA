#!/usr/bin/env bash
# Seed replication for the strand-conditioning result.
#
# The species-level bootstrap in the manuscript quantifies data-sampling
# variance only. It says nothing about training variance, so a single run per
# condition cannot support the claim that the strand bit reverses the pooled
# comparison. This sweep trains four additional seeds per condition and
# evaluates each on the identity-controlled test split.
#
# Seed 42 already exists for both conditions:
#   control -> experiments/20261007_094622_ident70
#   strand  -> experiments/20261007_150614_ident70_strand
#
# Epoch cap: the seed-42 strand run reached best val_loss at epoch 9 and then
# degraded monotonically through epoch 19. 13 epochs with patience 4 therefore
# covers the optimum with margin at ~40% of the wall-clock cost.

set -u
cd "$(dirname "$0")/.." || exit 1

LOG=experiments/seed_sweep.log
echo "=== seed sweep started $(date -Iseconds) ===" | tee -a "$LOG"

for SEED in 1 2 3 4; do
  for COND in none strand; do
    TAG="ident70_${COND}_s${SEED}"

    if ls -d experiments/*_"${TAG}" >/dev/null 2>&1; then
      echo "[skip-train] ${TAG} already present" | tee -a "$LOG"
    else
      echo "[train] ${TAG} $(date -Iseconds)" | tee -a "$LOG"
      python -m src.training.train \
        --config config/config_ident70.yaml \
        --condition "$COND" \
        --seed "$SEED" \
        --epochs 13 \
        --patience 4 \
        --tag "$TAG" >>"$LOG" 2>&1
      if [ $? -ne 0 ]; then
        echo "[FAIL-train] ${TAG}" | tee -a "$LOG"
        continue
      fi
    fi

    RUN=$(ls -d experiments/*_"${TAG}" 2>/dev/null | head -1)
    CKPT="${RUN}/checkpoints/best.pt"
    if [ ! -f "$CKPT" ]; then
      echo "[FAIL-ckpt] ${TAG}: no ${CKPT}" | tee -a "$LOG"
      continue
    fi

    if [ -f "data/qc_reports/model_eval_ident70_${TAG}.json" ]; then
      echo "[skip-eval] ${TAG} already evaluated" | tee -a "$LOG"
      continue
    fi

    echo "[eval] ${TAG} $(date -Iseconds)" | tee -a "$LOG"
    python scripts/evaluate_model_split.py \
      --checkpoint "$CKPT" \
      --tag ident70 \
      --test-split data/splits/test_ident70_per_gene.json \
      --train-split data/splits/train_ident70_per_gene.json \
      --condition "$COND" \
      --workers 20 \
      --out-suffix "_${TAG}" >>"$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "[FAIL-eval] ${TAG}" | tee -a "$LOG"
    fi
  done
done

echo "=== seed sweep finished $(date -Iseconds) ===" | tee -a "$LOG"
