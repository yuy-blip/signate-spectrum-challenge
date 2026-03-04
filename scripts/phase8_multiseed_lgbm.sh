#!/usr/bin/env bash
# Phase 8: Multi-seed runs for best GKF models
# Run AFTER Phase 7 to identify best configs, then seed-sweep them
#
# NOTE: Update the config overrides below based on Phase 7 results!
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Phase 8: Multi-seed for best GKF models"
echo "=========================================="

SEEDS=(0 1 2 3 4 5 6 7 8 9)

# --- 8a: Best PLS config × 10 seeds ---
# UPDATE: Replace n_components and window_length with Phase 7 best
echo "=== 8a: Best PLS × 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config configs/snv_sg1_pls.yaml \
    --override "model_params.n_components=20" \
    --override "seed=$seed" \
    --override "experiment_name=gkf_pls_best_s${seed}"
done

# --- 8b: Best Ridge × 10 seeds ---
echo "=== 8b: Best Ridge × 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_ridge.yaml \
    --override "model_params.alpha=10.0" \
    --override "seed=$seed" \
    --override "experiment_name=gkf_ridge_best_s${seed}"
done

# --- 8c: Best shallow LightGBM × 10 seeds ---
echo "=== 8c: Best shallow LightGBM × 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "seed=$seed" \
    --override "experiment_name=gkf_lgbm_best_s${seed}"
done

echo "Phase 8 complete!"
