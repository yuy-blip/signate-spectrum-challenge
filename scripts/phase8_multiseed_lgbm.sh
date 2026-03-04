#!/usr/bin/env bash
# Phase 8: Multi-seed LightGBM for ensemble diversity
# Run AFTER Phase 7 to find the best LightGBM config, then seed-sweep it
set -euo pipefail

echo "=========================================="
echo "Phase 8: Multi-seed LightGBM"
echo "=========================================="

BASE_CFG=configs/lgbm_pls_features.yaml
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# --- 8a: Best LightGBM config (lr=0.03, n=2000, default structure) ---
echo "=== 8a: Best LightGBM × 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config $BASE_CFG \
    --override model_params.n_estimators=2000 \
    --override model_params.learning_rate=0.03 \
    --override seed=$seed \
    --override experiment_name="lgbm_best_s${seed}"
done

# --- 8b: LightGBM no-bin config × 10 seeds ---
echo "=== 8b: LightGBM no-bin × 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config $BASE_CFG \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override model_params.n_estimators=2000 \
    --override model_params.learning_rate=0.03 \
    --override seed=$seed \
    --override experiment_name="lgbm_nobin_s${seed}"
done

# --- 8c: LightGBM + feature engineering × 5 seeds ---
echo "=== 8c: LightGBM + features × 5 seeds ==="
for seed in 0 1 2 3 4; do
  python -m spectral_challenge.cli cv \
    --config $BASE_CFG \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: spectral_stats, n_regions: 10}, {name: standard_scaler}]" \
    --override model_params.n_estimators=2000 \
    --override model_params.learning_rate=0.03 \
    --override seed=$seed \
    --override experiment_name="lgbm_feat_s${seed}"
done

echo "Phase 8 complete!"
