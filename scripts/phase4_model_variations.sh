#!/usr/bin/env bash
# Phase 4: Model variations (ElasticNet, LightGBM, Ridge sweeps)
# Purpose: Try different model families
set -euo pipefail

echo "=========================================="
echo "Phase 4: Model variations"
echo "=========================================="

# --- 4a: ElasticNet with SNV+SG(d1), sweep alpha and l1_ratio ---
echo "=== 4a: ElasticNet ==="
for alpha in 0.01 0.05 0.1 0.5 1.0; do
  for l1 in 0.1 0.3 0.5 0.7 0.9; do
    echo "--- ElasticNet alpha=$alpha l1=$l1 ---"
    python -m spectral_challenge.cli cv \
      --config configs/elastic_net.yaml \
      --override model_params.alpha=$alpha \
      --override model_params.l1_ratio=$l1 \
      --override experiment_name="enet_a${alpha}_l${l1}"
  done
done

# --- 4b: Ridge with SNV+SG(d1), sweep alpha ---
echo "=== 4b: Ridge + SG(d1) ==="
for alpha in 0.01 0.1 1.0 10.0 100.0 1000.0; do
  echo "--- Ridge alpha=$alpha ---"
  python -m spectral_challenge.cli cv \
    --config configs/baseline_ridge.yaml \
    --override model_params.alpha=$alpha \
    --override experiment_name="ridge_sg1_a${alpha}"
done

# --- 4c: Ridge with SNV+SG(d0), sweep alpha ---
echo "=== 4c: Ridge + SG(d0) ==="
for alpha in 0.01 0.1 1.0 10.0 100.0 1000.0; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_ridge.yaml \
    --override "preprocess[1].deriv=0" \
    --override model_params.alpha=$alpha \
    --override experiment_name="ridge_sg0_a${alpha}"
done

# --- 4d: LightGBM with SNV+SG(d1)+binning ---
echo "=== 4d: LightGBM ==="
for lr in 0.01 0.03 0.05; do
  for nest in 500 1000 2000; do
    echo "--- LightGBM lr=$lr n=$nest ---"
    python -m spectral_challenge.cli cv \
      --config configs/lgbm_pls_features.yaml \
      --override model_params.learning_rate=$lr \
      --override model_params.n_estimators=$nest \
      --override experiment_name="lgbm_lr${lr}_n${nest}"
  done
done

# --- 4e: LightGBM with raw features (no heavy preprocess) ---
echo "=== 4e: LightGBM raw ==="
for nest in 500 1000 2000; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_pls_features.yaml \
    --override "preprocess=[{name: standard_scaler}]" \
    --override model_params.n_estimators=$nest \
    --override model_params.learning_rate=0.03 \
    --override experiment_name="lgbm_raw_n${nest}"
done

echo "Phase 4 complete!"
