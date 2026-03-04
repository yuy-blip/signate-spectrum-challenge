#!/usr/bin/env bash
# Phase 5: Multi-seed ensemble for best configurations
# Purpose: Robust evaluation + generate diverse models for ensemble
# Run AFTER reviewing Phase 1-4 results to pick best configs
set -euo pipefail

echo "=========================================="
echo "Phase 5: Multi-seed best configs"
echo "=========================================="

SEEDS=(0 1 2 3 4 5 6 7 8 9)

# --- 5a: Best PLS (SNV+SG(d0), k=30) - from pls_refine results ---
echo "=== 5a: Best PLS k=30, 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_pls.yaml \
    --override model_params.n_components=30 \
    --override seed=$seed \
    --override experiment_name="best_pls_k30_s${seed}"
done

# --- 5b: Best PLS k=32 ---
echo "=== 5b: Best PLS k=32, 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_pls.yaml \
    --override model_params.n_components=32 \
    --override seed=$seed \
    --override experiment_name="best_pls_k32_s${seed}"
done

# --- 5c: Best PLS k=28 ---
echo "=== 5c: Best PLS k=28, 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_pls.yaml \
    --override model_params.n_components=28 \
    --override seed=$seed \
    --override experiment_name="best_pls_k28_s${seed}"
done

# --- 5d: SG(d1) PLS best k (update k after Phase 2) ---
echo "=== 5d: SNV+SG(d1) PLS k=28, 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config configs/snv_sg1_pls.yaml \
    --override model_params.n_components=28 \
    --override seed=$seed \
    --override experiment_name="best_sg1_k28_s${seed}"
done

# --- 5e: Ridge best alpha (update after Phase 4) ---
echo "=== 5e: Ridge alpha=1.0, 10 seeds ==="
for seed in "${SEEDS[@]}"; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_ridge.yaml \
    --override model_params.alpha=1.0 \
    --override seed=$seed \
    --override experiment_name="best_ridge_s${seed}"
done

echo "Phase 5 complete!"
