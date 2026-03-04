#!/usr/bin/env bash
# Phase 2: Preprocessing sweep (SG derivatives, window sizes)
# Purpose: Find the best preprocessing combination for PLS
set -euo pipefail

echo "=========================================="
echo "Phase 2: Preprocessing sweep"
echo "=========================================="

# --- 2a: SNV + SG 1st derivative + PLS, sweep k ---
echo "=== 2a: SNV + SG(d1) + PLS ==="
for k in 16 20 24 28 32 36 40 50 60; do
  echo "--- SNV+SG(d1) PLS k=$k ---"
  python -m spectral_challenge.cli cv \
    --config configs/snv_sg1_pls.yaml \
    --override model_params.n_components=$k \
    --override experiment_name="snv_sg1_k${k}"
done

# --- 2b: SNV + SG 2nd derivative + PLS, sweep k ---
echo "=== 2b: SNV + SG(d2) + PLS ==="
for k in 12 16 20 24 28 32 36; do
  echo "--- SNV+SG(d2) PLS k=$k ---"
  python -m spectral_challenge.cli cv \
    --config configs/snv_sg2_pls.yaml \
    --override model_params.n_components=$k \
    --override experiment_name="snv_sg2_k${k}"
done

# --- 2c: SG(d1) only (no SNV), sweep k ---
echo "=== 2c: SG(d1) only + PLS ==="
for k in 20 24 28 32 36 40; do
  echo "--- SG(d1) PLS k=$k ---"
  python -m spectral_challenge.cli cv \
    --config configs/sg1_only_pls.yaml \
    --override model_params.n_components=$k \
    --override experiment_name="sg1_only_k${k}"
done

# --- 2d: SG window size sweep (d0, best k=30 from previous) ---
echo "=== 2d: SG window sweep (d0) ==="
for w in 7 11 15 21 31; do
  echo "--- SNV+SG(d0,w=$w) PLS k=30 ---"
  python -m spectral_challenge.cli cv \
    --config configs/baseline_pls.yaml \
    --override "preprocess[1].window_length=$w" \
    --override model_params.n_components=30 \
    --override experiment_name="sg0_w${w}_k30"
done

# --- 2e: SG window size sweep (d1, best k=28 from previous) ---
echo "=== 2e: SG window sweep (d1) ==="
for w in 7 11 15 21 31; do
  echo "--- SNV+SG(d1,w=$w) PLS k=28 ---"
  python -m spectral_challenge.cli cv \
    --config configs/snv_sg1_pls.yaml \
    --override "preprocess[1].window_length=$w" \
    --override model_params.n_components=28 \
    --override experiment_name="sg1_w${w}_k28"
done

echo "Phase 2 complete!"
