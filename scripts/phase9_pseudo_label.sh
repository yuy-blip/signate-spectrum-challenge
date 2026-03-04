#!/usr/bin/env bash
# Phase 9: Pseudo-labeling experiments
# Semi-supervised: use confident test predictions as extra training data
set -euo pipefail

echo "=========================================="
echo "Phase 9: Pseudo-labeling"
echo "=========================================="

# --- 9a: LightGBM pseudo-labeling ---
echo "=== 9a: LightGBM pseudo-labeling ==="
for conf in 0.5 0.7 0.8 0.9; do
  for iters in 1 2 3; do
    echo "--- PL confidence=$conf iterations=$iters ---"
    python scripts/pseudo_labeling.py \
      --config configs/lgbm_pls_features.yaml \
      --confidence-threshold $conf \
      --iterations $iters \
      --seed 42
  done
done

# --- 9b: PLS pseudo-labeling ---
echo "=== 9b: PLS pseudo-labeling ==="
for conf in 0.7 0.8 0.9; do
  echo "--- PLS PL confidence=$conf ---"
  python scripts/pseudo_labeling.py \
    --config configs/baseline_pls.yaml \
    --confidence-threshold $conf \
    --iterations 2 \
    --seed 42
done

echo "Phase 9 complete!"
