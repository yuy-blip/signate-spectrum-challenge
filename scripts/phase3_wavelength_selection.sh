#!/usr/bin/env bash
# Phase 3: Wavelength region selection experiments
# Purpose: Focus on informative spectral regions, remove noise
set -euo pipefail

echo "=========================================="
echo "Phase 3: Wavelength selection"
echo "=========================================="

# --- 3a: Water absorption bands only ---
# O-H combination: ~5000-5500, O-H 1st overtone: ~6500-7500, C-H: ~8000-9000
echo "=== 3a: Water bands (OH combo + OH 1st + CH) ==="
for k in 12 16 20 24 28; do
  echo "--- Water bands PLS k=$k ---"
  python -m spectral_challenge.cli cv \
    --config configs/wn_select_pls.yaml \
    --override model_params.n_components=$k \
    --override experiment_name="wn_water_k${k}"
done

# --- 3b: Broader water bands ---
echo "=== 3b: Broad water bands ==="
for k in 16 20 24 28 32; do
  python -m spectral_challenge.cli cv \
    --config configs/wn_select_pls.yaml \
    --override "preprocess[0].ranges=[[4200,5800],[6000,7800],[8000,9500]]" \
    --override model_params.n_components=$k \
    --override experiment_name="wn_broad_k${k}"
done

# --- 3c: Trim noisy edges (keep 4200-9800) ---
echo "=== 3c: Trim edges ==="
for k in 24 28 30 32 36; do
  echo "--- Trimmed PLS k=$k ---"
  python -m spectral_challenge.cli cv \
    --config configs/trim_edges_pls.yaml \
    --override model_params.n_components=$k \
    --override experiment_name="trim_k${k}"
done

# --- 3d: Water bands + SG(d1) ---
echo "=== 3d: Water bands + SG(d1) ==="
for k in 12 16 20 24; do
  python -m spectral_challenge.cli cv \
    --config configs/wn_select_pls.yaml \
    --override "preprocess[2].deriv=1" \
    --override model_params.n_components=$k \
    --override experiment_name="wn_water_sg1_k${k}"
done

# --- 3e: Exclude edge noise only (remove <4100 and >9900) ---
echo "=== 3e: Exclude extreme edges ==="
for k in 24 28 32 36; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_pls.yaml \
    --override "preprocess=[{name: select_wn, ranges: [[4100, 9900]]}, {name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 0}, {name: standard_scaler}]" \
    --override model_params.n_components=$k \
    --override experiment_name="noedge_k${k}"
done

echo "Phase 3 complete!"
