#!/usr/bin/env bash
# Phase 1: GroupKFold experiments (honest evaluation by species)
# Purpose: Get true generalization performance when species differ
set -euo pipefail

echo "=========================================="
echo "Phase 1: GroupKFold by species"
echo "=========================================="

CFG=configs/gkf_pls.yaml

# GKF + SNV + SG(d0) + PLS: sweep k
for k in 20 24 28 30 32 36 40; do
  echo "--- GKF SNV+SG(d0) PLS k=$k ---"
  python -m spectral_challenge.cli cv \
    --config $CFG \
    --override model_params.n_components=$k \
    --override experiment_name="gkf_snv_sg0_k${k}"
done

# GKF + SNV + SG(d1) + PLS: sweep k
for k in 20 24 28 30 32 36 40; do
  echo "--- GKF SNV+SG(d1) PLS k=$k ---"
  python -m spectral_challenge.cli cv \
    --config $CFG \
    --override "preprocess[1].deriv=1" \
    --override model_params.n_components=$k \
    --override experiment_name="gkf_snv_sg1_k${k}"
done

# GKF + SNV + SG(d1) + Ridge
for alpha in 0.1 1.0 10.0 100.0; do
  echo "--- GKF SNV+SG(d1) Ridge alpha=$alpha ---"
  python -m spectral_challenge.cli cv \
    --config configs/baseline_ridge.yaml \
    --override split_method=group_kfold \
    --override "group_col=species number" \
    --override "model_params.alpha=$alpha" \
    --override experiment_name="gkf_ridge_a${alpha}"
done

echo "Phase 1 complete!"
