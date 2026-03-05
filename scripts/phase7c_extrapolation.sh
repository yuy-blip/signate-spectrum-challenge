#!/usr/bin/env bash
# Phase 7c: Extrapolation-aware strategy + ensemble diversity
#
# KEY INSIGHT: LightGBM cannot extrapolate beyond training range.
# ベイスギ max=298 but training max ~216 → Fold 2 RMSE 33+
# PLS/Ridge CAN extrapolate → may be better on Fold 2
# Strategy: best single models + LightGBM×PLS blend
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=========================================="
echo "Phase 7c: Extrapolation + Ensemble"
echo "=========================================="

# ===================================================================
# A. Multi-seed EMSC2 + LightGBM (current best 19.23)
# ===================================================================
echo "=== A: Multi-seed best config ==="

for seed in 0 1 2 3 4 5 7 9 11 13 17 21 33 55 77 99; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "seed=$seed" \
    --override "experiment_name=gkf_emsc2_lgbm_s${seed}"
done

# ===================================================================
# B. EMSC2 + LightGBM hyperparameter variations
# ===================================================================
echo "=== B: EMSC2 + LightGBM variations ==="

# Deeper trees (still bounded but might learn better)
for depth in 4 5 6; do
  for leaves in 15 31; do
    python -m spectral_challenge cv \
      --config configs/lgbm_shallow_gkf.yaml \
      --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
      --override "model_params.max_depth=$depth" \
      --override "model_params.num_leaves=$leaves" \
      --override "experiment_name=gkf_emsc2_lgbm_d${depth}_l${leaves}"
  done
done

# Lower regularization (allow more range)
for ra in 0.0 1.0; do
  for rl in 0.0 1.0; do
    python -m spectral_challenge cv \
      --config configs/lgbm_shallow_gkf.yaml \
      --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
      --override "model_params.reg_alpha=$ra" \
      --override "model_params.reg_lambda=$rl" \
      --override "experiment_name=gkf_emsc2_lgbm_ra${ra}_rl${rl}"
  done
done

# Different bin sizes with EMSC2
for bs in 4 6 12 16; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: $bs}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_emsc2_lgbm_bs${bs}"
done

# EMSC2 + different SG window lengths
for wl in 5 9 11 15; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: $wl, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_emsc2_lgbm_sgwl${wl}"
done

# ===================================================================
# C. PLS/Ridge with EMSC2 (for EXTRAPOLATION capability)
# ===================================================================
echo "=== C: Linear models with EMSC2 (extrapolation) ==="

# EMSC2 + SG1 + PLS (key: can extrapolate for ベイスギ!)
for nc in 3 5 7 10 12 15 20; do
  python -m spectral_challenge cv \
    --config configs/gkf_pls.yaml \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_emsc2_bin_pls_nc${nc}"
done

# EMSC2 + Ridge (extrapolation)
for alpha in 0.01 0.1 1.0 10.0 100.0 1000.0; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=ridge" \
    --override "model_params={alpha: $alpha}" \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_emsc2_ridge_a${alpha}"
done

# EMSC2 + ElasticNet (extrapolation + feature selection)
for alpha in 0.01 0.1 1.0; do
  for l1r in 0.1 0.5 0.9; do
    python -m spectral_challenge cv \
      --config configs/lgbm_shallow_gkf.yaml \
      --override "model_type=elastic_net" \
      --override "model_params={alpha: $alpha, l1_ratio: $l1r, max_iter: 10000}" \
      --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
      --override "experiment_name=gkf_emsc2_enet_a${alpha}_l${l1r}"
  done
done

# ===================================================================
# D. GKF-fixed target transforms (Phase 7b used wrong base config!)
# ===================================================================
echo "=== D: Target transforms (GKF-fixed) ==="

# log1p + PLS (GKF)
for nc in 5 10 15 20; do
  python -m spectral_challenge cv \
    --config configs/gkf_pls.yaml \
    --override "target_transform=log1p" \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_log1p_pls_nc${nc}"
done

# log1p + Ridge (GKF)
for alpha in 0.1 1.0 10.0 100.0; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=ridge" \
    --override "model_params={alpha: $alpha}" \
    --override "target_transform=log1p" \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_log1p_ridge_a${alpha}"
done

# log1p + EMSC2 + PLS (GKF)
for nc in 5 10 15; do
  python -m spectral_challenge cv \
    --config configs/gkf_pls.yaml \
    --override "target_transform=log1p" \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_log1p_emsc2_pls_nc${nc}"
done

# log1p + EMSC2 + Ridge
for alpha in 0.1 1.0 10.0 100.0; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=ridge" \
    --override "model_params={alpha: $alpha}" \
    --override "target_transform=log1p" \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_log1p_emsc2_ridge_a${alpha}"
done

# ===================================================================
# E. GKF-fixed SVR/Huber (with proper base config)
# ===================================================================
echo "=== E: SKIPPED (SVR/Huber too slow + poor accuracy) ==="

# ===================================================================
# F. Best GKF PLS/Ridge (for Fold 2 comparison)
# ===================================================================
echo "=== F: Best PLS/Ridge baseline for blend comparison ==="

# SNV + SG + PLS (best GKF PLS configs to compare Fold 2)
for nc in 5 10 15 20 25 28 30; do
  python -m spectral_challenge cv \
    --config configs/gkf_pls.yaml \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_snv_sg1_pls_nc${nc}"
done

# SNV + SG + Ridge
for alpha in 0.01 0.1 1.0 10.0 100.0; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=ridge" \
    --override "model_params={alpha: $alpha}" \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_snv_sg1_ridge_a${alpha}"
done

# SG1 only + Ridge (simplest possible)
for alpha in 0.1 1.0 10.0; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=ridge" \
    --override "model_params={alpha: $alpha}" \
    --override "preprocess=[{name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_sg1_ridge_a${alpha}"
done

# ===================================================================
# G. 2nd derivative approaches (removes baseline entirely)
# ===================================================================
echo "=== G: 2nd derivative (baseline-free) ==="

# SG2 + PLS
for nc in 3 5 7 10 15; do
  python -m spectral_challenge cv \
    --config configs/gkf_pls.yaml \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 21, polyorder: 3, deriv: 2}, {name: standard_scaler}]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_sg2_pls_nc${nc}"
done

# SG2 + LightGBM
for wl in 15 21 31; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: snv}, {name: sg, window_length: $wl, polyorder: 3, deriv: 2}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_sg2_wl${wl}_lgbm"
done

# SG2 + Ridge (extrapolation)
for alpha in 0.1 1.0 10.0; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=ridge" \
    --override "model_params={alpha: $alpha}" \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 21, polyorder: 3, deriv: 2}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_sg2_ridge_a${alpha}"
done

# EMSC2 + SG2 + LightGBM
python -m spectral_challenge cv \
  --config configs/lgbm_shallow_gkf.yaml \
  --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 21, polyorder: 3, deriv: 2}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
  --override "experiment_name=gkf_emsc2_sg2_lgbm"

# EMSC2 + SG2 + Ridge
for alpha in 0.1 1.0 10.0; do
  python -m spectral_challenge cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=ridge" \
    --override "model_params={alpha: $alpha}" \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 21, polyorder: 3, deriv: 2}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_emsc2_sg2_ridge_a${alpha}"
done

echo ""
echo "Phase 7c complete!"
echo ""
echo "Key analysis after this phase:"
echo "  1. Compare Fold 2 RMSE between LightGBM vs PLS/Ridge"
echo "  2. python scripts/analyze_results.py --top 30 --filter gkf"
echo "  3. python scripts/optimize_blend.py <best_lgbm_run> <best_pls_run> <best_ridge_run>"
