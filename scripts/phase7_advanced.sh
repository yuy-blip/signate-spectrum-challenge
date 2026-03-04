#!/usr/bin/env bash
# Phase 7: Advanced experiments
# - LightGBM deep tuning (more trees, different preprocess, feature engineering)
# - Target transforms
# - Feature engineering
set -euo pipefail

echo "=========================================="
echo "Phase 7: Advanced LightGBM & Feature Engineering"
echo "=========================================="

BASE_CFG=configs/lgbm_pls_features.yaml

# --- 7a: LightGBM deeper tuning ---
echo "=== 7a: LightGBM deep tuning ==="

# More trees, smaller learning rate
for nest in 3000 5000; do
  echo "--- LightGBM lr=0.01 n=$nest ---"
  python -m spectral_challenge.cli cv \
    --config $BASE_CFG \
    --override model_params.learning_rate=0.01 \
    --override model_params.n_estimators=$nest \
    --override experiment_name="lgbm_deep_lr0.01_n${nest}"
done

# Deeper trees
for depth in 8 10 -1; do
  for leaves in 31 63 127; do
    echo "--- LightGBM depth=$depth leaves=$leaves ---"
    python -m spectral_challenge.cli cv \
      --config $BASE_CFG \
      --override model_params.max_depth=$depth \
      --override model_params.num_leaves=$leaves \
      --override model_params.n_estimators=2000 \
      --override model_params.learning_rate=0.03 \
      --override experiment_name="lgbm_d${depth}_l${leaves}"
  done
done

# More regularization sweep
for alpha in 0.0 0.5 1.0 5.0; do
  for lam in 0.0 1.0 5.0 10.0; do
    echo "--- LightGBM reg_alpha=$alpha reg_lambda=$lam ---"
    python -m spectral_challenge.cli cv \
      --config $BASE_CFG \
      --override model_params.reg_alpha=$alpha \
      --override model_params.reg_lambda=$lam \
      --override model_params.n_estimators=2000 \
      --override model_params.learning_rate=0.03 \
      --override experiment_name="lgbm_ra${alpha}_rl${lam}"
  done
done

# --- 7b: LightGBM with different preprocessing ---
echo "=== 7b: LightGBM preprocessing variants ==="

# No binning (full features)
echo "--- LightGBM no binning ---"
python -m spectral_challenge.cli cv \
  --config $BASE_CFG \
  --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
  --override model_params.n_estimators=2000 \
  --override model_params.learning_rate=0.03 \
  --override experiment_name="lgbm_nobin"

# Different bin sizes
for bs in 2 4 16; do
  echo "--- LightGBM bin_size=$bs ---"
  python -m spectral_challenge.cli cv \
    --config $BASE_CFG \
    --override "preprocess[2].bin_size=$bs" \
    --override model_params.n_estimators=2000 \
    --override model_params.learning_rate=0.03 \
    --override experiment_name="lgbm_bin${bs}"
done

# SG(d0) + binning
echo "--- LightGBM SG(d0) ---"
python -m spectral_challenge.cli cv \
  --config $BASE_CFG \
  --override "preprocess[1].deriv=0" \
  --override model_params.n_estimators=2000 \
  --override model_params.learning_rate=0.03 \
  --override experiment_name="lgbm_sg0"

# SG(d2) + binning
echo "--- LightGBM SG(d2) ---"
python -m spectral_challenge.cli cv \
  --config $BASE_CFG \
  --override "preprocess=[{name: snv}, {name: sg, window_length: 15, polyorder: 3, deriv: 2}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
  --override model_params.n_estimators=2000 \
  --override model_params.learning_rate=0.03 \
  --override experiment_name="lgbm_sg2"

# MSC instead of SNV
echo "--- LightGBM MSC ---"
python -m spectral_challenge.cli cv \
  --config $BASE_CFG \
  --override "preprocess=[{name: msc}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
  --override model_params.n_estimators=2000 \
  --override model_params.learning_rate=0.03 \
  --override experiment_name="lgbm_msc"

# Absorbance + MSC
echo "--- LightGBM Absorbance+MSC ---"
python -m spectral_challenge.cli cv \
  --config $BASE_CFG \
  --override "preprocess=[{name: absorbance, base: e}, {name: msc}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
  --override model_params.n_estimators=2000 \
  --override model_params.learning_rate=0.03 \
  --override experiment_name="lgbm_abs_msc"

# --- 7c: Feature engineering ---
echo "=== 7c: Feature engineering ==="

# Band ratios + LightGBM
echo "--- LightGBM + band ratios ---"
python -m spectral_challenge.cli cv \
  --config $BASE_CFG \
  --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: band_ratio}, {name: standard_scaler}]" \
  --override model_params.n_estimators=2000 \
  --override model_params.learning_rate=0.03 \
  --override experiment_name="lgbm_bandratio"

# Spectral stats + LightGBM
echo "--- LightGBM + spectral stats ---"
python -m spectral_challenge.cli cv \
  --config $BASE_CFG \
  --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: spectral_stats, n_regions: 10}, {name: standard_scaler}]" \
  --override model_params.n_estimators=2000 \
  --override model_params.learning_rate=0.03 \
  --override experiment_name="lgbm_specstats"

# PCA features + LightGBM
echo "--- LightGBM + PCA features ---"
python -m spectral_challenge.cli cv \
  --config $BASE_CFG \
  --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: pca_features, n_components: 20}, {name: standard_scaler}]" \
  --override model_params.n_estimators=2000 \
  --override model_params.learning_rate=0.03 \
  --override experiment_name="lgbm_pca"

echo "Phase 7 complete!"
