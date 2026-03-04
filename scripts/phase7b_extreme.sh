#!/usr/bin/env bash
# Phase 7b: Extreme optimization for RMSE ≤ 10
#
# Current best: LightGBM GKF RMSE 20.30
# Target: RMSE ~10 (1st place level)
#
# Strategy:
#   A. LightGBM fine-tuning around best config (sg1_wl7)
#   B. EMSC preprocessing (better scatter correction than MSC)
#   C. Band ratios + spectral stats features
#   D. New models: RF, ExtraTrees, KernelRidge, XGBoost
#   E. Target transform (log1p/sqrt) for skewed 含水率
#   F. PCA-reduced features + various models
#   G. Diverse configs for ensemble
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Phase 7b: Push to RMSE ≤ 10"
echo "=========================================="

# Common GKF settings (applied via config defaults, all configs use group_kfold)

# ===================================================================
# A. LightGBM fine-tuning: sg1_wl5/7/9, bin 4/8/16, colsample, lr
# ===================================================================
echo "=== A: LightGBM fine-tuning ==="

# Window length fine-tuning
for wl in 5 7 9; do
  for bs in 4 6 8 12 16; do
    python -m spectral_challenge.cli cv \
      --config configs/lgbm_shallow_gkf.yaml \
      --override "preprocess=[{name: snv}, {name: sg, window_length: $wl, polyorder: 2, deriv: 1}, {name: binning, bin_size: $bs}, {name: standard_scaler}]" \
      --override "experiment_name=gkf_lgbm_wl${wl}_bs${bs}"
  done
done

# LightGBM hyperparameter refinement on best preprocessing (wl7, bs8)
for lr in 0.005 0.01 0.02 0.03; do
  for nest in 1000 2000 3000 5000; do
    for cs in 0.3 0.5 0.7; do
      python -m spectral_challenge.cli cv \
        --config configs/lgbm_shallow_gkf.yaml \
        --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
        --override "model_params.learning_rate=$lr" \
        --override "model_params.n_estimators=$nest" \
        --override "model_params.colsample_bytree=$cs" \
        --override "experiment_name=gkf_lgbm_lr${lr}_n${nest}_cs${cs}"
    done
  done
done

# LightGBM with different SG derivative orders
for deriv in 0 1 2; do
  for wl in 7 11 15 21; do
    python -m spectral_challenge.cli cv \
      --config configs/lgbm_shallow_gkf.yaml \
      --override "preprocess=[{name: snv}, {name: sg, window_length: $wl, polyorder: 2, deriv: $deriv}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
      --override "experiment_name=gkf_lgbm_d${deriv}_wl${wl}"
  done
done

# LightGBM without SNV (raw + SG)
for wl in 5 7 11; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: sg, window_length: $wl, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_lgbm_nosnv_wl${wl}"
done

# ===================================================================
# B. EMSC preprocessing (superior to MSC for heterogeneous samples)
# ===================================================================
echo "=== B: EMSC preprocessing ==="

for poly in 1 2 3; do
  # EMSC + PLS
  for nc in 5 10 15 20 25; do
    python -m spectral_challenge.cli cv \
      --config configs/gkf_pls.yaml \
      --override "preprocess=[{name: emsc, poly_order: $poly}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
      --override "model_params.n_components=$nc" \
      --override "experiment_name=gkf_emsc${poly}_sg1_pls_nc${nc}"
  done

  # EMSC + LightGBM
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: emsc, poly_order: $poly}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_emsc${poly}_lgbm"
done

# EMSC + SG2 (2nd derivative)
for nc in 5 10 15; do
  python -m spectral_challenge.cli cv \
    --config configs/gkf_pls.yaml \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 21, polyorder: 3, deriv: 2}, {name: standard_scaler}]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_emsc2_sg2_pls_nc${nc}"
done

# ===================================================================
# C. Feature engineering: band ratios + spectral stats
# ===================================================================
echo "=== C: Feature engineering ==="

# Band ratios + LightGBM
python -m spectral_challenge.cli cv \
  --config configs/lgbm_shallow_gkf.yaml \
  --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: band_ratio}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
  --override "experiment_name=gkf_lgbm_bandratio"

# Spectral stats only (replace spectrum with summary stats)
for nr in 8 16 32; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: spectral_stats, n_regions: $nr, append: false}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_lgbm_stats_r${nr}"
done

# Spectral stats appended to binned spectrum
for nr in 8 16; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: spectral_stats, n_regions: $nr, append: true}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_lgbm_stats_append_r${nr}"
done

# PCA features + LightGBM (replace spectrum with PCA)
for nc in 10 20 30 50; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: pca_features, n_components: $nc, append: false}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_lgbm_pca${nc}"
done

# ===================================================================
# D. New models
# ===================================================================
echo "=== D: New models ==="

# RandomForest
for nest in 500 1000; do
  for depth in 5 10 15; do
    for msl in 10 20; do
      python -m spectral_challenge.cli cv \
        --config configs/lgbm_shallow_gkf.yaml \
        --override "model_type=rf" \
        --override "model_params={n_estimators: $nest, max_depth: $depth, min_samples_leaf: $msl, max_features: 0.5, random_state: 42, n_jobs: -1}" \
        --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
        --override "experiment_name=gkf_rf_n${nest}_d${depth}_msl${msl}"
    done
  done
done

# ExtraTrees
for depth in 10 15 20; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=extra_trees" \
    --override "model_params={n_estimators: 1000, max_depth: $depth, min_samples_leaf: 10, max_features: 0.5, random_state: 42, n_jobs: -1}" \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_et_d${depth}"
done

# SVR with RBF kernel (needs binned features for speed)
for C in 1.0 10.0 100.0; do
  for eps in 0.1 1.0 5.0; do
    python -m spectral_challenge.cli cv \
      --config configs/lgbm_shallow_gkf.yaml \
      --override "model_type=svr" \
      --override "model_params={C: $C, epsilon: $eps, kernel: rbf}" \
      --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 16}, {name: standard_scaler}]" \
      --override "experiment_name=gkf_svr_C${C}_e${eps}"
  done
done

# KernelRidge
for alpha in 0.1 1.0 10.0; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "model_type=kernel_ridge" \
    --override "model_params={alpha: $alpha, kernel: rbf}" \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 16}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_kr_a${alpha}"
done

# Huber regression (robust to outliers like ベイスギ)
for eps_h in 1.35 1.5 2.0; do
  for alpha in 0.001 0.01 0.1; do
    python -m spectral_challenge.cli cv \
      --config configs/lgbm_shallow_gkf.yaml \
      --override "model_type=huber" \
      --override "model_params={epsilon: $eps_h, alpha: $alpha, max_iter: 10000}" \
      --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
      --override "experiment_name=gkf_huber_e${eps_h}_a${alpha}"
  done
done

# ===================================================================
# E. Target transforms (含水率 is right-skewed, max 298)
# ===================================================================
echo "=== E: Target transforms ==="

# Log1p transform + best models
for tt in log1p sqrt; do
  # PLS
  for nc in 5 10 15 20 28; do
    python -m spectral_challenge.cli cv \
      --config configs/snv_sg1_pls.yaml \
      --override "target_transform=$tt" \
      --override "model_params.n_components=$nc" \
      --override "experiment_name=gkf_${tt}_pls_nc${nc}"
  done

  # LightGBM
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "target_transform=$tt" \
    --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_${tt}_lgbm"

  # Ridge
  for alpha in 1.0 10.0 100.0; do
    python -m spectral_challenge.cli cv \
      --config configs/baseline_ridge.yaml \
      --override "target_transform=$tt" \
      --override "model_params.alpha=$alpha" \
      --override "experiment_name=gkf_${tt}_ridge_a${alpha}"
  done
done

# ===================================================================
# F. Combined strategies: EMSC + feature eng + target transform
# ===================================================================
echo "=== F: Combined strategies ==="

# EMSC + band_ratio + LightGBM + log1p
python -m spectral_challenge.cli cv \
  --config configs/lgbm_shallow_gkf.yaml \
  --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: band_ratio}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
  --override "target_transform=log1p" \
  --override "experiment_name=gkf_emsc_br_lgbm_log1p"

# EMSC + spectral_stats + PLS
for nc in 5 10 15; do
  python -m spectral_challenge.cli cv \
    --config configs/gkf_pls.yaml \
    --override "preprocess=[{name: emsc, poly_order: 2}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: spectral_stats, n_regions: 16, append: true}, {name: standard_scaler}]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_emsc_stats_pls_nc${nc}"
done

# MSC + SNV combined (double scatter correction)
python -m spectral_challenge.cli cv \
  --config configs/lgbm_shallow_gkf.yaml \
  --override "preprocess=[{name: msc}, {name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
  --override "experiment_name=gkf_msc_snv_lgbm"

# Absorbance + EMSC + LightGBM
python -m spectral_challenge.cli cv \
  --config configs/lgbm_shallow_gkf.yaml \
  --override "preprocess=[{name: absorbance}, {name: emsc, poly_order: 2}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
  --override "experiment_name=gkf_abs_emsc_lgbm"

# Water bands only + PLS
for nc in 5 10 15; do
  python -m spectral_challenge.cli cv \
    --config configs/wn_select_pls.yaml \
    --override "model_params.n_components=$nc" \
    --override "preprocess=[{name: select_wn, ranges: [[4800,5400],[6600,7200]]}, {name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_waterband_pls_nc${nc}"
done

# Water bands only + Ridge
for alpha in 1.0 10.0 100.0; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_ridge.yaml \
    --override "model_params.alpha=$alpha" \
    --override "preprocess=[{name: select_wn, ranges: [[4800,5400],[6600,7200]]}, {name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_waterband_ridge_a${alpha}"
done

# ===================================================================
# G. Diverse ensemble candidates
# ===================================================================
echo "=== G: Diverse ensemble candidates ==="

# Absorbance + MSC + SG1 + PLS (different from SNV-based)
for nc in 10 15 20 25 30; do
  python -m spectral_challenge.cli cv \
    --config configs/absorbance_msc_pls.yaml \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_abs_msc_pls_nc${nc}"
done

# Edge-trimmed + EMSC + PLS
for nc in 10 15 20; do
  python -m spectral_challenge.cli cv \
    --config configs/gkf_pls.yaml \
    --override "preprocess=[{name: select_wn, ranges: [[4200,9800]]}, {name: emsc, poly_order: 2}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_trim_emsc_pls_nc${nc}"
done

# LightGBM on full spectrum (no binning, very shallow)
for depth in 2 3; do
  for leaves in 4 7; do
    python -m spectral_challenge.cli cv \
      --config configs/lgbm_shallow_gkf.yaml \
      --override "preprocess=[{name: snv}, {name: sg, window_length: 7, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
      --override "model_params.max_depth=$depth" \
      --override "model_params.num_leaves=$leaves" \
      --override "model_params.colsample_bytree=0.3" \
      --override "experiment_name=gkf_lgbm_full_d${depth}_l${leaves}"
    done
done

echo ""
echo "Phase 7b complete!"
echo "Run: python scripts/analyze_results.py --top 50"
