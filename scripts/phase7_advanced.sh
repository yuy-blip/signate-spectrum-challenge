#!/usr/bin/env bash
# Phase 7: Cross-species generalization experiments (ALL GKF)
#
# Strategy: テストは完全未知の6樹種。個体暗記は無意味。
# Focus on:
#   - 散乱・ベースライン補正（SNV/MSC）で物理的個体差を打ち消す
#   - 1次/2次微分の窓幅探索でノイズ除去と情報抽出のバランスを取る
#   - LightGBMは極浅木+強正則化で樹種に依存しないパターンを学習
#   - PLS成分数の最適化
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

echo "=========================================="
echo "Phase 7: Cross-Species Generalization (GKF)"
echo "=========================================="

# ===================================================================
# 7a: PLS × 前処理バリエーション × 成分数
#     PLSは線形モデルで樹種暗記しにくい → GKFの主力候補
# ===================================================================
echo "=== 7a: PLS preprocessing & n_components sweep ==="

# SG1 window length sweep (deriv=1)
for wl in 7 11 15 21 25 31; do
  for nc in 5 10 15 20 25 30; do
    python -m spectral_challenge.cli cv \
      --config configs/snv_sg1_pls.yaml \
      --override "preprocess[1].window_length=$wl" \
      --override "model_params.n_components=$nc" \
      --override "experiment_name=gkf_snv_sg1_wl${wl}_nc${nc}"
  done
done

# SG2 window length sweep (deriv=2)
for wl in 11 15 21 25 31; do
  for nc in 5 10 15 20; do
    python -m spectral_challenge.cli cv \
      --config configs/snv_sg2_pls.yaml \
      --override "preprocess[1].window_length=$wl" \
      --override "model_params.n_components=$nc" \
      --override "experiment_name=gkf_snv_sg2_wl${wl}_nc${nc}"
  done
done

# SG0 (smoothing only, no derivative)
for wl in 7 11 15 21; do
  for nc in 10 20 28 35; do
    python -m spectral_challenge.cli cv \
      --config configs/baseline_pls.yaml \
      --override "preprocess[1].window_length=$wl" \
      --override "model_params.n_components=$nc" \
      --override "experiment_name=gkf_snv_sg0_wl${wl}_nc${nc}"
  done
done

# SG1 without SNV (raw derivative)
for wl in 7 11 15 21; do
  for nc in 10 20 28; do
    python -m spectral_challenge.cli cv \
      --config configs/sg1_only_pls.yaml \
      --override "preprocess[0].window_length=$wl" \
      --override "model_params.n_components=$nc" \
      --override "experiment_name=gkf_sg1only_wl${wl}_nc${nc}"
  done
done

# MSC + SG1 (MSC vs SNV comparison)
for wl in 7 11 15 21; do
  for nc in 10 15 20 25 30; do
    python -m spectral_challenge.cli cv \
      --config configs/absorbance_msc_pls.yaml \
      --override "preprocess=[{name: msc}, {name: sg, window_length: $wl, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
      --override "model_params.n_components=$nc" \
      --override "experiment_name=gkf_msc_sg1_wl${wl}_nc${nc}"
  done
done

# Absorbance + MSC + SG1
for nc in 10 15 20 25 30; do
  python -m spectral_challenge.cli cv \
    --config configs/absorbance_msc_pls.yaml \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_abs_msc_sg1_nc${nc}"
done

# Water absorption band selection + PLS
for nc in 5 10 15 20; do
  python -m spectral_challenge.cli cv \
    --config configs/wn_select_pls.yaml \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_wnsel_nc${nc}"

  # Broader water bands
  python -m spectral_challenge.cli cv \
    --config configs/wn_select_pls.yaml \
    --override "preprocess[0].ranges=[[4500,5500],[6000,7500],[8000,9500]]" \
    --override "model_params.n_components=$nc" \
    --override "experiment_name=gkf_wnsel_broad_nc${nc}"
done

# ===================================================================
# 7b: Ridge/ElasticNet × 正則化強度
#     線形モデル: 樹種に依存しない水の吸収パターンを捉える
# ===================================================================
echo "=== 7b: Ridge / ElasticNet regularization sweep ==="

for alpha in 0.01 0.1 1.0 10.0 100.0 1000.0; do
  python -m spectral_challenge.cli cv \
    --config configs/baseline_ridge.yaml \
    --override "model_params.alpha=$alpha" \
    --override "experiment_name=gkf_ridge_a${alpha}"
done

for alpha in 0.001 0.01 0.1 1.0 10.0; do
  for l1 in 0.1 0.3 0.5 0.7 0.9; do
    python -m spectral_challenge.cli cv \
      --config configs/elastic_net.yaml \
      --override "model_params.alpha=$alpha" \
      --override "model_params.l1_ratio=$l1" \
      --override "experiment_name=gkf_enet_a${alpha}_l1${l1}"
  done
done

# ===================================================================
# 7c: LightGBM 浅木 + 強正則化
#     深い木は個体暗記。浅い木で樹種横断パターンを学習
# ===================================================================
echo "=== 7c: LightGBM shallow + heavy regularization ==="

for depth in 2 3 4 5; do
  for leaves in 4 7 15; do
    for alpha in 1.0 5.0 10.0; do
      for lam in 5.0 10.0 50.0; do
        python -m spectral_challenge.cli cv \
          --config configs/lgbm_shallow_gkf.yaml \
          --override "model_params.max_depth=$depth" \
          --override "model_params.num_leaves=$leaves" \
          --override "model_params.reg_alpha=$alpha" \
          --override "model_params.reg_lambda=$lam" \
          --override "experiment_name=gkf_lgbm_d${depth}_l${leaves}_ra${alpha}_rl${lam}"
      done
    done
  done
done

# LightGBM with different preprocessing
for wl in 7 11 15 21; do
  python -m spectral_challenge.cli cv \
    --config configs/lgbm_shallow_gkf.yaml \
    --override "preprocess=[{name: snv}, {name: sg, window_length: $wl, polyorder: 2, deriv: 1}, {name: binning, bin_size: 8}, {name: standard_scaler}]" \
    --override "experiment_name=gkf_lgbm_sg1_wl${wl}"
done

# LightGBM no binning (full spectral features, very shallow)
python -m spectral_challenge.cli cv \
  --config configs/lgbm_shallow_gkf.yaml \
  --override "preprocess=[{name: snv}, {name: sg, window_length: 11, polyorder: 2, deriv: 1}, {name: standard_scaler}]" \
  --override "model_params.max_depth=2" \
  --override "model_params.num_leaves=4" \
  --override "experiment_name=gkf_lgbm_nobin_d2"

echo ""
echo "Phase 7 complete!"
echo "Run: python scripts/analyze_results.py --top 30"
