#!/usr/bin/env bash
# Phase 9 (was Phase 10): Final Ensemble & Submission
#
# Strategy: Combine diverse GKF-validated models
# - Stacking with Ridge meta-learner (safe for cross-species)
# - Optimized blend weights on GKF OOF
# - Simple average as baseline
#
# IMPORTANT: Update run directories below after Phases 7-8!
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Phase 9: Final Ensemble & Submission"
echo "=========================================="

# === STEP 1: Review results ===
echo "--- Step 1: Top GKF results ---"
python scripts/analyze_results.py --top 30

echo ""
echo "=== EDIT the variables below with best GKF run directories ==="
echo ""

# === EDIT THESE ===
# Pick 3-7 diverse models with GOOD GKF scores
ALL_RUNS=(
  # Example: "runs/gkf_snv_sg1_wl11_nc20_pls_YYYYMMDD_HHMMSS"
  # Example: "runs/gkf_ridge_a10_ridge_YYYYMMDD_HHMMSS"
  # Example: "runs/gkf_lgbm_d3_l7_lgbm_YYYYMMDD_HHMMSS"
)

if [ ${#ALL_RUNS[@]} -eq 0 ]; then
  echo "ERROR: No runs specified. Edit this script first!"
  exit 1
fi

# === Generate test predictions ===
echo "--- Generating test predictions ---"
for run_dir in "${ALL_RUNS[@]}"; do
  if [ ! -f "${run_dir}/test_preds.npy" ]; then
    config_file="${run_dir}/config.yaml"
    python -m spectral_challenge.cli predict \
      --config "$config_file" --run-dir "$run_dir"
  fi
done

# === Stacking (Ridge meta) ===
echo "--- Stacking (Ridge) ---"
python scripts/stacking.py --runs "${ALL_RUNS[@]}" --meta ridge --n-folds 5

# === Optimized blend ===
echo "--- Optimized blend ---"
python scripts/optimize_blend.py "${ALL_RUNS[@]}"

# === Simple average ===
echo "--- Simple average ---"
python scripts/ensemble_predictions.py "${ALL_RUNS[@]}"

echo ""
echo "Done! Check submissions/ and compare GKF OOF RMSE."
