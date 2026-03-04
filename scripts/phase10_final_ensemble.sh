#!/usr/bin/env bash
# Phase 10: Final Ensemble & Submission
# This is the ultimate step - combine everything for maximum accuracy
#
# Strategy:
#   1. Generate test predictions from ALL good runs
#   2. Stacking ensemble (Ridge meta-learner on diverse base models)
#   3. Optimized blend weights
#   4. Compare stacking vs. optimized blend vs. simple average
#   5. Submit the best one
#
# IMPORTANT: Update the run directories below after Phases 7-9 complete!
set -euo pipefail

echo "=========================================="
echo "Phase 10: Final Ensemble & Submission"
echo "=========================================="

# === STEP 1: Identify best runs ===
echo "--- Step 1: Analyze all results ---"
python scripts/analyze_results.py --top 50

echo ""
echo "=== EDIT the variables below with your best run directories ==="
echo ""

# === EDIT THESE AFTER REVIEWING RESULTS ===
# Pick 3-7 diverse models (different model types, different preprocessing)

# Best LightGBM runs (diverse seeds/configs)
LGBM_RUNS=(
  # Example: "runs/lgbm_best_s0_lgbm_YYYYMMDD_HHMMSS"
  # Example: "runs/lgbm_nobin_s0_lgbm_YYYYMMDD_HHMMSS"
)

# Best PLS runs
PLS_RUNS=(
  # Example: "runs/best_pls_k30_s0_pls_YYYYMMDD_HHMMSS"
)

# Best other runs (Ridge, ElasticNet, pseudo-labeled)
OTHER_RUNS=(
  # Example: "runs/ridge_sg0_a0.1_ridge_YYYYMMDD_HHMMSS"
)

ALL_RUNS=("${LGBM_RUNS[@]}" "${PLS_RUNS[@]}" "${OTHER_RUNS[@]}")

if [ ${#ALL_RUNS[@]} -eq 0 ]; then
  echo "ERROR: No runs specified. Edit this script first!"
  echo ""
  echo "Recommended approach:"
  echo "  1. python scripts/analyze_results.py --top 20"
  echo "  2. Pick ~5-7 diverse runs (different models + preprocessing)"
  echo "  3. Update ALL_RUNS in this script"
  echo "  4. Re-run this script"
  exit 1
fi

# === STEP 2: Generate test predictions for all selected runs ===
echo "--- Step 2: Generate test predictions ---"
for run_dir in "${ALL_RUNS[@]}"; do
  if [ ! -f "${run_dir}/test_preds.npy" ]; then
    echo "Predicting: $run_dir"
    config_file="${run_dir}/config.yaml"
    python -m spectral_challenge.cli predict \
      --config "$config_file" \
      --run-dir "$run_dir"
  else
    echo "Already predicted: $run_dir"
  fi
done

# === STEP 3: Stacking (Ridge meta-learner) ===
echo ""
echo "--- Step 3: Stacking Ensemble ---"
python scripts/stacking.py \
  --runs "${ALL_RUNS[@]}" \
  --meta ridge \
  --n-folds 5

# === STEP 4: Stacking (LightGBM meta-learner) ===
echo ""
echo "--- Step 4: LightGBM Stacking ---"
python scripts/stacking.py \
  --runs "${ALL_RUNS[@]}" \
  --meta lgbm \
  --n-folds 5

# === STEP 5: Optimized blend ===
echo ""
echo "--- Step 5: Optimized Blend ---"
python scripts/optimize_blend.py "${ALL_RUNS[@]}"

# === STEP 6: Simple average ===
echo ""
echo "--- Step 6: Simple Average ---"
python scripts/ensemble_predictions.py "${ALL_RUNS[@]}"

echo ""
echo "=========================================="
echo "  Phase 10 Complete!"
echo "=========================================="
echo ""
echo "Check submissions/ for all submission files."
echo "Compare OOF RMSE scores to decide which to submit:"
echo "  python scripts/analyze_results.py --filter stacking"
echo "  python scripts/analyze_results.py --filter optblend"
echo ""
echo "Submit the one with the lowest OOF RMSE!"
