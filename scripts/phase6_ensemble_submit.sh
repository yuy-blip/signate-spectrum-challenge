#!/usr/bin/env bash
# Phase 6: Generate predictions from best runs and create ensemble submission
# Usage:
#   1. First run: python scripts/analyze_results.py --top 20
#   2. Pick the best run dirs and update BEST_RUNS below
#   3. Then run: bash scripts/phase6_ensemble_submit.sh
set -euo pipefail

echo "=========================================="
echo "Phase 6: Ensemble + Submission"
echo "=========================================="

# === EDIT THESE: Put your best run directory names here ===
# Example: after reviewing results, pick top 3-5 diverse runs
BEST_RUNS=(
  # Uncomment and fill in after reviewing results:
  # "runs/best_pls_k30_s0_pls_YYYYMMDD_HHMMSS"
  # "runs/best_pls_k32_s4_pls_YYYYMMDD_HHMMSS"
  # "runs/best_sg1_k28_s3_pls_YYYYMMDD_HHMMSS"
)

if [ ${#BEST_RUNS[@]} -eq 0 ]; then
  echo "ERROR: No runs specified in BEST_RUNS array."
  echo "Edit this script and add your best run directories."
  echo ""
  echo "To find best runs:"
  echo "  python scripts/analyze_results.py --top 20"
  exit 1
fi

# Generate predictions for each run
for run_dir in "${BEST_RUNS[@]}"; do
  echo "--- Predicting: $run_dir ---"
  # Find the config used for this run
  config_file="${run_dir}/config.yaml"
  if [ ! -f "$config_file" ]; then
    echo "WARNING: No config.yaml in $run_dir, skipping"
    continue
  fi
  python -m spectral_challenge.cli predict \
    --config "$config_file" \
    --run-dir "$run_dir"
done

# Create ensemble
echo "--- Creating ensemble ---"
python scripts/ensemble_predictions.py "${BEST_RUNS[@]}"

echo "Phase 6 complete!"
echo "Submission file: submissions/ensemble_submission.csv"
