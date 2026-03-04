#!/usr/bin/env bash
# ULTIMATE RUN: Cross-species generalization strategy
#
# Phase 7: PLS/Ridge/LightGBM × preprocessing × hyperparams (ALL GKF)
# Phase 8: Multi-seed for best GKF configs
# Phase 9: Final ensemble & submission
#
# Usage:
#   bash scripts/run_ultimate.sh          # phases 7-8
#   bash scripts/run_ultimate.sh 8        # start from phase 8
set -euo pipefail

START="${1:-7}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "============================================================"
echo "  CROSS-SPECIES GENERALIZATION SUITE (ALL GKF)"
echo "  Train: 13 species → Test: 6 UNSEEN species"
echo "  Starting from Phase $START"
echo "  $(date)"
echo "============================================================"

if [ "$START" -le 7 ]; then
  echo "[$(date +%H:%M:%S)] Phase 7: Preprocessing & model sweep (GKF)..."
  bash scripts/phase7_advanced.sh 2>&1 | tee logs/phase7.log
fi

if [ "$START" -le 8 ]; then
  echo "[$(date +%H:%M:%S)] Phase 8: Multi-seed best configs..."
  bash scripts/phase8_multiseed_lgbm.sh 2>&1 | tee logs/phase8.log
fi

echo ""
echo "============================================================"
echo "  Phases 7-8 DONE! $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. python scripts/analyze_results.py --top 30"
echo "  2. Edit scripts/phase9_final_ensemble.sh with best runs"
echo "  3. bash scripts/phase9_final_ensemble.sh"
