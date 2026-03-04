#!/usr/bin/env python
"""Analyze all CV run results and print a ranked summary table.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --top 20
    python scripts/analyze_results.py --filter gkf
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders")
    parser.add_argument("--top", type=int, default=50, help="Show top N results")
    parser.add_argument("--filter", type=str, default="", help="Filter by experiment name substring")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results = []

    for run_path in sorted(runs_dir.iterdir()):
        metrics_file = run_path / "metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Extract experiment name from directory name
        run_name = run_path.name

        if args.filter and args.filter not in run_name:
            continue

        fold_rmses = metrics.get("fold_rmses", [])
        mean_rmse = metrics.get("mean_rmse", float("inf"))

        # Compute stats
        if fold_rmses:
            import statistics
            fold_std = statistics.stdev(fold_rmses) if len(fold_rmses) > 1 else 0
            fold_max = max(fold_rmses)
            fold_min = min(fold_rmses)
        else:
            fold_std = fold_max = fold_min = 0

        results.append({
            "run_name": run_name,
            "oof_rmse": mean_rmse,
            "fold_std": fold_std,
            "fold_min": fold_min,
            "fold_max": fold_max,
            "n_folds": len(fold_rmses),
            "fold_rmses": fold_rmses,
        })

    # Sort by oof_rmse
    results.sort(key=lambda x: x["oof_rmse"])

    if args.csv:
        print("rank,run_name,oof_rmse,fold_std,fold_min,fold_max")
        for i, r in enumerate(results[: args.top], 1):
            print(f"{i},{r['run_name']},{r['oof_rmse']:.6f},{r['fold_std']:.4f},{r['fold_min']:.4f},{r['fold_max']:.4f}")
    else:
        print(f"\n{'='*100}")
        print(f"{'Rank':>4}  {'OOF RMSE':>10}  {'Fold Std':>9}  {'Min':>8}  {'Max':>8}  Run Name")
        print(f"{'='*100}")
        for i, r in enumerate(results[: args.top], 1):
            # Highlight if fold_max is much worse than mean
            flag = " ⚠" if r["fold_max"] > r["oof_rmse"] * 2 else ""
            print(
                f"{i:>4}  {r['oof_rmse']:>10.4f}  {r['fold_std']:>9.4f}  "
                f"{r['fold_min']:>8.4f}  {r['fold_max']:>8.4f}  {r['run_name']}{flag}"
            )
        print(f"{'='*100}")
        print(f"Total runs: {len(results)}")


if __name__ == "__main__":
    main()
