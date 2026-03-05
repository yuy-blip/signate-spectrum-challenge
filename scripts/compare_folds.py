"""Compare fold-level RMSE between runs to identify extrapolation advantages.

Usage:
    python scripts/compare_folds.py --filter gkf --top 20
    python scripts/compare_folds.py --names "emsc2_lgbm" "snv_sg1_pls" "ridge"
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="gkf", help="Filter run names")
    parser.add_argument("--top", type=int, default=30, help="Top N runs")
    parser.add_argument("--fold", type=int, default=None, help="Sort by specific fold")
    parser.add_argument("--names", nargs="*", help="Filter by substrings")
    args = parser.parse_args()

    runs_dir = Path("runs")
    results = []

    for d in sorted(runs_dir.iterdir()):
        m = d / "metrics.json"
        if not m.exists():
            continue
        name = d.name
        if args.filter and args.filter not in name:
            continue
        if args.names and not any(n in name for n in args.names):
            continue
        data = json.loads(m.read_text())
        folds = data.get("fold_rmses", [])
        if not folds:
            continue
        results.append((data["mean_rmse"], name, folds))

    # Sort by specific fold or overall
    if args.fold is not None:
        results.sort(key=lambda x: x[2][args.fold] if len(x[2]) > args.fold else 999)
    else:
        results.sort()

    sort_label = f"Fold {args.fold}" if args.fold is not None else "Overall"
    print(f"\nSorted by: {sort_label}")
    print(f"{'Rank':>4}  {'Overall':>8}  {'Fold0':>7} {'Fold1':>7} {'Fold2':>7} {'Fold3':>7} {'Fold4':>7}  Run Name")
    print("-" * 120)
    for i, (rmse, name, folds) in enumerate(results[:args.top], 1):
        fold_str = " ".join(f"{f:7.2f}" for f in folds)
        # Mark Fold 2 if it's notably better or worse
        marker = ""
        if len(folds) > 2 and folds[2] < 30:
            marker = " <-- F2 good!"
        print(f"{i:4d}  {rmse:8.4f}  {fold_str}  {name}{marker}")

    # Summary: best Fold 2 performance
    if results:
        print(f"\n--- Best Fold 2 performers ---")
        fold2_sorted = sorted(results, key=lambda x: x[2][2] if len(x[2]) > 2 else 999)
        for i, (rmse, name, folds) in enumerate(fold2_sorted[:10], 1):
            print(f"  {i}. Fold2={folds[2]:7.2f}  Overall={rmse:8.4f}  {name}")


if __name__ == "__main__":
    main()
