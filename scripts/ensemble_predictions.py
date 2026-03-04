#!/usr/bin/env python
"""Create ensemble submission by averaging predictions from multiple runs.

Usage:
    python scripts/ensemble_predictions.py runs/run1 runs/run2 runs/run3
    python scripts/ensemble_predictions.py runs/run1 runs/run2 --weights 0.6 0.4
    python scripts/ensemble_predictions.py runs/run1 runs/run2 --output submissions/my_ensemble.csv
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Ensemble predictions from multiple runs")
    parser.add_argument("run_dirs", nargs="+", help="Run directories containing test_preds.npy")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="Weights for each run (default: equal)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    n_runs = len(run_dirs)

    # Validate
    for rd in run_dirs:
        pred_file = rd / "test_preds.npy"
        if not pred_file.exists():
            print(f"ERROR: {pred_file} not found. Run predict first:")
            print(f"  python -m spectral_challenge.cli predict --config {rd}/config.yaml --run-dir {rd}")
            sys.exit(1)

    # Load predictions
    all_preds = []
    for rd in run_dirs:
        preds = np.load(rd / "test_preds.npy")
        all_preds.append(preds)
        print(f"Loaded {rd.name}: shape={preds.shape}, mean={preds.mean():.4f}, std={preds.std():.4f}")

    all_preds = np.array(all_preds)  # (n_runs, n_samples)

    # Weighted average
    if args.weights:
        if len(args.weights) != n_runs:
            print(f"ERROR: {len(args.weights)} weights for {n_runs} runs")
            sys.exit(1)
        weights = np.array(args.weights)
        weights = weights / weights.sum()  # normalize
        ensemble_preds = np.average(all_preds, axis=0, weights=weights)
        print(f"\nWeighted ensemble ({n_runs} runs): {weights}")
    else:
        ensemble_preds = all_preds.mean(axis=0)
        print(f"\nEqual-weight ensemble ({n_runs} runs)")

    print(f"Ensemble: mean={ensemble_preds.mean():.4f}, std={ensemble_preds.std():.4f}")

    # Load test IDs
    data_dir = Path(args.data_dir)
    test_file = data_dir / "test.csv"
    encodings = ["utf-8", "cp932", "shift_jis"]
    for enc in encodings:
        try:
            df_test = pd.read_csv(test_file, encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    # Detect ID column
    id_col = None
    for candidate in ["sample number", "id", "ID", "sample_number"]:
        if candidate in df_test.columns:
            id_col = candidate
            break
    if id_col is None:
        id_col = df_test.columns[0]

    ids = df_test[id_col].values

    # Create submission
    submission = pd.DataFrame({
        id_col: ids,
        "含水率": ensemble_preds,
    })

    if args.output:
        out_path = Path(args.output)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("submissions") / f"ensemble_{n_runs}runs_{timestamp}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved to: {out_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()
