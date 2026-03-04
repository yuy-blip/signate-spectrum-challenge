"""CLI entry point.

Usage::

    python -m spectral_challenge.cli cv      --config configs/baseline_ridge.yaml
    python -m spectral_challenge.cli fit      --config configs/baseline_ridge.yaml
    python -m spectral_challenge.cli predict  --config configs/baseline_ridge.yaml --run-dir runs/xxx
    python -m spectral_challenge.cli submit   --config configs/baseline_ridge.yaml --run-dir runs/xxx

Override examples::

    python -m spectral_challenge.cli cv --config cfg.yaml --override model_params.n_components=60
    python -m spectral_challenge.cli cv --config cfg.yaml --set n_folds=3 --set seed=0
    for k in 10 20 30; do
        python -m spectral_challenge.cli cv --config cfg.yaml --override model_params.n_components=$k
    done
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spectral_challenge",
        description="SIGNATE Spectrum Challenge pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- cv ---
    p_cv = sub.add_parser("cv", help="Run cross-validation training")
    _add_common(p_cv)

    # --- fit ---
    p_fit = sub.add_parser("fit", help="Fit on full training data (no CV)")
    _add_common(p_fit)

    # --- predict ---
    p_pred = sub.add_parser("predict", help="Predict on test data using a trained run")
    _add_common(p_pred)
    p_pred.add_argument("--run-dir", type=str, required=True, help="Path to a completed run")

    # --- submit ---
    p_sub = sub.add_parser("submit", help="Generate submission CSV from a trained run")
    _add_common(p_sub)
    p_sub.add_argument("--run-dir", type=str, required=True, help="Path to a completed run")

    return parser.parse_args(argv)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--folds", type=int, default=None, help="Override number of CV folds")
    parser.add_argument("--outdir", type=str, default=None, help="Override output directory")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument(
        "--override", "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Hydra-style config override (repeatable). "
            "Examples: model_params.alpha=0.5  n_folds=3  shuffle=false"
        ),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from spectral_challenge.config import Config
    from spectral_challenge.logging_utils import setup_logger
    from spectral_challenge.paths import DATA_RAW, RUNS_DIR

    # Build config with CLI overrides
    overrides: dict = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.folds is not None:
        overrides["n_folds"] = args.folds

    cfg = Config.from_yaml(args.config, overrides, cli_overrides=args.overrides or None)

    data_dir = Path(args.data_dir) if args.data_dir else DATA_RAW

    # Resolve run directory
    if args.command in ("predict", "submit"):
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.experiment_name}_{cfg.model_type}_{timestamp}"
        run_dir = Path(args.outdir) if args.outdir else RUNS_DIR / run_name

    log = setup_logger(log_file=run_dir / "run.log" if args.command in ("cv", "fit") else None)

    # Set global seed
    np.random.seed(cfg.seed)

    if args.command == "cv":
        _cmd_cv(cfg, run_dir, data_dir, log)
    elif args.command == "fit":
        _cmd_fit(cfg, run_dir, data_dir, log)
    elif args.command == "predict":
        _cmd_predict(cfg, run_dir, data_dir, log)
    elif args.command == "submit":
        _cmd_submit(cfg, run_dir, data_dir, log)


def _cmd_cv(cfg, run_dir, data_dir, log):
    from spectral_challenge.data.load import load_train
    from spectral_challenge.train import run_cv

    log.info("Loading training data from %s", data_dir)
    X, y, ids = load_train(cfg, data_dir)
    log.info("X shape: %s, y shape: %s", X.shape, y.shape)

    groups = None
    if cfg.split_method == "group_kfold" and cfg.group_col:
        import pandas as pd

        df = pd.read_csv(data_dir / cfg.train_file)
        groups = df[cfg.group_col].values

    result = run_cv(cfg, X, y, run_dir, groups=groups)
    log.info("CV finished. Mean RMSE: %.6f", result["mean_rmse"])
    log.info("Run saved to %s", run_dir)


def _cmd_fit(cfg, run_dir, data_dir, log):
    """Fit on full training data (all folds = 1 model)."""
    import joblib

    from spectral_challenge.data.load import load_train
    from spectral_challenge.models.factory import create_model
    from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

    log.info("Loading training data from %s", data_dir)
    X, y, _ids = load_train(cfg, data_dir)
    log.info("X shape: %s, y shape: %s", X.shape, y.shape)

    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    pipe = build_preprocess_pipeline(cfg.preprocess)
    X_t = pipe.fit_transform(X)

    model = create_model(cfg.model_type, cfg.model_params)
    model.fit(X_t, y)

    joblib.dump(model, models_dir / "model_full.joblib")
    joblib.dump(pipe, models_dir / "pipe_full.joblib")
    log.info("Full-data model saved to %s", models_dir)


def _cmd_predict(cfg, run_dir, data_dir, log):
    from spectral_challenge.data.load import load_test
    from spectral_challenge.predict import predict_test

    log.info("Loading test data from %s", data_dir)
    X_test, ids = load_test(cfg, data_dir)
    log.info("X_test shape: %s", X_test.shape)

    preds = predict_test(X_test, run_dir)
    np.save(run_dir / "test_preds.npy", preds)
    log.info("Test predictions saved to %s/test_preds.npy", run_dir)


def _cmd_submit(cfg, run_dir, data_dir, log):
    from spectral_challenge.data.load import load_test
    from spectral_challenge.paths import SUBMISSIONS_DIR
    from spectral_challenge.predict import predict_test
    from spectral_challenge.submit import make_submission

    log.info("Loading test data from %s", data_dir)
    X_test, ids = load_test(cfg, data_dir)

    preds = predict_test(X_test, run_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUBMISSIONS_DIR / f"submission_{cfg.model_type}_{timestamp}.csv"
    make_submission(ids, preds, cfg.id_col, cfg.target_col, out_path)
    log.info("Submission written to %s", out_path)


if __name__ == "__main__":
    main()
