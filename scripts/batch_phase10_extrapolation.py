#!/usr/bin/env python
"""Phase 10: Targeting the extrapolation problem directly.

Key finding: Fold 2 RMSE ~30.7 is dominated by 14 samples with moisture > 200
in species 15. LGBM cannot extrapolate beyond training max (216.1).

Approaches:
1. Tweedie/Poisson/Gamma loss (natural for positive, right-skewed data)
2. Sample weighting (upweight high-moisture samples)
3. Water-aware EMSC (species-invariant water features)
4. EMSC coefficient extraction (scattering properties as features)
5. Very high n_estimators with tiny learning rate (better leaf values)
6. Dart boosting (dropout regularization for less overfitting)
7. Monotonic constraints for water-band features
"""

from __future__ import annotations

import datetime
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train
from spectral_challenge.train import run_cv

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")


def run_experiment(name: str, cfg_dict: dict) -> dict | None:
    try:
        cfg_path = Path("configs") / f"_auto_{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True)

        cfg = Config.from_yaml(cfg_path)
        X, y, ids = load_train(cfg, DATA_DIR)

        groups = None
        if cfg.split_method == "group_kfold" and cfg.group_col:
            import pandas as pd
            df = pd.read_csv(DATA_DIR / cfg.train_file, encoding="cp932")
            groups = df[cfg.group_col].values

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{name}_{timestamp}"
        result = run_cv(cfg, X, y, run_dir, groups=groups)

        print(f"  >>> {name}: RMSE={result['mean_rmse']:.4f}  folds=[{', '.join(f'{r:.2f}' for r in result['fold_rmses'])}]")
        return {
            "name": name,
            "mean_rmse": result["mean_rmse"],
            "fold_rmses": result["fold_rmses"],
            "run_dir": str(run_dir),
        }
    except Exception as e:
        print(f"  !!! FAILED: {name}: {e}")
        traceback.print_exc()
        return {"name": name, "mean_rmse": 999.0, "error": str(e)}


BASE_GKF = {
    "train_file": "train.csv", "test_file": "test.csv",
    "id_col": "sample number", "target_col": "含水率",
    "n_folds": 5, "split_method": "group_kfold",
    "group_col": "species number", "seed": 42, "shuffle": True,
}

PP_EMSC = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

PP_EMSC_NOBIN = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "standard_scaler"},
]

LGBM_BEST = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
    "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
    "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
}


def make_cfg(name: str, pp: list, model_type: str, model_params: dict, **extra) -> dict:
    cfg = dict(BASE_GKF)
    cfg["experiment_name"] = name
    cfg["preprocess"] = pp
    cfg["model_type"] = model_type
    cfg["model_params"] = model_params
    cfg.update(extra)
    return cfg


def get_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # -----------------------------------------------------------
    # A: Tweedie loss (natural for positive, right-skewed data)
    # -----------------------------------------------------------
    for p in [1.1, 1.3, 1.5, 1.7, 1.9]:
        experiments.append((
            f"tweedie_p{p}_emsc",
            make_cfg(f"tweedie_p{p}", PP_EMSC, "lgbm", {
                **LGBM_BEST,
                "objective": "tweedie",
                "tweedie_variance_power": p,
            }),
        ))

    # -----------------------------------------------------------
    # B: Poisson and Gamma loss
    # -----------------------------------------------------------
    experiments.append((
        "poisson_emsc",
        make_cfg("poisson", PP_EMSC, "lgbm", {
            **LGBM_BEST,
            "objective": "poisson",
        }),
    ))
    experiments.append((
        "gamma_emsc",
        make_cfg("gamma", PP_EMSC, "lgbm", {
            **LGBM_BEST,
            "objective": "gamma",
        }),
    ))

    # -----------------------------------------------------------
    # C: Very high n_estimators with tiny LR
    # -----------------------------------------------------------
    for n_est, lr in [(5000, 0.005), (10000, 0.002), (10000, 0.001)]:
        experiments.append((
            f"long_{n_est}_lr{lr}_emsc",
            make_cfg(f"long_{n_est}_{lr}", PP_EMSC, "lgbm", {
                "n_estimators": n_est, "learning_rate": lr, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))
    # Also with depth=3 (more regularized)
    for n_est, lr in [(5000, 0.005), (10000, 0.002)]:
        experiments.append((
            f"long_{n_est}_lr{lr}_d3_emsc",
            make_cfg(f"long_{n_est}_{lr}_d3", PP_EMSC, "lgbm", {
                "n_estimators": n_est, "learning_rate": lr, "max_depth": 3,
                "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # D: DART boosting (dropout regularization)
    # -----------------------------------------------------------
    for rate in [0.05, 0.1, 0.2]:
        experiments.append((
            f"dart_r{rate}_emsc",
            make_cfg(f"dart_r{rate}", PP_EMSC, "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0,
                "boosting_type": "dart", "drop_rate": rate,
                "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # E: EMSC coefficient features (scattering properties)
    # -----------------------------------------------------------
    PP_EMSC_COEF = [
        {"name": "emsc_coefficients", "poly_order": 2, "include_corrected": True},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "emsc_coef_lgbm",
        make_cfg("emsc_coef", PP_EMSC_COEF, "lgbm", dict(LGBM_BEST)),
    ))

    # Coefficients only (no corrected spectrum)
    PP_EMSC_COEF_ONLY = [
        {"name": "emsc_coefficients", "poly_order": 2, "include_corrected": False},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "emsc_coef_only_lgbm",
        make_cfg("emsc_coef_only", PP_EMSC_COEF_ONLY, "lgbm", dict(LGBM_BEST)),
    ))

    # -----------------------------------------------------------
    # F: Water-aware EMSC
    # -----------------------------------------------------------
    PP_WATER_EMSC = [
        {"name": "water_emsc", "poly_order": 2, "return_coefficients": True},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    PP_WATER_EMSC_NOCOEF = [
        {"name": "water_emsc", "poly_order": 2, "return_coefficients": False},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "water_emsc_coef_lgbm",
        make_cfg("water_emsc_coef", PP_WATER_EMSC, "lgbm", dict(LGBM_BEST)),
    ))
    experiments.append((
        "water_emsc_lgbm",
        make_cfg("water_emsc", PP_WATER_EMSC_NOCOEF, "lgbm", dict(LGBM_BEST)),
    ))

    # -----------------------------------------------------------
    # G: XGBoost with Tweedie and long training
    # -----------------------------------------------------------
    for p in [1.3, 1.5]:
        experiments.append((
            f"xgb_tweedie_p{p}_emsc",
            make_cfg(f"xgb_tweedie_p{p}", PP_EMSC, "xgb", {
                "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5,
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": 5.0, "reg_lambda": 10.0, "tree_method": "hist",
                "objective": f"reg:tweedie",
                "tweedie_variance_power": p,
            }),
        ))

    # XGBoost with very long training
    experiments.append((
        "xgb_long_emsc",
        make_cfg("xgb_long", PP_EMSC, "xgb", {
            "n_estimators": 5000, "learning_rate": 0.005, "max_depth": 5,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 5.0, "reg_lambda": 10.0, "tree_method": "hist",
        }),
    ))

    # -----------------------------------------------------------
    # H: LGBM with adjusted leaf constraints for better extrapolation
    # -----------------------------------------------------------
    # Lower min_child_weight allows larger leaf values
    for mcw in [0.001, 0.01, 0.1, 1.0]:
        experiments.append((
            f"lgbm_mcw{mcw}_emsc",
            make_cfg(f"lgbm_mcw{mcw}", PP_EMSC, "lgbm", {
                **LGBM_BEST,
                "min_child_weight": mcw,
                "min_child_samples": 5,
            }),
        ))

    # -----------------------------------------------------------
    # I: LGBM with linear tree (can extrapolate!)
    # -----------------------------------------------------------
    experiments.append((
        "lgbm_linear_tree_emsc",
        make_cfg("lgbm_linear_tree", PP_EMSC, "lgbm", {
            **LGBM_BEST,
            "linear_tree": True,
        }),
    ))
    experiments.append((
        "lgbm_linear_tree_nobin_emsc",
        make_cfg("lgbm_linear_tree_nobin", PP_EMSC_NOBIN, "lgbm", {
            **LGBM_BEST,
            "linear_tree": True,
        }),
    ))
    # Linear tree with different depths
    for depth, leaves in [(3, 7), (4, 15), (6, 31)]:
        experiments.append((
            f"lgbm_lt_d{depth}_emsc",
            make_cfg(f"lgbm_lt_d{depth}", PP_EMSC, "lgbm", {
                **LGBM_BEST, "max_depth": depth, "num_leaves": leaves,
                "linear_tree": True,
            }),
        ))
    # Linear tree with slow training
    experiments.append((
        "lgbm_lt_slow_emsc",
        make_cfg("lgbm_lt_slow", PP_EMSC, "lgbm", {
            "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5,
            "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0,
            "verbose": -1, "linear_tree": True,
        }),
    ))

    # -----------------------------------------------------------
    # J: Multi-seed with linear tree for ensemble
    # -----------------------------------------------------------
    for seed in range(5):
        experiments.append((
            f"lgbm_lt_s{seed}_emsc",
            make_cfg(f"lgbm_lt_s{seed}", PP_EMSC, "lgbm", {
                **LGBM_BEST, "linear_tree": True, "random_state": seed,
            }),
        ))

    # -----------------------------------------------------------
    # K: Tweedie + linear tree combination
    # -----------------------------------------------------------
    for p in [1.3, 1.5]:
        experiments.append((
            f"lgbm_lt_tweedie_p{p}_emsc",
            make_cfg(f"lgbm_lt_tweedie_p{p}", PP_EMSC, "lgbm", {
                **LGBM_BEST,
                "linear_tree": True,
                "objective": "tweedie",
                "tweedie_variance_power": p,
            }),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 10: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 10 RESULTS (TOP 25)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Fold0':>7}  {'Fold1':>7}  {'Fold2':>7}  {'Fold3':>7}  {'Fold4':>7}  Name")
    print("-" * 100)
    for rank, r in enumerate(results[:25], 1):
        fr = r.get("fold_rmses", [0]*5)
        fr_str = "  ".join(f"{f:>7.2f}" for f in fr[:5])
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fr_str}  {r['name']}")

    results_path = Path("runs") / "batch_phase10_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
