#!/usr/bin/env python
"""Batch experiment runner — systematically explore the entire search space.

Creates configs on-the-fly, runs CV, and collects all results.
Designed for aggressive exploration toward RMSE ~10.
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
    """Run a single experiment and return metrics."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    try:
        # Save config
        cfg_path = Path("configs") / f"_auto_{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True)

        cfg = Config.from_yaml(cfg_path)
        X, y, ids = load_train(cfg, DATA_DIR)

        # Load groups for GKF
        groups = None
        if cfg.split_method == "group_kfold" and cfg.group_col:
            import pandas as pd
            df = pd.read_csv(DATA_DIR / cfg.train_file, encoding="cp932")
            groups = df[cfg.group_col].values

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{name}_{timestamp}"

        result = run_cv(cfg, X, y, run_dir, groups=groups)

        print(f"  >>> {name}: OOF RMSE = {result['mean_rmse']:.4f}")
        print(f"      Fold RMSEs: {[f'{r:.2f}' for r in result['fold_rmses']]}")
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


# ============================================================
# BASE CONFIG TEMPLATE
# ============================================================

BASE_GKF = {
    "train_file": "train.csv",
    "test_file": "test.csv",
    "id_col": "sample number",
    "target_col": "含水率",
    "n_folds": 5,
    "split_method": "group_kfold",
    "group_col": "species number",
    "seed": 42,
    "shuffle": True,
}


def make_cfg(name_hint: str, preprocess: list, model_type: str,
             model_params: dict, target_transform: str = "none",
             target_transform_lambda: float = 0.5, **extra) -> dict:
    """Build a config dict."""
    cfg = dict(BASE_GKF)
    cfg["experiment_name"] = name_hint
    cfg["preprocess"] = preprocess
    cfg["model_type"] = model_type
    cfg["model_params"] = model_params
    cfg["target_transform"] = target_transform
    cfg["target_transform_lambda"] = target_transform_lambda
    cfg.update(extra)
    return cfg


# ============================================================
# PREPROCESSING RECIPES
# ============================================================

PP_SNV_SG1 = [
    {"name": "snv"},
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
    {"name": "standard_scaler"},
]

PP_SNV_SG1_BIN = [
    {"name": "snv"},
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

PP_SNV_SG2 = [
    {"name": "snv"},
    {"name": "sg", "window_length": 15, "polyorder": 3, "deriv": 2},
    {"name": "standard_scaler"},
]

PP_MSC_SG1 = [
    {"name": "msc"},
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
    {"name": "standard_scaler"},
]

PP_ABS_SNV_SG1 = [
    {"name": "absorbance"},
    {"name": "snv"},
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
    {"name": "standard_scaler"},
]

PP_SNV_SG1_FEAT = [
    {"name": "snv"},
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
    {"name": "band_ratio"},
    {"name": "spectral_stats", "n_regions": 10},
    {"name": "standard_scaler"},
]

PP_SNV_SG1_PCA = [
    {"name": "snv"},
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
    {"name": "pca_features", "n_components": 30},
    {"name": "standard_scaler"},
]

PP_SNV_SMOOTH_ONLY = [
    {"name": "snv"},
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 0},
    {"name": "standard_scaler"},
]

PP_RAW_SCALER = [
    {"name": "standard_scaler"},
]

PP_SNV_SG1_BIN4 = [
    {"name": "snv"},
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 4},
    {"name": "standard_scaler"},
]


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

def get_all_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # -----------------------------------------------------------
    # SECTION A: Target Transform — THE BIGGEST WIN EXPECTED
    # -----------------------------------------------------------

    # A1: log1p + PLS (various n_components)
    for nc in [10, 20, 28, 40, 50]:
        experiments.append((
            f"log1p_pls_nc{nc}",
            make_cfg(f"log1p_pls_nc{nc}", PP_SNV_SG1, "pls",
                     {"n_components": nc}, target_transform="log1p"),
        ))

    # A2: log1p + LightGBM
    experiments.append((
        "log1p_lgbm_shallow",
        make_cfg("log1p_lgbm_shallow", PP_SNV_SG1_BIN, "lgbm", {
            "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
            "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0,
            "verbose": -1,
        }, target_transform="log1p"),
    ))

    experiments.append((
        "log1p_lgbm_deep",
        make_cfg("log1p_lgbm_deep", PP_SNV_SG1_BIN, "lgbm", {
            "n_estimators": 3000, "learning_rate": 0.005, "max_depth": 5,
            "num_leaves": 15, "min_child_samples": 10, "subsample": 0.8,
            "colsample_bytree": 0.6, "reg_alpha": 2.0, "reg_lambda": 5.0,
            "verbose": -1,
        }, target_transform="log1p"),
    ))

    # A3: log1p + Ridge
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        experiments.append((
            f"log1p_ridge_a{alpha}",
            make_cfg(f"log1p_ridge_a{alpha}", PP_SNV_SG1, "ridge",
                     {"alpha": alpha}, target_transform="log1p"),
        ))

    # A4: sqrt + PLS
    for nc in [20, 28, 40]:
        experiments.append((
            f"sqrt_pls_nc{nc}",
            make_cfg(f"sqrt_pls_nc{nc}", PP_SNV_SG1, "pls",
                     {"n_components": nc}, target_transform="sqrt"),
        ))

    # A5: log1p + ElasticNet
    for alpha in [0.01, 0.1, 1.0]:
        for l1 in [0.1, 0.5, 0.9]:
            experiments.append((
                f"log1p_enet_a{alpha}_l1{l1}",
                make_cfg(f"log1p_enet_a{alpha}_l1{l1}", PP_SNV_SG1, "elastic_net",
                         {"alpha": alpha, "l1_ratio": l1, "max_iter": 5000},
                         target_transform="log1p"),
            ))

    # A6: boxcox + PLS
    for lam in [0.0, 0.25, 0.5]:
        experiments.append((
            f"boxcox_pls_lam{lam}_nc28",
            make_cfg(f"boxcox_pls_lam{lam}_nc28", PP_SNV_SG1, "pls",
                     {"n_components": 28}, target_transform="boxcox",
                     target_transform_lambda=lam),
        ))

    # -----------------------------------------------------------
    # SECTION B: No Transform — Various model/preprocess combos
    # -----------------------------------------------------------

    # B1: PLS variants
    for nc in [15, 20, 28, 40]:
        experiments.append((
            f"notx_pls_nc{nc}_snvsg1",
            make_cfg(f"notx_pls_nc{nc}", PP_SNV_SG1, "pls", {"n_components": nc}),
        ))

    # B2: Different preprocessing + PLS28
    for pp_name, pp in [("msc_sg1", PP_MSC_SG1), ("abs_snv_sg1", PP_ABS_SNV_SG1),
                         ("snv_sg2", PP_SNV_SG2), ("snv_smooth", PP_SNV_SMOOTH_ONLY),
                         ("raw", PP_RAW_SCALER)]:
        experiments.append((
            f"notx_pls28_{pp_name}",
            make_cfg(f"pls28_{pp_name}", pp, "pls", {"n_components": 28}),
        ))

    # B3: LightGBM without transform
    experiments.append((
        "notx_lgbm_shallow",
        make_cfg("notx_lgbm_shallow", PP_SNV_SG1_BIN, "lgbm", {
            "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
            "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0,
            "verbose": -1,
        }),
    ))

    # -----------------------------------------------------------
    # SECTION C: XGBoost
    # -----------------------------------------------------------

    experiments.append((
        "log1p_xgb_shallow",
        make_cfg("log1p_xgb_shallow", PP_SNV_SG1_BIN, "xgb", {
            "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 5.0, "reg_lambda": 10.0,
            "tree_method": "hist",
        }, target_transform="log1p"),
    ))

    experiments.append((
        "log1p_xgb_deep",
        make_cfg("log1p_xgb_deep", PP_SNV_SG1_BIN, "xgb", {
            "n_estimators": 3000, "learning_rate": 0.005, "max_depth": 5,
            "subsample": 0.8, "colsample_bytree": 0.6,
            "reg_alpha": 2.0, "reg_lambda": 5.0,
            "tree_method": "hist",
        }, target_transform="log1p"),
    ))

    experiments.append((
        "notx_xgb_shallow",
        make_cfg("notx_xgb_shallow", PP_SNV_SG1_BIN, "xgb", {
            "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 5.0, "reg_lambda": 10.0,
            "tree_method": "hist",
        }),
    ))

    # -----------------------------------------------------------
    # SECTION D: Feature Engineering + Best models
    # -----------------------------------------------------------

    # D1: Band ratios + PLS
    experiments.append((
        "log1p_pls28_feat",
        make_cfg("log1p_pls28_feat", PP_SNV_SG1_FEAT, "pls",
                 {"n_components": 28}, target_transform="log1p"),
    ))

    # D2: PCA features + PLS
    experiments.append((
        "log1p_pls28_pca",
        make_cfg("log1p_pls28_pca", PP_SNV_SG1_PCA, "pls",
                 {"n_components": 28}, target_transform="log1p"),
    ))

    # D3: Feature eng + LightGBM
    experiments.append((
        "log1p_lgbm_feat",
        make_cfg("log1p_lgbm_feat", PP_SNV_SG1_FEAT, "lgbm", {
            "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 4,
            "num_leaves": 15, "min_child_samples": 15, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 3.0, "reg_lambda": 5.0,
            "verbose": -1,
        }, target_transform="log1p"),
    ))

    # D4: Bin4 + LightGBM
    experiments.append((
        "log1p_lgbm_bin4",
        make_cfg("log1p_lgbm_bin4", PP_SNV_SG1_BIN4, "lgbm", {
            "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
            "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0,
            "verbose": -1,
        }, target_transform="log1p"),
    ))

    # -----------------------------------------------------------
    # SECTION E: Random Forest / ExtraTrees
    # -----------------------------------------------------------

    experiments.append((
        "log1p_rf",
        make_cfg("log1p_rf", PP_SNV_SG1_BIN, "rf", {
            "n_estimators": 500, "max_depth": 10, "min_samples_leaf": 5,
            "max_features": 0.5, "n_jobs": -1,
        }, target_transform="log1p"),
    ))

    experiments.append((
        "log1p_et",
        make_cfg("log1p_et", PP_SNV_SG1_BIN, "extra_trees", {
            "n_estimators": 500, "max_depth": 10, "min_samples_leaf": 5,
            "max_features": 0.5, "n_jobs": -1,
        }, target_transform="log1p"),
    ))

    # -----------------------------------------------------------
    # SECTION F: KernelRidge
    # -----------------------------------------------------------

    for alpha in [0.1, 1.0, 10.0]:
        experiments.append((
            f"log1p_kr_a{alpha}",
            make_cfg(f"log1p_kr_a{alpha}", PP_SNV_SG1, "kernel_ridge",
                     {"alpha": alpha, "kernel": "rbf", "gamma": None},
                     target_transform="log1p"),
        ))

    # -----------------------------------------------------------
    # SECTION G: Multi-seed for best configs (robustness check)
    # -----------------------------------------------------------
    for seed in [0, 123, 777]:
        experiments.append((
            f"log1p_pls28_seed{seed}",
            make_cfg(f"log1p_pls28_seed{seed}", PP_SNV_SG1, "pls",
                     {"n_components": 28}, target_transform="log1p", seed=seed),
        ))

    # -----------------------------------------------------------
    # SECTION H: Lasso
    # -----------------------------------------------------------
    for alpha in [0.001, 0.01, 0.1]:
        experiments.append((
            f"log1p_lasso_a{alpha}",
            make_cfg(f"log1p_lasso_a{alpha}", PP_SNV_SG1, "lasso",
                     {"alpha": alpha, "max_iter": 5000}, target_transform="log1p"),
        ))

    # -----------------------------------------------------------
    # SECTION I: Wide PLS search with log1p
    # -----------------------------------------------------------
    for nc in [5, 8, 12, 15, 60, 80]:
        experiments.append((
            f"log1p_pls_nc{nc}_wide",
            make_cfg(f"log1p_pls_nc{nc}_wide", PP_SNV_SG1, "pls",
                     {"n_components": nc}, target_transform="log1p"),
        ))

    # -----------------------------------------------------------
    # SECTION J: MSC + log1p combos
    # -----------------------------------------------------------
    for nc in [20, 28, 40]:
        experiments.append((
            f"log1p_pls_nc{nc}_msc",
            make_cfg(f"log1p_pls_nc{nc}_msc", PP_MSC_SG1, "pls",
                     {"n_components": nc}, target_transform="log1p"),
        ))

    experiments.append((
        "log1p_lgbm_msc",
        make_cfg("log1p_lgbm_msc", [
            {"name": "msc"},
            {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ], "lgbm", {
            "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
            "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0,
            "verbose": -1,
        }, target_transform="log1p"),
    ))

    return experiments


def main():
    experiments = get_all_experiments()
    print(f"Total experiments to run: {len(experiments)}")
    print()

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running {name}...")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    # Sort by RMSE
    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  FINAL RESULTS — ALL EXPERIMENTS")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Fold Std':>10}  {'Min':>8}  {'Max':>8}  Name")
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        fold_rmses = r.get("fold_rmses", [])
        fold_std = np.std(fold_rmses) if fold_rmses else 0
        fold_min = min(fold_rmses) if fold_rmses else 0
        fold_max = max(fold_rmses) if fold_rmses else 0
        marker = " ***" if rank <= 5 else ""
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fold_std:>10.4f}  {fold_min:>8.2f}  {fold_max:>8.2f}  {r['name']}{marker}")

    # Save results
    results_path = Path("runs") / "batch_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
