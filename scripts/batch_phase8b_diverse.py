#!/usr/bin/env python
"""Phase 8b: CatBoost + GaussianProcess + diverse models for stacking."""

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


def make_cfg(name: str, pp: list, model_type: str, model_params: dict, **extra) -> dict:
    cfg = dict(BASE_GKF)
    cfg["experiment_name"] = name
    cfg["preprocess"] = pp
    cfg["model_type"] = model_type
    cfg["model_params"] = model_params
    cfg.update(extra)
    return cfg


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

PP_SNV_SG1_BIN = [
    {"name": "snv"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]


def get_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # -----------------------------------------------------------
    # A: CatBoost with EMSC
    # -----------------------------------------------------------
    catboost_configs = [
        ("cb_d3", {"depth": 3, "iterations": 2000, "learning_rate": 0.01, "l2_leaf_reg": 10, "verbose": 0}),
        ("cb_d4", {"depth": 4, "iterations": 2000, "learning_rate": 0.01, "l2_leaf_reg": 10, "verbose": 0}),
        ("cb_d5", {"depth": 5, "iterations": 2000, "learning_rate": 0.01, "l2_leaf_reg": 10, "verbose": 0}),
        ("cb_d6", {"depth": 6, "iterations": 2000, "learning_rate": 0.01, "l2_leaf_reg": 10, "verbose": 0}),
        ("cb_d3_lr05", {"depth": 3, "iterations": 400, "learning_rate": 0.05, "l2_leaf_reg": 5, "verbose": 0}),
        ("cb_d5_lr05", {"depth": 5, "iterations": 400, "learning_rate": 0.05, "l2_leaf_reg": 5, "verbose": 0}),
        ("cb_d3_l2_1", {"depth": 3, "iterations": 2000, "learning_rate": 0.01, "l2_leaf_reg": 1, "verbose": 0}),
        ("cb_d3_l2_50", {"depth": 3, "iterations": 2000, "learning_rate": 0.01, "l2_leaf_reg": 50, "verbose": 0}),
        ("cb_d4_ss07", {"depth": 4, "iterations": 2000, "learning_rate": 0.01, "l2_leaf_reg": 10, "subsample": 0.7, "verbose": 0}),
    ]

    for name_suffix, params in catboost_configs:
        for pp_name, pp in [("emsc", PP_EMSC), ("snv", PP_SNV_SG1_BIN)]:
            experiments.append((
                f"{name_suffix}_{pp_name}",
                make_cfg(f"{name_suffix}_{pp_name}", pp, "catboost", params),
            ))

    # CatBoost multi-seed
    for seed in range(5):
        experiments.append((
            f"cb_d4_emsc_s{seed}",
            make_cfg(f"cb_d4_s{seed}", PP_EMSC, "catboost", {
                "depth": 4, "iterations": 2000, "learning_rate": 0.01,
                "l2_leaf_reg": 10, "random_seed": seed, "verbose": 0,
            }),
        ))

    # -----------------------------------------------------------
    # B: GaussianProcessRegressor (on PCA-reduced features for speed)
    # -----------------------------------------------------------
    # GP is too slow on full features, use PCA first
    PP_EMSC_PCA = [
        {"name": "emsc"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "pca_features", "n_components": 30, "append": False},
        {"name": "standard_scaler"},
    ]
    PP_EMSC_PCA50 = [
        {"name": "emsc"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "pca_features", "n_components": 50, "append": False},
        {"name": "standard_scaler"},
    ]

    # KernelRidge with EMSC (since GP is same idea but faster)
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        experiments.append((
            f"kr_rbf_a{alpha}_emsc_pca",
            make_cfg(f"kr_a{alpha}_emsc_pca", PP_EMSC_PCA, "kernel_ridge", {
                "alpha": alpha, "kernel": "rbf",
            }),
        ))

    # -----------------------------------------------------------
    # C: Ridge/ElasticNet with EMSC (for linear diversity)
    # -----------------------------------------------------------
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        experiments.append((
            f"ridge_a{alpha}_emsc",
            make_cfg(f"ridge_a{alpha}_emsc", PP_EMSC_NOBIN, "ridge", {"alpha": alpha}),
        ))

    for alpha in [0.01, 0.1]:
        for l1 in [0.1, 0.5, 0.9]:
            experiments.append((
                f"enet_a{alpha}_l1{l1}_emsc",
                make_cfg(f"enet_a{alpha}_l1{l1}_emsc", PP_EMSC_NOBIN, "elastic_net", {
                    "alpha": alpha, "l1_ratio": l1, "max_iter": 5000,
                }),
            ))

    # -----------------------------------------------------------
    # D: PLS with EMSC (for linear spectral model diversity)
    # -----------------------------------------------------------
    for nc in [10, 15, 20, 28, 35, 50]:
        experiments.append((
            f"pls_nc{nc}_emsc",
            make_cfg(f"pls_nc{nc}_emsc", PP_EMSC_NOBIN, "pls", {"n_components": nc}),
        ))

    # -----------------------------------------------------------
    # E: Lasso with EMSC
    # -----------------------------------------------------------
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        experiments.append((
            f"lasso_a{alpha}_emsc",
            make_cfg(f"lasso_a{alpha}_emsc", PP_EMSC_NOBIN, "lasso", {
                "alpha": alpha, "max_iter": 5000,
            }),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 8b: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 8b RESULTS (TOP 25)")
    print("=" * 80)
    for rank, r in enumerate(results[:25], 1):
        fr = r.get("fold_rmses", [])
        fmin = min(fr) if fr else 0
        fmax = max(fr) if fr else 0
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fmin:>8.2f}  {fmax:>8.2f}  {r['name']}")

    results_path = Path("runs") / "batch_phase8b_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
