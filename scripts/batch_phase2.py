#!/usr/bin/env python
"""Phase 2: Aggressive hyperparameter tuning + diversity for stacking.

Focus on:
1. LightGBM wider hyperparameter search
2. XGBoost variants
3. Different preprocessing + tree models
4. Quick stacking test
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


def make_cfg(name: str, pp: list, model_type: str, model_params: dict, **extra) -> dict:
    cfg = dict(BASE_GKF)
    cfg["experiment_name"] = name
    cfg["preprocess"] = pp
    cfg["model_type"] = model_type
    cfg["model_params"] = model_params
    cfg.update(extra)
    return cfg


# Preprocessing recipes
PP_SNV_SG1 = [{"name": "snv"}, {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}, {"name": "standard_scaler"}]
PP_SNV_SG1_BIN8 = [{"name": "snv"}, {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]
PP_SNV_SG1_BIN4 = [{"name": "snv"}, {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 4}, {"name": "standard_scaler"}]
PP_SNV_SG1_BIN16 = [{"name": "snv"}, {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 16}, {"name": "standard_scaler"}]
PP_MSC_SG1_BIN8 = [{"name": "msc"}, {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]
PP_SNV_SG0_BIN8 = [{"name": "snv"}, {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 0}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]
PP_SNV_SG1_FEAT_BIN8 = [{"name": "snv"}, {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "band_ratio"}, {"name": "spectral_stats", "n_regions": 10}, {"name": "standard_scaler"}]
PP_RAW_BIN8 = [{"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]
PP_SNV_SG1_BIN2 = [{"name": "snv"}, {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 2}, {"name": "standard_scaler"}]

# SG window variations
PP_SNV_SG1_W7 = [{"name": "snv"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]
PP_SNV_SG1_W15 = [{"name": "snv"}, {"name": "sg", "window_length": 15, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]
PP_SNV_SG1_W21 = [{"name": "snv"}, {"name": "sg", "window_length": 21, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]


def get_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # -----------------------------------------------------------
    # A: LightGBM hyperparameter exploration
    # -----------------------------------------------------------

    lgbm_configs = [
        # (name_suffix, params)
        ("d2_l4", {"max_depth": 2, "num_leaves": 4, "min_child_samples": 30}),
        ("d3_l7", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 20}),
        ("d3_l7_mcs30", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 30}),
        ("d4_l15", {"max_depth": 4, "num_leaves": 15, "min_child_samples": 15}),
        ("d5_l20", {"max_depth": 5, "num_leaves": 20, "min_child_samples": 10}),
        ("d6_l31", {"max_depth": 6, "num_leaves": 31, "min_child_samples": 10}),
        ("d3_l7_ss05", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 20, "subsample": 0.5}),
        ("d3_l7_ss09", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 20, "subsample": 0.9}),
        ("d3_l7_cs03", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 20, "colsample_bytree": 0.3}),
        ("d3_l7_cs08", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 20, "colsample_bytree": 0.8}),
        ("d3_l7_ra1", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 20, "reg_alpha": 1.0, "reg_lambda": 1.0}),
        ("d3_l7_ra10_rl50", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 20, "reg_alpha": 10.0, "reg_lambda": 50.0}),
        ("dart", {"max_depth": 3, "num_leaves": 7, "min_child_samples": 20, "boosting_type": "dart", "drop_rate": 0.1}),
    ]

    for suffix, extra_params in lgbm_configs:
        base_params = {
            "n_estimators": 2000, "learning_rate": 0.01,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
        }
        base_params.update(extra_params)
        experiments.append((
            f"lgbm_{suffix}",
            make_cfg(f"lgbm_{suffix}", PP_SNV_SG1_BIN8, "lgbm", base_params),
        ))

    # LightGBM learning rate sweep
    for lr in [0.005, 0.02, 0.05]:
        n_est = int(2000 * 0.01 / lr)  # Scale iterations inversely
        experiments.append((
            f"lgbm_lr{lr}",
            make_cfg(f"lgbm_lr{lr}", PP_SNV_SG1_BIN8, "lgbm", {
                "n_estimators": n_est, "learning_rate": lr, "max_depth": 3,
                "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0,
                "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # B: XGBoost hyperparameter exploration
    # -----------------------------------------------------------

    xgb_configs = [
        ("d2", {"max_depth": 2}),
        ("d3", {"max_depth": 3}),
        ("d4", {"max_depth": 4}),
        ("d5", {"max_depth": 5}),
        ("d3_ss05", {"max_depth": 3, "subsample": 0.5}),
        ("d3_cs03", {"max_depth": 3, "colsample_bytree": 0.3}),
        ("d3_ra10_rl50", {"max_depth": 3, "reg_alpha": 10.0, "reg_lambda": 50.0}),
    ]

    for suffix, extra_params in xgb_configs:
        base_params = {
            "n_estimators": 2000, "learning_rate": 0.01,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 5.0, "reg_lambda": 10.0,
            "tree_method": "hist",
        }
        base_params.update(extra_params)
        experiments.append((
            f"xgb_{suffix}",
            make_cfg(f"xgb_{suffix}", PP_SNV_SG1_BIN8, "xgb", base_params),
        ))

    # -----------------------------------------------------------
    # C: Preprocessing variations with best LGBM
    # -----------------------------------------------------------

    best_lgbm = {
        "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
        "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
        "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0,
        "verbose": -1,
    }

    for pp_name, pp in [
        ("bin4", PP_SNV_SG1_BIN4),
        ("bin16", PP_SNV_SG1_BIN16),
        ("bin2", PP_SNV_SG1_BIN2),
        ("msc_bin8", PP_MSC_SG1_BIN8),
        ("smooth_bin8", PP_SNV_SG0_BIN8),
        ("feat_bin8", PP_SNV_SG1_FEAT_BIN8),
        ("raw_bin8", PP_RAW_BIN8),
        ("nobin", PP_SNV_SG1),
        ("w7", PP_SNV_SG1_W7),
        ("w15", PP_SNV_SG1_W15),
        ("w21", PP_SNV_SG1_W21),
    ]:
        experiments.append((
            f"lgbm_pp_{pp_name}",
            make_cfg(f"lgbm_pp_{pp_name}", pp, "lgbm", dict(best_lgbm)),
        ))

    # -----------------------------------------------------------
    # D: RF / ExtraTrees (fixed)
    # -----------------------------------------------------------

    for n_est in [300, 500, 1000]:
        experiments.append((
            f"rf_n{n_est}",
            make_cfg(f"rf_n{n_est}", PP_SNV_SG1_BIN8, "rf", {
                "n_estimators": n_est, "max_depth": 15,
                "min_samples_leaf": 3, "max_features": "sqrt",
                "n_jobs": -1, "random_state": 42,
            }),
        ))

    experiments.append((
        "et_500",
        make_cfg("et_500", PP_SNV_SG1_BIN8, "extra_trees", {
            "n_estimators": 500, "max_depth": 15,
            "min_samples_leaf": 3, "max_features": "sqrt",
            "n_jobs": -1, "random_state": 42,
        }),
    ))

    # -----------------------------------------------------------
    # E: Multi-seed ensemble (for best configs)
    # -----------------------------------------------------------

    # Not applicable for GKF (deterministic) — skip
    # But we can vary the model seed for tree models
    for seed in [0, 42, 123, 777, 2024]:
        experiments.append((
            f"lgbm_seed{seed}",
            make_cfg(f"lgbm_seed{seed}", PP_SNV_SG1_BIN8, "lgbm", {
                "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
                "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0,
                "verbose": -1, "random_state": seed,
            }),
        ))

    # -----------------------------------------------------------
    # F: PLS for stacking diversity (no transform)
    # -----------------------------------------------------------
    for nc in [10, 15, 20, 28, 35]:
        experiments.append((
            f"pls_nc{nc}_div",
            make_cfg(f"pls_nc{nc}_div", PP_SNV_SG1, "pls", {"n_components": nc}),
        ))

    # -----------------------------------------------------------
    # G: Ridge for stacking diversity
    # -----------------------------------------------------------
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        experiments.append((
            f"ridge_a{alpha}_div",
            make_cfg(f"ridge_a{alpha}_div", PP_SNV_SG1, "ridge", {"alpha": alpha}),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 2: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 2 RESULTS")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Min':>8}  {'Max':>8}  Name")
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        fr = r.get("fold_rmses", [])
        fmin = min(fr) if fr else 0
        fmax = max(fr) if fr else 0
        marker = " ***" if rank <= 10 else ""
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fmin:>8.2f}  {fmax:>8.2f}  {r['name']}{marker}")

    # Save
    results_path = Path("runs") / "batch_phase2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
