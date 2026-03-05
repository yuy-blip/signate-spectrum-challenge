#!/usr/bin/env python
"""Phase 5: EMSC-based pipeline deep optimization.

EMSC gave 18.92 — massive improvement! Now tune everything around EMSC.
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


def get_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # Base EMSC pipelines
    def emsc_pp(sg_w=7, sg_p=2, sg_d=1, bin_size=8, extra_steps=None):
        pp = [{"name": "emsc"}]
        pp.append({"name": "sg", "window_length": sg_w, "polyorder": sg_p, "deriv": sg_d})
        if bin_size:
            pp.append({"name": "binning", "bin_size": bin_size})
        if extra_steps:
            pp.extend(extra_steps)
        pp.append({"name": "standard_scaler"})
        return pp

    # LGBM variants
    LGBM_BASE = {
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3,
        "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
        "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
    }
    LGBM_DEEP = {
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
        "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
        "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
    }
    LGBM_SHALLOW = {
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 2,
        "num_leaves": 4, "min_child_samples": 30, "subsample": 0.7,
        "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
    }
    LGBM_SLOW = {
        "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
        "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
        "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
    }
    LGBM_VERYBIG = {
        "n_estimators": 5000, "learning_rate": 0.005, "max_depth": 4,
        "num_leaves": 15, "min_child_samples": 15, "subsample": 0.7,
        "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
    }
    XGB_BASE = {
        "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
        "subsample": 0.7, "colsample_bytree": 0.5,
        "reg_alpha": 5.0, "reg_lambda": 10.0, "tree_method": "hist",
    }

    # -----------------------------------------------------------
    # A: EMSC + SG window sweep
    # -----------------------------------------------------------
    for sg_w in [5, 7, 9, 11, 15, 21]:
        experiments.append((
            f"emsc_sg1_w{sg_w}_b8_lgbm",
            make_cfg(f"emsc_w{sg_w}", emsc_pp(sg_w=sg_w), "lgbm", dict(LGBM_BASE)),
        ))

    # -----------------------------------------------------------
    # B: EMSC + binning sweep
    # -----------------------------------------------------------
    for bin_s in [0, 2, 4, 8, 12, 16, 24, 32]:
        experiments.append((
            f"emsc_sg1w7_b{bin_s}_lgbm",
            make_cfg(f"emsc_b{bin_s}", emsc_pp(bin_size=bin_s if bin_s > 0 else None), "lgbm", dict(LGBM_BASE)),
        ))

    # -----------------------------------------------------------
    # C: EMSC + derivative order sweep
    # -----------------------------------------------------------
    for deriv in [0, 1, 2]:
        sg_p = max(deriv + 1, 2)
        experiments.append((
            f"emsc_sg{deriv}_w7_lgbm",
            make_cfg(f"emsc_d{deriv}", emsc_pp(sg_d=deriv, sg_p=sg_p), "lgbm", dict(LGBM_BASE)),
        ))

    # -----------------------------------------------------------
    # D: EMSC + LGBM hyperparameter sweep
    # -----------------------------------------------------------
    for hp_name, hp in [("base", LGBM_BASE), ("deep", LGBM_DEEP), ("shallow", LGBM_SHALLOW),
                         ("slow", LGBM_SLOW), ("verybig", LGBM_VERYBIG)]:
        experiments.append((
            f"emsc_w7b8_{hp_name}",
            make_cfg(f"emsc_{hp_name}", emsc_pp(), "lgbm", dict(hp)),
        ))

    # -----------------------------------------------------------
    # E: EMSC + XGBoost
    # -----------------------------------------------------------
    experiments.append((
        "emsc_w7b8_xgb",
        make_cfg("emsc_xgb", emsc_pp(), "xgb", dict(XGB_BASE)),
    ))
    experiments.append((
        "emsc_w7b8_xgb_d5",
        make_cfg("emsc_xgb_d5", emsc_pp(), "xgb", {**XGB_BASE, "max_depth": 5}),
    ))

    # -----------------------------------------------------------
    # F: EMSC + RF
    # -----------------------------------------------------------
    experiments.append((
        "emsc_w7b8_rf",
        make_cfg("emsc_rf", emsc_pp(), "rf", {
            "n_estimators": 500, "max_depth": 15,
            "min_samples_leaf": 3, "max_features": "sqrt",
            "n_jobs": -1, "random_state": 42,
        }),
    ))

    # -----------------------------------------------------------
    # G: EMSC + regularization sweep
    # -----------------------------------------------------------
    for ra, rl in [(0.5, 0.5), (1, 1), (1, 5), (3, 10), (5, 10), (10, 50)]:
        experiments.append((
            f"emsc_ra{ra}_rl{rl}",
            make_cfg(f"emsc_ra{ra}_rl{rl}", emsc_pp(), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3,
                "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": ra, "reg_lambda": rl, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # H: EMSC + learning rate sweep
    # -----------------------------------------------------------
    for lr in [0.01, 0.02, 0.03, 0.05, 0.1]:
        n_est = max(100, int(400 * 0.05 / lr))
        experiments.append((
            f"emsc_lr{lr}",
            make_cfg(f"emsc_lr{lr}", emsc_pp(), "lgbm", {
                "n_estimators": n_est, "learning_rate": lr, "max_depth": 3,
                "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # I: EMSC + subsample/colsample sweep
    # -----------------------------------------------------------
    for ss, cs in [(0.5, 0.3), (0.6, 0.4), (0.7, 0.5), (0.8, 0.7), (0.9, 0.8)]:
        experiments.append((
            f"emsc_ss{ss}_cs{cs}",
            make_cfg(f"emsc_ss{ss}_cs{cs}", emsc_pp(), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3,
                "num_leaves": 7, "min_child_samples": 20, "subsample": ss,
                "colsample_bytree": cs, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # J: EMSC + feature engineering
    # -----------------------------------------------------------
    experiments.append((
        "emsc_feat_lgbm",
        make_cfg("emsc_feat", emsc_pp(extra_steps=[
            {"name": "band_ratio"}, {"name": "spectral_stats", "n_regions": 10}
        ]), "lgbm", dict(LGBM_BASE)),
    ))
    experiments.append((
        "emsc_pca_lgbm",
        make_cfg("emsc_pca", emsc_pp(extra_steps=[
            {"name": "pca_features", "n_components": 20}
        ]), "lgbm", dict(LGBM_BASE)),
    ))

    # -----------------------------------------------------------
    # K: EMSC + 2nd derivative
    # -----------------------------------------------------------
    for sg_w in [11, 15, 21]:
        experiments.append((
            f"emsc_2nd_w{sg_w}_lgbm",
            make_cfg(f"emsc_2nd_w{sg_w}", emsc_pp(sg_w=sg_w, sg_d=2, sg_p=3), "lgbm", dict(LGBM_BASE)),
        ))

    # -----------------------------------------------------------
    # L: Multi-seed EMSC + LGBM
    # -----------------------------------------------------------
    for seed in [0, 42, 123, 456, 777, 2024, 9999]:
        experiments.append((
            f"emsc_seed{seed}",
            make_cfg(f"emsc_seed{seed}", emsc_pp(), "lgbm", {
                **LGBM_BASE, "random_state": seed,
            }),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 5: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 5 RESULTS (TOP 25)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Min':>8}  {'Max':>8}  Name")
    print("-" * 80)
    for rank, r in enumerate(results[:25], 1):
        fr = r.get("fold_rmses", [])
        fmin = min(fr) if fr else 0
        fmax = max(fr) if fr else 0
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fmin:>8.2f}  {fmax:>8.2f}  {r['name']}")

    results_path = Path("runs") / "batch_phase5_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
