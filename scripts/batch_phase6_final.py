#!/usr/bin/env python
"""Phase 6: Final tuning around EMSC + deep LGBM (17.79 baseline).

Focus on:
1. Fine-tune depth/leaves around depth=5
2. EMSC + deep with various reg/ss/cs
3. EMSC + no-binning + deep (18.19 was close)
4. Multi-model diversity for stacking
5. Seed ensemble for final predictions
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


def emsc_pp(sg_w=7, sg_p=2, sg_d=1, bin_size=8, extra_steps=None):
    pp = [{"name": "emsc"}]
    pp.append({"name": "sg", "window_length": sg_w, "polyorder": sg_p, "deriv": sg_d})
    if bin_size and bin_size > 0:
        pp.append({"name": "binning", "bin_size": bin_size})
    if extra_steps:
        pp.extend(extra_steps)
    pp.append({"name": "standard_scaler"})
    return pp


def get_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # -----------------------------------------------------------
    # A: Depth/leaves fine-tuning around best (depth=5, leaves=20)
    # -----------------------------------------------------------
    for depth, leaves, mcs in [
        (4, 12, 15), (4, 15, 10), (4, 15, 15),
        (5, 15, 10), (5, 15, 15), (5, 20, 10), (5, 20, 15), (5, 20, 20),
        (5, 25, 10), (5, 31, 10),
        (6, 20, 10), (6, 31, 10), (6, 31, 15),
        (7, 40, 10), (7, 63, 10),
        (8, 50, 10), (8, 63, 10),
    ]:
        experiments.append((
            f"emsc_d{depth}_l{leaves}_mcs{mcs}",
            make_cfg(f"emsc_d{depth}_l{leaves}", emsc_pp(), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": depth,
                "num_leaves": leaves, "min_child_samples": mcs, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # B: Learning rate sweep with depth=5
    # -----------------------------------------------------------
    for lr in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        n_est = max(100, int(400 * 0.05 / lr))
        experiments.append((
            f"emsc_deep_lr{lr}",
            make_cfg(f"emsc_deep_lr{lr}", emsc_pp(), "lgbm", {
                "n_estimators": n_est, "learning_rate": lr, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # C: Regularization with depth=5
    # -----------------------------------------------------------
    for ra, rl in [(0.1, 0.1), (0.5, 0.5), (1, 1), (2, 2), (3, 5), (5, 10), (10, 20)]:
        experiments.append((
            f"emsc_deep_ra{ra}_rl{rl}",
            make_cfg(f"emsc_deep_ra{ra}_rl{rl}", emsc_pp(), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": ra, "reg_lambda": rl, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # D: Subsample/colsample with depth=5
    # -----------------------------------------------------------
    for ss, cs in [(0.5, 0.3), (0.5, 0.5), (0.6, 0.4), (0.7, 0.5), (0.8, 0.6), (0.8, 0.8), (0.9, 0.7)]:
        experiments.append((
            f"emsc_deep_ss{ss}_cs{cs}",
            make_cfg(f"emsc_deep_ss{ss}_cs{cs}", emsc_pp(), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": ss,
                "colsample_bytree": cs, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # E: No-binning + depth=5 (since no-bin was 18.19)
    # -----------------------------------------------------------
    for depth, leaves in [(4, 15), (5, 20), (5, 31), (6, 31)]:
        experiments.append((
            f"emsc_nobin_d{depth}_l{leaves}",
            make_cfg(f"emsc_nobin_d{depth}", emsc_pp(bin_size=0), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": depth,
                "num_leaves": leaves, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # F: Bin size sweep with depth=5
    # -----------------------------------------------------------
    for bin_s in [2, 4, 6, 8, 12, 16]:
        experiments.append((
            f"emsc_deep_b{bin_s}",
            make_cfg(f"emsc_deep_b{bin_s}", emsc_pp(bin_size=bin_s), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # G: SG window with depth=5
    # -----------------------------------------------------------
    for sg_w in [5, 7, 9, 11, 15]:
        experiments.append((
            f"emsc_deep_sgw{sg_w}",
            make_cfg(f"emsc_deep_sgw{sg_w}", emsc_pp(sg_w=sg_w), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # H: XGBoost deep with EMSC
    # -----------------------------------------------------------
    for d in [3, 4, 5, 6]:
        experiments.append((
            f"emsc_xgb_d{d}",
            make_cfg(f"emsc_xgb_d{d}", emsc_pp(), "xgb", {
                "n_estimators": 2000, "learning_rate": 0.01, "max_depth": d,
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": 5.0, "reg_lambda": 10.0, "tree_method": "hist",
            }),
        ))
    # XGBoost with higher LR
    for d in [4, 5]:
        experiments.append((
            f"emsc_xgb_d{d}_lr05",
            make_cfg(f"emsc_xgb_d{d}_lr05", emsc_pp(), "xgb", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": d,
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": 1.0, "reg_lambda": 1.0, "tree_method": "hist",
            }),
        ))

    # -----------------------------------------------------------
    # I: RF with EMSC deep
    # -----------------------------------------------------------
    for n_est in [300, 500, 1000]:
        experiments.append((
            f"emsc_rf_{n_est}",
            make_cfg(f"emsc_rf_{n_est}", emsc_pp(), "rf", {
                "n_estimators": n_est, "max_depth": 15,
                "min_samples_leaf": 3, "max_features": "sqrt",
                "n_jobs": -1, "random_state": 42,
            }),
        ))

    # -----------------------------------------------------------
    # J: Multi-seed with EMSC deep for ensemble
    # -----------------------------------------------------------
    for seed in range(10):
        experiments.append((
            f"emsc_deep_s{seed}",
            make_cfg(f"emsc_deep_s{seed}", emsc_pp(), "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0,
                "verbose": -1, "random_state": seed,
            }),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 6: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 6 RESULTS (TOP 30)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Min':>8}  {'Max':>8}  Name")
    print("-" * 80)
    for rank, r in enumerate(results[:30], 1):
        fr = r.get("fold_rmses", [])
        fmin = min(fr) if fr else 0
        fmax = max(fr) if fr else 0
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fmin:>8.2f}  {fmax:>8.2f}  {r['name']}")

    results_path = Path("runs") / "batch_phase6_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
