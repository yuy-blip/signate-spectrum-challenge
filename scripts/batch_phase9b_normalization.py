#!/usr/bin/env python
"""Phase 9b: Advanced normalization for cross-species generalization.

New preprocessing approaches:
- Area normalization (L1 norm per row)
- Continuum removal (convex hull envelope normalization)
- Max normalization
- Range normalization
- Combinations with EMSC and derivatives
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


LGBM_BEST = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
    "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
    "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
}

LGBM_SLOW = {
    "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5,
    "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
    "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
}


def get_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # -----------------------------------------------------------
    # A: EMSC + Area normalization (double normalization)
    # -----------------------------------------------------------
    for bin_size in [0, 8]:
        pp = [
            {"name": "emsc"},
            {"name": "area_normalize"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        ]
        if bin_size:
            pp.append({"name": "binning", "bin_size": bin_size})
        pp.append({"name": "standard_scaler"})
        bn = f"b{bin_size}" if bin_size else "nobin"
        experiments.append((f"emsc_area_{bn}_lgbm", make_cfg(f"emsc_area_{bn}", pp, "lgbm", dict(LGBM_BEST))))

    # -----------------------------------------------------------
    # B: Continuum removal + derivatives
    # -----------------------------------------------------------
    for deriv in [0, 1]:
        pp = [{"name": "continuum_removal"}]
        if deriv:
            pp.append({"name": "sg", "window_length": 7, "polyorder": 2, "deriv": deriv})
        pp.extend([{"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}])
        experiments.append((f"cr_d{deriv}_lgbm", make_cfg(f"cr_d{deriv}", pp, "lgbm", dict(LGBM_BEST))))

    # Continuum removal after EMSC
    for deriv in [0, 1]:
        pp = [{"name": "emsc"}, {"name": "continuum_removal"}]
        if deriv:
            pp.append({"name": "sg", "window_length": 7, "polyorder": 2, "deriv": deriv})
        pp.extend([{"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}])
        experiments.append((f"emsc_cr_d{deriv}_lgbm", make_cfg(f"emsc_cr_d{deriv}", pp, "lgbm", dict(LGBM_BEST))))

    # -----------------------------------------------------------
    # C: Area normalize only (no EMSC) + derivatives
    # -----------------------------------------------------------
    for deriv in [1, 2]:
        sg_p = max(deriv + 1, 2)
        sg_w = 7 if deriv == 1 else 15
        pp = [
            {"name": "area_normalize"},
            {"name": "sg", "window_length": sg_w, "polyorder": sg_p, "deriv": deriv},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]
        experiments.append((f"area_d{deriv}_lgbm", make_cfg(f"area_d{deriv}", pp, "lgbm", dict(LGBM_BEST))))

    # -----------------------------------------------------------
    # D: Max normalize + derivatives
    # -----------------------------------------------------------
    pp = [
        {"name": "max_normalize"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append(("maxnorm_d1_lgbm", make_cfg("maxnorm_d1", pp, "lgbm", dict(LGBM_BEST))))

    # -----------------------------------------------------------
    # E: Range normalize + EMSC + derivative
    # -----------------------------------------------------------
    pp = [
        {"name": "range_normalize"},
        {"name": "emsc"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append(("range_emsc_d1_lgbm", make_cfg("range_emsc_d1", pp, "lgbm", dict(LGBM_BEST))))

    # -----------------------------------------------------------
    # F: Absorbance-based approaches (physical model)
    # -----------------------------------------------------------
    # Absorbance + EMSC
    pp = [
        {"name": "absorbance"},
        {"name": "emsc"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append(("abs_emsc_d1_lgbm", make_cfg("abs_emsc_d1", pp, "lgbm", dict(LGBM_BEST))))

    # Absorbance + area normalize
    pp = [
        {"name": "absorbance"},
        {"name": "area_normalize"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append(("abs_area_d1_lgbm", make_cfg("abs_area_d1", pp, "lgbm", dict(LGBM_BEST))))

    # -----------------------------------------------------------
    # G: Best normalization approaches with sqrt target transform
    # -----------------------------------------------------------
    for norm_name, norm_pp in [
        ("emsc_area", [{"name": "emsc"}, {"name": "area_normalize"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]),
        ("cr", [{"name": "continuum_removal"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]),
        ("emsc_cr", [{"name": "emsc"}, {"name": "continuum_removal"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]),
    ]:
        experiments.append((
            f"sqrt_{norm_name}_lgbm",
            make_cfg(f"sqrt_{norm_name}", norm_pp, "lgbm", dict(LGBM_BEST), target_transform="sqrt"),
        ))

    # -----------------------------------------------------------
    # H: Multi-depth with best new normalizations
    # -----------------------------------------------------------
    for depth, leaves in [(3, 7), (4, 15), (6, 31), (7, 40)]:
        # EMSC + area normalize
        pp = [
            {"name": "emsc"},
            {"name": "area_normalize"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]
        experiments.append((
            f"emsc_area_d{depth}_lgbm",
            make_cfg(f"emsc_area_d{depth}", pp, "lgbm", {
                **LGBM_BEST, "max_depth": depth, "num_leaves": leaves,
            }),
        ))

    # -----------------------------------------------------------
    # I: Slow training with new normalizations
    # -----------------------------------------------------------
    for norm_name, norm_pp in [
        ("emsc_area", [{"name": "emsc"}, {"name": "area_normalize"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]),
        ("emsc_cr", [{"name": "emsc"}, {"name": "continuum_removal"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]),
    ]:
        experiments.append((
            f"{norm_name}_lgbm_slow",
            make_cfg(f"{norm_name}_slow", norm_pp, "lgbm", dict(LGBM_SLOW)),
        ))

    # -----------------------------------------------------------
    # J: XGBoost with new normalizations
    # -----------------------------------------------------------
    for norm_name, norm_pp in [
        ("emsc_area", [{"name": "emsc"}, {"name": "area_normalize"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]),
        ("cr", [{"name": "continuum_removal"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1}, {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}]),
    ]:
        experiments.append((
            f"{norm_name}_xgb",
            make_cfg(f"{norm_name}_xgb", norm_pp, "xgb", {
                "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5,
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": 5.0, "reg_lambda": 10.0, "tree_method": "hist",
            }),
        ))

    # -----------------------------------------------------------
    # K: Multi-seed with best new normalizations for ensemble
    # -----------------------------------------------------------
    pp_emsc_area = [
        {"name": "emsc"},
        {"name": "area_normalize"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    for seed in range(5):
        experiments.append((
            f"emsc_area_s{seed}_lgbm",
            make_cfg(f"emsc_area_s{seed}", pp_emsc_area, "lgbm", {
                **LGBM_BEST, "random_state": seed,
            }),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 9b: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 9b RESULTS (TOP 25)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Fold0':>7}  {'Fold1':>7}  {'Fold2':>7}  {'Fold3':>7}  {'Fold4':>7}  Name")
    print("-" * 100)
    for rank, r in enumerate(results[:25], 1):
        fr = r.get("fold_rmses", [0]*5)
        fr_str = "  ".join(f"{f:>7.2f}" for f in fr[:5])
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fr_str}  {r['name']}")

    results_path = Path("runs") / "batch_phase9b_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
