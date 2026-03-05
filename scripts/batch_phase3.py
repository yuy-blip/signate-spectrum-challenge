#!/usr/bin/env python
"""Phase 3: Combine best findings + stacking + seed ensemble.

Best findings so far:
- SG window_length=7 (not 11)
- bin_size=16 (not 8)
- lr=0.05 with fewer trees
- depth 5, num_leaves 20 also good

Strategy:
1. Combine best PP + HP settings
2. Multi-seed ensemble for submission
3. Quick stacking from saved runs
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

    # Best PP: SG w7 + bin16
    PP_W7_BIN16 = [
        {"name": "snv"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16}, {"name": "standard_scaler"},
    ]
    PP_W7_BIN8 = [
        {"name": "snv"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"},
    ]
    PP_W7_BIN12 = [
        {"name": "snv"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 12}, {"name": "standard_scaler"},
    ]
    PP_W7_BIN24 = [
        {"name": "snv"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 24}, {"name": "standard_scaler"},
    ]
    PP_W7_BIN32 = [
        {"name": "snv"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 32}, {"name": "standard_scaler"},
    ]
    PP_W7_NOBIN = [
        {"name": "snv"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "standard_scaler"},
    ]
    PP_W5_BIN16 = [
        {"name": "snv"}, {"name": "sg", "window_length": 5, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16}, {"name": "standard_scaler"},
    ]
    PP_W9_BIN16 = [
        {"name": "snv"}, {"name": "sg", "window_length": 9, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16}, {"name": "standard_scaler"},
    ]
    PP_W7_BIN16_FEAT = [
        {"name": "snv"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16}, {"name": "band_ratio"}, {"name": "spectral_stats", "n_regions": 10},
        {"name": "standard_scaler"},
    ]
    PP_MSC_W7_BIN16 = [
        {"name": "msc"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16}, {"name": "standard_scaler"},
    ]

    # -----------------------------------------------------------
    # A: Combined best PP + HP for LightGBM
    # -----------------------------------------------------------

    lgbm_hp_variants = [
        # Best from Phase 2
        ("base", {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 7, "min_child_samples": 20}),
        ("deep", {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 10}),
        ("deeper", {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "min_child_samples": 10}),
        ("lr03", {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 3, "num_leaves": 7, "min_child_samples": 20}),
        ("lr1", {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "num_leaves": 7, "min_child_samples": 20}),
        ("lr03_deep", {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 5, "num_leaves": 20, "min_child_samples": 10}),
        ("verybig", {"n_estimators": 5000, "learning_rate": 0.005, "max_depth": 4, "num_leaves": 15, "min_child_samples": 15}),
    ]

    for pp_name, pp in [("w7_b16", PP_W7_BIN16), ("w7_b8", PP_W7_BIN8), ("w7_b12", PP_W7_BIN12),
                         ("w7_b24", PP_W7_BIN24), ("w7_b32", PP_W7_BIN32), ("w5_b16", PP_W5_BIN16),
                         ("w9_b16", PP_W9_BIN16), ("w7_nobin", PP_W7_NOBIN)]:
        for hp_name, hp in lgbm_hp_variants:
            full_params = {
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
            }
            full_params.update(hp)
            experiments.append((
                f"lgbm_{pp_name}_{hp_name}",
                make_cfg(f"lgbm_{pp_name}_{hp_name}", pp, "lgbm", full_params),
            ))

    # -----------------------------------------------------------
    # B: Best LGBM with feature engineering
    # -----------------------------------------------------------
    for hp_name, hp in [("base", lgbm_hp_variants[0][1]), ("deep", lgbm_hp_variants[1][1])]:
        full_params = {
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
        }
        full_params.update(hp)
        experiments.append((
            f"lgbm_w7b16_feat_{hp_name}",
            make_cfg(f"lgbm_w7b16_feat_{hp_name}", PP_W7_BIN16_FEAT, "lgbm", full_params),
        ))

    # -----------------------------------------------------------
    # C: MSC instead of SNV
    # -----------------------------------------------------------
    for hp_name, hp in [("base", lgbm_hp_variants[0][1]), ("deep", lgbm_hp_variants[1][1])]:
        full_params = {
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
        }
        full_params.update(hp)
        experiments.append((
            f"lgbm_msc_w7b16_{hp_name}",
            make_cfg(f"lgbm_msc_w7b16_{hp_name}", PP_MSC_W7_BIN16, "lgbm", full_params),
        ))

    # -----------------------------------------------------------
    # D: XGBoost with best PP
    # -----------------------------------------------------------
    for pp_name, pp in [("w7_b16", PP_W7_BIN16), ("w7_b8", PP_W7_BIN8)]:
        for hp_name, hp in [
            ("d3", {"max_depth": 3}),
            ("d4", {"max_depth": 4}),
            ("d5", {"max_depth": 5}),
            ("lr05", {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 400}),
        ]:
            base_params = {
                "n_estimators": 2000, "learning_rate": 0.01,
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": 5.0, "reg_lambda": 10.0, "tree_method": "hist",
            }
            base_params.update(hp)
            experiments.append((
                f"xgb_{pp_name}_{hp_name}",
                make_cfg(f"xgb_{pp_name}_{hp_name}", pp, "xgb", base_params),
            ))

    # -----------------------------------------------------------
    # E: RF with best PP
    # -----------------------------------------------------------
    for pp_name, pp in [("w7_b16", PP_W7_BIN16), ("w7_b8", PP_W7_BIN8)]:
        experiments.append((
            f"rf_{pp_name}",
            make_cfg(f"rf_{pp_name}", pp, "rf", {
                "n_estimators": 500, "max_depth": 15,
                "min_samples_leaf": 3, "max_features": "sqrt",
                "n_jobs": -1, "random_state": 42,
            }),
        ))

    # -----------------------------------------------------------
    # F: Regularization sweep on best config
    # -----------------------------------------------------------
    for ra, rl in [(1, 1), (1, 5), (3, 10), (5, 10), (10, 10), (10, 50), (0.5, 0.5)]:
        experiments.append((
            f"lgbm_w7b16_ra{ra}_rl{rl}",
            make_cfg(f"lgbm_w7b16_ra{ra}_rl{rl}", PP_W7_BIN16, "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3,
                "num_leaves": 7, "min_child_samples": 20,
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": ra, "reg_lambda": rl, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # G: Subsample/colsample sweep
    # -----------------------------------------------------------
    for ss, cs in [(0.5, 0.3), (0.5, 0.5), (0.6, 0.4), (0.8, 0.6), (0.9, 0.8), (1.0, 1.0)]:
        experiments.append((
            f"lgbm_w7b16_ss{ss}_cs{cs}",
            make_cfg(f"lgbm_w7b16_ss{ss}_cs{cs}", PP_W7_BIN16, "lgbm", {
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3,
                "num_leaves": 7, "min_child_samples": 20,
                "subsample": ss, "colsample_bytree": cs,
                "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
            }),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 3: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 3 RESULTS (TOP 20)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Min':>8}  {'Max':>8}  Name")
    print("-" * 80)
    for rank, r in enumerate(results[:20], 1):
        fr = r.get("fold_rmses", [])
        fmin = min(fr) if fr else 0
        fmax = max(fr) if fr else 0
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fmin:>8.2f}  {fmax:>8.2f}  {r['name']}")

    # Save
    results_path = Path("runs") / "batch_phase3_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
