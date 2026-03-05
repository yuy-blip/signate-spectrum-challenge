#!/usr/bin/env python
"""Phase 9: Cross-species generalization — targeting Fold 2 bottleneck.

Key insight: Species 15 (ベイスギ) has extreme moisture values (up to 298.6).
Fold 2 (species 15/17/19) has RMSE ~30.6 while other folds are 10-15.

Approaches:
1. Target transforms (sqrt, log1p) to compress high-value range
2. Huber loss (robust to outliers, should help with extreme values)
3. Water-band wavelength selection (species-invariant moisture features)
4. Higher EMSC poly_order (more aggressive baseline removal)
5. SVR with RBF kernel (different inductive bias)
6. Feature engineering: water band ratios + spectral stats
7. Aggressive regularization tuned for generalization
8. 2nd derivative (removes more species-specific baseline effects)
9. Combination of multiple preprocessing approaches
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


# --- Preprocessing pipelines ---

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

PP_EMSC_2ND = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 15, "polyorder": 3, "deriv": 2},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

PP_EMSC_P3 = [
    {"name": "emsc", "poly_order": 3},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

PP_EMSC_P4 = [
    {"name": "emsc", "poly_order": 4},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

# Water band focused: O-H overtone (~7000), O-H combination (~5200), C-H/O-H (~4000-4500)
PP_EMSC_WATER = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "wavelength_selector", "ranges": [[6500, 7500], [4800, 5600], [4000, 4600]]},
    {"name": "standard_scaler"},
]

# Wider water bands
PP_EMSC_WATER_WIDE = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "wavelength_selector", "ranges": [[6000, 7800], [4500, 5800], [3999, 4700]]},
    {"name": "standard_scaler"},
]

# Exclude noisy regions (keep 4000-8000 only)
PP_EMSC_MID = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "wavelength_selector", "ranges": [[4000, 8000]]},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

# EMSC + feature engineering (band ratios + stats)
PP_EMSC_FEAT = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "band_ratio"},
    {"name": "spectral_stats", "n_regions": 10},
    {"name": "standard_scaler"},
]

# EMSC + PCA (for SVR, KernelRidge — needs lower dimensions)
PP_EMSC_PCA20 = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "pca_features", "n_components": 20, "append": False},
    {"name": "standard_scaler"},
]

PP_EMSC_PCA50 = [
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "pca_features", "n_components": 50, "append": False},
    {"name": "standard_scaler"},
]

# Double scatter correction: SNV then EMSC
PP_SNV_EMSC = [
    {"name": "snv"},
    {"name": "emsc"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

# EMSC then SNV
PP_EMSC_SNV = [
    {"name": "emsc"},
    {"name": "snv"},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]


# Best LGBM config
LGBM_BEST = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
    "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
    "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
}


def get_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # -----------------------------------------------------------
    # A: Target transforms with best LGBM (sqrt should help compress 298→17)
    # -----------------------------------------------------------
    for transform in ["sqrt", "log1p"]:
        for pp_name, pp in [("emsc", PP_EMSC), ("emsc_nobin", PP_EMSC_NOBIN)]:
            experiments.append((
                f"tt_{transform}_{pp_name}_lgbm",
                make_cfg(f"tt_{transform}_{pp_name}", pp, "lgbm", dict(LGBM_BEST),
                         target_transform=transform),
            ))

    # sqrt with different depths
    for depth, leaves in [(3, 7), (4, 15), (5, 20), (6, 31), (7, 40)]:
        experiments.append((
            f"tt_sqrt_emsc_d{depth}",
            make_cfg(f"tt_sqrt_d{depth}", PP_EMSC, "lgbm", {
                **LGBM_BEST, "max_depth": depth, "num_leaves": leaves,
            }, target_transform="sqrt"),
        ))

    # -----------------------------------------------------------
    # B: LGBM with Huber loss (robust regression)
    # -----------------------------------------------------------
    for alpha in [0.7, 0.8, 0.9, 0.95, 0.99]:
        experiments.append((
            f"huber_a{alpha}_emsc",
            make_cfg(f"huber_a{alpha}", PP_EMSC, "lgbm", {
                **LGBM_BEST,
                "objective": "huber",
                "huber_delta": alpha * 100,  # scaled to target range
            }),
        ))

    # Huber with different deltas
    for delta in [5, 10, 20, 30, 50, 100]:
        experiments.append((
            f"huber_d{delta}_emsc",
            make_cfg(f"huber_d{delta}", PP_EMSC, "lgbm", {
                **LGBM_BEST,
                "objective": "huber",
                "huber_delta": delta,
            }),
        ))

    # -----------------------------------------------------------
    # C: Water band wavelength selection
    # -----------------------------------------------------------
    for pp_name, pp in [("water", PP_EMSC_WATER), ("water_wide", PP_EMSC_WATER_WIDE),
                         ("mid", PP_EMSC_MID)]:
        experiments.append((
            f"{pp_name}_lgbm",
            make_cfg(f"{pp_name}_lgbm", pp, "lgbm", dict(LGBM_BEST)),
        ))
        # Also with slower training for more features
        experiments.append((
            f"{pp_name}_lgbm_slow",
            make_cfg(f"{pp_name}_lgbm_slow", pp, "lgbm", {
                "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # D: Higher EMSC poly_order (more aggressive baseline removal)
    # -----------------------------------------------------------
    for pp_name, pp in [("emsc_p3", PP_EMSC_P3), ("emsc_p4", PP_EMSC_P4)]:
        experiments.append((
            f"{pp_name}_lgbm",
            make_cfg(f"{pp_name}_lgbm", pp, "lgbm", dict(LGBM_BEST)),
        ))
        experiments.append((
            f"{pp_name}_lgbm_slow",
            make_cfg(f"{pp_name}_lgbm_slow", pp, "lgbm", {
                "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5,
                "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # E: 2nd derivative with EMSC (more species-invariant)
    # -----------------------------------------------------------
    for depth, leaves in [(3, 7), (5, 20), (6, 31)]:
        experiments.append((
            f"emsc_2nd_d{depth}_lgbm",
            make_cfg(f"emsc_2nd_d{depth}", PP_EMSC_2ND, "lgbm", {
                **LGBM_BEST, "max_depth": depth, "num_leaves": leaves,
            }),
        ))

    # -----------------------------------------------------------
    # F: SVR with RBF kernel (different inductive bias, on PCA)
    # -----------------------------------------------------------
    for C in [1.0, 10.0, 100.0]:
        for eps in [0.1, 1.0]:
            experiments.append((
                f"svr_C{C}_e{eps}_pca20",
                make_cfg(f"svr_C{C}_e{eps}", PP_EMSC_PCA20, "svr", {
                    "kernel": "rbf", "C": C, "epsilon": eps,
                }),
            ))
    for C in [10.0, 100.0]:
        experiments.append((
            f"svr_C{C}_pca50",
            make_cfg(f"svr_C{C}_pca50", PP_EMSC_PCA50, "svr", {
                "kernel": "rbf", "C": C, "epsilon": 1.0,
            }),
        ))

    # -----------------------------------------------------------
    # G: Feature engineering with EMSC
    # -----------------------------------------------------------
    experiments.append((
        "emsc_feat_lgbm",
        make_cfg("emsc_feat_lgbm", PP_EMSC_FEAT, "lgbm", dict(LGBM_BEST)),
    ))
    experiments.append((
        "emsc_feat_lgbm_deep",
        make_cfg("emsc_feat_deep", PP_EMSC_FEAT, "lgbm", {
            **LGBM_BEST, "max_depth": 6, "num_leaves": 31,
        }),
    ))

    # -----------------------------------------------------------
    # H: Double scatter correction (SNV + EMSC or EMSC + SNV)
    # -----------------------------------------------------------
    for pp_name, pp in [("snv_emsc", PP_SNV_EMSC), ("emsc_snv", PP_EMSC_SNV)]:
        experiments.append((
            f"{pp_name}_lgbm",
            make_cfg(f"{pp_name}_lgbm", pp, "lgbm", dict(LGBM_BEST)),
        ))

    # -----------------------------------------------------------
    # I: XGBoost with sqrt target transform
    # -----------------------------------------------------------
    for depth in [4, 5, 6]:
        experiments.append((
            f"tt_sqrt_emsc_xgb_d{depth}",
            make_cfg(f"tt_sqrt_xgb_d{depth}", PP_EMSC, "xgb", {
                "n_estimators": 2000, "learning_rate": 0.01, "max_depth": depth,
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": 5.0, "reg_lambda": 10.0, "tree_method": "hist",
            }, target_transform="sqrt"),
        ))

    # -----------------------------------------------------------
    # J: LGBM with MAE objective (median regression — robust to outliers)
    # -----------------------------------------------------------
    experiments.append((
        "mae_emsc_lgbm",
        make_cfg("mae_emsc", PP_EMSC, "lgbm", {
            **LGBM_BEST, "objective": "mae",
        }),
    ))

    # -----------------------------------------------------------
    # K: LGBM with quantile regression (predict different quantiles)
    # -----------------------------------------------------------
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        experiments.append((
            f"quantile_a{alpha}_emsc",
            make_cfg(f"quantile_a{alpha}", PP_EMSC, "lgbm", {
                **LGBM_BEST,
                "objective": "quantile",
                "alpha": alpha,
            }),
        ))

    # -----------------------------------------------------------
    # L: PLS with different preprocessing (linear model diversity)
    # -----------------------------------------------------------
    for nc in [15, 20, 30]:
        for pp_name, pp in [("water", PP_EMSC_WATER), ("water_wide", PP_EMSC_WATER_WIDE)]:
            experiments.append((
                f"pls_nc{nc}_{pp_name}",
                make_cfg(f"pls_nc{nc}_{pp_name}", pp, "pls", {"n_components": nc}),
            ))

    # -----------------------------------------------------------
    # M: KernelRidge on water bands
    # -----------------------------------------------------------
    for alpha in [0.1, 1.0, 10.0]:
        experiments.append((
            f"kr_a{alpha}_water_pca",
            make_cfg(f"kr_a{alpha}_water", [
                {"name": "emsc"},
                {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
                {"name": "wavelength_selector", "ranges": [[6500, 7500], [4800, 5600], [4000, 4600]]},
                {"name": "pca_features", "n_components": 20, "append": False},
                {"name": "standard_scaler"},
            ], "kernel_ridge", {"alpha": alpha, "kernel": "rbf"}),
        ))

    # -----------------------------------------------------------
    # N: Combined: sqrt transform + Huber + EMSC
    # -----------------------------------------------------------
    for delta in [10, 30, 50]:
        experiments.append((
            f"sqrt_huber_d{delta}_emsc",
            make_cfg(f"sqrt_huber_d{delta}", PP_EMSC, "lgbm", {
                **LGBM_BEST,
                "objective": "huber",
                "huber_delta": delta,
            }, target_transform="sqrt"),
        ))

    # -----------------------------------------------------------
    # O: Multi-seed ensemble with sqrt transform (best diversity)
    # -----------------------------------------------------------
    for seed in range(10):
        experiments.append((
            f"sqrt_emsc_lgbm_s{seed}",
            make_cfg(f"sqrt_s{seed}", PP_EMSC, "lgbm", {
                **LGBM_BEST, "random_state": seed,
            }, target_transform="sqrt"),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 9: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 9 RESULTS (TOP 30)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Fold0':>7}  {'Fold1':>7}  {'Fold2':>7}  {'Fold3':>7}  {'Fold4':>7}  Name")
    print("-" * 100)
    for rank, r in enumerate(results[:30], 1):
        fr = r.get("fold_rmses", [0]*5)
        fr_str = "  ".join(f"{f:>7.2f}" for f in fr[:5])
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fr_str}  {r['name']}")

    results_path = Path("runs") / "batch_phase9_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
