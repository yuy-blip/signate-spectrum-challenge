#!/usr/bin/env python
"""Phase 7: Deep EMSC optimization + wavelength selection + poly_order tuning.

Current best: 17.48 (ensemble of 4 EMSC models)
Single best: 17.745

Focus on:
1. EMSC poly_order variation (1, 2, 3, 4, 5)
2. EMSC + wavelength selection for water bands
3. EMSC + feature engineering for diversity
4. Different model architectures for ensemble diversity
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


LGBM_DEEP = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
    "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
    "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
}

LGBM_DEEP_MCS20 = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
    "num_leaves": 20, "min_child_samples": 20, "subsample": 0.7,
    "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
}

XGB_DEEP = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
    "subsample": 0.7, "colsample_bytree": 0.5,
    "reg_alpha": 1.0, "reg_lambda": 1.0, "tree_method": "hist",
}


def get_experiments() -> list[tuple[str, dict]]:
    experiments = []

    # -----------------------------------------------------------
    # A: EMSC poly_order variation
    # -----------------------------------------------------------
    for poly in [1, 2, 3, 4, 5]:
        pp = [
            {"name": "emsc", "poly_order": poly},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]
        for hp_name, hp in [("deep", LGBM_DEEP), ("mcs20", LGBM_DEEP_MCS20)]:
            experiments.append((
                f"emsc_p{poly}_{hp_name}",
                make_cfg(f"emsc_p{poly}", pp, "lgbm", dict(hp)),
            ))
        experiments.append((
            f"emsc_p{poly}_xgb",
            make_cfg(f"emsc_p{poly}_xgb", pp, "xgb", dict(XGB_DEEP)),
        ))

    # -----------------------------------------------------------
    # B: EMSC + wavelength selection (water bands)
    # Wavelengths: 9993.77 → 3999.82 cm⁻¹, ~3.85 cm⁻¹/step, 1557 features
    # Water band 1: ~7000 cm⁻¹ → idx ~(9993.77-7000)/3.85 ≈ 778
    # Water band 2: ~5200 cm⁻¹ → idx ~(9993.77-5200)/3.85 ≈ 1245
    # Water band 3: ~4500 cm⁻¹ → idx ~(9993.77-4500)/3.85 ≈ 1427
    # -----------------------------------------------------------
    for wn_name, start, end in [
        ("water_all", 700, 1557),
        ("water_focus", 700, 1400),
        ("mid_nir", 500, 1200),
        ("upper_trim", 100, 1557),  # trim noisy edges
        ("lower_focus", 900, 1557),
    ]:
        pp = [
            {"name": "wavelength_selector", "start_idx": start, "end_idx": end},
            {"name": "emsc"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]
        experiments.append((
            f"wn_{wn_name}_emsc_deep",
            make_cfg(f"wn_{wn_name}_emsc", pp, "lgbm", dict(LGBM_DEEP)),
        ))

    # EMSC first, then wavelength select
    for wn_name, start, end in [
        ("water_all", 700, 1557),
        ("water_focus", 700, 1400),
        ("upper_trim", 100, 1557),
    ]:
        pp = [
            {"name": "emsc"},
            {"name": "wavelength_selector", "start_idx": start, "end_idx": end},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]
        experiments.append((
            f"emsc_wn_{wn_name}_deep",
            make_cfg(f"emsc_wn_{wn_name}", pp, "lgbm", dict(LGBM_DEEP)),
        ))

    # -----------------------------------------------------------
    # C: EMSC + feature eng for diversity
    # -----------------------------------------------------------
    pp_feat = [
        {"name": "emsc"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "band_ratio"},
        {"name": "spectral_stats", "n_regions": 10},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "emsc_feat_deep",
        make_cfg("emsc_feat_deep", pp_feat, "lgbm", dict(LGBM_DEEP)),
    ))

    pp_pca = [
        {"name": "emsc"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "pca_features", "n_components": 30},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "emsc_pca_deep",
        make_cfg("emsc_pca_deep", pp_pca, "lgbm", dict(LGBM_DEEP)),
    ))

    # -----------------------------------------------------------
    # D: SNV + EMSC combined (apply both corrections)
    # -----------------------------------------------------------
    pp_snv_emsc = [
        {"name": "snv"},
        {"name": "emsc"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "snv_emsc_deep",
        make_cfg("snv_emsc_deep", pp_snv_emsc, "lgbm", dict(LGBM_DEEP)),
    ))

    pp_emsc_snv = [
        {"name": "emsc"},
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "emsc_snv_deep",
        make_cfg("emsc_snv_deep", pp_emsc_snv, "lgbm", dict(LGBM_DEEP)),
    ))

    # -----------------------------------------------------------
    # E: EMSC + absorbance
    # -----------------------------------------------------------
    pp_abs_emsc = [
        {"name": "absorbance"},
        {"name": "emsc"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "abs_emsc_deep",
        make_cfg("abs_emsc_deep", pp_abs_emsc, "lgbm", dict(LGBM_DEEP)),
    ))

    # -----------------------------------------------------------
    # F: Very deep LGBM with high regularization + EMSC
    # -----------------------------------------------------------
    for depth, leaves in [(8, 63), (10, 100), (12, 200)]:
        experiments.append((
            f"emsc_verydeep_d{depth}_l{leaves}",
            make_cfg(f"emsc_verydeep_d{depth}", [
                {"name": "emsc"},
                {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
                {"name": "binning", "bin_size": 8},
                {"name": "standard_scaler"},
            ], "lgbm", {
                "n_estimators": 800, "learning_rate": 0.03, "max_depth": depth,
                "num_leaves": leaves, "min_child_samples": 5, "subsample": 0.7,
                "colsample_bytree": 0.5, "reg_alpha": 5.0, "reg_lambda": 10.0, "verbose": -1,
            }),
        ))

    # -----------------------------------------------------------
    # G: Bagging ensemble (same config, different seeds)
    # -----------------------------------------------------------
    for seed in range(20):
        experiments.append((
            f"emsc_bag_s{seed}",
            make_cfg(f"emsc_bag_s{seed}", [
                {"name": "emsc"},
                {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
                {"name": "binning", "bin_size": 8},
                {"name": "standard_scaler"},
            ], "lgbm", {
                **LGBM_DEEP, "random_state": seed * 17 + 3,
                "subsample": 0.6 + 0.02 * (seed % 5),
                "colsample_bytree": 0.4 + 0.02 * (seed % 5),
            }),
        ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 7: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 7 RESULTS (TOP 25)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Min':>8}  {'Max':>8}  Name")
    print("-" * 80)
    for rank, r in enumerate(results[:25], 1):
        fr = r.get("fold_rmses", [])
        fmin = min(fr) if fr else 0
        fmax = max(fr) if fr else 0
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fmin:>8.2f}  {fmax:>8.2f}  {r['name']}")

    # Seed ensemble analysis
    bag_results = [r for r in results if 'emsc_bag_s' in r['name'] and r['mean_rmse'] < 25]
    if bag_results:
        print(f"\n--- Bag Ensemble Analysis ---")
        import pandas as pd
        df_train = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
        y = df_train['含水率'].values
        from spectral_challenge.metrics import rmse

        bag_oofs = []
        for r in sorted(bag_results, key=lambda x: x['mean_rmse']):
            oof = np.load(Path(r['run_dir']) / 'oof_preds.npy')
            bag_oofs.append(oof)

        for k in [3, 5, 8, 10, 15, 20]:
            if k <= len(bag_oofs):
                avg = np.column_stack(bag_oofs[:k]).mean(axis=1)
                print(f"  Top-{k:2d} bag: {rmse(y, avg):.4f}")

    results_path = Path("runs") / "batch_phase7_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
