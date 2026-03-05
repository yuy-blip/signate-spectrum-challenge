#!/usr/bin/env python
"""Phase 4: Physics-driven approach — water absorption band focus.

Key insight: Test species are COMPLETELY different from train.
Must capture universal water-NIR relationship.

Water absorption bands in NIR (wavenumber):
- ~7000 cm⁻¹ (1st overtone OH stretch, ~1430nm)
- ~5200 cm⁻¹ (combination band, ~1920nm)
- ~4000-4500 cm⁻¹ (combination band, ~2200-2500nm)
- ~6900 cm⁻¹ (O-H stretch)
- ~5150 cm⁻¹ (H-O-H bend + O-H stretch)

Our wavelength range: 3999.82 — 9993.77 cm⁻¹
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

    # Water absorption band regions (in column indices — need to map from wavenumbers)
    # Our wavelengths: 9993.77 → 3999.82 cm⁻¹ (1557 features, descending)
    # Spacing: ~(9993.77 - 3999.82) / 1556 ≈ 3.85 cm⁻¹ per step

    # Water band 1: 6800-7200 cm⁻¹ → indices ~(9993.77-7200)/3.85 to (9993.77-6800)/3.85
    #   = ~725 to ~830
    # Water band 2: 5000-5400 cm⁻¹ → ~1194 to ~1298
    # Water band 3: 4000-4500 cm⁻¹ → ~1427 to ~1557

    # Wavelength selection configs
    # Focus on water bands
    WATER_BANDS = {"start_idx": 700, "end_idx": 1557}  # ~7100 to 3999 cm⁻¹
    WATER_NARROW = {"start_idx": 700, "end_idx": 1350}  # ~7100 to 5000 cm⁻¹

    # -----------------------------------------------------------
    # A: 2nd derivative — more species-invariant
    # -----------------------------------------------------------

    PP_2ND_DERIV = [
        {"name": "snv"},
        {"name": "sg", "window_length": 15, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    PP_2ND_DERIV_W7 = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    PP_2ND_DERIV_W11 = [
        {"name": "snv"},
        {"name": "sg", "window_length": 11, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    PP_2ND_DERIV_W21 = [
        {"name": "snv"},
        {"name": "sg", "window_length": 21, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    PP_2ND_DERIV_BIN16 = [
        {"name": "snv"},
        {"name": "sg", "window_length": 15, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 16},
        {"name": "standard_scaler"},
    ]

    # 1st derivative (our best so far)
    PP_1ST_W7_B16 = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16},
        {"name": "standard_scaler"},
    ]

    PP_1ST_W7_B8 = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    # Absorbance + derivative
    PP_ABS_1ST = [
        {"name": "absorbance"},
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    PP_ABS_2ND = [
        {"name": "absorbance"},
        {"name": "snv"},
        {"name": "sg", "window_length": 15, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    # Double derivative (1st then 2nd)
    PP_DOUBLE_DERIV = [
        {"name": "snv"},
        {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
        {"name": "sg", "window_length": 11, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    # MSC + 2nd derivative
    PP_MSC_2ND = [
        {"name": "msc"},
        {"name": "sg", "window_length": 15, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    # SNV only (no derivative)
    PP_SNV_ONLY_BIN8 = [
        {"name": "snv"},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    # LGBM best params
    LGBM_BEST = {
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3,
        "num_leaves": 7, "min_child_samples": 20, "subsample": 0.7,
        "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
    }

    LGBM_DEEP = {
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
        "num_leaves": 20, "min_child_samples": 10, "subsample": 0.7,
        "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 1.0, "verbose": -1,
    }

    XGB_BEST = {
        "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 3,
        "subsample": 0.7, "colsample_bytree": 0.5,
        "reg_alpha": 5.0, "reg_lambda": 10.0, "tree_method": "hist",
    }

    # A: 2nd derivative with various window sizes
    for pp_name, pp in [
        ("2nd_w7", PP_2ND_DERIV_W7),
        ("2nd_w11", PP_2ND_DERIV_W11),
        ("2nd_w15", PP_2ND_DERIV),
        ("2nd_w21", PP_2ND_DERIV_W21),
        ("2nd_w15_b16", PP_2ND_DERIV_BIN16),
    ]:
        for model_name, model_type, model_params in [
            ("lgbm", "lgbm", LGBM_BEST),
            ("lgbm_deep", "lgbm", LGBM_DEEP),
            ("xgb", "xgb", XGB_BEST),
        ]:
            experiments.append((
                f"p4_{pp_name}_{model_name}",
                make_cfg(f"p4_{pp_name}_{model_name}", pp, model_type, model_params),
            ))

    # B: Absorbance-based preprocessing
    for pp_name, pp in [("abs_1st", PP_ABS_1ST), ("abs_2nd", PP_ABS_2ND)]:
        for model_name, model_type, model_params in [
            ("lgbm", "lgbm", LGBM_BEST),
            ("xgb", "xgb", XGB_BEST),
        ]:
            experiments.append((
                f"p4_{pp_name}_{model_name}",
                make_cfg(f"p4_{pp_name}_{model_name}", pp, model_type, model_params),
            ))

    # C: MSC + 2nd derivative
    for model_name, model_type, model_params in [
        ("lgbm", "lgbm", LGBM_BEST),
        ("xgb", "xgb", XGB_BEST),
    ]:
        experiments.append((
            f"p4_msc_2nd_{model_name}",
            make_cfg(f"p4_msc_2nd_{model_name}", PP_MSC_2ND, model_type, model_params),
        ))

    # D: Wavelength selection — water bands only
    for wn_range_name, start, end in [
        ("water_wide", 700, 1557),   # 7100-3999 cm⁻¹
        ("water_narrow", 700, 1350), # 7100-5000 cm⁻¹
        ("water_peak1", 700, 850),   # 7100-6800 cm⁻¹
        ("water_peak2", 1150, 1350), # 5400-5000 cm⁻¹
        ("lower_half", 780, 1557),   # ~6993 to 3999 cm⁻¹
    ]:
        pp = [
            {"name": "wavelength_selector", "start_idx": start, "end_idx": end},
            {"name": "snv"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "standard_scaler"},
        ]
        experiments.append((
            f"p4_wn_{wn_range_name}_lgbm",
            make_cfg(f"p4_wn_{wn_range_name}", pp, "lgbm", LGBM_BEST),
        ))

    # E: SNV only (no derivative) to see if derivatives help
    experiments.append((
        "p4_snv_only_lgbm",
        make_cfg("p4_snv_only", PP_SNV_ONLY_BIN8, "lgbm", LGBM_BEST),
    ))

    # F: Double derivative
    experiments.append((
        "p4_double_deriv_lgbm",
        make_cfg("p4_double_deriv", PP_DOUBLE_DERIV, "lgbm", LGBM_BEST),
    ))

    # G: Wavelength selection + 2nd derivative combos
    for wn_name, start, end in [
        ("water_wide", 700, 1557),
        ("lower_half", 780, 1557),
    ]:
        pp = [
            {"name": "wavelength_selector", "start_idx": start, "end_idx": end},
            {"name": "snv"},
            {"name": "sg", "window_length": 15, "polyorder": 3, "deriv": 2},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]
        experiments.append((
            f"p4_wn_{wn_name}_2nd_lgbm",
            make_cfg(f"p4_wn_{wn_name}_2nd", pp, "lgbm", LGBM_BEST),
        ))

    # H: EMSC preprocessing (if available)
    for pp in [
        [{"name": "emsc"}, {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
         {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}],
    ]:
        experiments.append((
            "p4_emsc_1st_lgbm",
            make_cfg("p4_emsc_1st", pp, "lgbm", LGBM_BEST),
        ))

    # I: More regularized models for generalization
    for ra, rl in [(10, 50), (20, 100), (50, 200)]:
        experiments.append((
            f"p4_heavyreg_ra{ra}_rl{rl}",
            make_cfg(f"p4_heavyreg", PP_1ST_W7_B16, "lgbm", {
                "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 2,
                "num_leaves": 4, "min_child_samples": 30, "subsample": 0.5,
                "colsample_bytree": 0.3, "reg_alpha": ra, "reg_lambda": rl, "verbose": -1,
            }),
        ))

    # J: Feature interactions via band ratios + 1st deriv
    PP_FEAT_HEAVY = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16},
        {"name": "band_ratio"},
        {"name": "spectral_stats", "n_regions": 12, "stats": ["mean", "std", "slope", "max", "min", "skew"]},
        {"name": "pca_features", "n_components": 20},
        {"name": "standard_scaler"},
    ]
    experiments.append((
        "p4_feat_heavy_lgbm",
        make_cfg("p4_feat_heavy", PP_FEAT_HEAVY, "lgbm", LGBM_BEST),
    ))
    experiments.append((
        "p4_feat_heavy_xgb",
        make_cfg("p4_feat_heavy", PP_FEAT_HEAVY, "xgb", XGB_BEST),
    ))

    return experiments


def main():
    experiments = get_experiments()
    print(f"Phase 4: {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 4 RESULTS (TOP 20)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'RMSE':>10}  {'Min':>8}  {'Max':>8}  Name")
    print("-" * 80)
    for rank, r in enumerate(results[:25], 1):
        fr = r.get("fold_rmses", [])
        fmin = min(fr) if fr else 0
        fmax = max(fr) if fr else 0
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fmin:>8.2f}  {fmax:>8.2f}  {r['name']}")

    results_path = Path("runs") / "batch_phase4_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
