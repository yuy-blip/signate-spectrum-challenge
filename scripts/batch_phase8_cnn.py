#!/usr/bin/env python
"""Phase 8: 1D CNN experiments + diverse model stacking.

Goal: CNN captures different patterns than tree models → better stacking.
Also: CatBoost, wider EMSC search, final comprehensive stacking.
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
from spectral_challenge.metrics import rmse

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

    # Base EMSC preprocessing
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

    PP_SNV_SG1 = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "standard_scaler"},
    ]

    PP_SNV_SG1_BIN = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    # -----------------------------------------------------------
    # A: 1D CNN with EMSC preprocessing
    # -----------------------------------------------------------

    cnn_architectures = [
        # (name, n_filters, kernel_sizes, fc_sizes, dropout)
        ("small", [16, 32], [7, 5], [64], 0.3),
        ("medium", [32, 64, 128], [7, 5, 3], [128, 64], 0.3),
        ("large", [64, 128, 256], [11, 7, 5], [256, 128], 0.3),
        ("wide", [64, 128], [11, 7], [128, 64], 0.3),
        ("deep", [32, 64, 128, 256], [7, 5, 3, 3], [128, 64], 0.3),
        ("lowdrop", [32, 64, 128], [7, 5, 3], [128, 64], 0.1),
        ("highdrop", [32, 64, 128], [7, 5, 3], [128, 64], 0.5),
    ]

    for arch_name, n_filters, kernel_sizes, fc_sizes, dropout in cnn_architectures:
        for pp_name, pp in [("emsc", PP_EMSC), ("emsc_nobin", PP_EMSC_NOBIN)]:
            experiments.append((
                f"cnn_{arch_name}_{pp_name}",
                make_cfg(f"cnn_{arch_name}_{pp_name}", pp, "cnn1d", {
                    "n_filters": n_filters,
                    "kernel_sizes": kernel_sizes,
                    "fc_sizes": fc_sizes,
                    "dropout": dropout,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "epochs": 300,
                    "batch_size": 64,
                    "patience": 30,
                    "seed": 42,
                }),
            ))

    # -----------------------------------------------------------
    # B: CNN with different learning rates
    # -----------------------------------------------------------
    for lr in [5e-4, 1e-3, 2e-3, 5e-3]:
        experiments.append((
            f"cnn_med_lr{lr}_emsc",
            make_cfg(f"cnn_med_lr{lr}", PP_EMSC, "cnn1d", {
                "n_filters": [32, 64, 128],
                "kernel_sizes": [7, 5, 3],
                "fc_sizes": [128, 64],
                "dropout": 0.3,
                "lr": lr,
                "weight_decay": 1e-4,
                "epochs": 300,
                "batch_size": 64,
                "patience": 30,
                "seed": 42,
            }),
        ))

    # -----------------------------------------------------------
    # C: CNN with different weight decay
    # -----------------------------------------------------------
    for wd in [1e-5, 1e-4, 1e-3, 1e-2]:
        experiments.append((
            f"cnn_med_wd{wd}_emsc",
            make_cfg(f"cnn_med_wd{wd}", PP_EMSC, "cnn1d", {
                "n_filters": [32, 64, 128],
                "kernel_sizes": [7, 5, 3],
                "fc_sizes": [128, 64],
                "dropout": 0.3,
                "lr": 1e-3,
                "weight_decay": wd,
                "epochs": 300,
                "batch_size": 64,
                "patience": 30,
                "seed": 42,
            }),
        ))

    # -----------------------------------------------------------
    # D: CNN with SNV (for diversity in stacking)
    # -----------------------------------------------------------
    for arch_name, n_filters, kernel_sizes, fc_sizes, dropout in [
        ("small", [16, 32], [7, 5], [64], 0.3),
        ("medium", [32, 64, 128], [7, 5, 3], [128, 64], 0.3),
        ("large", [64, 128, 256], [11, 7, 5], [256, 128], 0.3),
    ]:
        experiments.append((
            f"cnn_{arch_name}_snv",
            make_cfg(f"cnn_{arch_name}_snv", PP_SNV_SG1, "cnn1d", {
                "n_filters": n_filters,
                "kernel_sizes": kernel_sizes,
                "fc_sizes": fc_sizes,
                "dropout": dropout,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 300,
                "batch_size": 64,
                "patience": 30,
                "seed": 42,
            }),
        ))

    # -----------------------------------------------------------
    # E: CNN multi-seed for bagging
    # -----------------------------------------------------------
    for seed in range(10):
        experiments.append((
            f"cnn_med_emsc_s{seed}",
            make_cfg(f"cnn_med_s{seed}", PP_EMSC, "cnn1d", {
                "n_filters": [32, 64, 128],
                "kernel_sizes": [7, 5, 3],
                "fc_sizes": [128, 64],
                "dropout": 0.3,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 300,
                "batch_size": 64,
                "patience": 30,
                "seed": seed,
            }),
        ))

    # -----------------------------------------------------------
    # F: CNN with batch size variations
    # -----------------------------------------------------------
    for bs in [32, 128, 256]:
        experiments.append((
            f"cnn_med_bs{bs}_emsc",
            make_cfg(f"cnn_med_bs{bs}", PP_EMSC, "cnn1d", {
                "n_filters": [32, 64, 128],
                "kernel_sizes": [7, 5, 3],
                "fc_sizes": [128, 64],
                "dropout": 0.3,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 300,
                "batch_size": bs,
                "patience": 30,
                "seed": 42,
            }),
        ))

    # -----------------------------------------------------------
    # G: CNN longer training
    # -----------------------------------------------------------
    experiments.append((
        "cnn_med_long_emsc",
        make_cfg("cnn_med_long", PP_EMSC, "cnn1d", {
            "n_filters": [32, 64, 128],
            "kernel_sizes": [7, 5, 3],
            "fc_sizes": [128, 64],
            "dropout": 0.3,
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "epochs": 500,
            "batch_size": 64,
            "patience": 50,
            "seed": 42,
        }),
    ))

    return experiments


def main():
    import pandas as pd

    experiments = get_experiments()
    print(f"Phase 8 (CNN): {len(experiments)} experiments")

    results = []
    for i, (name, cfg_dict) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        result = run_experiment(name, cfg_dict)
        if result:
            results.append(result)

    results.sort(key=lambda r: r["mean_rmse"])

    print("\n" + "=" * 80)
    print("  PHASE 8 (CNN) RESULTS (TOP 25)")
    print("=" * 80)
    for rank, r in enumerate(results[:25], 1):
        fr = r.get("fold_rmses", [])
        fmin = min(fr) if fr else 0
        fmax = max(fr) if fr else 0
        print(f"{rank:>4}  {r['mean_rmse']:>10.4f}  {fmin:>8.2f}  {fmax:>8.2f}  {r['name']}")

    # Save results
    results_path = Path("runs") / "batch_phase8_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Quick stacking test: best CNN + best LGBM
    print("\n--- CNN + LGBM Stacking Test ---")
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    y = df["含水率"].values

    # Load best CNN OOF
    cnn_results = [r for r in results if r["mean_rmse"] < 30 and "error" not in r]
    if cnn_results:
        best_cnn = cnn_results[0]
        cnn_oof = np.load(Path(best_cnn["run_dir"]) / "oof_preds.npy")
        print(f"Best CNN: {best_cnn['name']} RMSE={best_cnn['mean_rmse']:.4f}")

        # Load best LGBM EMSC OOF
        lgbm_runs = []
        for f in RUNS_DIR.glob("*/metrics.json"):
            if "emsc" in f.parent.name and "cnn" not in f.parent.name:
                with open(f) as fh:
                    m = json.load(fh)
                if m.get("mean_rmse", 999) < 20:
                    oof = np.load(f.parent / "oof_preds.npy")
                    lgbm_runs.append({"name": f.parent.name, "rmse": m["mean_rmse"], "oof": oof})

        lgbm_runs.sort(key=lambda x: x["rmse"])
        if lgbm_runs:
            best_lgbm_oof = lgbm_runs[0]["oof"]
            print(f"Best LGBM: {lgbm_runs[0]['name']} RMSE={lgbm_runs[0]['rmse']:.4f}")

            # Try various blend weights
            for w_cnn in [0.1, 0.2, 0.3, 0.4, 0.5]:
                blend = w_cnn * cnn_oof + (1 - w_cnn) * best_lgbm_oof
                r = rmse(y, blend)
                print(f"  CNN:{w_cnn:.1f} + LGBM:{1-w_cnn:.1f} = {r:.4f}")

            # Greedy add CNN to top LGBM models
            print("\n  Greedy: Add CNN to LGBM ensemble")
            # Top-4 LGBM avg
            top4_lgbm = np.column_stack([r["oof"] for r in lgbm_runs[:4]]).mean(axis=1)
            base_rmse = rmse(y, top4_lgbm)
            print(f"  Top-4 LGBM avg: {base_rmse:.4f}")

            for w_cnn in [0.05, 0.1, 0.15, 0.2, 0.3]:
                blend = w_cnn * cnn_oof + (1 - w_cnn) * top4_lgbm
                r = rmse(y, blend)
                print(f"  CNN:{w_cnn:.2f} + Top4LGBM:{1-w_cnn:.2f} = {r:.4f}")

            # Multi-CNN avg + LGBM
            if len(cnn_results) >= 3:
                cnn_avg = np.column_stack([
                    np.load(Path(r["run_dir"]) / "oof_preds.npy")
                    for r in cnn_results[:5]
                ]).mean(axis=1)
                cnn_avg_rmse = rmse(y, cnn_avg)
                print(f"\n  Top-5 CNN avg: {cnn_avg_rmse:.4f}")
                for w_cnn in [0.1, 0.2, 0.3]:
                    blend = w_cnn * cnn_avg + (1 - w_cnn) * top4_lgbm
                    r = rmse(y, blend)
                    print(f"  CNNavg:{w_cnn:.2f} + Top4LGBM:{1-w_cnn:.2f} = {r:.4f}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
