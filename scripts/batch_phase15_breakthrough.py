#!/usr/bin/env python
"""Phase 15: Breakthrough experiments — combining insights from Gemini, ChatGPT, and domain knowledge.

KEY EXPERIMENTS:
A. Wet Basis Transform (物理ベースのターゲット変換)
B. SNV + SG → PLS (EMSCなし — PLS破壊仮説の検証)
C. PLS残差学習 (PLS base + LGBM residual)
D. 外挿型擬似ラベル (PLS/Ridgeで200%超の擬似ラベル生成)
E. Adversarial Validation (樹種バレ波長の除去)
F. Band ratio特徴量
G. Water Difference Vector (物理ベースのデータ合成)
H. OPLS (直交PLS)
I. Combined best approaches
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")

# ============================================================================
# Common utilities
# ============================================================================

def load_data():
    cfg = Config(
        train_file="train.csv", test_file="test.csv",
        id_col="sample number", target_col="含水率",
        group_col="species number",
    )
    X_train, y_train, ids = load_train(cfg, DATA_DIR)
    X_test, test_ids = load_test(cfg, DATA_DIR)
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    groups = df["species number"].values
    return X_train, y_train, groups, X_test, test_ids


def save_result(name, oof_preds, fold_rmses, y_train, extra_metrics=None):
    mean_score = rmse(y_train, oof_preds)
    run_dir = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "oof_preds.npy", oof_preds)
    metrics = {"mean_rmse": float(mean_score),
               "fold_rmses": [float(x) for x in fold_rmses]}
    if extra_metrics:
        metrics.update(extra_metrics)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return mean_score, run_dir


BEST_PREPROCESS = [
    {"name": "emsc", "poly_order": 2},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

LGBM_PARAMS = {
    "n_estimators": 400, "max_depth": 5, "num_leaves": 20,
    "learning_rate": 0.05, "min_child_samples": 20,
    "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "verbose": -1, "n_jobs": -1,
}


# ============================================================================
# Wet Basis Transform utilities
# ============================================================================

def dry_to_wet(y_dry):
    """Convert dry-basis moisture (%) to wet-basis (0-100%)."""
    return (y_dry / (100.0 + y_dry)) * 100.0

def wet_to_dry(y_wet):
    """Convert wet-basis moisture (%) back to dry-basis (%)."""
    y_wet = np.clip(y_wet, 0, 99.9)  # prevent division by zero
    return (100.0 * y_wet) / (100.0 - y_wet)


# ============================================================================
# Section A: Wet Basis Transform
# ============================================================================

def section_a_wet_basis(X_train, y_train, groups, results):
    """Test wet-basis target transform with LGBM."""
    print("\n=== Section A: Wet Basis Transform ===")
    gkf = GroupKFold(n_splits=5)

    # A1: Wet basis + best LGBM pipeline
    for label, preprocess_cfg in [
        ("wb_emsc", BEST_PREPROCESS),
        ("wb_snv_sg", [
            {"name": "snv"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
    ]:
        for model_type, model_params in [
            ("lgbm", LGBM_PARAMS),
            ("lgbm", {**LGBM_PARAMS, "n_estimators": 800, "learning_rate": 0.02}),
        ]:
            name = f"{label}_lr{model_params['learning_rate']}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            y_wet = dry_to_wet(y_train)

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr_wet = y_wet[train_idx]
                y_val = y_train[val_idx]

                pipe = build_preprocess_pipeline(preprocess_cfg)
                X_tr_t = pipe.fit_transform(X_tr)
                X_val_t = pipe.transform(X_val)

                model = create_model(model_type, model_params)
                model.fit(X_tr_t, y_tr_wet)
                pred_wet = model.predict(X_val_t).ravel()
                pred_dry = wet_to_dry(pred_wet)

                oof[val_idx] = pred_dry
                fold_rmses.append(rmse(y_val, pred_dry))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append(("A_wet_basis", name, score, fold_rmses))


# ============================================================================
# Section B: SNV + SG → PLS (without EMSC — test PLS destruction hypothesis)
# ============================================================================

def section_b_pls_proper(X_train, y_train, groups, results):
    """Test PLS with proper NIR preprocessing (no EMSC)."""
    print("\n=== Section B: PLS with proper preprocessing (no EMSC) ===")
    gkf = GroupKFold(n_splits=5)

    # Different preprocessing pipelines for PLS
    pls_pipelines = {
        "snv_sg1_b8": [
            {"name": "snv"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ],
        "snv_sg2_b8": [
            {"name": "snv"},
            {"name": "sg", "window_length": 11, "polyorder": 3, "deriv": 2},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ],
        "msc_sg1_b8": [
            {"name": "msc"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ],
        "sg1_only_b8": [
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ],
        "snv_sg1_b16": [
            {"name": "snv"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 16},
            {"name": "standard_scaler"},
        ],
        "snv_sg1_nobin": [
            {"name": "snv"},
            {"name": "sg", "window_length": 15, "polyorder": 2, "deriv": 1},
            {"name": "standard_scaler"},
        ],
    }

    for pipe_name, preprocess_cfg in pls_pipelines.items():
        for nc in [2, 3, 5, 8, 10, 15, 20]:
            name = f"pls_{pipe_name}_nc{nc}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                pipe = build_preprocess_pipeline(preprocess_cfg)
                X_tr_t = pipe.fit_transform(X_tr)
                X_val_t = pipe.transform(X_val)

                model = PLSRegression(n_components=nc, max_iter=1000)
                model.fit(X_tr_t, y_tr)
                pred = model.predict(X_val_t).ravel()

                oof[val_idx] = pred
                fold_rmses.append(rmse(y_val, pred))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append(("B_pls", name, score, fold_rmses))

    # Also test PLS with wet basis
    print("\n  --- PLS + Wet Basis ---")
    y_wet = dry_to_wet(y_train)
    for nc in [3, 5, 10, 15]:
        name = f"pls_snv_sg1_b8_wb_nc{nc}"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr_wet = y_wet[train_idx]
            y_val = y_train[val_idx]

            pipe = build_preprocess_pipeline(pls_pipelines["snv_sg1_b8"])
            X_tr_t = pipe.fit_transform(X_tr)
            X_val_t = pipe.transform(X_val)

            model = PLSRegression(n_components=nc, max_iter=1000)
            model.fit(X_tr_t, y_tr_wet)
            pred_wet = model.predict(X_val_t).ravel()
            pred_dry = wet_to_dry(pred_wet)

            oof[val_idx] = pred_dry
            fold_rmses.append(rmse(y_val, pred_dry))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append(("B_pls_wb", name, score, fold_rmses))


# ============================================================================
# Section C: PLS Residual Learning (PLS base + LGBM residual)
# ============================================================================

def section_c_residual_learning(X_train, y_train, groups, results):
    """PLS predicts base trend (can extrapolate), LGBM corrects residual."""
    print("\n=== Section C: PLS Residual Learning ===")
    gkf = GroupKFold(n_splits=5)

    # Use SNV pipeline for PLS (not EMSC)
    pls_preprocess = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    for nc in [2, 3, 5, 8, 10]:
        for use_wet in [False, True]:
            wb_suffix = "_wb" if use_wet else ""
            name = f"resid_pls{nc}{wb_suffix}_lgbm"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            y_target = dry_to_wet(y_train) if use_wet else y_train

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr = y_target[train_idx]
                y_val_orig = y_train[val_idx]

                # Step 1: PLS base prediction
                pipe_pls = build_preprocess_pipeline(pls_preprocess)
                X_tr_pls = pipe_pls.fit_transform(X_tr)
                X_val_pls = pipe_pls.transform(X_val)

                pls = PLSRegression(n_components=nc, max_iter=1000)
                pls.fit(X_tr_pls, y_tr)
                pls_train_pred = pls.predict(X_tr_pls).ravel()
                pls_val_pred = pls.predict(X_val_pls).ravel()

                # Step 2: LGBM on residual
                residual = y_tr - pls_train_pred

                pipe_lgbm = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_lgbm = pipe_lgbm.fit_transform(X_tr)
                X_val_lgbm = pipe_lgbm.transform(X_val)

                lgbm = create_model("lgbm", LGBM_PARAMS)
                lgbm.fit(X_tr_lgbm, residual)
                lgbm_val_pred = lgbm.predict(X_val_lgbm).ravel()

                # Step 3: Combine
                combined = pls_val_pred + lgbm_val_pred

                if use_wet:
                    combined = wet_to_dry(combined)

                oof[val_idx] = combined
                fold_rmses.append(rmse(y_val_orig, combined))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append(("C_residual", name, score, fold_rmses))

    # Also test Ridge residual
    for alpha in [10.0, 100.0, 1000.0]:
        for use_wet in [False, True]:
            wb_suffix = "_wb" if use_wet else ""
            name = f"resid_ridge{alpha:.0f}{wb_suffix}_lgbm"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            y_target = dry_to_wet(y_train) if use_wet else y_train

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr = y_target[train_idx]
                y_val_orig = y_train[val_idx]

                # Ridge base (use SNV pipeline)
                pipe_ridge = build_preprocess_pipeline(pls_preprocess)
                X_tr_r = pipe_ridge.fit_transform(X_tr)
                X_val_r = pipe_ridge.transform(X_val)

                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_r, y_tr)
                ridge_train_pred = ridge.predict(X_tr_r).ravel()
                ridge_val_pred = ridge.predict(X_val_r).ravel()

                # LGBM on residual
                residual = y_tr - ridge_train_pred

                pipe_lgbm = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_lgbm = pipe_lgbm.fit_transform(X_tr)
                X_val_lgbm = pipe_lgbm.transform(X_val)

                lgbm = create_model("lgbm", LGBM_PARAMS)
                lgbm.fit(X_tr_lgbm, residual)
                lgbm_val_pred = lgbm.predict(X_val_lgbm).ravel()

                combined = ridge_val_pred + lgbm_val_pred
                if use_wet:
                    combined = wet_to_dry(combined)

                oof[val_idx] = combined
                fold_rmses.append(rmse(y_val_orig, combined))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append(("C_residual_ridge", name, score, fold_rmses))


# ============================================================================
# Section D: Extrapolation-aware pseudo-labels
# ============================================================================

def section_d_extrapolation_pseudolabels(X_train, y_train, groups, X_test, results):
    """Use PLS/Ridge (which CAN extrapolate) to generate pseudo-labels >200%."""
    print("\n=== Section D: Extrapolation-aware Pseudo-labels ===")
    gkf = GroupKFold(n_splits=5)

    pls_preprocess = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    # Step 1: Generate pseudo-labels with PLS (can extrapolate)
    print("\n  --- Generating PLS-based pseudo-labels ---")
    for nc in [3, 5, 10]:
        pipe_full = build_preprocess_pipeline(pls_preprocess)
        X_all = np.vstack([X_train, X_test])
        X_all_t = pipe_full.fit_transform(X_all)
        X_train_t = X_all_t[:len(y_train)]
        X_test_t = X_all_t[len(y_train):]

        pls = PLSRegression(n_components=nc, max_iter=1000)
        pls.fit(X_train_t, y_train)
        pls_test_pred = pls.predict(X_test_t).ravel()
        print(f"  PLS nc={nc}: test range=[{pls_test_pred.min():.1f}, {pls_test_pred.max():.1f}]")

        # Also with wet basis
        y_wet = dry_to_wet(y_train)
        pls_wb = PLSRegression(n_components=nc, max_iter=1000)
        pls_wb.fit(X_train_t, y_wet)
        pls_test_pred_wb = wet_to_dry(pls_wb.predict(X_test_t).ravel())
        print(f"  PLS nc={nc} (wb): test range=[{pls_test_pred_wb.min():.1f}, {pls_test_pred_wb.max():.1f}]")

    # Step 2: Generate LGBM pseudo-labels for comparison
    print("\n  --- Generating LGBM pseudo-labels ---")
    pipe_lgbm_full = build_preprocess_pipeline(BEST_PREPROCESS)
    X_all = np.vstack([X_train, X_test])
    X_all_t_lgbm = pipe_lgbm_full.fit_transform(X_all)
    X_train_t_lgbm = X_all_t_lgbm[:len(y_train)]
    X_test_t_lgbm = X_all_t_lgbm[len(y_train):]

    lgbm_full = create_model("lgbm", LGBM_PARAMS)
    lgbm_full.fit(X_train_t_lgbm, y_train)
    lgbm_test_pred = lgbm_full.predict(X_test_t_lgbm).ravel()
    print(f"  LGBM: test range=[{lgbm_test_pred.min():.1f}, {lgbm_test_pred.max():.1f}]")

    # Step 3: Blend PLS + LGBM pseudo-labels
    print("\n  --- Testing pseudo-label strategies ---")

    pseudo_strategies = {}
    # Strategy 1: PLS-only pseudo-labels
    for nc in [5, 10]:
        pipe = build_preprocess_pipeline(pls_preprocess)
        X_all_t = pipe.fit_transform(np.vstack([X_train, X_test]))
        pls_m = PLSRegression(n_components=nc, max_iter=1000)
        pls_m.fit(X_all_t[:len(y_train)], y_train)
        pseudo_strategies[f"pls{nc}"] = pls_m.predict(X_all_t[len(y_train):]).ravel()

    # Strategy 2: LGBM pseudo-labels
    pseudo_strategies["lgbm"] = lgbm_test_pred

    # Strategy 3: Blended (LGBM for most, PLS for high-value extrapolation)
    for nc in [5, 10]:
        pls_pred = pseudo_strategies[f"pls{nc}"]
        blended = lgbm_test_pred.copy()
        # Where PLS predicts > LGBM max, use PLS value (extrapolation)
        lgbm_max = lgbm_test_pred.max()
        extrap_mask = pls_pred > lgbm_max * 0.95
        blended[extrap_mask] = 0.5 * lgbm_test_pred[extrap_mask] + 0.5 * pls_pred[extrap_mask]
        pseudo_strategies[f"blend_pls{nc}"] = blended
        print(f"  Blend PLS{nc}: {extrap_mask.sum()} samples extrapolated, range=[{blended.min():.1f}, {blended.max():.1f}]")

    # Strategy 4: PLS wet-basis pseudo-labels
    for nc in [5, 10]:
        pipe = build_preprocess_pipeline(pls_preprocess)
        X_all_t = pipe.fit_transform(np.vstack([X_train, X_test]))
        y_wet = dry_to_wet(y_train)
        pls_m = PLSRegression(n_components=nc, max_iter=1000)
        pls_m.fit(X_all_t[:len(y_train)], y_wet)
        pred_wet = pls_m.predict(X_all_t[len(y_train):]).ravel()
        pseudo_strategies[f"pls{nc}_wb"] = wet_to_dry(pred_wet)
        print(f"  PLS{nc}_wb: range=[{pseudo_strategies[f'pls{nc}_wb'].min():.1f}, {pseudo_strategies[f'pls{nc}_wb'].max():.1f}]")

    # Run CV for each strategy
    for strat_name, test_pseudo in pseudo_strategies.items():
        for pw in [0.3, 0.5, 1.0]:
            name = f"pl_{strat_name}_w{pw}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                X_tr_aug = np.vstack([X_tr, X_test])
                y_tr_aug = np.concatenate([y_tr, test_pseudo])
                weights = np.concatenate([np.ones(len(y_tr)), np.full(len(test_pseudo), pw)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_t = pipe.fit_transform(X_tr_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model("lgbm", LGBM_PARAMS)
                model.fit(X_tr_t, y_tr_aug, sample_weight=weights)
                pred = model.predict(X_val_t).ravel()

                oof[val_idx] = pred
                fold_rmses.append(rmse(y_val, pred))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append(("D_pseudo", name, score, fold_rmses))

    # Also test: wet basis + pseudo-labels
    print("\n  --- Wet basis + pseudo-labels ---")
    for strat_name in ["lgbm", "blend_pls5"]:
        test_pseudo = pseudo_strategies[strat_name]
        test_pseudo_wet = dry_to_wet(test_pseudo)
        y_wet = dry_to_wet(y_train)

        for pw in [0.3, 0.5]:
            name = f"pl_wb_{strat_name}_w{pw}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_val_orig = y_train[val_idx]

                X_tr_aug = np.vstack([X_tr, X_test])
                y_tr_aug = np.concatenate([y_wet[train_idx], test_pseudo_wet])
                weights = np.concatenate([np.ones(len(train_idx)), np.full(len(test_pseudo_wet), pw)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_t = pipe.fit_transform(X_tr_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model("lgbm", LGBM_PARAMS)
                model.fit(X_tr_t, y_tr_aug, sample_weight=weights)
                pred_wet = model.predict(X_val_t).ravel()
                pred_dry = wet_to_dry(pred_wet)

                oof[val_idx] = pred_dry
                fold_rmses.append(rmse(y_val_orig, pred_dry))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append(("D_pseudo_wb", name, score, fold_rmses))


# ============================================================================
# Section E: Adversarial Validation
# ============================================================================

def section_e_adversarial_validation(X_train, y_train, groups, X_test, results):
    """Remove features that distinguish train vs test (species-specific wavelengths)."""
    print("\n=== Section E: Adversarial Validation ===")
    gkf = GroupKFold(n_splits=5)

    # Step 1: Build adversarial classifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_val_score

    # Preprocess first
    pipe_av = build_preprocess_pipeline(BEST_PREPROCESS)
    X_all_raw = np.vstack([X_train, X_test])
    X_all_t = pipe_av.fit_transform(X_all_raw)

    labels = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    clf = LGBMClassifier(n_estimators=200, max_depth=3, verbose=-1, n_jobs=-1)
    auc_scores = cross_val_score(clf, X_all_t, labels, cv=5, scoring="roc_auc")
    print(f"  Adversarial AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    # Get feature importances
    clf.fit(X_all_t, labels)
    importances = clf.feature_importances_

    # Try removing top N% most discriminative features
    for remove_pct in [10, 20, 30, 50]:
        threshold = np.percentile(importances, 100 - remove_pct)
        keep_mask = importances < threshold
        n_keep = keep_mask.sum()
        print(f"  Removing top {remove_pct}% features: keeping {n_keep}/{len(importances)}")

        name = f"av_rm{remove_pct}pct"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            pipe = build_preprocess_pipeline(BEST_PREPROCESS)
            X_tr_t = pipe.fit_transform(X_tr)
            X_val_t = pipe.transform(X_val)

            # Apply feature mask
            X_tr_t = X_tr_t[:, keep_mask]
            X_val_t = X_val_t[:, keep_mask]

            model = create_model("lgbm", LGBM_PARAMS)
            model.fit(X_tr_t, y_tr)
            pred = model.predict(X_val_t).ravel()

            oof[val_idx] = pred
            fold_rmses.append(rmse(y_val, pred))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append(("E_av", name, score, fold_rmses))

    # AV + wet basis
    print("\n  --- AV + Wet Basis ---")
    threshold_20 = np.percentile(importances, 80)
    keep_mask_20 = importances < threshold_20

    for remove_pct, mask in [(20, keep_mask_20)]:
        name = f"av_rm{remove_pct}_wb"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []
        y_wet = dry_to_wet(y_train)

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_val_orig = y_train[val_idx]

            pipe = build_preprocess_pipeline(BEST_PREPROCESS)
            X_tr_t = pipe.fit_transform(X_tr)
            X_val_t = pipe.transform(X_val)

            X_tr_t = X_tr_t[:, mask]
            X_val_t = X_val_t[:, mask]

            model = create_model("lgbm", LGBM_PARAMS)
            model.fit(X_tr_t, y_wet[train_idx])
            pred_wet = model.predict(X_val_t).ravel()
            pred_dry = wet_to_dry(pred_wet)

            oof[val_idx] = pred_dry
            fold_rmses.append(rmse(y_val_orig, pred_dry))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append(("E_av_wb", name, score, fold_rmses))


# ============================================================================
# Section F: Band Ratio Features
# ============================================================================

def section_f_band_ratios(X_train, y_train, groups, results):
    """Add physically-motivated band ratio features for water."""
    print("\n=== Section F: Band Ratio Features ===")
    gkf = GroupKFold(n_splits=5)

    # Get wavelength info from data
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932", nrows=1)
    float_cols = [c for c in df.columns if c.replace('.', '').replace('-', '').isdigit() or
                  (c.replace('.', '', 1).isdigit())]
    wavelengths = np.array([float(c) for c in float_cols if '.' in c or c.isdigit()])
    # These are wavenumbers (cm⁻¹), sort to find indices near water bands
    print(f"  Wavelength range: {wavelengths.min():.0f} - {wavelengths.max():.0f} cm⁻¹")

    # Find indices closest to water absorption bands
    def find_nearest_idx(arr, val):
        return np.argmin(np.abs(arr - val))

    water_bands = {
        "5200": find_nearest_idx(wavelengths, 5200),
        "6900": find_nearest_idx(wavelengths, 6900),
        "5000": find_nearest_idx(wavelengths, 5000),
        "7100": find_nearest_idx(wavelengths, 7100),
        "4500": find_nearest_idx(wavelengths, 4500),
    }
    print(f"  Water band indices: {water_bands}")

    def add_band_ratios(X):
        """Add band ratio features to X."""
        extra = []
        # Ratio: ~5200 / ~6900
        extra.append(X[:, water_bands["5200"]] / (X[:, water_bands["6900"]] + 1e-10))
        # Difference
        extra.append(X[:, water_bands["5200"]] - X[:, water_bands["6900"]])
        # Ratio: ~5000 / ~7100
        extra.append(X[:, water_bands["5000"]] / (X[:, water_bands["7100"]] + 1e-10))
        # Normalized difference
        s52 = X[:, water_bands["5200"]]
        s69 = X[:, water_bands["6900"]]
        extra.append((s52 - s69) / (s52 + s69 + 1e-10))
        # Slope between water bands
        extra.append((s52 - s69) / (6900 - 5200))
        return np.column_stack([X] + extra)

    # Test with band ratios added to standard pipeline
    for label, preprocess_cfg in [
        ("br_emsc", BEST_PREPROCESS),
        ("br_snv", [
            {"name": "snv"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
    ]:
        name = f"{label}_lgbm"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Add band ratios BEFORE preprocessing (on raw spectra)
            X_tr_br = add_band_ratios(X_tr)
            X_val_br = add_band_ratios(X_val)

            # Preprocess (band ratios are at the end, binning won't touch them properly)
            # So we process spectra separately, then add ratios
            pipe = build_preprocess_pipeline(preprocess_cfg)
            X_tr_t = pipe.fit_transform(X_tr)
            X_val_t = pipe.transform(X_val)

            # Add raw band ratios (from raw spectra)
            from sklearn.preprocessing import StandardScaler
            raw_ratios_tr = X_tr_br[:, -5:]
            raw_ratios_val = X_val_br[:, -5:]
            scaler_br = StandardScaler()
            raw_ratios_tr_s = scaler_br.fit_transform(raw_ratios_tr)
            raw_ratios_val_s = scaler_br.transform(raw_ratios_val)

            X_tr_final = np.hstack([X_tr_t, raw_ratios_tr_s])
            X_val_final = np.hstack([X_val_t, raw_ratios_val_s])

            model = create_model("lgbm", LGBM_PARAMS)
            model.fit(X_tr_final, y_tr)
            pred = model.predict(X_val_final).ravel()

            oof[val_idx] = pred
            fold_rmses.append(rmse(y_val, pred))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append(("F_band_ratio", name, score, fold_rmses))


# ============================================================================
# Section G: Water Difference Vector (Physical Data Augmentation)
# ============================================================================

def section_g_water_difference(X_train, y_train, groups, results):
    """Synthesize high-moisture spectra using water difference vectors."""
    print("\n=== Section G: Water Difference Vector Augmentation ===")
    gkf = GroupKFold(n_splits=5)

    # For each species, compute a "water vector" from high-low moisture pairs
    unique_species = np.unique(groups)
    print(f"  Species: {unique_species}")

    # Build augmented data per fold
    for n_aug in [20, 50, 100]:
        for extrap_factor in [1.0, 1.5, 2.0]:
            name = f"wdv_n{n_aug}_f{extrap_factor}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                g_tr = groups[train_idx]

                # Generate synthetic high-moisture samples
                synth_X = []
                synth_y = []

                for sp in np.unique(g_tr):
                    sp_mask = g_tr == sp
                    X_sp = X_tr[sp_mask]
                    y_sp = y_tr[sp_mask]

                    if len(y_sp) < 5:
                        continue

                    # Find high and low moisture samples
                    high_idx = np.argsort(y_sp)[-5:]  # top 5
                    low_idx = np.argsort(y_sp)[:5]   # bottom 5

                    # Compute water difference vectors
                    for hi in high_idx:
                        for lo in low_idx:
                            delta_y = y_sp[hi] - y_sp[lo]
                            if delta_y < 10:
                                continue
                            delta_x = X_sp[hi] - X_sp[lo]  # water vector

                            # Extrapolate: add water vector to high samples
                            new_x = X_sp[hi] + extrap_factor * delta_x
                            new_y = y_sp[hi] + extrap_factor * delta_y
                            synth_X.append(new_x)
                            synth_y.append(new_y)

                if not synth_X:
                    # fallback: use LGBM without augmentation
                    pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                    X_tr_t = pipe.fit_transform(X_tr)
                    X_val_t = pipe.transform(X_val)
                    model = create_model("lgbm", LGBM_PARAMS)
                    model.fit(X_tr_t, y_tr)
                    oof[val_idx] = model.predict(X_val_t).ravel()
                    fold_rmses.append(rmse(y_val, oof[val_idx]))
                    continue

                synth_X = np.array(synth_X)
                synth_y = np.array(synth_y)

                # Randomly sample n_aug from synthetic data
                if len(synth_X) > n_aug:
                    idx = np.random.choice(len(synth_X), n_aug, replace=False)
                    synth_X = synth_X[idx]
                    synth_y = synth_y[idx]

                print(f"    Fold {fold_idx}: {len(synth_X)} synthetic, y_range=[{synth_y.min():.0f}, {synth_y.max():.0f}]")

                # Combine
                X_tr_aug = np.vstack([X_tr, synth_X])
                y_tr_aug = np.concatenate([y_tr, synth_y])
                weights = np.concatenate([np.ones(len(y_tr)), np.full(len(synth_y), 0.5)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_t = pipe.fit_transform(X_tr_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model("lgbm", LGBM_PARAMS)
                model.fit(X_tr_t, y_tr_aug, sample_weight=weights)
                pred = model.predict(X_val_t).ravel()

                oof[val_idx] = pred
                fold_rmses.append(rmse(y_val, pred))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append(("G_wdv", name, score, fold_rmses))


# ============================================================================
# Section H: OPLS (Orthogonal PLS)
# ============================================================================

def section_h_opls(X_train, y_train, groups, results):
    """Test OPLS — remove orthogonal (species/scatter) variance."""
    print("\n=== Section H: OPLS-like approach ===")
    gkf = GroupKFold(n_splits=5)

    # Manual OPLS implementation (no pyopls dependency needed)
    # OPLS = PLS + orthogonal signal correction (OSC)
    # Simplified: use PLS to find predictive direction, project out orthogonal

    class SimpleOPLS:
        """Simplified OPLS: PLS + orthogonal signal correction."""
        def __init__(self, n_components=5, n_ortho=3):
            self.n_components = n_components
            self.n_ortho = n_ortho

        def fit(self, X, y):
            self.X_mean_ = X.mean(axis=0)
            self.X_std_ = X.std(axis=0)
            self.X_std_[self.X_std_ == 0] = 1.0

            X_c = (X - self.X_mean_) / self.X_std_
            y_c = y - y.mean()
            self.y_mean_ = y.mean()

            # OSC: remove directions orthogonal to y
            self.ortho_weights_ = []
            self.ortho_loadings_ = []

            for _ in range(self.n_ortho):
                # Weight vector w = X'y / ||X'y||
                w = X_c.T @ y_c
                w = w / (np.linalg.norm(w) + 1e-10)

                # Score t = Xw
                t = X_c @ w

                # Orthogonal weight: remove variation in X not correlated with t
                # Loading p = X't / (t't)
                p = X_c.T @ t / (t @ t + 1e-10)

                # Orthogonal component: direction in X space orthogonal to w
                p_orth = p - w * (w @ p)  # project out the predictive direction
                p_orth = p_orth / (np.linalg.norm(p_orth) + 1e-10)

                # Orthogonal score
                t_orth = X_c @ p_orth

                # Remove orthogonal component
                X_c = X_c - np.outer(t_orth, p_orth)

                self.ortho_weights_.append(p_orth)
                self.ortho_loadings_.append(p_orth)

            # Fit PLS on cleaned X
            self.pls_ = PLSRegression(n_components=min(self.n_components, X_c.shape[1]),
                                       max_iter=1000)
            self.pls_.fit(X_c, y)
            return self

        def predict(self, X):
            X_c = (X - self.X_mean_) / self.X_std_

            # Remove orthogonal components
            for p_orth in self.ortho_weights_:
                t_orth = X_c @ p_orth
                X_c = X_c - np.outer(t_orth, p_orth)

            return self.pls_.predict(X_c).ravel()

    pls_preprocess = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    for nc in [3, 5, 10]:
        for n_ortho in [1, 2, 3, 5]:
            name = f"opls_nc{nc}_no{n_ortho}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                pipe = build_preprocess_pipeline(pls_preprocess)
                X_tr_t = pipe.fit_transform(X_tr)
                X_val_t = pipe.transform(X_val)

                model = SimpleOPLS(n_components=nc, n_ortho=n_ortho)
                model.fit(X_tr_t, y_tr)
                pred = model.predict(X_val_t)

                oof[val_idx] = pred
                fold_rmses.append(rmse(y_val, pred))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append(("H_opls", name, score, fold_rmses))


# ============================================================================
# Section I: Combined Best Approaches
# ============================================================================

def section_i_combined(X_train, y_train, groups, X_test, results):
    """Combine the most promising approaches."""
    print("\n=== Section I: Combined Approaches ===")
    gkf = GroupKFold(n_splits=5)

    # I1: Wet basis + pseudo-labels + multi-seed ensemble
    print("\n  --- I1: Wet basis + pseudo-labels + multi-seed ---")

    # First generate pseudo-labels with full training data
    pipe_full = build_preprocess_pipeline(BEST_PREPROCESS)
    X_all = np.vstack([X_train, X_test])
    X_all_t = pipe_full.fit_transform(X_all)
    lgbm_full = create_model("lgbm", LGBM_PARAMS)
    lgbm_full.fit(X_all_t[:len(y_train)], y_train)
    test_pseudo = lgbm_full.predict(X_all_t[len(y_train):]).ravel()
    test_pseudo_wet = dry_to_wet(test_pseudo)
    y_wet = dry_to_wet(y_train)

    for pw in [0.3, 0.5]:
        seed_oofs = []
        for seed in range(5):
            params = dict(LGBM_PARAMS)
            params["random_state"] = seed * 7 + 42
            params["bagging_seed"] = seed * 11 + 17

            oof = np.full(len(y_train), np.nan)
            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]

                X_tr_aug = np.vstack([X_tr, X_test])
                y_tr_aug = np.concatenate([y_wet[train_idx], test_pseudo_wet])
                weights = np.concatenate([np.ones(len(train_idx)), np.full(len(test_pseudo_wet), pw)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_t = pipe.fit_transform(X_tr_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model("lgbm", params)
                model.fit(X_tr_t, y_tr_aug, sample_weight=weights)
                pred_wet = model.predict(X_val_t).ravel()
                oof[val_idx] = wet_to_dry(pred_wet)

            seed_oofs.append(oof)

        avg_oof = np.mean(seed_oofs, axis=0)
        score = rmse(y_train, avg_oof)
        fold_rmses_comb = []
        for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            fold_rmses_comb.append(rmse(y_train[val_idx], avg_oof[val_idx]))

        name = f"combined_wb_pl_5seed_w{pw}"
        save_result(name, avg_oof, fold_rmses_comb, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses_comb)}]")
        results.append(("I_combined", name, score, fold_rmses_comb))

    # I2: Residual learning + wet basis + pseudo-labels
    print("\n  --- I2: PLS residual + wet basis + pseudo ---")
    pls_preprocess = [
        {"name": "snv"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]

    for nc in [3, 5]:
        name = f"resid_pls{nc}_wb_pl"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_val_orig = y_train[val_idx]

            # PLS base (wet basis)
            pipe_pls = build_preprocess_pipeline(pls_preprocess)
            X_tr_pls = pipe_pls.fit_transform(X_tr)
            X_val_pls = pipe_pls.transform(X_val)

            pls = PLSRegression(n_components=nc, max_iter=1000)
            pls.fit(X_tr_pls, y_wet[train_idx])
            pls_train_pred = pls.predict(X_tr_pls).ravel()
            pls_val_pred = pls.predict(X_val_pls).ravel()

            # LGBM residual with pseudo-labels
            residual = y_wet[train_idx] - pls_train_pred

            # Get PLS prediction for test too
            pipe_pls2 = build_preprocess_pipeline(pls_preprocess)
            X_all_pls = pipe_pls2.fit_transform(np.vstack([X_tr, X_test]))
            pls2 = PLSRegression(n_components=nc, max_iter=1000)
            pls2.fit(X_all_pls[:len(train_idx)], y_wet[train_idx])
            test_pls_pred = pls2.predict(X_all_pls[len(train_idx):]).ravel()
            test_residual = test_pseudo_wet - test_pls_pred

            X_tr_aug = np.vstack([X_tr, X_test])
            resid_aug = np.concatenate([residual, test_residual])
            weights = np.concatenate([np.ones(len(train_idx)), np.full(len(X_test), 0.3)])

            pipe_lgbm = build_preprocess_pipeline(BEST_PREPROCESS)
            X_tr_lgbm = pipe_lgbm.fit_transform(X_tr_aug)
            X_val_lgbm = pipe_lgbm.transform(X_val)

            lgbm = create_model("lgbm", LGBM_PARAMS)
            lgbm.fit(X_tr_lgbm, resid_aug, sample_weight=weights)
            lgbm_val_pred = lgbm.predict(X_val_lgbm).ravel()

            combined_wet = pls_val_pred + lgbm_val_pred
            combined_dry = wet_to_dry(combined_wet)

            oof[val_idx] = combined_dry
            fold_rmses.append(rmse(y_val_orig, combined_dry))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append(("I_combined_resid", name, score, fold_rmses))

    # I3: Wet basis only (no pseudo-labels, for clean comparison)
    print("\n  --- I3: Wet basis + various LGBM configs ---")
    for n_est, lr in [(400, 0.05), (800, 0.02), (1500, 0.01)]:
        name = f"wb_lgbm_n{n_est}_lr{lr}"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        params = dict(LGBM_PARAMS)
        params["n_estimators"] = n_est
        params["learning_rate"] = lr

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_val_orig = y_train[val_idx]

            pipe = build_preprocess_pipeline(BEST_PREPROCESS)
            X_tr_t = pipe.fit_transform(X_tr)
            X_val_t = pipe.transform(X_val)

            model = create_model("lgbm", params)
            model.fit(X_tr_t, y_wet[train_idx])
            pred_wet = model.predict(X_val_t).ravel()
            pred_dry = wet_to_dry(pred_wet)

            oof[val_idx] = pred_dry
            fold_rmses.append(rmse(y_val_orig, pred_dry))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append(("I_wb_lgbm", name, score, fold_rmses))


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 15: Breakthrough Experiments")
    print("=" * 70)

    np.random.seed(42)

    X_train, y_train, groups, X_test, test_ids = load_data()
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"y range: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"y wet range: [{dry_to_wet(y_train).min():.1f}, {dry_to_wet(y_train).max():.1f}]")
    print(f"Baseline (best so far): CV RMSE = 16.14")

    results = []

    try:
        section_a_wet_basis(X_train, y_train, groups, results)
    except Exception as e:
        print(f"  Section A ERROR: {e}")
        traceback.print_exc()

    try:
        section_b_pls_proper(X_train, y_train, groups, results)
    except Exception as e:
        print(f"  Section B ERROR: {e}")
        traceback.print_exc()

    try:
        section_c_residual_learning(X_train, y_train, groups, results)
    except Exception as e:
        print(f"  Section C ERROR: {e}")
        traceback.print_exc()

    try:
        section_d_extrapolation_pseudolabels(X_train, y_train, groups, X_test, results)
    except Exception as e:
        print(f"  Section D ERROR: {e}")
        traceback.print_exc()

    try:
        section_e_adversarial_validation(X_train, y_train, groups, X_test, results)
    except Exception as e:
        print(f"  Section E ERROR: {e}")
        traceback.print_exc()

    try:
        section_f_band_ratios(X_train, y_train, groups, results)
    except Exception as e:
        print(f"  Section F ERROR: {e}")
        traceback.print_exc()

    try:
        section_g_water_difference(X_train, y_train, groups, results)
    except Exception as e:
        print(f"  Section G ERROR: {e}")
        traceback.print_exc()

    try:
        section_h_opls(X_train, y_train, groups, results)
    except Exception as e:
        print(f"  Section H ERROR: {e}")
        traceback.print_exc()

    try:
        section_i_combined(X_train, y_train, groups, X_test, results)
    except Exception as e:
        print(f"  Section I ERROR: {e}")
        traceback.print_exc()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("PHASE 15 FINAL SUMMARY")
    print("=" * 70)
    results.sort(key=lambda x: x[2])
    for section, name, score, folds in results:
        fold_str = ', '.join(f'{f:.1f}' for f in folds)
        marker = " ★" if score < 16.14 else ""
        print(f"  {score:.4f}  [{fold_str}]  {section} / {name}{marker}")

    if results:
        best = results[0]
        print(f"\n{'='*40}")
        print(f"BEST: {best[2]:.4f} ({best[0]} / {best[1]})")
        print(f"Baseline: 16.14 (Phase 12 pseudo-label)")
        improvement = 16.14 - best[2]
        print(f"Improvement: {improvement:+.4f}")
        print(f"{'='*40}")

        # Count improvements
        improved = [r for r in results if r[2] < 16.14]
        print(f"\n{len(improved)} / {len(results)} experiments beat baseline 16.14")


if __name__ == "__main__":
    main()
