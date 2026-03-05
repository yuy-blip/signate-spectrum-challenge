#!/usr/bin/env python
"""Phase 15c: Targeted WDV + Iterative PL — the breakthrough combo.

Phase 15b discovered:
- WDV + PL → RMSE 15.58 (Fold 2: 25.2) — HUGE improvement
- Iterative PL (5 rounds) → 15.85
- d7/l32 LGBM + PL → 15.87

This phase focuses on:
1. Targeted WDV: only synthesize from high-moisture (>100%) samples
2. WDV-enhanced pseudo-labels: use WDV-trained model to generate better PLs
3. Multi-round: WDV → PL → WDV → PL iterations
4. Hyperparameter sweep for the WDV+PL pipeline
5. Grand ensemble of WDV+PL variants
6. Ordinal expectation (classification approach)
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
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")

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


def save_result(name, oof, fold_rmses, y_train, extra=None):
    score = rmse(y_train, oof)
    run_dir = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "oof_preds.npy", oof)
    metrics = {"mean_rmse": float(score), "fold_rmses": [float(x) for x in fold_rmses]}
    if extra:
        metrics.update(extra)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return score, run_dir


def generate_wdv(X_tr, y_tr, groups_tr, n_aug, extrap_factor, min_moisture=0):
    """Generate Water Difference Vector synthetic data.

    Args:
        min_moisture: Only use source samples with moisture > this value.
                      Set >100 for "targeted WDV" to only extrapolate high-moisture samples.
    """
    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]

        if len(y_sp) < 3:
            continue

        # Filter: only use high-moisture samples as base
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 2:
            continue

        # Top samples to extrapolate from
        sorted_idx = np.argsort(y_sp)
        high_idx = sorted_idx[high_mask[sorted_idx]][-5:]  # top 5 above threshold
        low_idx = sorted_idx[:5]  # bottom 5 (for computing water vector)

        for hi in high_idx:
            for lo in low_idx:
                dy = y_sp[hi] - y_sp[lo]
                if dy < 10:
                    continue
                dx = X_sp[hi] - X_sp[lo]
                synth_X.append(X_sp[hi] + extrap_factor * dx)
                synth_y.append(y_sp[hi] + extrap_factor * dy)

    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)

    synth_X = np.array(synth_X)
    synth_y = np.array(synth_y)

    if len(synth_X) > n_aug:
        idx = np.random.choice(len(synth_X), n_aug, replace=False)
        synth_X, synth_y = synth_X[idx], synth_y[idx]

    return synth_X, synth_y


def generate_pseudo_labels(X_train, y_train, X_test, preprocess_cfg, model_type, model_params,
                           synth_X=None, synth_y=None, synth_weight=0.3):
    """Generate pseudo-labels, optionally with WDV-augmented training."""
    pipe = build_preprocess_pipeline(preprocess_cfg)

    if synth_X is not None and len(synth_X) > 0:
        X_full = np.vstack([X_train, synth_X, X_test])
        y_full = np.concatenate([y_train, synth_y])
        w_full = np.concatenate([np.ones(len(y_train)), np.full(len(synth_y), synth_weight)])
    else:
        X_full = np.vstack([X_train, X_test])
        y_full = y_train
        w_full = np.ones(len(y_train))

    X_full_t = pipe.fit_transform(X_full)
    n_train = len(y_full)

    model = create_model(model_type, model_params)
    try:
        model.fit(X_full_t[:n_train], y_full, sample_weight=w_full)
    except TypeError:
        model.fit(X_full_t[:n_train], y_full)

    return model.predict(X_full_t[n_train:]).ravel()


# ============================================================================
# Section 1: Targeted WDV (high-moisture only)
# ============================================================================

def section1_targeted_wdv(X_train, y_train, groups, X_test, results):
    """Targeted WDV: only extrapolate from high-moisture samples."""
    print("\n=== Section 1: Targeted WDV ===")
    gkf = GroupKFold(n_splits=5)

    # Generate base pseudo-labels
    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)
    print(f"  Base pseudo range: [{pseudo.min():.1f}, {pseudo.max():.1f}]")

    for min_moist in [50, 80, 100, 120, 150]:
        for n_aug in [10, 20, 50]:
            for ef in [0.5, 1.0, 1.5]:
                for pw in [0.5, 1.0]:
                    name = f"twdv_m{min_moist}_n{n_aug}_f{ef}_pw{pw}"
                    oof = np.full(len(y_train), np.nan)
                    fold_rmses = []

                    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]
                        g_tr = groups[train_idx]

                        synth_X, synth_y = generate_wdv(X_tr, y_tr, g_tr, n_aug, ef, min_moisture=min_moist)

                        if len(synth_X) > 0:
                            X_aug = np.vstack([X_tr, X_test, synth_X])
                            y_aug = np.concatenate([y_tr, pseudo, synth_y])
                            w = np.concatenate([
                                np.ones(len(y_tr)),
                                np.full(len(pseudo), pw),
                                np.full(len(synth_y), 0.3),
                            ])
                        else:
                            X_aug = np.vstack([X_tr, X_test])
                            y_aug = np.concatenate([y_tr, pseudo])
                            w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw)])

                        pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                        X_tr_t = pipe.fit_transform(X_aug)
                        X_val_t = pipe.transform(X_val)

                        model = create_model("lgbm", LGBM_PARAMS)
                        model.fit(X_tr_t, y_aug, sample_weight=w)
                        oof[val_idx] = model.predict(X_val_t).ravel()
                        fold_rmses.append(rmse(y_val, oof[val_idx]))

                    score, _ = save_result(name, oof, fold_rmses, y_train)
                    print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
                    results.append((name, score, fold_rmses))


# ============================================================================
# Section 2: WDV-enhanced pseudo-labels
# ============================================================================

def section2_wdv_enhanced_pl(X_train, y_train, groups, X_test, results):
    """Use WDV-augmented model to generate better pseudo-labels with higher range."""
    print("\n=== Section 2: WDV-enhanced Pseudo-labels ===")
    gkf = GroupKFold(n_splits=5)

    # Generate WDV synthetic data from full training set
    synth_X, synth_y = generate_wdv(X_train, y_train, groups, n_aug=30, extrap_factor=1.0, min_moisture=80)
    print(f"  WDV synthetic: {len(synth_y)} samples, y_range=[{synth_y.min():.0f}, {synth_y.max():.0f}]")

    # Generate enhanced pseudo-labels using WDV-augmented model
    pseudo_wdv = generate_pseudo_labels(
        X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS,
        synth_X=synth_X, synth_y=synth_y, synth_weight=0.3
    )
    pseudo_base = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    print(f"  Base pseudo range: [{pseudo_base.min():.1f}, {pseudo_base.max():.1f}]")
    print(f"  WDV-enhanced pseudo range: [{pseudo_wdv.min():.1f}, {pseudo_wdv.max():.1f}]")

    # Test different WDV-derived pseudo-label strategies
    for pseudo_name, pseudo in [("pl_base", pseudo_base), ("pl_wdv", pseudo_wdv)]:
        for n_aug in [0, 10, 20, 30]:
            for ef in [0.5, 1.0]:
                for pw in [0.5, 1.0]:
                    min_m = 80
                    name = f"wdv_epl_{pseudo_name}_n{n_aug}_f{ef}_pw{pw}"
                    oof = np.full(len(y_train), np.nan)
                    fold_rmses = []

                    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]
                        g_tr = groups[train_idx]

                        if n_aug > 0:
                            s_X, s_y = generate_wdv(X_tr, y_tr, g_tr, n_aug, ef, min_moisture=min_m)
                        else:
                            s_X, s_y = np.empty((0, X_tr.shape[1])), np.empty(0)

                        if len(s_X) > 0:
                            X_aug = np.vstack([X_tr, X_test, s_X])
                            y_aug = np.concatenate([y_tr, pseudo, s_y])
                            w = np.concatenate([
                                np.ones(len(y_tr)),
                                np.full(len(pseudo), pw),
                                np.full(len(s_y), 0.3),
                            ])
                        else:
                            X_aug = np.vstack([X_tr, X_test])
                            y_aug = np.concatenate([y_tr, pseudo])
                            w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw)])

                        pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                        X_tr_t = pipe.fit_transform(X_aug)
                        X_val_t = pipe.transform(X_val)

                        model = create_model("lgbm", LGBM_PARAMS)
                        model.fit(X_tr_t, y_aug, sample_weight=w)
                        oof[val_idx] = model.predict(X_val_t).ravel()
                        fold_rmses.append(rmse(y_val, oof[val_idx]))

                    score, _ = save_result(name, oof, fold_rmses, y_train)
                    print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
                    results.append((name, score, fold_rmses))


# ============================================================================
# Section 3: Multi-round WDV → PL iterations
# ============================================================================

def section3_iterative_wdv_pl(X_train, y_train, groups, X_test, results):
    """Iterative: WDV-model → pseudo-labels → retrain → refined pseudo-labels."""
    print("\n=== Section 3: Iterative WDV + PL ===")
    gkf = GroupKFold(n_splits=5)

    for n_rounds in [2, 3, 5]:
        for n_aug in [20, 30]:
            for pw in [0.5, 1.0]:
                name = f"iter_wdv_pl_r{n_rounds}_n{n_aug}_pw{pw}"

                # Round 0: initial pseudo-labels with WDV
                synth_X, synth_y = generate_wdv(X_train, y_train, groups, n_aug, 1.0, min_moisture=80)
                pseudo = generate_pseudo_labels(
                    X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS,
                    synth_X=synth_X, synth_y=synth_y
                )
                print(f"  {name} Round 0: pseudo range=[{pseudo.min():.1f}, {pseudo.max():.1f}]")

                # Iterate
                for r in range(1, n_rounds):
                    X_aug = np.vstack([X_train, X_test])
                    y_aug = np.concatenate([y_train, pseudo])
                    w = np.concatenate([np.ones(len(y_train)), np.full(len(pseudo), pw)])

                    if len(synth_X) > 0:
                        X_aug = np.vstack([X_aug, synth_X])
                        y_aug = np.concatenate([y_aug, synth_y])
                        w = np.concatenate([w, np.full(len(synth_y), 0.3)])

                    pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                    X_aug_t = pipe.fit_transform(X_aug)

                    model = create_model("lgbm", LGBM_PARAMS)
                    model.fit(X_aug_t, y_aug, sample_weight=w)

                    X_test_t = pipe.transform(X_test)
                    pseudo = model.predict(X_test_t).ravel()
                    print(f"  {name} Round {r}: pseudo range=[{pseudo.min():.1f}, {pseudo.max():.1f}]")

                # Final CV
                oof = np.full(len(y_train), np.nan)
                fold_rmses = []
                for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    g_tr = groups[train_idx]

                    s_X, s_y = generate_wdv(X_tr, y_tr, g_tr, n_aug, 1.0, min_moisture=80)

                    if len(s_X) > 0:
                        X_aug = np.vstack([X_tr, X_test, s_X])
                        y_aug = np.concatenate([y_tr, pseudo, s_y])
                        w = np.concatenate([
                            np.ones(len(y_tr)),
                            np.full(len(pseudo), pw),
                            np.full(len(s_y), 0.3),
                        ])
                    else:
                        X_aug = np.vstack([X_tr, X_test])
                        y_aug = np.concatenate([y_tr, pseudo])
                        w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw)])

                    pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                    X_tr_t = pipe.fit_transform(X_aug)
                    X_val_t = pipe.transform(X_val)

                    model = create_model("lgbm", LGBM_PARAMS)
                    model.fit(X_tr_t, y_aug, sample_weight=w)
                    oof[val_idx] = model.predict(X_val_t).ravel()
                    fold_rmses.append(rmse(y_val, oof[val_idx]))

                score, _ = save_result(name, oof, fold_rmses, y_train)
                print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
                results.append((name, score, fold_rmses))


# ============================================================================
# Section 4: Best config sweep for WDV+PL
# ============================================================================

def section4_wdv_pl_lgbm_sweep(X_train, y_train, groups, X_test, results):
    """Hyperparameter sweep on the best WDV+PL pipeline."""
    print("\n=== Section 4: LGBM Sweep with WDV+PL ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    configs = [
        ("d7_l32", {**LGBM_PARAMS, "max_depth": 7, "num_leaves": 32}),
        ("d7_l64", {**LGBM_PARAMS, "max_depth": 7, "num_leaves": 64}),
        ("n800_lr02", {**LGBM_PARAMS, "n_estimators": 800, "learning_rate": 0.02}),
        ("n800_lr02_d7", {**LGBM_PARAMS, "n_estimators": 800, "learning_rate": 0.02, "max_depth": 7, "num_leaves": 32}),
        ("n1500_lr01", {**LGBM_PARAMS, "n_estimators": 1500, "learning_rate": 0.01}),
        ("n400_ss08", {**LGBM_PARAMS, "subsample": 0.8, "colsample_bytree": 0.8}),
        ("n400_mcs10", {**LGBM_PARAMS, "min_child_samples": 10}),
        ("n400_d5_l31", {**LGBM_PARAMS, "num_leaves": 31}),
    ]

    for config_name, params in configs:
        name = f"wdvpl_{config_name}"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            g_tr = groups[train_idx]

            s_X, s_y = generate_wdv(X_tr, y_tr, g_tr, 20, 1.0, min_moisture=80)

            if len(s_X) > 0:
                X_aug = np.vstack([X_tr, X_test, s_X])
                y_aug = np.concatenate([y_tr, pseudo, s_y])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0), np.full(len(s_y), 0.3)])
            else:
                X_aug = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

            pipe = build_preprocess_pipeline(BEST_PREPROCESS)
            X_tr_t = pipe.fit_transform(X_aug)
            X_val_t = pipe.transform(X_val)

            model = create_model("lgbm", params)
            model.fit(X_tr_t, y_aug, sample_weight=w)
            oof[val_idx] = model.predict(X_val_t).ravel()
            fold_rmses.append(rmse(y_val, oof[val_idx]))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append((name, score, fold_rmses))


# ============================================================================
# Section 5: Grand ensemble of WDV+PL variants
# ============================================================================

def section5_grand_wdv_pl_ensemble(X_train, y_train, groups, X_test, results):
    """Grand ensemble: multi-seed × multi-config × WDV+PL."""
    print("\n=== Section 5: Grand WDV+PL Ensemble ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    all_oofs = []

    configs = [
        LGBM_PARAMS,
        {**LGBM_PARAMS, "max_depth": 7, "num_leaves": 32},
        {**LGBM_PARAMS, "n_estimators": 800, "learning_rate": 0.02},
    ]

    pipelines = [
        BEST_PREPROCESS,
        [{"name": "snv"}, {"name": "emsc", "poly_order": 2},
         {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
         {"name": "binning", "bin_size": 8}, {"name": "standard_scaler"}],
    ]

    total = 0
    for pipe_cfg in pipelines:
        for params in configs:
            for seed in range(5):
                p = dict(params)
                p["random_state"] = seed * 7 + 42
                p["bagging_seed"] = seed * 11 + 17

                oof = np.full(len(y_train), np.nan)
                for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr = y_train[train_idx]
                    g_tr = groups[train_idx]

                    s_X, s_y = generate_wdv(X_tr, y_tr, g_tr, 20, 1.0, min_moisture=80)

                    if len(s_X) > 0:
                        X_aug = np.vstack([X_tr, X_test, s_X])
                        y_aug = np.concatenate([y_tr, pseudo, s_y])
                        w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0), np.full(len(s_y), 0.3)])
                    else:
                        X_aug = np.vstack([X_tr, X_test])
                        y_aug = np.concatenate([y_tr, pseudo])
                        w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

                    pipe = build_preprocess_pipeline(pipe_cfg)
                    X_tr_t = pipe.fit_transform(X_aug)
                    X_val_t = pipe.transform(X_val)

                    model = create_model("lgbm", p)
                    model.fit(X_tr_t, y_aug, sample_weight=w)
                    oof[val_idx] = model.predict(X_val_t).ravel()

                all_oofs.append(oof)
                total += 1

    # Add XGBoost diversity
    for seed in range(3):
        xgb_params = {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbosity": 0, "n_jobs": -1, "random_state": seed * 13 + 7,
        }
        oof = np.full(len(y_train), np.nan)
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]
            g_tr = groups[train_idx]

            s_X, s_y = generate_wdv(X_tr, y_tr, g_tr, 20, 1.0, min_moisture=80)

            if len(s_X) > 0:
                X_aug = np.vstack([X_tr, X_test, s_X])
                y_aug = np.concatenate([y_tr, pseudo, s_y])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0), np.full(len(s_y), 0.3)])
            else:
                X_aug = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

            pipe = build_preprocess_pipeline(BEST_PREPROCESS)
            X_tr_t = pipe.fit_transform(X_aug)
            X_val_t = pipe.transform(X_val)

            model = create_model("xgb", xgb_params)
            model.fit(X_tr_t, y_aug, sample_weight=w)
            oof[val_idx] = model.predict(X_val_t).ravel()

        all_oofs.append(oof)
        total += 1

    # Average
    grand_oof = np.mean(all_oofs, axis=0)
    fold_rmses = []
    for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_rmses.append(rmse(y_train[val_idx], grand_oof[val_idx]))
    score = rmse(y_train, grand_oof)
    save_result("grand_wdv_pl_ensemble", grand_oof, fold_rmses, y_train)
    print(f"  Grand WDV+PL Ensemble ({total} models): RMSE={score:.4f}")
    print(f"    folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
    results.append(("grand_wdv_pl_ensemble", score, fold_rmses))

    # Optimized weights
    from scipy.optimize import minimize

    def ens_rmse(weights):
        weights = np.abs(weights) / np.abs(weights).sum()
        pred = sum(w * o for w, o in zip(weights, all_oofs))
        return rmse(y_train, pred)

    w0 = np.ones(total) / total
    res = minimize(ens_rmse, w0, method="Nelder-Mead", options={"maxiter": 5000})
    opt_w = np.abs(res.x) / np.abs(res.x).sum()
    opt_oof = sum(w * o for w, o in zip(opt_w, all_oofs))

    fold_rmses_opt = []
    for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_rmses_opt.append(rmse(y_train[val_idx], opt_oof[val_idx]))
    score_opt = rmse(y_train, opt_oof)
    save_result("grand_wdv_pl_opt", opt_oof, fold_rmses_opt, y_train)
    print(f"  Grand WDV+PL Optimized: RMSE={score_opt:.4f}")
    print(f"    folds=[{', '.join(f'{f:.1f}' for f in fold_rmses_opt)}]")
    results.append(("grand_wdv_pl_opt", score_opt, fold_rmses_opt))


# ============================================================================
# Section 6: Ordinal Expectation (regression as classification)
# ============================================================================

def section6_ordinal_expectation(X_train, y_train, groups, X_test, results):
    """Turn regression into classification with expectation-based prediction."""
    print("\n=== Section 6: Ordinal Expectation ===")
    from lightgbm import LGBMClassifier
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    # Create bins
    for n_bins in [15, 20, 30]:
        # Create bin edges from combined train + pseudo data
        all_y = np.concatenate([y_train, pseudo])
        bin_edges = np.percentile(all_y, np.linspace(0, 100, n_bins + 1))
        # Add extra bins for extrapolation
        bin_edges = np.concatenate([[0], bin_edges[1:-1], [400]])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        n_classes = len(bin_centers)

        name = f"ordinal_{n_bins}bins"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # With pseudo-labels
            X_aug = np.vstack([X_tr, X_test])
            y_aug = np.concatenate([y_tr, pseudo])
            w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

            # Discretize targets
            y_binned = np.digitize(y_aug, bin_edges[1:])
            y_binned = np.clip(y_binned, 0, n_classes - 1)

            pipe = build_preprocess_pipeline(BEST_PREPROCESS)
            X_tr_t = pipe.fit_transform(X_aug)
            X_val_t = pipe.transform(X_val)

            clf = LGBMClassifier(
                n_estimators=400, max_depth=5, num_leaves=20,
                learning_rate=0.05, min_child_samples=20,
                subsample=0.7, colsample_bytree=0.7,
                verbose=-1, n_jobs=-1, num_class=n_classes,
                objective="multiclass",
            )
            clf.fit(X_tr_t, y_binned, sample_weight=w)

            # Predict probabilities
            proba = clf.predict_proba(X_val_t)  # (n_val, n_classes)
            # Expected value
            pred = proba @ bin_centers

            oof[val_idx] = pred
            fold_rmses.append(rmse(y_val, pred))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append((name, score, fold_rmses))

    # Ordinal + WDV
    print("\n  --- Ordinal + WDV ---")
    for n_bins in [20, 30]:
        all_y = np.concatenate([y_train, pseudo])
        bin_edges = np.percentile(all_y, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.concatenate([[0], bin_edges[1:-1], [400]])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        n_classes = len(bin_centers)

        name = f"ordinal_{n_bins}bins_wdv"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            g_tr = groups[train_idx]

            s_X, s_y = generate_wdv(X_tr, y_tr, g_tr, 20, 1.0, min_moisture=80)

            if len(s_X) > 0:
                X_aug = np.vstack([X_tr, X_test, s_X])
                y_aug = np.concatenate([y_tr, pseudo, s_y])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0), np.full(len(s_y), 0.3)])
            else:
                X_aug = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

            y_binned = np.digitize(y_aug, bin_edges[1:])
            y_binned = np.clip(y_binned, 0, n_classes - 1)

            pipe = build_preprocess_pipeline(BEST_PREPROCESS)
            X_tr_t = pipe.fit_transform(X_aug)
            X_val_t = pipe.transform(X_val)

            clf = LGBMClassifier(
                n_estimators=400, max_depth=5, num_leaves=20,
                learning_rate=0.05, min_child_samples=20,
                subsample=0.7, colsample_bytree=0.7,
                verbose=-1, n_jobs=-1, num_class=n_classes,
                objective="multiclass",
            )
            clf.fit(X_tr_t, y_binned, sample_weight=w)
            proba = clf.predict_proba(X_val_t)
            pred = proba @ bin_centers

            oof[val_idx] = pred
            fold_rmses.append(rmse(y_val, pred))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append((name, score, fold_rmses))


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 15c: Targeted WDV + Iterative PL")
    print("=" * 70)

    np.random.seed(42)
    X_train, y_train, groups, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Current best: RMSE = 15.58 (WDV+PL from Phase 15b)")

    results = []

    for section_name, fn in [
        ("Section 1: Targeted WDV", section1_targeted_wdv),
        ("Section 2: WDV-enhanced PL", section2_wdv_enhanced_pl),
        ("Section 3: Iterative WDV+PL", section3_iterative_wdv_pl),
        ("Section 4: LGBM Sweep", section4_wdv_pl_lgbm_sweep),
        ("Section 5: Grand Ensemble", section5_grand_wdv_pl_ensemble),
        ("Section 6: Ordinal Expectation", section6_ordinal_expectation),
    ]:
        try:
            fn(X_train, y_train, groups, X_test, results)
        except Exception as e:
            print(f"  {section_name} ERROR: {e}")
            traceback.print_exc()

    # FINAL SUMMARY
    print("\n\n" + "=" * 70)
    print("PHASE 15c FINAL SUMMARY")
    print("=" * 70)
    results.sort(key=lambda x: x[1])
    for name, score, folds in results[:30]:
        fold_str = ', '.join(f'{f:.1f}' for f in folds)
        marker = " ★★" if score < 15.0 else " ★" if score < 15.58 else " ●" if score < 16.04 else ""
        print(f"  {score:.4f}  [{fold_str}]  {name}{marker}")

    if results:
        best = results[0]
        print(f"\nBEST: {best[1]:.4f} ({best[0]})")
        print(f"Phase 15b best: 15.58")
        print(f"Phase 15 best: 16.04")
        print(f"Phase 12 best: 16.14")
        print(f"No-PL baseline: 17.75")


if __name__ == "__main__":
    main()
