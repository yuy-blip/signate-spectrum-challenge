#!/usr/bin/env python
"""Phase 15b: Refined experiments based on Phase 15 findings.

KEY FINDINGS FROM PHASE 15:
- PLS is useless for this task (cross-species generalization)
- Wet basis alone hurts (inverse transform amplifies Fold 2 errors)
- Pseudo-labels w=1.0 gave marginal improvement (16.04 vs 16.14)
- WDV n=20 f=1.0 improved Fold 2 to 27.4 (from ~30.7)
- Adversarial validation AUC=0.85 but feature removal didn't help

NEW EXPERIMENTS:
1. Iterative pseudo-labeling (multiple rounds)
2. WDV + pseudo-labels combination
3. LGBM hyperparameter tuning with pseudo-labels
4. Multi-seed ensemble with pseudo-labels
5. Stacking: different preprocessing pipelines
6. Confidence-weighted pseudo-labels with variance estimation
7. SNV+EMSC (double correction) + pseudo-labels
8. Feature selection: LGBM importance-based
9. Larger ensembles
10. XGBoost + CatBoost pseudo-labels
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


def generate_pseudo_labels(X_train, y_train, X_test, preprocess_cfg, model_type, model_params):
    """Generate pseudo-labels using full training data."""
    pipe = build_preprocess_pipeline(preprocess_cfg)
    X_all = np.vstack([X_train, X_test])
    X_all_t = pipe.fit_transform(X_all)

    model = create_model(model_type, model_params)
    model.fit(X_all_t[:len(y_train)], y_train)
    return model.predict(X_all_t[len(y_train):]).ravel()


# ============================================================================
# Section 1: Iterative pseudo-labeling
# ============================================================================

def section1_iterative_pl(X_train, y_train, groups, X_test, results):
    """Multi-round pseudo-labeling: each round improves labels."""
    print("\n=== Section 1: Iterative Pseudo-labeling (multiple rounds) ===")
    gkf = GroupKFold(n_splits=5)

    for n_rounds in [2, 3, 5]:
        for pw in [0.5, 1.0]:
            name = f"iter_pl_r{n_rounds}_w{pw}"

            # Generate initial pseudo-labels
            pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)
            print(f"  Round 0: pseudo range=[{pseudo.min():.1f}, {pseudo.max():.1f}]")

            # Iteratively refine
            for r in range(1, n_rounds):
                X_aug = np.vstack([X_train, X_test])
                y_aug = np.concatenate([y_train, pseudo])
                weights = np.concatenate([np.ones(len(y_train)), np.full(len(pseudo), pw)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_aug_t = pipe.fit_transform(X_aug)

                model = create_model("lgbm", LGBM_PARAMS)
                model.fit(X_aug_t, y_aug, sample_weight=weights)

                # Re-predict test with refined model
                X_test_t = pipe.transform(X_test)
                pseudo = model.predict(X_test_t).ravel()
                print(f"  Round {r}: pseudo range=[{pseudo.min():.1f}, {pseudo.max():.1f}]")

            # Final CV with refined pseudo-labels
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []
            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

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
# Section 2: WDV + pseudo-labels
# ============================================================================

def section2_wdv_pseudo(X_train, y_train, groups, X_test, results):
    """Combine water difference vector augmentation with pseudo-labels."""
    print("\n=== Section 2: WDV + Pseudo-labels ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    for n_aug in [10, 20]:
        for extrap_factor in [0.5, 1.0]:
            for pw in [0.5, 1.0]:
                name = f"wdv_pl_n{n_aug}_f{extrap_factor}_w{pw}"
                oof = np.full(len(y_train), np.nan)
                fold_rmses = []

                for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    g_tr = groups[train_idx]

                    # Generate WDV synthetic data
                    synth_X, synth_y = [], []
                    for sp in np.unique(g_tr):
                        sp_mask = g_tr == sp
                        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
                        if len(y_sp) < 5:
                            continue
                        high_idx = np.argsort(y_sp)[-3:]
                        low_idx = np.argsort(y_sp)[:3]
                        for hi in high_idx:
                            for lo in low_idx:
                                dy = y_sp[hi] - y_sp[lo]
                                if dy < 10:
                                    continue
                                dx = X_sp[hi] - X_sp[lo]
                                synth_X.append(X_sp[hi] + extrap_factor * dx)
                                synth_y.append(y_sp[hi] + extrap_factor * dy)

                    if synth_X:
                        synth_X = np.array(synth_X)
                        synth_y = np.array(synth_y)
                        if len(synth_X) > n_aug:
                            idx = np.random.choice(len(synth_X), n_aug, replace=False)
                            synth_X, synth_y = synth_X[idx], synth_y[idx]

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
# Section 3: LGBM hyperparameter sweep with pseudo-labels
# ============================================================================

def section3_lgbm_tuning(X_train, y_train, groups, X_test, results):
    """Try different LGBM configs with pseudo-labels."""
    print("\n=== Section 3: LGBM Tuning with Pseudo-labels ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    configs = [
        ("n800_lr02", {**LGBM_PARAMS, "n_estimators": 800, "learning_rate": 0.02}),
        ("n1500_lr01", {**LGBM_PARAMS, "n_estimators": 1500, "learning_rate": 0.01}),
        ("n400_d7_l32", {**LGBM_PARAMS, "max_depth": 7, "num_leaves": 32}),
        ("n400_d3_l12", {**LGBM_PARAMS, "max_depth": 3, "num_leaves": 12}),
        ("n400_mcs10", {**LGBM_PARAMS, "min_child_samples": 10}),
        ("n400_mcs30", {**LGBM_PARAMS, "min_child_samples": 30}),
        ("n400_ss08", {**LGBM_PARAMS, "subsample": 0.8, "colsample_bytree": 0.8}),
        ("n400_ss06", {**LGBM_PARAMS, "subsample": 0.6, "colsample_bytree": 0.6}),
        ("n400_ra05_rl05", {**LGBM_PARAMS, "reg_alpha": 0.5, "reg_lambda": 0.5}),
        ("n400_ra001_rl01", {**LGBM_PARAMS, "reg_alpha": 0.01, "reg_lambda": 0.1}),
        ("dart_n400", {**LGBM_PARAMS, "boosting_type": "dart", "n_estimators": 400}),
        ("goss_n400", {**LGBM_PARAMS, "boosting_type": "goss", "n_estimators": 400}),
    ]

    for config_name, params in configs:
        for pw in [0.5, 1.0]:
            name = f"pl_{config_name}_w{pw}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                X_aug = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_t = pipe.fit_transform(X_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model("lgbm", params)
                try:
                    model.fit(X_tr_t, y_aug, sample_weight=w)
                except TypeError:
                    model.fit(X_tr_t, y_aug)
                oof[val_idx] = model.predict(X_val_t).ravel()
                fold_rmses.append(rmse(y_val, oof[val_idx]))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append((name, score, fold_rmses))


# ============================================================================
# Section 4: Multi-seed ensemble with pseudo-labels
# ============================================================================

def section4_multiseed_pl(X_train, y_train, groups, X_test, results):
    """Multi-seed averaging with pseudo-labels."""
    print("\n=== Section 4: Multi-seed Ensemble + Pseudo-labels ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    for n_seeds in [5, 10, 20]:
        for pw in [0.5, 1.0]:
            name = f"mseed{n_seeds}_pl_w{pw}"
            seed_oofs = []

            for seed in range(n_seeds):
                params = dict(LGBM_PARAMS)
                params["random_state"] = seed * 7 + 42
                params["bagging_seed"] = seed * 11 + 17

                oof = np.full(len(y_train), np.nan)
                for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                    X_aug = np.vstack([X_tr, X_test])
                    y_aug = np.concatenate([y_tr, pseudo])
                    w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw)])

                    pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                    X_tr_t = pipe.fit_transform(X_aug)
                    X_val_t = pipe.transform(X_val)

                    model = create_model("lgbm", params)
                    model.fit(X_tr_t, y_aug, sample_weight=w)
                    oof[val_idx] = model.predict(X_val_t).ravel()

                seed_oofs.append(oof)

            avg_oof = np.mean(seed_oofs, axis=0)
            fold_rmses = []
            for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                fold_rmses.append(rmse(y_train[val_idx], avg_oof[val_idx]))

            score, _ = save_result(name, avg_oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append((name, score, fold_rmses))


# ============================================================================
# Section 5: Multi-pipeline stacking
# ============================================================================

def section5_multi_pipeline(X_train, y_train, groups, X_test, results):
    """Different preprocessing pipelines with pseudo-labels, then average."""
    print("\n=== Section 5: Multi-pipeline Ensemble ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    pipelines = {
        "emsc_b8": BEST_PREPROCESS,
        "emsc_b4": [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 4},
            {"name": "standard_scaler"},
        ],
        "emsc_b16": [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 16},
            {"name": "standard_scaler"},
        ],
        "snv_emsc_b8": [
            {"name": "snv"},
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ],
        "emsc_sg2_b8": [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 11, "polyorder": 3, "deriv": 2},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ],
        "emsc_nobin": [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 15, "polyorder": 2, "deriv": 1},
            {"name": "standard_scaler"},
        ],
    }

    pipe_oofs = {}
    for pipe_name, preprocess_cfg in pipelines.items():
        name = f"pipe_{pipe_name}_pl"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            X_aug = np.vstack([X_tr, X_test])
            y_aug = np.concatenate([y_tr, pseudo])
            w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

            pipe = build_preprocess_pipeline(preprocess_cfg)
            X_tr_t = pipe.fit_transform(X_aug)
            X_val_t = pipe.transform(X_val)

            model = create_model("lgbm", LGBM_PARAMS)
            model.fit(X_tr_t, y_aug, sample_weight=w)
            oof[val_idx] = model.predict(X_val_t).ravel()
            fold_rmses.append(rmse(y_val, oof[val_idx]))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        pipe_oofs[pipe_name] = oof
        results.append((name, score, fold_rmses))

    # Average all pipelines
    all_oofs = np.array(list(pipe_oofs.values()))
    avg_oof = np.mean(all_oofs, axis=0)
    fold_rmses = []
    for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_rmses.append(rmse(y_train[val_idx], avg_oof[val_idx]))
    score = rmse(y_train, avg_oof)
    save_result("multi_pipe_avg_pl", avg_oof, fold_rmses, y_train)
    print(f"  multi_pipe_avg_pl: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
    results.append(("multi_pipe_avg_pl", score, fold_rmses))

    # Optimize weights
    from scipy.optimize import minimize

    def ensemble_rmse(weights):
        weights = np.abs(weights) / np.abs(weights).sum()
        pred = np.zeros(len(y_train))
        for i, k in enumerate(pipe_oofs):
            pred += weights[i] * pipe_oofs[k]
        return rmse(y_train, pred)

    n_pipes = len(pipe_oofs)
    w0 = np.ones(n_pipes) / n_pipes
    res = minimize(ensemble_rmse, w0, method="Nelder-Mead")
    opt_weights = np.abs(res.x) / np.abs(res.x).sum()

    opt_oof = np.zeros(len(y_train))
    for i, k in enumerate(pipe_oofs):
        opt_oof += opt_weights[i] * pipe_oofs[k]
        print(f"    {k}: weight={opt_weights[i]:.3f}")

    fold_rmses = []
    for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_rmses.append(rmse(y_train[val_idx], opt_oof[val_idx]))
    score = rmse(y_train, opt_oof)
    save_result("multi_pipe_opt_pl", opt_oof, fold_rmses, y_train)
    print(f"  multi_pipe_opt_pl: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
    results.append(("multi_pipe_opt_pl", score, fold_rmses))


# ============================================================================
# Section 6: XGBoost / CatBoost with pseudo-labels
# ============================================================================

def section6_other_models(X_train, y_train, groups, X_test, results):
    """Test XGBoost and CatBoost with pseudo-labels."""
    print("\n=== Section 6: XGBoost / CatBoost with Pseudo-labels ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    models = {
        "xgb_pl": ("xgb", {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbosity": 0, "n_jobs": -1,
        }),
        "xgb800_pl": ("xgb", {
            "n_estimators": 800, "max_depth": 5, "learning_rate": 0.02,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbosity": 0, "n_jobs": -1,
        }),
        "cb_pl": ("catboost", {
            "iterations": 400, "depth": 5, "learning_rate": 0.05,
            "verbose": 0, "thread_count": -1,
        }),
    }

    model_oofs = {}
    for config_name, (model_type, params) in models.items():
        for pw in [0.5, 1.0]:
            name = f"{config_name}_w{pw}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                X_aug = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_t = pipe.fit_transform(X_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model(model_type, params)
                try:
                    model.fit(X_tr_t, y_aug, sample_weight=w)
                except TypeError:
                    model.fit(X_tr_t, y_aug)
                oof[val_idx] = model.predict(X_val_t).ravel()
                fold_rmses.append(rmse(y_val, oof[val_idx]))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            if pw == 1.0:
                model_oofs[config_name] = oof
            results.append((name, score, fold_rmses))

    # Also get LGBM oof for ensemble
    oof_lgbm = np.full(len(y_train), np.nan)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]

        X_aug = np.vstack([X_tr, X_test])
        y_aug = np.concatenate([y_tr, pseudo])
        w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

        pipe = build_preprocess_pipeline(BEST_PREPROCESS)
        X_tr_t = pipe.fit_transform(X_aug)
        X_val_t = pipe.transform(X_val)

        model = create_model("lgbm", LGBM_PARAMS)
        model.fit(X_tr_t, y_aug, sample_weight=w)
        oof_lgbm[val_idx] = model.predict(X_val_t).ravel()

    model_oofs["lgbm_pl"] = oof_lgbm

    # Cross-model ensemble
    if len(model_oofs) >= 2:
        from scipy.optimize import minimize

        def ens_rmse(weights):
            weights = np.abs(weights) / np.abs(weights).sum()
            pred = np.zeros(len(y_train))
            for i, k in enumerate(model_oofs):
                pred += weights[i] * model_oofs[k]
            return rmse(y_train, pred)

        n = len(model_oofs)
        w0 = np.ones(n) / n
        res = minimize(ens_rmse, w0, method="Nelder-Mead")
        opt_w = np.abs(res.x) / np.abs(res.x).sum()

        ens_oof = np.zeros(len(y_train))
        for i, k in enumerate(model_oofs):
            ens_oof += opt_w[i] * model_oofs[k]
            print(f"    {k}: weight={opt_w[i]:.3f}")

        fold_rmses = []
        for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            fold_rmses.append(rmse(y_train[val_idx], ens_oof[val_idx]))
        score = rmse(y_train, ens_oof)
        save_result("cross_model_ens_pl", ens_oof, fold_rmses, y_train)
        print(f"  cross_model_ens_pl: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append(("cross_model_ens_pl", score, fold_rmses))


# ============================================================================
# Section 7: Grand ensemble (multi-seed × multi-pipe × multi-model)
# ============================================================================

def section7_grand_ensemble(X_train, y_train, groups, X_test, results):
    """Combine everything: multi-seed, multi-pipeline, multi-model."""
    print("\n=== Section 7: Grand Ensemble ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    pipelines = [
        BEST_PREPROCESS,
        [{"name": "emsc", "poly_order": 2},
         {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
         {"name": "binning", "bin_size": 4},
         {"name": "standard_scaler"}],
        [{"name": "snv"},
         {"name": "emsc", "poly_order": 2},
         {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
         {"name": "binning", "bin_size": 8},
         {"name": "standard_scaler"}],
    ]

    all_oofs = []
    for pipe_idx, preprocess_cfg in enumerate(pipelines):
        for seed in range(5):
            params = dict(LGBM_PARAMS)
            params["random_state"] = seed * 7 + 42
            params["bagging_seed"] = seed * 11 + 17

            oof = np.full(len(y_train), np.nan)
            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr = y_train[train_idx]

                X_aug = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

                pipe = build_preprocess_pipeline(preprocess_cfg)
                X_tr_t = pipe.fit_transform(X_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model("lgbm", params)
                model.fit(X_tr_t, y_aug, sample_weight=w)
                oof[val_idx] = model.predict(X_val_t).ravel()

            all_oofs.append(oof)
            score = rmse(y_train, oof)

    # Grand average
    grand_oof = np.mean(all_oofs, axis=0)
    fold_rmses = []
    for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_rmses.append(rmse(y_train[val_idx], grand_oof[val_idx]))
    score = rmse(y_train, grand_oof)
    save_result("grand_ensemble_3pipe_5seed", grand_oof, fold_rmses, y_train)
    print(f"  grand_ensemble (3 pipes × 5 seeds = {len(all_oofs)} models): RMSE={score:.4f}")
    print(f"    folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
    results.append(("grand_ensemble_3pipe_5seed", score, fold_rmses))

    # Add XGBoost to grand ensemble
    for seed in range(3):
        xgb_params = {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbosity": 0, "n_jobs": -1,
            "random_state": seed * 13 + 7,
        }
        oof = np.full(len(y_train), np.nan)
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

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

    # Grand + XGB average
    grand_oof2 = np.mean(all_oofs, axis=0)
    fold_rmses = []
    for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_rmses.append(rmse(y_train[val_idx], grand_oof2[val_idx]))
    score = rmse(y_train, grand_oof2)
    save_result("grand_ensemble_plus_xgb", grand_oof2, fold_rmses, y_train)
    print(f"  grand_ensemble + 3 XGB = {len(all_oofs)} models: RMSE={score:.4f}")
    print(f"    folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
    results.append(("grand_ensemble_plus_xgb", score, fold_rmses))


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 15b: Refined Experiments")
    print("=" * 70)

    np.random.seed(42)
    X_train, y_train, groups, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Baseline: 16.04 (Phase 15 best)")

    results = []

    sections = [
        ("Section 1: Iterative PL", section1_iterative_pl),
        ("Section 2: WDV + PL", section2_wdv_pseudo),
        ("Section 3: LGBM Tuning + PL", section3_lgbm_tuning),
        ("Section 4: Multi-seed + PL", section4_multiseed_pl),
        ("Section 5: Multi-pipeline", section5_multi_pipeline),
        ("Section 6: XGB/CB + PL", section6_other_models),
        ("Section 7: Grand Ensemble", section7_grand_ensemble),
    ]

    for section_name, section_fn in sections:
        try:
            if section_fn in [section1_iterative_pl, section2_wdv_pseudo,
                              section3_lgbm_tuning, section4_multiseed_pl,
                              section5_multi_pipeline, section6_other_models,
                              section7_grand_ensemble]:
                section_fn(X_train, y_train, groups, X_test, results)
            else:
                section_fn(X_train, y_train, groups, results)
        except Exception as e:
            print(f"  {section_name} ERROR: {e}")
            traceback.print_exc()

    # FINAL SUMMARY
    print("\n\n" + "=" * 70)
    print("PHASE 15b FINAL SUMMARY")
    print("=" * 70)
    results.sort(key=lambda x: x[1])
    for name, score, folds in results:
        fold_str = ', '.join(f'{f:.1f}' for f in folds)
        marker = " ★" if score < 16.04 else " ●" if score < 16.14 else ""
        print(f"  {score:.4f}  [{fold_str}]  {name}{marker}")

    if results:
        best = results[0]
        print(f"\nBEST: {best[1]:.4f} ({best[0]})")
        print(f"Phase 15 best: 16.04")
        print(f"Phase 12 best: 16.14")
        print(f"No-PL baseline: 17.745")

        improved = sum(1 for r in results if r[1] < 16.04)
        print(f"\n{improved} / {len(results)} beat Phase 15 best (16.04)")


if __name__ == "__main__":
    main()
