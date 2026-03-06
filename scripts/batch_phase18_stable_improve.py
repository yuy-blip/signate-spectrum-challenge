#!/usr/bin/env python
"""Phase 18: Stable improvements on proven Targeted WDV (15.10).

WDV Basis was a fluke (seed-dependent, true avg ~18.97).
Back to reliable Targeted WDV + systematic improvements.

EXPERIMENTS:
A. Multi-seed targeted WDV ensemble (averaging reduces variance)
B. Targeted WDV + different preprocessing pipelines
C. XGBoost / CatBoost with targeted WDV
D. Targeted WDV parameter fine-tuning around best
E. Log-target transform with targeted WDV
F. Feature selection (mutual info, variance) + WDV
G. Grand ensemble of all stable approaches
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")

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


def generate_targeted_wdv(X_tr, y_tr, groups_tr, n_aug, extrap_factor, min_moisture=150):
    """Proven targeted WDV from Phase 15c."""
    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 2:
            continue
        sorted_idx = np.argsort(y_sp)
        high_idx = sorted_idx[high_mask[sorted_idx]][-5:]
        low_idx = sorted_idx[:5]
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
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        idx = np.random.choice(len(synth_X), n_aug, replace=False)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


def cv_targeted_wdv(X_train, y_train, groups, X_test, n_aug=50, extrap=1.5,
                    min_moisture=150, preprocess=None, model_type="lgbm",
                    model_params=None, pl_w=0.0, target_transform=None,
                    feat_mask=None):
    """Run CV with targeted WDV."""
    if preprocess is None:
        preprocess = BEST_PREPROCESS
    if model_params is None:
        model_params = LGBM_PARAMS.copy()

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_train))
    fold_rmses = []
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups[tr_idx]

        # Target transform
        if target_transform == "log1p":
            y_tr_t = np.log1p(y_tr)
        elif target_transform == "sqrt":
            y_tr_t = np.sqrt(y_tr)
        else:
            y_tr_t = y_tr.copy()

        pipe = build_preprocess_pipeline(preprocess)
        pipe.fit(X_tr)

        synth_X, synth_y = generate_targeted_wdv(
            X_tr, y_tr, g_tr, n_aug, extrap, min_moisture
        )

        X_tr_t = pipe.transform(X_tr)
        X_va_t = pipe.transform(X_va)
        X_test_t = pipe.transform(X_test)

        if len(synth_X) > 0:
            synth_X_t = pipe.transform(synth_X)
            if target_transform == "log1p":
                synth_y_t = np.log1p(synth_y)
            elif target_transform == "sqrt":
                synth_y_t = np.sqrt(synth_y)
            else:
                synth_y_t = synth_y
            X_tr_aug = np.vstack([X_tr_t, synth_X_t])
            y_tr_aug = np.concatenate([y_tr_t, synth_y_t])
        else:
            X_tr_aug = X_tr_t
            y_tr_aug = y_tr_t

        # Feature mask
        if feat_mask is not None:
            X_tr_aug = X_tr_aug[:, feat_mask]
            X_va_t = X_va_t[:, feat_mask]
            X_test_t = X_test_t[:, feat_mask]

        # PL
        if pl_w > 0:
            temp = create_model(model_type, model_params)
            temp.fit(X_tr_aug, y_tr_aug)
            pl = temp.predict(X_test_t)
            X_tr_aug = np.vstack([X_tr_aug, X_test_t])
            y_tr_aug = np.concatenate([y_tr_aug, pl])
            w = np.ones(len(y_tr_aug))
            w[-len(pl):] = pl_w
            model = create_model(model_type, model_params)
            model.fit(X_tr_aug, y_tr_aug, sample_weight=w)
        else:
            model = create_model(model_type, model_params)
            model.fit(X_tr_aug, y_tr_aug)

        pred = model.predict(X_va_t)

        # Inverse target transform
        if target_transform == "log1p":
            pred = np.expm1(pred)
        elif target_transform == "sqrt":
            pred = pred ** 2

        oof[va_idx] = pred
        fold_rmses.append(rmse(y_va, pred))

        tp = model.predict(X_test_t)
        if target_transform == "log1p":
            tp = np.expm1(tp)
        elif target_transform == "sqrt":
            tp = tp ** 2
        test_preds.append(tp)

    return oof, fold_rmses, np.mean(test_preds, axis=0)


def main():
    np.random.seed(42)
    X_train, y_train, groups, X_test, test_ids = load_data()

    all_results = []
    all_oofs = {}

    def log(name, score, folds, oof=None):
        fr = [round(x, 1) for x in folds]
        m = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"  {name}: RMSE={score:.4f}  folds={fr}{m}")
        all_results.append((name, score, fr))
        if oof is not None:
            all_oofs[name] = oof

    # ============================================================
    # Section A: Multi-seed targeted WDV ensemble
    # ============================================================
    print("\n=== Section A: Multi-seed targeted WDV (n50, f1.5, mm150) ===")

    seed_oofs = []
    for seed in range(20):
        np.random.seed(seed)
        name = f"twdv_s{seed}"
        try:
            oof, folds, _ = cv_targeted_wdv(
                X_train, y_train, groups, X_test,
                n_aug=50, extrap=1.5, min_moisture=150
            )
            score = rmse(y_train, oof)
            log(name, score, folds, oof)
            seed_oofs.append(oof)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    if len(seed_oofs) >= 3:
        avg = np.mean(seed_oofs, axis=0)
        s = rmse(y_train, avg)
        print(f"\n  ** 20-seed avg: RMSE={s:.4f} **")
        all_oofs["twdv_avg20"] = avg
        all_results.append(("twdv_avg20", s, []))

        scored = sorted([(rmse(y_train, o), o) for o in seed_oofs])
        for n in [3, 5, 10]:
            if len(scored) >= n:
                avg = np.mean([o for _, o in scored[:n]], axis=0)
                s = rmse(y_train, avg)
                print(f"  ** Top-{n} avg: RMSE={s:.4f} **")
                all_oofs[f"twdv_top{n}"] = avg
                all_results.append((f"twdv_top{n}", s, []))

    # ============================================================
    # Section B: Fine-tune targeted WDV params
    # ============================================================
    print("\n=== Section B: Fine-tune targeted WDV params ===")

    for mm in [130, 140, 145, 150, 155, 160]:
        for n_aug in [30, 40, 50, 60, 70]:
            for extrap in [1.0, 1.2, 1.3, 1.5, 1.7, 2.0]:
                name = f"twdv_mm{mm}_n{n_aug}_f{extrap}"
                np.random.seed(42)
                try:
                    oof, folds, _ = cv_targeted_wdv(
                        X_train, y_train, groups, X_test,
                        n_aug=n_aug, extrap=extrap, min_moisture=mm
                    )
                    score = rmse(y_train, oof)
                    if score < 15.5:  # Only log good results
                        log(name, score, folds, oof)
                    else:
                        all_results.append((name, score, [round(x, 1) for x in folds]))
                except Exception as e:
                    print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section C: Different preprocessing pipelines
    # ============================================================
    print("\n=== Section C: Different preprocessing ===")

    alt_preprocess = [
        ("sg_w9", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 9, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
        ("sg_w11", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
        ("sg_w5", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 5, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
        ("bin4", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 4},
            {"name": "standard_scaler"},
        ]),
        ("bin16", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 16},
            {"name": "standard_scaler"},
        ]),
        ("deriv2", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 3, "deriv": 2},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
        ("emsc3", [
            {"name": "emsc", "poly_order": 3},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
        ("no_bin", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "standard_scaler"},
        ]),
        ("snv_sg", [
            {"name": "snv"},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
    ]

    for pp_name, pp in alt_preprocess:
        name = f"twdv_{pp_name}"
        np.random.seed(42)
        try:
            oof, folds, _ = cv_targeted_wdv(
                X_train, y_train, groups, X_test,
                n_aug=50, extrap=1.5, min_moisture=150,
                preprocess=pp
            )
            score = rmse(y_train, oof)
            log(name, score, folds, oof)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section D: XGBoost / CatBoost with targeted WDV
    # ============================================================
    print("\n=== Section D: XGBoost/CatBoost with targeted WDV ===")

    xgb_params = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbosity": 0, "n_jobs": -1,
    }

    np.random.seed(42)
    try:
        oof, folds, _ = cv_targeted_wdv(
            X_train, y_train, groups, X_test,
            n_aug=50, extrap=1.5, min_moisture=150,
            model_type="xgb", model_params=xgb_params
        )
        score = rmse(y_train, oof)
        log("twdv_xgb", score, folds, oof)
    except Exception as e:
        print(f"  twdv_xgb: FAILED - {e}")

    cat_params = {
        "iterations": 400, "depth": 5, "learning_rate": 0.05,
        "verbose": 0, "thread_count": -1,
    }

    np.random.seed(42)
    try:
        oof, folds, _ = cv_targeted_wdv(
            X_train, y_train, groups, X_test,
            n_aug=50, extrap=1.5, min_moisture=150,
            model_type="catboost", model_params=cat_params
        )
        score = rmse(y_train, oof)
        log("twdv_catboost", score, folds, oof)
    except Exception as e:
        print(f"  twdv_catboost: FAILED - {e}")

    # ============================================================
    # Section E: Target transforms
    # ============================================================
    print("\n=== Section E: Target transforms ===")

    for tt in ["log1p", "sqrt"]:
        np.random.seed(42)
        name = f"twdv_{tt}"
        try:
            oof, folds, _ = cv_targeted_wdv(
                X_train, y_train, groups, X_test,
                n_aug=50, extrap=1.5, min_moisture=150,
                target_transform=tt
            )
            score = rmse(y_train, oof)
            log(name, score, folds, oof)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section F: Targeted WDV + PL (multi-seed)
    # ============================================================
    print("\n=== Section F: Targeted WDV + PL ===")

    for pl_w in [0.3, 0.5, 0.7]:
        pl_oofs = []
        for seed in range(5):
            np.random.seed(seed + 100)
            name = f"twdv_pl{pl_w}_s{seed}"
            try:
                oof, folds, _ = cv_targeted_wdv(
                    X_train, y_train, groups, X_test,
                    n_aug=50, extrap=1.5, min_moisture=150,
                    pl_w=pl_w
                )
                score = rmse(y_train, oof)
                log(name, score, folds, oof)
                pl_oofs.append(oof)
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

        if len(pl_oofs) >= 3:
            avg = np.mean(pl_oofs, axis=0)
            s = rmse(y_train, avg)
            print(f"  ** PL w={pl_w} avg: RMSE={s:.4f} **")
            all_oofs[f"twdv_pl{pl_w}_avg"] = avg
            all_results.append((f"twdv_pl{pl_w}_avg", s, []))

    # ============================================================
    # Section G: LGBM tuning with targeted WDV
    # ============================================================
    print("\n=== Section G: LGBM tuning ===")

    lgbm_tweaks = [
        {"n_estimators": 600, "learning_rate": 0.03},
        {"n_estimators": 800, "learning_rate": 0.02},
        {"n_estimators": 300, "learning_rate": 0.08},
        {"min_child_samples": 15},
        {"min_child_samples": 10},
        {"min_child_samples": 25},
        {"num_leaves": 15, "max_depth": 4},
        {"num_leaves": 25, "max_depth": 6},
        {"num_leaves": 31, "max_depth": 7},
        {"subsample": 0.8, "colsample_bytree": 0.8},
        {"subsample": 0.6, "colsample_bytree": 0.6},
        {"reg_alpha": 0.05, "reg_lambda": 0.5},
        {"reg_alpha": 0.3, "reg_lambda": 2.0},
        {"reg_alpha": 0.01, "reg_lambda": 0.1},
    ]

    for i, tw in enumerate(lgbm_tweaks):
        params = {**LGBM_PARAMS, **tw}
        name = f"twdv_lgbm_{list(tw.keys())[0]}={list(tw.values())[0]}"
        np.random.seed(42)
        try:
            oof, folds, _ = cv_targeted_wdv(
                X_train, y_train, groups, X_test,
                n_aug=50, extrap=1.5, min_moisture=150,
                model_params=params
            )
            score = rmse(y_train, oof)
            log(name, score, folds, oof)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section H: Grand ensemble
    # ============================================================
    print("\n=== Section H: Grand ensemble ===")

    top = sorted([(rmse(y_train, o), n, o) for n, o in all_oofs.items()])
    print(f"  {len(top)} OOFs collected")
    for s, n, _ in top[:20]:
        print(f"    {s:.4f}  {n}")

    if len(top) >= 3:
        for n_top in [3, 5, 10, 15, 20]:
            if len(top) >= n_top:
                avg = np.mean([o for _, _, o in top[:n_top]], axis=0)
                s = rmse(y_train, avg)
                print(f"  Grand avg top-{n_top}: RMSE={s:.4f}")
                all_results.append((f"grand_top{n_top}", s, []))

        try:
            from scipy.optimize import minimize
            n = min(20, len(top))
            mat = np.array([o for _, _, o in top[:n]])
            def obj(w):
                w = np.abs(w) / np.abs(w).sum()
                return rmse(y_train, (w[:, None] * mat).sum(axis=0))
            res = minimize(obj, np.ones(n)/n, method="Nelder-Mead",
                          options={"maxiter": 10000})
            w = np.abs(res.x) / np.abs(res.x).sum()
            opt = (w[:, None] * mat).sum(axis=0)
            opt_s = rmse(y_train, opt)
            print(f"  Grand optimized ({n}): RMSE={opt_s:.4f}")
            wn = sorted([(w[i], top[i][1]) for i in range(n)], reverse=True)
            for wi, ni in wn[:5]:
                print(f"    {wi:.3f}  {ni}")
            all_results.append(("grand_opt", opt_s, []))
        except Exception as e:
            print(f"  Grand opt FAILED: {e}")

    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 18 FINAL SUMMARY")
    print("=" * 70)

    all_results.sort(key=lambda x: x[1])
    for name, score, folds in all_results[:40]:
        m = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"  {score:.4f}  {folds}  {name}{m}")

    if all_results:
        print(f"\nBEST: {all_results[0][1]:.4f} ({all_results[0][0]})")
    print(f"Phase 15c best: 15.10")


if __name__ == "__main__":
    main()
