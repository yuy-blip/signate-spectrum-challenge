#!/usr/bin/env python
"""Phase 17: Deep optimization of WDV Basis breakthrough.

Phase 16 best: wdv_basis_k5_n30_f1.5 → RMSE 14.57 [12.4, 9.1, 24.9, 11.6, 8.8]

EXPERIMENTS:
A. Fine-tune WDV basis params around k=5, n=30, f=1.5
B. WDV basis + pseudo-labeling (combine two best techniques)
C. WDV basis + min_child_samples tuning
D. Multi-seed WDV basis ensemble
E. WDV basis in EMSC-corrected space
F. WDV basis with different randomness strategies
G. Grand ensemble of top WDV basis configs
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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


def generate_wdv_basis(X_tr, y_tr, groups_tr, n_aug, extrap_factor,
                       min_moisture=0, n_basis=5, noise_std=0.3, seed=42):
    """WDV with PCA basis — optimized version with controllable randomness."""
    rng = np.random.RandomState(seed)
    all_deltas = []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue
        sorted_idx = np.argsort(y_sp)
        for i in range(len(sorted_idx)):
            for j in range(i + 1, len(sorted_idx)):
                dy = y_sp[sorted_idx[j]] - y_sp[sorted_idx[i]]
                if dy < 10:
                    continue
                dx = X_sp[sorted_idx[j]] - X_sp[sorted_idx[i]]
                all_deltas.append(dx / dy)

    if len(all_deltas) < n_basis:
        return np.empty((0, X_tr.shape[1])), np.empty(0)

    all_deltas = np.array(all_deltas)
    pca = PCA(n_components=min(n_basis, len(all_deltas), all_deltas.shape[1]))
    pca.fit(all_deltas)

    mean_delta = all_deltas.mean(axis=0)
    base_coeffs = pca.transform(mean_delta.reshape(1, -1))[0]

    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 1:
            continue
        high_samples = np.where(high_mask)[0]
        for hi_idx in high_samples:
            target_dy = extrap_factor * (y_sp[hi_idx] * 0.5 + 50)
            coeffs = base_coeffs * (1 + noise_std * rng.randn(len(base_coeffs)))
            delta = pca.inverse_transform(coeffs.reshape(1, -1))[0]
            new_x = X_sp[hi_idx] + delta * target_dy
            new_y = y_sp[hi_idx] + target_dy
            synth_X.append(new_x)
            synth_y.append(new_y)

    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        idx = rng.choice(len(synth_X), n_aug, replace=False)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


def generate_wdv_basis_emsc(X_tr, y_tr, groups_tr, pipe, n_aug, extrap_factor,
                            min_moisture=0, n_basis=5, noise_std=0.3, seed=42):
    """WDV basis in EMSC-corrected (preprocessed) space."""
    rng = np.random.RandomState(seed)
    X_tr_t = pipe.transform(X_tr)
    all_deltas = []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr_t[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue
        sorted_idx = np.argsort(y_sp)
        for i in range(len(sorted_idx)):
            for j in range(i + 1, len(sorted_idx)):
                dy = y_sp[sorted_idx[j]] - y_sp[sorted_idx[i]]
                if dy < 10:
                    continue
                dx = X_sp[sorted_idx[j]] - X_sp[sorted_idx[i]]
                all_deltas.append(dx / dy)

    if len(all_deltas) < n_basis:
        return np.empty((0, X_tr_t.shape[1])), np.empty(0)

    all_deltas = np.array(all_deltas)
    pca = PCA(n_components=min(n_basis, len(all_deltas), all_deltas.shape[1]))
    pca.fit(all_deltas)
    mean_delta = all_deltas.mean(axis=0)
    base_coeffs = pca.transform(mean_delta.reshape(1, -1))[0]

    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr_t[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 1:
            continue
        for hi_idx in np.where(high_mask)[0]:
            target_dy = extrap_factor * (y_sp[hi_idx] * 0.5 + 50)
            coeffs = base_coeffs * (1 + noise_std * rng.randn(len(base_coeffs)))
            delta = pca.inverse_transform(coeffs.reshape(1, -1))[0]
            new_x = X_sp[hi_idx] + delta * target_dy
            new_y = y_sp[hi_idx] + target_dy
            synth_X.append(new_x)
            synth_y.append(new_y)

    if not synth_X:
        return np.empty((0, X_tr_t.shape[1])), np.empty(0)
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        idx = rng.choice(len(synth_X), n_aug, replace=False)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


def cv_with_wdv_basis(X_train, y_train, groups, n_aug, extrap_factor,
                      min_moisture=150, n_basis=5, noise_std=0.3,
                      lgbm_overrides=None, seed=42, pl_weight=0.0,
                      X_test=None, preprocess=None):
    """Run CV with WDV basis augmentation."""
    if preprocess is None:
        preprocess = BEST_PREPROCESS
    params = {**LGBM_PARAMS}
    if lgbm_overrides:
        params.update(lgbm_overrides)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_train))
    fold_rmses = []
    test_preds_list = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups[tr_idx]

        pipe = build_preprocess_pipeline(preprocess)
        pipe.fit(X_tr)

        # Generate WDV basis augmentation in raw space
        synth_X, synth_y = generate_wdv_basis(
            X_tr, y_tr, g_tr, n_aug, extrap_factor,
            min_moisture, n_basis, noise_std, seed + fold
        )

        # Preprocess
        X_tr_t = pipe.transform(X_tr)
        X_va_t = pipe.transform(X_va)

        if len(synth_X) > 0:
            synth_X_t = pipe.transform(synth_X)
            X_tr_aug = np.vstack([X_tr_t, synth_X_t])
            y_tr_aug = np.concatenate([y_tr, synth_y])
        else:
            X_tr_aug = X_tr_t
            y_tr_aug = y_tr

        # Add pseudo-labels if weight > 0
        if pl_weight > 0 and X_test is not None:
            X_test_t = pipe.transform(X_test)
            # Train temp model to generate PLs
            temp_model = create_model("lgbm", params)
            temp_model.fit(X_tr_aug, y_tr_aug)
            pl_preds = temp_model.predict(X_test_t)
            X_tr_aug = np.vstack([X_tr_aug, X_test_t])
            y_tr_aug = np.concatenate([y_tr_aug, pl_preds])
            # Use sample weights
            w = np.ones(len(y_tr_aug))
            w[-len(pl_preds):] = pl_weight
            model = create_model("lgbm", params)
            model.fit(X_tr_aug, y_tr_aug, sample_weight=w)
        else:
            model = create_model("lgbm", params)
            model.fit(X_tr_aug, y_tr_aug)

        pred = model.predict(X_va_t)
        oof[va_idx] = pred
        fold_rmses.append(rmse(y_va, pred))

        if X_test is not None:
            X_test_t = pipe.transform(X_test)
            test_preds_list.append(model.predict(X_test_t))

    test_preds = np.mean(test_preds_list, axis=0) if test_preds_list else None
    return oof, fold_rmses, test_preds


def cv_with_wdv_basis_emsc(X_train, y_train, groups, n_aug, extrap_factor,
                           min_moisture=150, n_basis=5, noise_std=0.3,
                           lgbm_overrides=None, seed=42):
    """CV with WDV basis in EMSC-corrected space."""
    params = {**LGBM_PARAMS}
    if lgbm_overrides:
        params.update(lgbm_overrides)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_train))
    fold_rmses = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups[tr_idx]

        pipe = build_preprocess_pipeline(BEST_PREPROCESS)
        pipe.fit(X_tr)

        # Generate WDV basis in preprocessed space
        synth_X_t, synth_y = generate_wdv_basis_emsc(
            X_tr, y_tr, g_tr, pipe, n_aug, extrap_factor,
            min_moisture, n_basis, noise_std, seed + fold
        )

        X_tr_t = pipe.transform(X_tr)
        X_va_t = pipe.transform(X_va)

        if len(synth_X_t) > 0:
            X_tr_aug = np.vstack([X_tr_t, synth_X_t])
            y_tr_aug = np.concatenate([y_tr, synth_y])
        else:
            X_tr_aug = X_tr_t
            y_tr_aug = y_tr

        model = create_model("lgbm", params)
        model.fit(X_tr_aug, y_tr_aug)
        pred = model.predict(X_va_t)
        oof[va_idx] = pred
        fold_rmses.append(rmse(y_va, pred))

    return oof, fold_rmses


def main():
    np.random.seed(42)
    X_train, y_train, groups, X_test, test_ids = load_data()

    all_results = []
    all_oofs = {}

    def log_result(name, score, fold_rmses, oof=None):
        fr = [round(x, 1) for x in fold_rmses]
        marker = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"  {name}: RMSE={score:.4f}  folds={fr}{marker}")
        all_results.append((name, score, fr))
        if oof is not None:
            all_oofs[name] = oof

    # ============================================================
    # Section A: Fine-tune WDV basis around best config
    # Best: k=5, n=30, f=1.5 → 14.57
    # ============================================================
    print("\n=== Section A: WDV basis fine-tuning ===")

    for n_basis in [4, 5, 6, 7]:
        for n_aug in [20, 25, 30, 35, 40]:
            for extrap in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
                name = f"wb_k{n_basis}_n{n_aug}_f{extrap}"
                try:
                    oof, folds, _ = cv_with_wdv_basis(
                        X_train, y_train, groups, n_aug, extrap,
                        min_moisture=150, n_basis=n_basis, noise_std=0.3,
                        X_test=X_test
                    )
                    score = rmse(y_train, oof)
                    log_result(name, score, folds, oof)
                except Exception as e:
                    print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section B: WDV basis + pseudo-labeling
    # ============================================================
    print("\n=== Section B: WDV basis + pseudo-labeling ===")

    for n_basis in [5, 6]:
        for n_aug in [25, 30]:
            for extrap in [1.4, 1.5, 1.6]:
                for pl_w in [0.3, 0.5, 0.7]:
                    name = f"wb_pl_k{n_basis}_n{n_aug}_f{extrap}_pw{pl_w}"
                    try:
                        oof, folds, _ = cv_with_wdv_basis(
                            X_train, y_train, groups, n_aug, extrap,
                            min_moisture=150, n_basis=n_basis, noise_std=0.3,
                            pl_weight=pl_w, X_test=X_test
                        )
                        score = rmse(y_train, oof)
                        log_result(name, score, folds, oof)
                    except Exception as e:
                        print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section C: WDV basis + LGBM tuning
    # ============================================================
    print("\n=== Section C: WDV basis + LGBM tuning ===")

    lgbm_configs = [
        {"min_child_samples": 15, "reg_alpha": 0.05},
        {"min_child_samples": 10, "reg_alpha": 0.1},
        {"min_child_samples": 10, "reg_alpha": 0.05},
        {"min_child_samples": 5, "reg_alpha": 0.1},
        {"min_child_samples": 5, "reg_alpha": 0.05},
        {"n_estimators": 600, "learning_rate": 0.03},
        {"n_estimators": 300, "learning_rate": 0.08},
        {"num_leaves": 25, "max_depth": 6},
        {"num_leaves": 15, "max_depth": 4},
        {"subsample": 0.8, "colsample_bytree": 0.8},
        {"subsample": 0.6, "colsample_bytree": 0.6},
        {"reg_lambda": 0.5, "reg_alpha": 0.05},
        {"reg_lambda": 2.0, "reg_alpha": 0.3},
    ]

    for i, lgbm_ov in enumerate(lgbm_configs):
        name = f"wb_lgbm{i}_{list(lgbm_ov.keys())[0]}={list(lgbm_ov.values())[0]}"
        try:
            oof, folds, _ = cv_with_wdv_basis(
                X_train, y_train, groups, 30, 1.5,
                min_moisture=150, n_basis=5, noise_std=0.3,
                lgbm_overrides=lgbm_ov, X_test=X_test
            )
            score = rmse(y_train, oof)
            log_result(name, score, folds, oof)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section D: Multi-seed WDV basis ensemble
    # ============================================================
    print("\n=== Section D: Multi-seed ensemble ===")

    # Top configs from Section A (we'll use best known + variations)
    top_configs = [
        {"n_basis": 5, "n_aug": 30, "extrap": 1.5},
        {"n_basis": 5, "n_aug": 25, "extrap": 1.5},
        {"n_basis": 6, "n_aug": 30, "extrap": 1.5},
        {"n_basis": 5, "n_aug": 30, "extrap": 1.4},
    ]

    seed_oofs = []
    for seed in [42, 123, 456, 789, 2024]:
        name = f"wb_seed{seed}"
        try:
            oof, folds, _ = cv_with_wdv_basis(
                X_train, y_train, groups, 30, 1.5,
                min_moisture=150, n_basis=5, noise_std=0.3,
                seed=seed, X_test=X_test
            )
            score = rmse(y_train, oof)
            log_result(name, score, folds, oof)
            seed_oofs.append(oof)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    if len(seed_oofs) >= 3:
        avg_oof = np.mean(seed_oofs, axis=0)
        avg_score = rmse(y_train, avg_oof)
        print(f"  multi_seed_avg ({len(seed_oofs)} seeds): RMSE={avg_score:.4f}")
        all_oofs["multi_seed_avg"] = avg_oof
        all_results.append(("multi_seed_avg", avg_score, []))

    # ============================================================
    # Section E: WDV basis in EMSC space
    # ============================================================
    print("\n=== Section E: WDV basis in EMSC space ===")

    for n_basis in [5, 7]:
        for n_aug in [25, 30, 40]:
            for extrap in [1.3, 1.5, 1.7]:
                name = f"wb_emsc_k{n_basis}_n{n_aug}_f{extrap}"
                try:
                    oof, folds = cv_with_wdv_basis_emsc(
                        X_train, y_train, groups, n_aug, extrap,
                        min_moisture=150, n_basis=n_basis, noise_std=0.3
                    )
                    score = rmse(y_train, oof)
                    log_result(name, score, folds, oof)
                except Exception as e:
                    print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section F: Noise variation
    # ============================================================
    print("\n=== Section F: Noise std variation ===")

    for noise in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        name = f"wb_noise{noise}"
        try:
            oof, folds, _ = cv_with_wdv_basis(
                X_train, y_train, groups, 30, 1.5,
                min_moisture=150, n_basis=5, noise_std=noise,
                X_test=X_test
            )
            score = rmse(y_train, oof)
            log_result(name, score, folds, oof)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Different min_moisture values
    print("\n  --- min_moisture sweep ---")
    for mm in [100, 120, 130, 140, 150, 160, 170, 180]:
        name = f"wb_mm{mm}"
        try:
            oof, folds, _ = cv_with_wdv_basis(
                X_train, y_train, groups, 30, 1.5,
                min_moisture=mm, n_basis=5, noise_std=0.3,
                X_test=X_test
            )
            score = rmse(y_train, oof)
            log_result(name, score, folds, oof)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # ============================================================
    # Section G: Grand ensemble
    # ============================================================
    print("\n=== Section G: Grand ensemble ===")

    # Collect top OOFs
    top_oofs = sorted(
        [(name, oof) for name, oof in all_oofs.items()],
        key=lambda x: rmse(y_train, x[1])
    )[:20]

    if len(top_oofs) >= 3:
        # Simple average of top N
        for n_top in [3, 5, 7, 10]:
            if len(top_oofs) >= n_top:
                avg_oof = np.mean([o for _, o in top_oofs[:n_top]], axis=0)
                score = rmse(y_train, avg_oof)
                names = [n for n, _ in top_oofs[:n_top]]
                print(f"  Grand avg top-{n_top}: RMSE={score:.4f}")
                print(f"    configs: {names}")
                all_results.append((f"grand_top{n_top}", score, []))

        # Optimized weights via scipy
        try:
            from scipy.optimize import minimize

            oof_matrix = np.array([o for _, o in top_oofs[:10]])

            def obj(w):
                w = np.abs(w) / np.abs(w).sum()
                pred = (w[:, None] * oof_matrix).sum(axis=0)
                return rmse(y_train, pred)

            n = min(10, len(top_oofs))
            w0 = np.ones(n) / n
            res = minimize(obj, w0, method="Nelder-Mead",
                          options={"maxiter": 5000})
            w_opt = np.abs(res.x) / np.abs(res.x).sum()
            opt_oof = (w_opt[:, None] * oof_matrix[:n]).sum(axis=0)
            opt_score = rmse(y_train, opt_oof)
            print(f"  Grand optimized ({n} models): RMSE={opt_score:.4f}")
            print(f"    weights: {[f'{w:.3f}' for w in w_opt]}")
            all_results.append(("grand_optimized", opt_score, []))
        except Exception as e:
            print(f"  Grand optimized: FAILED - {e}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 17 FINAL SUMMARY")
    print("=" * 70)

    all_results.sort(key=lambda x: x[1])
    for name, score, folds in all_results[:50]:
        marker = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"  {score:.4f}  {folds}  {name}{marker}")

    if all_results:
        print(f"\nBEST: {all_results[0][1]:.4f} ({all_results[0][0]})")
    print(f"Phase 16 best: 14.57")


if __name__ == "__main__":
    main()
