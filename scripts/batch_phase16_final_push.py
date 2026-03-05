#!/usr/bin/env python
"""Phase 16: Final push — combining all LLM insights + own ideas.

EXPERIMENTS:
A. WDV in EMSC-corrected space (ChatGPT: cleaner water vectors)
B. WDV basis (ChatGPT: PCA on difference vectors, multi-directional)
C. min_child_samples extreme tuning (Gemini: unlock leaf constraints)
D. PCA → KRR as PL teacher (ChatGPT: break 203% ceiling)
E. Prediction stretching for pseudo-labels (own idea)
F. Two-expert model (high/low moisture split)
G. Autoencoder features (Gemini: unsupervised 1D-CNN feature extraction)
H. KNN regression blend (own idea: fundamentally different extrapolation)
I. Grand combined: best of everything
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
from sklearn.decomposition import PCA, NMF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
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


def generate_wdv_raw(X_tr, y_tr, groups_tr, n_aug, extrap_factor, min_moisture=0):
    """Generate WDV in raw spectrum space (current approach)."""
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


def generate_wdv_emsc(X_tr, y_tr, groups_tr, n_aug, extrap_factor, min_moisture=0, pipe=None):
    """Generate WDV in EMSC-corrected space, then inverse-transform back."""
    # This version computes difference vectors in preprocessed space
    # and adds them to preprocessed high-moisture samples
    # Returns preprocessed synthetic data (to be used WITHOUT further preprocessing)
    if pipe is None:
        return np.empty((0, 10)), np.empty(0)  # fallback
    X_tr_t = pipe.transform(X_tr)  # pipe must already be fitted
    synth_X_t, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp_t, y_sp = X_tr_t[sp_mask], y_tr[sp_mask]
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
                dx = X_sp_t[hi] - X_sp_t[lo]
                synth_X_t.append(X_sp_t[hi] + extrap_factor * dx)
                synth_y.append(y_sp[hi] + extrap_factor * dy)
    if not synth_X_t:
        return np.empty((0, X_tr_t.shape[1])), np.empty(0)
    synth_X_t, synth_y = np.array(synth_X_t), np.array(synth_y)
    if len(synth_X_t) > n_aug:
        idx = np.random.choice(len(synth_X_t), n_aug, replace=False)
        synth_X_t, synth_y = synth_X_t[idx], synth_y[idx]
    return synth_X_t, synth_y


def generate_wdv_basis(X_tr, y_tr, groups_tr, n_aug, extrap_factor, min_moisture=0, n_basis=3):
    """WDV with PCA basis: collect all difference vectors, PCA to K directions."""
    # Collect all difference vectors
    all_deltas, all_dy = [], []
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
                # Normalize by dy to get "per-unit-moisture" vector
                all_deltas.append(dx / dy)
                all_dy.append(dy)

    if len(all_deltas) < n_basis:
        return generate_wdv_raw(X_tr, y_tr, groups_tr, n_aug, extrap_factor, min_moisture)

    all_deltas = np.array(all_deltas)

    # PCA on normalized difference vectors
    pca = PCA(n_components=min(n_basis, len(all_deltas), all_deltas.shape[1]))
    pca.fit(all_deltas)
    basis = pca.components_  # (K, n_features)

    # Synthesize using basis vectors
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
            # Random coefficients for each basis vector
            target_dy = extrap_factor * (y_sp[hi_idx] * 0.5 + 50)  # adaptive delta
            # Project the mean delta vector onto basis
            mean_delta = all_deltas.mean(axis=0)
            coeffs = pca.transform(mean_delta.reshape(1, -1))[0]
            # Add some randomness
            coeffs *= (1 + 0.3 * np.random.randn(len(coeffs)))
            # Reconstruct
            delta = pca.inverse_transform(coeffs.reshape(1, -1))[0]
            new_x = X_sp[hi_idx] + delta * target_dy
            new_y = y_sp[hi_idx] + target_dy
            synth_X.append(new_x)
            synth_y.append(new_y)

    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        idx = np.random.choice(len(synth_X), n_aug, replace=False)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


def generate_pseudo_labels(X_train, y_train, X_test, preprocess_cfg, model_type, model_params):
    pipe = build_preprocess_pipeline(preprocess_cfg)
    X_all = np.vstack([X_train, X_test])
    X_all_t = pipe.fit_transform(X_all)
    model = create_model(model_type, model_params)
    model.fit(X_all_t[:len(y_train)], y_train)
    return model.predict(X_all_t[len(y_train):]).ravel()


def run_wdv_pl_cv(name, X_train, y_train, groups, X_test, pseudo,
                   preprocess_cfg, model_params, wdv_fn, wdv_kwargs,
                   pw=1.0, wdv_weight=0.3, wdv_in_emsc_space=False):
    """Generic WDV+PL CV runner."""
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y_train), np.nan)
    fold_rmses = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        g_tr = groups[train_idx]

        if wdv_in_emsc_space:
            # First fit pipeline on train, then generate WDV in that space
            pipe_fit = build_preprocess_pipeline(preprocess_cfg)
            X_tr_t = pipe_fit.fit_transform(X_tr)
            X_val_t = pipe_fit.transform(X_val)
            X_test_t = pipe_fit.transform(X_test)
            pseudo_t = pseudo  # pseudo labels stay the same

            s_X_t, s_y = generate_wdv_emsc(X_tr, y_tr, g_tr, pipe=pipe_fit, **wdv_kwargs)

            if len(s_X_t) > 0:
                X_tr_final = np.vstack([X_tr_t, X_test_t, s_X_t])
                y_tr_final = np.concatenate([y_tr, pseudo, s_y])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw),
                                    np.full(len(s_y), wdv_weight)])
            else:
                X_tr_final = np.vstack([X_tr_t, X_test_t])
                y_tr_final = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw)])

            model = create_model("lgbm", model_params)
            model.fit(X_tr_final, y_tr_final, sample_weight=w)
            oof[val_idx] = model.predict(X_val_t).ravel()
        else:
            # Standard: WDV in raw space, then preprocess together
            s_X, s_y = wdv_fn(X_tr, y_tr, g_tr, **wdv_kwargs)

            if len(s_X) > 0:
                X_aug = np.vstack([X_tr, X_test, s_X])
                y_aug = np.concatenate([y_tr, pseudo, s_y])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw),
                                    np.full(len(s_y), wdv_weight)])
            else:
                X_aug = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), pw)])

            pipe = build_preprocess_pipeline(preprocess_cfg)
            X_tr_t = pipe.fit_transform(X_aug)
            X_val_t = pipe.transform(X_val)

            model = create_model("lgbm", model_params)
            model.fit(X_tr_t, y_aug, sample_weight=w)
            oof[val_idx] = model.predict(X_val_t).ravel()

        fold_rmses.append(rmse(y_val, oof[val_idx]))

    score, _ = save_result(name, oof, fold_rmses, y_train)
    print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
    return score, fold_rmses


# ============================================================================
# Section A: WDV in EMSC-corrected space
# ============================================================================

def section_a(X_train, y_train, groups, X_test, results):
    print("\n=== Section A: WDV in EMSC-corrected space ===")
    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    for n_aug in [20, 50]:
        for ef in [1.0, 1.5]:
            for mm in [80, 150]:
                name = f"wdv_emsc_n{n_aug}_f{ef}_m{mm}"
                score, folds = run_wdv_pl_cv(
                    name, X_train, y_train, groups, X_test, pseudo,
                    BEST_PREPROCESS, LGBM_PARAMS,
                    wdv_fn=None,
                    wdv_kwargs={"n_aug": n_aug, "extrap_factor": ef, "min_moisture": mm},
                    pw=0.5, wdv_in_emsc_space=True,
                )
                results.append((name, score, folds))

    # Compare: same params, raw space (control)
    print("\n  --- Control: WDV in raw space ---")
    for n_aug in [20, 50]:
        for ef in [1.0, 1.5]:
            name = f"wdv_raw_n{n_aug}_f{ef}_m150"
            score, folds = run_wdv_pl_cv(
                name, X_train, y_train, groups, X_test, pseudo,
                BEST_PREPROCESS, LGBM_PARAMS,
                wdv_fn=generate_wdv_raw,
                wdv_kwargs={"n_aug": n_aug, "extrap_factor": ef, "min_moisture": 150},
                pw=0.5,
            )
            results.append((name, score, folds))


# ============================================================================
# Section B: WDV Basis (PCA on difference vectors)
# ============================================================================

def section_b(X_train, y_train, groups, X_test, results):
    print("\n=== Section B: WDV Basis (PCA on difference vectors) ===")
    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    for n_basis in [3, 5, 8]:
        for n_aug in [30, 50]:
            for ef in [1.0, 1.5]:
                name = f"wdv_basis_k{n_basis}_n{n_aug}_f{ef}"
                score, folds = run_wdv_pl_cv(
                    name, X_train, y_train, groups, X_test, pseudo,
                    BEST_PREPROCESS, LGBM_PARAMS,
                    wdv_fn=generate_wdv_basis,
                    wdv_kwargs={"n_aug": n_aug, "extrap_factor": ef, "min_moisture": 100, "n_basis": n_basis},
                    pw=0.5,
                )
                results.append((name, score, folds))


# ============================================================================
# Section C: min_child_samples extreme tuning
# ============================================================================

def section_c(X_train, y_train, groups, X_test, results):
    print("\n=== Section C: min_child_samples extreme tuning ===")
    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    for mcs in [1, 2, 3, 5, 10, 15]:
        for ra in [0.1, 0.5, 1.0, 3.0]:
            params = {**LGBM_PARAMS, "min_child_samples": mcs, "reg_alpha": ra}
            name = f"mcs{mcs}_ra{ra}"

            # Without WDV (pure PL)
            score, folds = run_wdv_pl_cv(
                f"pl_{name}", X_train, y_train, groups, X_test, pseudo,
                BEST_PREPROCESS, params,
                wdv_fn=generate_wdv_raw,
                wdv_kwargs={"n_aug": 0, "extrap_factor": 1.0, "min_moisture": 999},
                pw=1.0,
            )
            results.append((f"pl_{name}", score, folds))

    # Best mcs with WDV
    print("\n  --- MCS + WDV ---")
    for mcs in [1, 3, 5]:
        for ra in [0.5, 1.0]:
            params = {**LGBM_PARAMS, "min_child_samples": mcs, "reg_alpha": ra}
            name = f"wdv_mcs{mcs}_ra{ra}"
            score, folds = run_wdv_pl_cv(
                name, X_train, y_train, groups, X_test, pseudo,
                BEST_PREPROCESS, params,
                wdv_fn=generate_wdv_raw,
                wdv_kwargs={"n_aug": 50, "extrap_factor": 1.5, "min_moisture": 150},
                pw=0.5,
            )
            results.append((name, score, folds))


# ============================================================================
# Section D: PCA → KRR as PL teacher (break 203% ceiling)
# ============================================================================

def section_d(X_train, y_train, groups, X_test, results):
    print("\n=== Section D: PCA → KRR/KNN as PL teacher ===")
    gkf = GroupKFold(n_splits=5)

    # Generate KRR-based pseudo-labels
    print("\n  --- KRR pseudo-label generation ---")
    for n_pca in [30, 50, 80]:
        for alpha in [0.1, 1.0, 10.0]:
            for gamma in [0.01, 0.1, 1.0]:
                try:
                    pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                    X_all = np.vstack([X_train, X_test])
                    X_all_t = pipe.fit_transform(X_all)

                    pca = PCA(n_components=n_pca)
                    X_all_pca = pca.fit_transform(X_all_t)

                    krr = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
                    krr.fit(X_all_pca[:len(y_train)], y_train)
                    krr_pseudo = krr.predict(X_all_pca[len(y_train):])

                    print(f"  KRR pca{n_pca}_a{alpha}_g{gamma}: range=[{krr_pseudo.min():.1f}, {krr_pseudo.max():.1f}]")

                    if krr_pseudo.max() > 210:  # Only interesting if it extrapolates
                        # Test this as PL
                        oof = np.full(len(y_train), np.nan)
                        fold_rmses = []
                        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                            X_tr, X_val = X_train[train_idx], X_train[val_idx]
                            y_tr, y_val = y_train[train_idx], y_train[val_idx]

                            X_aug = np.vstack([X_tr, X_test])
                            y_aug = np.concatenate([y_tr, krr_pseudo])
                            w = np.concatenate([np.ones(len(y_tr)), np.full(len(krr_pseudo), 1.0)])

                            p = build_preprocess_pipeline(BEST_PREPROCESS)
                            X_tr_t = p.fit_transform(X_aug)
                            X_val_t = p.transform(X_val)

                            model = create_model("lgbm", LGBM_PARAMS)
                            model.fit(X_tr_t, y_aug, sample_weight=w)
                            oof[val_idx] = model.predict(X_val_t).ravel()
                            fold_rmses.append(rmse(y_val, oof[val_idx]))

                        name = f"krr_pl_pca{n_pca}_a{alpha}_g{gamma}"
                        score, _ = save_result(name, oof, fold_rmses, y_train)
                        print(f"    → {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
                        results.append((name, score, fold_rmses))
                except Exception as e:
                    pass  # KRR can fail with bad params

    # Also try blending KRR + LGBM pseudo-labels
    print("\n  --- Blended KRR + LGBM pseudo-labels ---")
    lgbm_pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    for n_pca in [50]:
        for alpha in [1.0]:
            for gamma in [0.1]:
                try:
                    pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                    X_all = np.vstack([X_train, X_test])
                    X_all_t = pipe.fit_transform(X_all)
                    pca = PCA(n_components=n_pca)
                    X_all_pca = pca.fit_transform(X_all_t)
                    krr = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
                    krr.fit(X_all_pca[:len(y_train)], y_train)
                    krr_pseudo = krr.predict(X_all_pca[len(y_train):])

                    # Blend: use LGBM for most, KRR for high-end extrapolation
                    for blend_threshold in [150, 180]:
                        for krr_weight in [0.3, 0.5, 0.7]:
                            blended = lgbm_pseudo.copy()
                            high_mask = lgbm_pseudo > blend_threshold
                            blended[high_mask] = (1 - krr_weight) * lgbm_pseudo[high_mask] + krr_weight * krr_pseudo[high_mask]

                            oof = np.full(len(y_train), np.nan)
                            fold_rmses = []
                            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                                X_aug = np.vstack([X_tr, X_test])
                                y_aug = np.concatenate([y_tr, blended])
                                w = np.concatenate([np.ones(len(y_tr)), np.full(len(blended), 1.0)])

                                p = build_preprocess_pipeline(BEST_PREPROCESS)
                                X_tr_t = p.fit_transform(X_aug)
                                X_val_t = p.transform(X_val)

                                model = create_model("lgbm", LGBM_PARAMS)
                                model.fit(X_tr_t, y_aug, sample_weight=w)
                                oof[val_idx] = model.predict(X_val_t).ravel()
                                fold_rmses.append(rmse(y_val, oof[val_idx]))

                            name = f"blend_krr_lgbm_t{blend_threshold}_kw{krr_weight}"
                            score, _ = save_result(name, oof, fold_rmses, y_train)
                            print(f"    → {name}: RMSE={score:.4f}")
                            results.append((name, score, fold_rmses))
                except Exception as e:
                    print(f"  Blend ERROR: {e}")


# ============================================================================
# Section E: Prediction stretching for pseudo-labels
# ============================================================================

def section_e(X_train, y_train, groups, X_test, results):
    print("\n=== Section E: Prediction Stretching for PL ===")
    gkf = GroupKFold(n_splits=5)

    lgbm_pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)
    print(f"  LGBM pseudo range: [{lgbm_pseudo.min():.1f}, {lgbm_pseudo.max():.1f}]")

    for threshold in [150, 170, 180, 190]:
        for stretch in [1.3, 1.5, 2.0, 2.5, 3.0]:
            stretched = lgbm_pseudo.copy()
            mask = stretched > threshold
            stretched[mask] = threshold + (stretched[mask] - threshold) * stretch
            new_max = stretched.max()

            name = f"stretch_t{threshold}_s{stretch}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                X_aug = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, stretched])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(stretched), 1.0)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_t = pipe.fit_transform(X_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model("lgbm", LGBM_PARAMS)
                model.fit(X_tr_t, y_aug, sample_weight=w)
                oof[val_idx] = model.predict(X_val_t).ravel()
                fold_rmses.append(rmse(y_val, oof[val_idx]))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name} (max={new_max:.0f}): RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append((name, score, fold_rmses))

    # Stretched PL + WDV
    print("\n  --- Stretched PL + WDV ---")
    for threshold, stretch in [(170, 2.0), (180, 1.5), (150, 2.5)]:
        stretched = lgbm_pseudo.copy()
        mask = stretched > threshold
        stretched[mask] = threshold + (stretched[mask] - threshold) * stretch

        name = f"stretch_wdv_t{threshold}_s{stretch}"
        score, folds = run_wdv_pl_cv(
            name, X_train, y_train, groups, X_test, stretched,
            BEST_PREPROCESS, LGBM_PARAMS,
            wdv_fn=generate_wdv_raw,
            wdv_kwargs={"n_aug": 50, "extrap_factor": 1.5, "min_moisture": 150},
            pw=0.5,
        )
        results.append((name, score, folds))


# ============================================================================
# Section F: Two-expert model
# ============================================================================

def section_f(X_train, y_train, groups, X_test, results):
    print("\n=== Section F: Two-expert model ===")
    gkf = GroupKFold(n_splits=5)

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    for split_val in [80, 100, 120]:
        for wdv_n in [30, 50]:
            name = f"2expert_s{split_val}_n{wdv_n}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                g_tr = groups[train_idx]

                # Expert 1: Low moisture (no WDV, clean)
                pipe_low = build_preprocess_pipeline(BEST_PREPROCESS)
                X_aug_low = np.vstack([X_tr, X_test])
                y_aug_low = np.concatenate([y_tr, pseudo])
                w_low = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0)])

                X_tr_low = pipe_low.fit_transform(X_aug_low)
                X_val_low = pipe_low.transform(X_val)

                model_low = create_model("lgbm", LGBM_PARAMS)
                model_low.fit(X_tr_low, y_aug_low, sample_weight=w_low)
                pred_low = model_low.predict(X_val_low).ravel()

                # Expert 2: High moisture (with WDV, aggressive extrapolation)
                s_X, s_y = generate_wdv_raw(X_tr, y_tr, g_tr, wdv_n, 1.5, min_moisture=100)
                if len(s_X) > 0:
                    X_aug_high = np.vstack([X_tr, X_test, s_X])
                    y_aug_high = np.concatenate([y_tr, pseudo, s_y])
                    w_high = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 1.0),
                                             np.full(len(s_y), 0.5)])
                else:
                    X_aug_high = X_aug_low
                    y_aug_high = y_aug_low
                    w_high = w_low

                params_high = {**LGBM_PARAMS, "min_child_samples": 5, "reg_alpha": 0.5}
                pipe_high = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_high = pipe_high.fit_transform(X_aug_high)
                X_val_high = pipe_high.transform(X_val)

                model_high = create_model("lgbm", params_high)
                model_high.fit(X_tr_high, y_aug_high, sample_weight=w_high)
                pred_high = model_high.predict(X_val_high).ravel()

                # Adaptive blend: use low expert for low predictions, high expert for high
                alpha = np.clip((pred_low - split_val) / 50.0, 0, 1)  # sigmoid-like
                pred_blend = (1 - alpha) * pred_low + alpha * pred_high

                oof[val_idx] = pred_blend
                fold_rmses.append(rmse(y_val, pred_blend))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append((name, score, fold_rmses))


# ============================================================================
# Section G: Autoencoder features
# ============================================================================

def section_g(X_train, y_train, groups, X_test, results):
    print("\n=== Section G: Autoencoder features ===")
    gkf = GroupKFold(n_splits=5)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  PyTorch not available, skipping")
        return

    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    # Preprocess all data
    pipe_ae = build_preprocess_pipeline(BEST_PREPROCESS)
    X_all = np.vstack([X_train, X_test])
    X_all_t = pipe_ae.fit_transform(X_all)
    n_features = X_all_t.shape[1]

    for bottleneck in [16, 32, 64]:
        print(f"\n  --- Autoencoder bottleneck={bottleneck} ---")

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, bottleneck_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, bottleneck_dim),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        # Train autoencoder on ALL data (unsupervised)
        X_tensor = torch.FloatTensor(X_all_t)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        ae = Autoencoder(n_features, bottleneck)
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        ae.train()
        for epoch in range(100):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon, _ = ae(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Extract features
        ae.eval()
        with torch.no_grad():
            _, Z_all = ae(X_tensor)
            Z_all = Z_all.numpy()

        Z_train = Z_all[:len(y_train)]
        Z_test = Z_all[len(y_train):]
        print(f"  AE features: {Z_all.shape}")

        # Test 1: AE features only
        name = f"ae{bottleneck}_only"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            Z_tr = np.vstack([Z_train[train_idx], Z_test])
            Z_val = Z_train[val_idx]
            y_aug = np.concatenate([y_train[train_idx], pseudo])
            w = np.concatenate([np.ones(len(train_idx)), np.full(len(pseudo), 1.0)])

            model = create_model("lgbm", LGBM_PARAMS)
            model.fit(Z_tr, y_aug, sample_weight=w)
            oof[val_idx] = model.predict(Z_val).ravel()
            fold_rmses.append(rmse(y_train[val_idx], oof[val_idx]))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}")
        results.append((name, score, fold_rmses))

        # Test 2: AE features concatenated with original
        name = f"ae{bottleneck}_concat"
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            g_tr = groups[train_idx]
            y_tr = y_train[train_idx]

            s_X, s_y = generate_wdv_raw(X_tr, y_tr, g_tr, 50, 1.5, min_moisture=150)

            if len(s_X) > 0:
                X_aug_raw = np.vstack([X_tr, X_test, s_X])
                y_aug = np.concatenate([y_tr, pseudo, s_y])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 0.5),
                                    np.full(len(s_y), 0.3)])
            else:
                X_aug_raw = np.vstack([X_tr, X_test])
                y_aug = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 0.5)])

            pipe_c = build_preprocess_pipeline(BEST_PREPROCESS)
            X_aug_t = pipe_c.fit_transform(X_aug_raw)
            X_val_t = pipe_c.transform(X_val)

            # AE features for aug data (only for train+test, not WDV synthetic)
            n_real = len(y_tr) + len(pseudo)
            Z_aug = np.zeros((len(X_aug_t), bottleneck))
            Z_aug[:len(y_tr)] = Z_train[train_idx]
            Z_aug[len(y_tr):n_real] = Z_test
            # For WDV synthetic, use nearest neighbor AE features
            if len(s_X) > 0:
                from sklearn.neighbors import NearestNeighbors
                nn_model = NearestNeighbors(n_neighbors=1)
                nn_model.fit(Z_train)
                # Use the AE features of nearest training sample
                pipe_s = build_preprocess_pipeline(BEST_PREPROCESS)
                # Can't easily get AE features for synthetic data, use zeros
                # This is fine since LGBM can handle it

            X_aug_final = np.hstack([X_aug_t, Z_aug])
            X_val_final = np.hstack([X_val_t, Z_train[val_idx]])

            model = create_model("lgbm", LGBM_PARAMS)
            model.fit(X_aug_final, y_aug, sample_weight=w)
            oof[val_idx] = model.predict(X_val_final).ravel()
            fold_rmses.append(rmse(y_train[val_idx], oof[val_idx]))

        score, _ = save_result(name, oof, fold_rmses, y_train)
        print(f"  {name}: RMSE={score:.4f}")
        results.append((name, score, fold_rmses))


# ============================================================================
# Section H: KNN regression blend
# ============================================================================

def section_h(X_train, y_train, groups, X_test, results):
    print("\n=== Section H: KNN Regression ===")
    gkf = GroupKFold(n_splits=5)
    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    for k in [5, 10, 20, 50]:
        for metric in ["euclidean", "cosine"]:
            name = f"knn_k{k}_{metric}"
            oof = np.full(len(y_train), np.nan)
            fold_rmses = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Use PCA-reduced features for KNN
                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_aug = np.vstack([X_tr, X_test])
                X_aug_t = pipe.fit_transform(X_aug)
                X_val_t = pipe.transform(X_val)

                pca = PCA(n_components=50)
                X_aug_pca = pca.fit_transform(X_aug_t)
                X_val_pca = pca.transform(X_val_t)

                y_aug = np.concatenate([y_tr, pseudo])

                knn = KNeighborsRegressor(n_neighbors=k, metric=metric, weights='distance')
                knn.fit(X_aug_pca, y_aug)
                oof[val_idx] = knn.predict(X_val_pca).ravel()
                fold_rmses.append(rmse(y_val, oof[val_idx]))

            score, _ = save_result(name, oof, fold_rmses, y_train)
            print(f"  {name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
            results.append((name, score, fold_rmses))

    # KNN + LGBM blend
    print("\n  --- KNN + LGBM blend ---")
    for k in [10, 20]:
        name = f"blend_knn{k}_lgbm"
        oof_knn = np.full(len(y_train), np.nan)
        oof_lgbm = np.full(len(y_train), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            g_tr = groups[train_idx]

            # KNN
            pipe_knn = build_preprocess_pipeline(BEST_PREPROCESS)
            X_aug = np.vstack([X_tr, X_test])
            X_aug_t = pipe_knn.fit_transform(X_aug)
            X_val_t_knn = pipe_knn.transform(X_val)
            pca = PCA(n_components=50)
            X_aug_pca = pca.fit_transform(X_aug_t)
            X_val_pca = pca.transform(X_val_t_knn)
            y_aug = np.concatenate([y_tr, pseudo])
            knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean', weights='distance')
            knn.fit(X_aug_pca, y_aug)
            oof_knn[val_idx] = knn.predict(X_val_pca).ravel()

            # LGBM with WDV
            s_X, s_y = generate_wdv_raw(X_tr, y_tr, g_tr, 50, 1.5, min_moisture=150)
            if len(s_X) > 0:
                X_aug2 = np.vstack([X_tr, X_test, s_X])
                y_aug2 = np.concatenate([y_tr, pseudo, s_y])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 0.5),
                                    np.full(len(s_y), 0.3)])
            else:
                X_aug2 = np.vstack([X_tr, X_test])
                y_aug2 = np.concatenate([y_tr, pseudo])
                w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 0.5)])

            pipe_lgbm = build_preprocess_pipeline(BEST_PREPROCESS)
            X_aug2_t = pipe_lgbm.fit_transform(X_aug2)
            X_val_lgbm = pipe_lgbm.transform(X_val)
            model = create_model("lgbm", LGBM_PARAMS)
            model.fit(X_aug2_t, y_aug2, sample_weight=w)
            oof_lgbm[val_idx] = model.predict(X_val_lgbm).ravel()

        # Optimize blend
        from scipy.optimize import minimize_scalar

        def blend_rmse(w_knn):
            pred = w_knn * oof_knn + (1 - w_knn) * oof_lgbm
            return rmse(y_train, pred)

        res = minimize_scalar(blend_rmse, bounds=(0, 0.5), method='bounded')
        best_w = res.x
        blended = best_w * oof_knn + (1 - best_w) * oof_lgbm

        fold_rmses = []
        for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            fold_rmses.append(rmse(y_train[val_idx], blended[val_idx]))

        score = rmse(y_train, blended)
        save_result(name, blended, fold_rmses, y_train, {"knn_weight": float(best_w)})
        print(f"  {name}: RMSE={score:.4f} (knn_w={best_w:.3f})  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append((name, score, fold_rmses))


# ============================================================================
# Section I: Grand Combined
# ============================================================================

def section_i(X_train, y_train, groups, X_test, results):
    print("\n=== Section I: Grand Combined ===")
    gkf = GroupKFold(n_splits=5)
    pseudo = generate_pseudo_labels(X_train, y_train, X_test, BEST_PREPROCESS, "lgbm", LGBM_PARAMS)

    # Collect all OOF predictions from diverse approaches
    all_oofs = []

    # Config variants
    configs = [
        (LGBM_PARAMS, 50, 1.5, 150),  # base WDV
        ({**LGBM_PARAMS, "min_child_samples": 5, "reg_alpha": 0.5}, 50, 1.5, 150),  # low mcs
        ({**LGBM_PARAMS, "n_estimators": 800, "learning_rate": 0.02}, 50, 1.5, 150),
        ({**LGBM_PARAMS, "max_depth": 7, "num_leaves": 32}, 30, 1.0, 80),
        (LGBM_PARAMS, 30, 1.0, 80),  # different WDV params
    ]

    for cfg_idx, (params, n_aug, ef, mm) in enumerate(configs):
        for seed in range(3):
            p = dict(params)
            p["random_state"] = seed * 7 + 42
            p["bagging_seed"] = seed * 11 + 17

            oof = np.full(len(y_train), np.nan)
            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr = y_train[train_idx]
                g_tr = groups[train_idx]

                s_X, s_y = generate_wdv_raw(X_tr, y_tr, g_tr, n_aug, ef, min_moisture=mm)
                if len(s_X) > 0:
                    X_aug = np.vstack([X_tr, X_test, s_X])
                    y_aug = np.concatenate([y_tr, pseudo, s_y])
                    w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 0.5),
                                        np.full(len(s_y), 0.3)])
                else:
                    X_aug = np.vstack([X_tr, X_test])
                    y_aug = np.concatenate([y_tr, pseudo])
                    w = np.concatenate([np.ones(len(y_tr)), np.full(len(pseudo), 0.5)])

                pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                X_tr_t = pipe.fit_transform(X_aug)
                X_val_t = pipe.transform(X_val)

                model = create_model("lgbm", p)
                model.fit(X_tr_t, y_aug, sample_weight=w)
                oof[val_idx] = model.predict(X_val_t).ravel()

            all_oofs.append(oof)

    # Grand average
    grand_oof = np.mean(all_oofs, axis=0)
    fold_rmses = []
    for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_rmses.append(rmse(y_train[val_idx], grand_oof[val_idx]))
    score = rmse(y_train, grand_oof)
    save_result("grand_p16_avg", grand_oof, fold_rmses, y_train)
    print(f"  Grand P16 avg ({len(all_oofs)} models): RMSE={score:.4f}")
    print(f"    folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
    results.append(("grand_p16_avg", score, fold_rmses))

    # Optimized
    from scipy.optimize import minimize
    def ens_rmse(weights):
        weights = np.abs(weights) / np.abs(weights).sum()
        pred = sum(w * o for w, o in zip(weights, all_oofs))
        return rmse(y_train, pred)
    w0 = np.ones(len(all_oofs)) / len(all_oofs)
    res = minimize(ens_rmse, w0, method="Nelder-Mead", options={"maxiter": 5000})
    opt_w = np.abs(res.x) / np.abs(res.x).sum()
    opt_oof = sum(w * o for w, o in zip(opt_w, all_oofs))
    fold_rmses_opt = []
    for _, (_, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_rmses_opt.append(rmse(y_train[val_idx], opt_oof[val_idx]))
    score_opt = rmse(y_train, opt_oof)
    save_result("grand_p16_opt", opt_oof, fold_rmses_opt, y_train)
    print(f"  Grand P16 opt: RMSE={score_opt:.4f}")
    results.append(("grand_p16_opt", score_opt, fold_rmses_opt))


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 16: Final Push — All Ideas Combined")
    print("=" * 70)

    np.random.seed(42)
    X_train, y_train, groups, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Current best: RMSE = 15.10 (Phase 15c)")

    results = []

    sections = [
        ("A: WDV EMSC space", section_a),
        ("B: WDV Basis", section_b),
        ("C: min_child_samples", section_c),
        ("D: KRR PL teacher", section_d),
        ("E: Prediction stretching", section_e),
        ("F: Two-expert", section_f),
        ("G: Autoencoder", section_g),
        ("H: KNN blend", section_h),
        ("I: Grand combined", section_i),
    ]

    for name, fn in sections:
        try:
            fn(X_train, y_train, groups, X_test, results)
        except Exception as e:
            print(f"  {name} ERROR: {e}")
            traceback.print_exc()

    # FINAL SUMMARY
    print("\n\n" + "=" * 70)
    print("PHASE 16 FINAL SUMMARY")
    print("=" * 70)
    results.sort(key=lambda x: x[1])
    for name, score, folds in results[:40]:
        fold_str = ', '.join(f'{f:.1f}' for f in folds)
        marker = " ★★" if score < 15.0 else " ★" if score < 15.10 else ""
        print(f"  {score:.4f}  [{fold_str}]  {name}{marker}")

    if results:
        best = results[0]
        print(f"\nBEST: {best[1]:.4f} ({best[0]})")
        print(f"Phase 15c best: 15.10")
        improved = sum(1 for r in results if r[1] < 15.10)
        print(f"{improved} / {len(results)} beat Phase 15c best")


if __name__ == "__main__":
    main()
