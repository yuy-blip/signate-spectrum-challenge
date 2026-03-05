#!/usr/bin/env python
"""Phase 21: MEGA STRATEGY — Stacking + Ceiling Breaking + Species 15 Attack.

Three strategies executed together:
  1. Stacking: Diverse base models → Ridge/ElasticNet meta-learner (GroupKFold)
  2. Ceiling Breaking: Residual correction with water-band linear model + extreme WDV
  3. Species 15 Attack: High-moisture upweighted models + species-adapted WDV

All OOF predictions are collected and combined in a final mega-ensemble.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")

# ==========================================================================
# Constants
# ==========================================================================
BEST_PREPROCESS = [
    {"name": "emsc", "poly_order": 2},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

LGBM_BASE = {
    "n_estimators": 400, "max_depth": 5, "num_leaves": 20,
    "learning_rate": 0.05, "min_child_samples": 20,
    "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "verbose": -1, "n_jobs": -1,
}

# Water absorption band indices (approximate for NIR)
# ~1400-1500nm and ~1900-2000nm regions are key water bands
WATER_BAND_INDICES = None  # Will be auto-detected


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

    # Get wavelength column names for water band detection
    float_cols = [c for c in df.columns if _is_float(c)]
    wavelengths = np.array([float(c) for c in float_cols])

    return X_train, y_train, groups, X_test, test_ids, wavelengths


def _is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def detect_water_bands(wavelengths, X_train, y_train):
    """Detect which feature indices correspond to water absorption bands."""
    # Compute correlation between each feature and target
    corrs = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1]
                       for i in range(X_train.shape[1])])
    # Top 100 most correlated features (water-sensitive)
    top_idx = np.argsort(np.abs(corrs))[-100:]
    return top_idx


# ==========================================================================
# Universal WDV (from Phase 20)
# ==========================================================================
def generate_universal_wdv(X_tr, y_tr, groups_tr, n_aug=30,
                           extrap_factor=1.5, min_moisture=150,
                           dy_scale=0.3, dy_offset=30):
    species_deltas = []
    species_dy = []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 5:
            continue
        median_y = np.median(y_sp)
        high = X_sp[y_sp >= median_y]
        low = X_sp[y_sp < median_y]
        if len(high) < 2 or len(low) < 2:
            continue
        delta = high.mean(axis=0) - low.mean(axis=0)
        dy = y_sp[y_sp >= median_y].mean() - y_sp[y_sp < median_y].mean()
        if dy > 5:
            species_deltas.append(delta / dy)
            species_dy.append(dy)

    if len(species_deltas) < 2:
        return np.empty((0, X_tr.shape[1])), np.empty(0)

    species_deltas = np.array(species_deltas)
    if len(species_deltas) >= 3:
        pca = PCA(n_components=1)
        pca.fit(species_deltas)
        water_vec = pca.components_[0]
        if np.corrcoef(species_deltas @ water_vec, species_dy)[0, 1] < 0:
            water_vec = -water_vec
    else:
        water_vec = species_deltas.mean(axis=0)
        water_vec /= np.linalg.norm(water_vec) + 1e-8

    from numpy.polynomial.polynomial import polyfit
    proj = X_tr @ water_vec
    coeffs = polyfit(proj, y_tr, 1)
    scale = coeffs[1]

    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 1:
            continue
        for hi_idx in np.where(high_mask)[0]:
            target_dy = extrap_factor * (y_sp[hi_idx] * dy_scale + dy_offset)
            step = target_dy / (scale + 1e-8)
            synth_X.append(X_sp[hi_idx] + step * water_vec)
            synth_y.append(y_sp[hi_idx] + target_dy)

    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        idx = np.linspace(0, len(synth_X) - 1, n_aug, dtype=int)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


# ==========================================================================
# Core CV function for diverse models
# ==========================================================================
def cv_model(X_train, y_train, groups, X_test,
             preprocess=None, lgbm_params=None,
             n_aug=0, extrap=1.5, min_moisture=150,
             dy_scale=0.3, dy_offset=30,
             pl_w=0.0, pl_rounds=1,
             sample_weight_fn=None,
             model_type="lgbm", model_params=None,
             clip_min=0.0, clip_max=500.0):
    """Generic CV with many knobs for diversity."""
    if preprocess is None:
        preprocess = BEST_PREPROCESS
    if lgbm_params is None and model_type == "lgbm":
        lgbm_params = {**LGBM_BASE}
    if model_params is None:
        model_params = lgbm_params if model_type == "lgbm" else {}

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_train))
    fold_rmses = []
    test_preds_all = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups[tr_idx]

        pipe = build_preprocess_pipeline(preprocess)
        pipe.fit(X_tr)

        # WDV augmentation
        if n_aug > 0:
            synth_X, synth_y = generate_universal_wdv(
                X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset
            )
        else:
            synth_X, synth_y = np.empty((0, X_tr.shape[1])), np.empty(0)

        X_tr_t = pipe.transform(X_tr)
        X_va_t = pipe.transform(X_va)
        X_test_t = pipe.transform(X_test)

        if len(synth_X) > 0:
            synth_X_t = pipe.transform(synth_X)
            X_tr_aug = np.vstack([X_tr_t, synth_X_t])
            y_tr_aug = np.concatenate([y_tr, synth_y])
        else:
            X_tr_aug = X_tr_t
            y_tr_aug = y_tr

        # Sample weights
        sw = None
        if sample_weight_fn is not None:
            base_sw = sample_weight_fn(y_tr)
            if len(synth_y) > 0:
                synth_sw = sample_weight_fn(synth_y)
                sw = np.concatenate([base_sw, synth_sw])
            else:
                sw = base_sw

        # Pseudo-labeling rounds
        current_X = X_tr_aug
        current_y = y_tr_aug
        current_sw = sw
        test_pred = None

        for pl_round in range(pl_rounds):
            if pl_round > 0 and pl_w > 0 and test_pred is not None:
                # Use previous round's test preds as PL
                current_X = np.vstack([X_tr_aug, X_test_t])
                current_y = np.concatenate([y_tr_aug, test_pred])
                w = np.ones(len(current_y))
                w[-len(test_pred):] = pl_w
                if current_sw is not None:
                    w[:len(current_sw)] = current_sw[:]  # keep original weights for train
                current_sw = w
            elif pl_round == 0 and pl_w > 0:
                # First round PL
                temp_model = create_model(model_type, model_params)
                if current_sw is not None:
                    temp_model.fit(current_X, current_y, sample_weight=current_sw)
                else:
                    temp_model.fit(current_X, current_y)
                pl_pred = temp_model.predict(X_test_t)
                current_X = np.vstack([X_tr_aug, X_test_t])
                current_y = np.concatenate([y_tr_aug, pl_pred])
                w = np.ones(len(current_y))
                w[-len(pl_pred):] = pl_w
                if sw is not None:
                    w[:len(sw)] = sw
                current_sw = w

            model = create_model(model_type, model_params)
            if current_sw is not None:
                model.fit(current_X, current_y, sample_weight=current_sw)
            else:
                model.fit(current_X, current_y)
            test_pred = model.predict(X_test_t).ravel()

        val_pred = model.predict(X_va_t).ravel()
        val_pred = np.clip(val_pred, clip_min, clip_max)
        test_pred = np.clip(test_pred, clip_min, clip_max)

        oof[va_idx] = val_pred
        fold_rmses.append(rmse(y_va, val_pred))
        test_preds_all.append(test_pred)

    return oof, fold_rmses, np.mean(test_preds_all, axis=0)


# ==========================================================================
# STRATEGY 1: Diverse Stacking Base Models
# ==========================================================================
def build_diverse_models(X_train, y_train, groups, X_test, wavelengths):
    """Build a collection of diverse base models for stacking."""
    print("\n" + "=" * 70)
    print("STRATEGY 1: DIVERSE STACKING")
    print("=" * 70)

    models = {}

    # --- Model 1: Best known config (UW + iterPL) ---
    print("\n  [1/12] Best config: UW30 + iterPL2 pw0.5")
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["best_uw30_iterpl2_pw05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 2: Different PL weight ---
    print("\n  [2/12] UW30 + PL pw0.3")
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=30, extrap=1.5, pl_w=0.3, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["uw30_iterpl2_pw03"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 3: Higher extrap ---
    print("\n  [3/12] UW40 extrap=2.0 + PL pw0.5")
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=40, extrap=2.0, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["uw40_extrap2_iterpl2"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 4: Binning(4) pipeline ---
    print("\n  [4/12] Binning(4) + UW + PL")
    pp_bin4 = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 4},
        {"name": "standard_scaler"},
    ]
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        preprocess=pp_bin4, n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["bin4_uw30_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 5: SG window=9 ---
    print("\n  [5/12] SG(9) + UW + PL")
    pp_sg9 = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 9, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        preprocess=pp_sg9, n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["sg9_uw30_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 6: Deeper LGBM ---
    print("\n  [6/12] Deep LGBM (d7/l32) + UW + PL")
    deep_params = {**LGBM_BASE, "max_depth": 7, "num_leaves": 32,
                   "n_estimators": 600, "learning_rate": 0.03}
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        lgbm_params=deep_params, n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["deep_d7l32_uw30_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 7: Shallow LGBM ---
    print("\n  [7/12] Shallow LGBM (d4/l15) + UW + PL")
    shallow_params = {**LGBM_BASE, "max_depth": 4, "num_leaves": 15,
                      "n_estimators": 800, "learning_rate": 0.02}
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        lgbm_params=shallow_params, n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["shallow_d4l15_uw30_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 8: More regularized LGBM ---
    print("\n  [8/12] Regularized LGBM + UW + PL")
    reg_params = {**LGBM_BASE, "reg_alpha": 0.5, "reg_lambda": 3.0,
                  "min_child_samples": 30}
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        lgbm_params=reg_params, n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["reg_heavy_uw30_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 9: No WDV, just PL (different error profile) ---
    print("\n  [9/12] No WDV, PL only")
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=0, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["no_wdv_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 10: XGBoost ---
    print("\n  [10/12] XGBoost + UW + PL")
    xgb_params = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbosity": 0, "n_jobs": -1,
    }
    try:
        oof, folds, tp = cv_model(
            X_train, y_train, groups, X_test,
            model_type="xgb", model_params=xgb_params,
            n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
        )
        score = rmse(y_train, oof)
        print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
        models["xgb_uw30_pl05"] = {"oof": oof, "test": tp, "rmse": score}
    except Exception as e:
        print(f"         FAILED: {e}")

    # --- Model 11: Binning(16) wider bins ---
    print("\n  [11/12] Binning(16) + UW + PL")
    pp_bin16 = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16},
        {"name": "standard_scaler"},
    ]
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        preprocess=pp_bin16, n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["bin16_uw30_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- Model 12: 3-round iterative PL ---
    print("\n  [12/12] UW30 + 3-round iterPL")
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=3
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["uw30_iterpl3_pw05"] = {"oof": oof, "test": tp, "rmse": score}

    return models


# ==========================================================================
# STRATEGY 2: Ceiling Breaking
# ==========================================================================
def build_ceiling_breakers(X_train, y_train, groups, X_test, wavelengths):
    """Models specifically designed to break the prediction ceiling."""
    print("\n" + "=" * 70)
    print("STRATEGY 2: CEILING BREAKING")
    print("=" * 70)

    models = {}

    # --- A. Extreme WDV extrap (3.0, 4.0) ---
    # These are individually noisy but in stacking they provide high-value info
    print("\n  [A1] Extreme WDV extrap=3.0 + PL")
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=50, extrap=3.0, min_moisture=130, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["extreme_wdv_f3_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    print("\n  [A2] Extreme WDV extrap=4.0 + PL")
    oof, folds, tp = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=50, extrap=4.0, min_moisture=130, pl_w=0.5, pl_rounds=2
    )
    score = rmse(y_train, oof)
    print(f"         RMSE={score:.4f} folds={[round(f,1) for f in folds]}")
    models["extreme_wdv_f4_pl05"] = {"oof": oof, "test": tp, "rmse": score}

    # --- B. Residual correction with water band linear model ---
    print("\n  [B] Residual correction: LGBM base + Ridge on water features")
    gkf = GroupKFold(n_splits=5)

    # First pass: get base LGBM predictions
    oof_base, folds_base, tp_base = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )
    base_score = rmse(y_train, oof_base)
    print(f"    Base LGBM RMSE: {base_score:.4f}")

    # Compute residuals
    residuals = y_train - oof_base

    # Build water-band features for residual prediction
    # Use preprocessed features projected onto water direction
    pipe_full = build_preprocess_pipeline(BEST_PREPROCESS)
    pipe_full.fit(X_train)
    X_train_t = pipe_full.transform(X_train)
    X_test_t = pipe_full.transform(X_test)

    # Add raw-level water features: variance, range, specific bands
    def water_features(X_raw, X_proc):
        """Extract water-sensitive features."""
        feats = []
        # Processed feature stats
        feats.append(X_proc.mean(axis=1, keepdims=True))
        feats.append(X_proc.std(axis=1, keepdims=True))
        feats.append(X_proc.max(axis=1, keepdims=True))
        feats.append(X_proc.min(axis=1, keepdims=True))
        feats.append((X_proc.max(axis=1) - X_proc.min(axis=1)).reshape(-1, 1))
        # Raw spectral features
        n_feat = X_raw.shape[1]
        # Split into quarters
        q = n_feat // 4
        for i in range(4):
            feats.append(X_raw[:, i*q:(i+1)*q].mean(axis=1, keepdims=True))
            feats.append(X_raw[:, i*q:(i+1)*q].std(axis=1, keepdims=True))
        # PCA top 5 of processed
        pca = PCA(n_components=min(5, X_proc.shape[1]))
        pca.fit(X_proc)
        feats.append(pca.transform(X_proc))
        return np.hstack(feats), pca

    wf_train, pca_wf = water_features(X_train, X_train_t)
    # For test, need to use same PCA
    wf_test_parts = []
    wf_test_parts.append(X_test_t.mean(axis=1, keepdims=True))
    wf_test_parts.append(X_test_t.std(axis=1, keepdims=True))
    wf_test_parts.append(X_test_t.max(axis=1, keepdims=True))
    wf_test_parts.append(X_test_t.min(axis=1, keepdims=True))
    wf_test_parts.append((X_test_t.max(axis=1) - X_test_t.min(axis=1)).reshape(-1, 1))
    n_feat = X_test.shape[1]
    q = n_feat // 4
    for i in range(4):
        wf_test_parts.append(X_test[:, i*q:(i+1)*q].mean(axis=1, keepdims=True))
        wf_test_parts.append(X_test[:, i*q:(i+1)*q].std(axis=1, keepdims=True))
    wf_test_parts.append(pca_wf.transform(X_test_t))
    wf_test = np.hstack(wf_test_parts)

    # CV residual prediction
    oof_resid = np.zeros(len(y_train))
    test_resid_preds = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        # Train Ridge on residuals using water features
        ridge = Ridge(alpha=10.0)
        ridge.fit(wf_train[tr_idx], residuals[tr_idx])
        oof_resid[va_idx] = ridge.predict(wf_train[va_idx])
        test_resid_preds.append(ridge.predict(wf_test))

    # Corrected predictions
    oof_corrected = oof_base + oof_resid
    corrected_score = rmse(y_train, oof_corrected)
    tp_corrected = tp_base + np.mean(test_resid_preds, axis=0)
    print(f"    Residual-corrected RMSE: {corrected_score:.4f}")

    # Also try different alpha scales for the correction
    for alpha_scale in [0.3, 0.5, 0.7, 1.0]:
        oof_sc = oof_base + alpha_scale * oof_resid
        sc_score = rmse(y_train, oof_sc)
        print(f"    Alpha={alpha_scale}: RMSE={sc_score:.4f}")
        if sc_score < corrected_score:
            corrected_score = sc_score
            oof_corrected = oof_sc
            tp_corrected = tp_base + alpha_scale * np.mean(test_resid_preds, axis=0)

    folds_corr = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        folds_corr.append(rmse(y_train[va_idx], oof_corrected[va_idx]))
    print(f"    Best corrected: RMSE={corrected_score:.4f} folds={[round(f,1) for f in folds_corr]}")
    models["residual_corrected"] = {"oof": oof_corrected, "test": tp_corrected, "rmse": corrected_score}

    # --- C. Post-processing: stretch high predictions ---
    print("\n  [C] Post-processing: stretch high predictions")

    # Get base predictions
    oof_base2, folds_b2, tp_base2 = cv_model(
        X_train, y_train, groups, X_test,
        n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2
    )

    best_stretch_score = rmse(y_train, oof_base2)
    best_stretch_oof = oof_base2
    best_stretch_tp = tp_base2
    best_stretch_name = "no_stretch"

    for threshold_pct in [80, 85, 90, 95]:
        threshold = np.percentile(oof_base2, threshold_pct)
        for stretch in [1.05, 1.1, 1.15, 1.2, 1.3]:
            oof_s = oof_base2.copy()
            mask = oof_s > threshold
            oof_s[mask] = threshold + (oof_s[mask] - threshold) * stretch
            s = rmse(y_train, oof_s)
            if s < best_stretch_score:
                best_stretch_score = s
                best_stretch_oof = oof_s
                tp_s = tp_base2.copy()
                mask_t = tp_s > threshold
                tp_s[mask_t] = threshold + (tp_s[mask_t] - threshold) * stretch
                best_stretch_tp = tp_s
                best_stretch_name = f"stretch_p{threshold_pct}_s{stretch}"

    print(f"    Best stretch: {best_stretch_name} RMSE={best_stretch_score:.4f}")
    if best_stretch_score < rmse(y_train, oof_base2):
        models["post_stretch"] = {"oof": best_stretch_oof, "test": best_stretch_tp, "rmse": best_stretch_score}

    # --- D. Quantile-based calibration ---
    print("\n  [D] Quantile calibration")
    # For each fold, learn a monotonic mapping from predicted to actual
    oof_calib = np.zeros(len(y_train))
    test_calib_preds = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        # Sort by prediction
        pred_tr = oof_base[tr_idx]
        actual_tr = y_train[tr_idx]
        sort_idx = np.argsort(pred_tr)
        pred_sorted = pred_tr[sort_idx]
        actual_sorted = actual_tr[sort_idx]

        # Isotonic-like: moving average calibration
        from scipy.interpolate import interp1d
        # Smooth with window
        window = max(5, len(pred_sorted) // 20)
        pred_smooth = np.convolve(pred_sorted, np.ones(window)/window, mode='valid')
        actual_smooth = np.convolve(actual_sorted, np.ones(window)/window, mode='valid')

        # Create interpolator
        try:
            calib_fn = interp1d(pred_smooth, actual_smooth,
                               bounds_error=False, fill_value='extrapolate', kind='linear')
            oof_calib[va_idx] = calib_fn(oof_base[va_idx])
            test_calib_preds.append(calib_fn(tp_base))
        except Exception:
            oof_calib[va_idx] = oof_base[va_idx]
            test_calib_preds.append(tp_base)

    calib_score = rmse(y_train, oof_calib)
    print(f"    Calibrated RMSE: {calib_score:.4f}")
    if calib_score < base_score:
        models["calibrated"] = {"oof": oof_calib, "test": np.mean(test_calib_preds, axis=0), "rmse": calib_score}

    return models


# ==========================================================================
# STRATEGY 3: Species 15 Targeted Attack
# ==========================================================================
def build_species15_attackers(X_train, y_train, groups, X_test, wavelengths):
    """Models specifically targeting Species 15 improvement."""
    print("\n" + "=" * 70)
    print("STRATEGY 3: SPECIES 15 ATTACK")
    print("=" * 70)

    models = {}

    # --- A. High-moisture upweighting ---
    print("\n  [A] High-moisture sample upweighting")

    def high_moisture_weight(y, threshold=100, boost=3.0):
        w = np.ones(len(y))
        w[y > threshold] = boost
        return w

    for threshold in [80, 100, 120, 150]:
        for boost in [2.0, 3.0, 5.0]:
            name = f"upweight_t{threshold}_b{boost}"
            wfn = lambda y, t=threshold, b=boost: high_moisture_weight(y, t, b)
            oof, folds, tp = cv_model(
                X_train, y_train, groups, X_test,
                n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2,
                sample_weight_fn=wfn
            )
            score = rmse(y_train, oof)
            if score < 16.5:
                print(f"    {name}: RMSE={score:.4f} Fold2={folds[2]:.1f}")
                models[name] = {"oof": oof, "test": tp, "rmse": score}

    # --- B. WDV with lower min_moisture to help more species ---
    print("\n  [B] Broader WDV coverage")
    for mm in [50, 80, 100]:
        for n_aug in [40, 60]:
            name = f"broad_wdv_mm{mm}_n{n_aug}"
            oof, folds, tp = cv_model(
                X_train, y_train, groups, X_test,
                n_aug=n_aug, extrap=1.5, min_moisture=mm,
                pl_w=0.5, pl_rounds=2
            )
            score = rmse(y_train, oof)
            print(f"    {name}: RMSE={score:.4f} Fold2={folds[2]:.1f}")
            if score < 17.0:
                models[name] = {"oof": oof, "test": tp, "rmse": score}

    # --- C. Species-wise WDV: Different extrap by moisture range ---
    print("\n  [C] Adaptive WDV (extra extrap for high moisture species)")

    # Custom WDV that uses higher extrap for species with max(y) > 200
    gkf = GroupKFold(n_splits=5)
    oof_adaptive = np.zeros(len(y_train))
    fold_rmses_adaptive = []
    test_preds_adaptive = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups[tr_idx]

        pipe = build_preprocess_pipeline(BEST_PREPROCESS)
        pipe.fit(X_tr)

        # Standard WDV
        synth_X1, synth_y1 = generate_universal_wdv(
            X_tr, y_tr, g_tr, n_aug=30, extrap_factor=1.5, min_moisture=150
        )
        # Extra aggressive WDV for high moisture
        synth_X2, synth_y2 = generate_universal_wdv(
            X_tr, y_tr, g_tr, n_aug=30, extrap_factor=3.0, min_moisture=180,
            dy_scale=0.5, dy_offset=50
        )

        X_tr_t = pipe.transform(X_tr)
        X_va_t = pipe.transform(X_va)
        X_test_t = pipe.transform(X_test)

        all_X = [X_tr_t]
        all_y = [y_tr]
        for sx, sy in [(synth_X1, synth_y1), (synth_X2, synth_y2)]:
            if len(sx) > 0:
                all_X.append(pipe.transform(sx))
                all_y.append(sy)

        X_aug = np.vstack(all_X)
        y_aug = np.concatenate(all_y)

        # PL
        temp = create_model("lgbm", LGBM_BASE)
        temp.fit(X_aug, y_aug)
        pl = temp.predict(X_test_t)

        X_final = np.vstack([X_aug, X_test_t])
        y_final = np.concatenate([y_aug, pl])
        w = np.ones(len(y_final))
        w[-len(pl):] = 0.5

        model = create_model("lgbm", LGBM_BASE)
        model.fit(X_final, y_final, sample_weight=w)

        oof_adaptive[va_idx] = model.predict(X_va_t)
        fold_rmses_adaptive.append(rmse(y_va, oof_adaptive[va_idx]))
        test_preds_adaptive.append(model.predict(X_test_t))

    adaptive_score = rmse(y_train, oof_adaptive)
    print(f"    Adaptive WDV: RMSE={adaptive_score:.4f} folds={[round(f,1) for f in fold_rmses_adaptive]}")
    models["adaptive_wdv"] = {
        "oof": oof_adaptive,
        "test": np.mean(test_preds_adaptive, axis=0),
        "rmse": adaptive_score
    }

    # --- D. Fold 2-optimized: maximize Fold 2 performance specifically ---
    print("\n  [D] Fold2-optimized config sweeps")
    best_f2 = 999
    best_f2_model = None

    configs = [
        {"n_aug": 50, "extrap": 2.5, "min_moisture": 130, "dy_scale": 0.4, "dy_offset": 40},
        {"n_aug": 60, "extrap": 2.0, "min_moisture": 120, "dy_scale": 0.5, "dy_offset": 30},
        {"n_aug": 40, "extrap": 3.0, "min_moisture": 140, "dy_scale": 0.3, "dy_offset": 50},
        {"n_aug": 70, "extrap": 2.5, "min_moisture": 100, "dy_scale": 0.4, "dy_offset": 40},
    ]

    for i, cfg in enumerate(configs):
        name = f"fold2opt_{i}"
        oof, folds, tp = cv_model(
            X_train, y_train, groups, X_test,
            pl_w=0.5, pl_rounds=2, **cfg
        )
        score = rmse(y_train, oof)
        f2 = folds[2]
        print(f"    {name}: RMSE={score:.4f} Fold2={f2:.1f}")
        models[name] = {"oof": oof, "test": tp, "rmse": score}
        if f2 < best_f2:
            best_f2 = f2
            best_f2_model = name

    print(f"    Best Fold 2: {best_f2:.1f} ({best_f2_model})")

    return models


# ==========================================================================
# MEGA ENSEMBLE: Stack all models
# ==========================================================================
def mega_ensemble(all_models, y_train, groups, X_test_shape):
    """Build the final stacked ensemble from all strategies."""
    print("\n" + "=" * 70)
    print("MEGA ENSEMBLE: Stacking all models")
    print("=" * 70)

    # Collect all OOFs and test predictions
    names = list(all_models.keys())
    oofs = np.column_stack([all_models[n]["oof"] for n in names])
    tests = np.column_stack([all_models[n]["test"] for n in names])
    rmses = [all_models[n]["rmse"] for n in names]

    print(f"\n  Total models: {len(names)}")
    print(f"  OOF matrix shape: {oofs.shape}")
    for n, r in sorted(zip(names, rmses), key=lambda x: x[1]):
        print(f"    {r:.4f}  {n}")

    results = {}
    gkf = GroupKFold(n_splits=5)

    # --- 1. Simple average ---
    avg = oofs.mean(axis=1)
    avg_score = rmse(y_train, avg)
    avg_test = tests.mean(axis=1)
    print(f"\n  Simple average: RMSE={avg_score:.4f}")
    results["simple_avg"] = {"rmse": avg_score, "test": avg_test, "oof": avg}

    # --- 2. Inverse-RMSE weighted average ---
    inv_w = np.array([1.0 / r for r in rmses])
    inv_w /= inv_w.sum()
    wavg = (oofs * inv_w).sum(axis=1)
    wavg_score = rmse(y_train, wavg)
    wavg_test = (tests * inv_w).sum(axis=1)
    print(f"  Inv-RMSE weighted: RMSE={wavg_score:.4f}")
    results["inv_rmse_avg"] = {"rmse": wavg_score, "test": wavg_test, "oof": wavg}

    # --- 3. Greedy selection + average ---
    print("\n  --- Greedy selection ---")
    order = sorted(range(len(names)), key=lambda i: rmses[i])
    selected = [order[0]]
    best_greedy = rmses[order[0]]

    for step in range(len(order) - 1):
        cur_avg = oofs[:, selected].mean(axis=1)
        cur_rmse = rmse(y_train, cur_avg)
        best_score = cur_rmse
        best_idx = -1

        for i in order:
            if i in selected:
                continue
            new_avg = (cur_avg * len(selected) + oofs[:, i]) / (len(selected) + 1)
            s = rmse(y_train, new_avg)
            if s < best_score - 0.001:
                best_score = s
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)
            print(f"    +{len(selected)}: {names[best_idx][:40]:40s} ens={best_score:.4f}")
        else:
            break

    greedy_avg = oofs[:, selected].mean(axis=1)
    greedy_score = rmse(y_train, greedy_avg)
    greedy_test = tests[:, selected].mean(axis=1)
    print(f"  Greedy avg ({len(selected)} models): RMSE={greedy_score:.4f}")
    results["greedy_avg"] = {"rmse": greedy_score, "test": greedy_test, "oof": greedy_avg}

    # --- 4. Ridge stacking (GKF) ---
    print("\n  --- Ridge stacking (GKF) ---")
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        oof_stack = np.zeros(len(y_train))
        test_stack_preds = []
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(oofs, y_train, groups)):
            meta = Ridge(alpha=alpha)
            meta.fit(oofs[tr_idx], y_train[tr_idx])
            oof_stack[va_idx] = meta.predict(oofs[va_idx])
            test_stack_preds.append(meta.predict(tests))
            fold_scores.append(rmse(y_train[va_idx], oof_stack[va_idx]))
        s = rmse(y_train, oof_stack)
        print(f"    Ridge alpha={alpha:5.1f}: RMSE={s:.4f} folds={[round(f,1) for f in fold_scores]}")
        results[f"ridge_a{alpha}"] = {
            "rmse": s, "test": np.mean(test_stack_preds, axis=0), "oof": oof_stack
        }

    # --- 5. ElasticNet stacking ---
    print("\n  --- ElasticNet stacking ---")
    for alpha in [0.01, 0.1, 1.0]:
        for l1 in [0.1, 0.5, 0.9]:
            oof_stack = np.zeros(len(y_train))
            test_stack_preds = []
            fold_scores = []
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(oofs, y_train, groups)):
                meta = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000)
                meta.fit(oofs[tr_idx], y_train[tr_idx])
                oof_stack[va_idx] = meta.predict(oofs[va_idx])
                test_stack_preds.append(meta.predict(tests))
                fold_scores.append(rmse(y_train[va_idx], oof_stack[va_idx]))
            s = rmse(y_train, oof_stack)
            if s < 16.0:
                print(f"    EN a={alpha} l1={l1}: RMSE={s:.4f} folds={[round(f,1) for f in fold_scores]}")
                results[f"enet_a{alpha}_l{l1}"] = {
                    "rmse": s, "test": np.mean(test_stack_preds, axis=0), "oof": oof_stack
                }

    # --- 6. BayesianRidge stacking ---
    print("\n  --- BayesianRidge stacking ---")
    oof_stack = np.zeros(len(y_train))
    test_stack_preds = []
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(oofs, y_train, groups)):
        meta = BayesianRidge()
        meta.fit(oofs[tr_idx], y_train[tr_idx])
        oof_stack[va_idx] = meta.predict(oofs[va_idx])
        test_stack_preds.append(meta.predict(tests))
        fold_scores.append(rmse(y_train[va_idx], oof_stack[va_idx]))
    s = rmse(y_train, oof_stack)
    print(f"    BayesianRidge: RMSE={s:.4f} folds={[round(f,1) for f in fold_scores]}")
    results["bayesian_ridge"] = {
        "rmse": s, "test": np.mean(test_stack_preds, axis=0), "oof": oof_stack
    }

    # --- 7. Nelder-Mead weight optimization ---
    print("\n  --- Nelder-Mead weight optimization ---")
    n = len(names)
    def obj(w):
        w_pos = np.abs(w)
        w_norm = w_pos / (w_pos.sum() + 1e-8)
        return rmse(y_train, (oofs * w_norm).sum(axis=1))

    best_opt_rmse = 999
    best_opt_w = np.ones(n) / n
    for trial in range(100):
        w0 = np.random.dirichlet(np.ones(n) * 2)
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9})
        if res.fun < best_opt_rmse:
            best_opt_rmse = res.fun
            w = np.abs(res.x)
            best_opt_w = w / w.sum()

    opt_pred = (oofs * best_opt_w).sum(axis=1)
    opt_test = (tests * best_opt_w).sum(axis=1)
    print(f"    Optimized weights: RMSE={best_opt_rmse:.4f}")
    print(f"    Top weights:")
    for idx in np.argsort(-best_opt_w)[:8]:
        if best_opt_w[idx] > 0.01:
            print(f"      {best_opt_w[idx]:.3f}  {names[idx]}")
    results["nelder_mead_opt"] = {"rmse": best_opt_rmse, "test": opt_test, "oof": opt_pred}

    # --- 8. Greedy + Ridge two-level stacking ---
    print("\n  --- Two-level: Greedy subsets + Ridge ---")
    # Use greedy-selected models only for Ridge stacking
    sub_oofs = oofs[:, selected]
    sub_tests = tests[:, selected]
    for alpha in [1.0, 5.0, 10.0]:
        oof_stack = np.zeros(len(y_train))
        test_stack_preds = []
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(sub_oofs, y_train, groups)):
            meta = Ridge(alpha=alpha)
            meta.fit(sub_oofs[tr_idx], y_train[tr_idx])
            oof_stack[va_idx] = meta.predict(sub_oofs[va_idx])
            test_stack_preds.append(meta.predict(sub_tests))
            fold_scores.append(rmse(y_train[va_idx], oof_stack[va_idx]))
        s = rmse(y_train, oof_stack)
        print(f"    Greedy+Ridge a={alpha}: RMSE={s:.4f} folds={[round(f,1) for f in fold_scores]}")
        results[f"greedy_ridge_a{alpha}"] = {
            "rmse": s, "test": np.mean(test_stack_preds, axis=0), "oof": oof_stack
        }

    return results


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 21: MEGA STRATEGY")
    print("  1. Diverse Stacking")
    print("  2. Ceiling Breaking")
    print("  3. Species 15 Attack")
    print("  4. Mega Ensemble")
    print("=" * 70)

    X_train, y_train, groups, X_test, test_ids, wavelengths = load_data()
    print(f"\nData: train={X_train.shape}, test={X_test.shape}")
    print(f"Target range: [{y_train.min():.1f}, {y_train.max():.1f}], mean={y_train.mean():.1f}")

    # Run all three strategies
    all_models = {}

    stacking_models = build_diverse_models(X_train, y_train, groups, X_test, wavelengths)
    all_models.update(stacking_models)

    ceiling_models = build_ceiling_breakers(X_train, y_train, groups, X_test, wavelengths)
    all_models.update(ceiling_models)

    species15_models = build_species15_attackers(X_train, y_train, groups, X_test, wavelengths)
    all_models.update(species15_models)

    # Mega ensemble
    ensemble_results = mega_ensemble(all_models, y_train, groups, X_test.shape)

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 21 FINAL SUMMARY")
    print("=" * 70)

    # Combine base models and ensemble results for ranking
    all_final = {}
    for n, m in all_models.items():
        all_final[f"base:{n}"] = m
    for n, r in ensemble_results.items():
        all_final[f"ens:{n}"] = r

    ranked = sorted(all_final.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:30]:
        star = " ★★" if data["rmse"] < 14.5 else (" ★" if data["rmse"] < 15.0 else "")
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 20 best: 15.63 (uw_iterpl2_pw0.5)")

    # Save best submission
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save top 3 submissions
    for i, (name, data) in enumerate(ranked[:3]):
        sub = pd.DataFrame({
            "sample number": test_ids.values,
            "含水率": data["test"]
        })
        path = submissions_dir / f"submission_phase21_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"\n  Saved: {path} (RMSE={data['rmse']:.4f}, {name})")

    # Save all OOF predictions for future use
    oof_dir = Path("runs") / f"phase21_{ts}"
    oof_dir.mkdir(parents=True, exist_ok=True)

    import json
    summary = {
        "total_models": len(all_models),
        "total_ensembles": len(ensemble_results),
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked[:30]},
    }
    with open(oof_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save OOF for potential future stacking
    for name, data in all_models.items():
        np.save(oof_dir / f"oof_{name}.npy", data["oof"])
        np.save(oof_dir / f"test_{name}.npy", data["test"])

    print(f"\n  All artifacts saved to: {oof_dir}")
    print("\n" + "=" * 70)
    print("PHASE 21 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
