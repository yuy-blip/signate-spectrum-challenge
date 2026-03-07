#!/usr/bin/env python
"""Phase 26: The Spectral Decoupling — 全AI連合攻撃.

Current best: 13.61 (Phase 25 Conditional Ensemble t=150)

=== Gemini Ideas ===
  A. MCR-ALS: Decompose spectra into pure water/wood components
  B. Wavelet (DWT): Multi-resolution features preserving fine peaks
  C. Test-reference EMSC: Domain adaptation using test mean as reference

=== ChatGPT Ideas ===
  D. Tail calibrator: Rule-based residual correction for 200+ region
  E. Conditional Ensemble gate optimization: Multi-threshold, sigmoid gates

=== Claude Ideas ===
  F. Low-moisture bias correction: Fix +2.87 bias in 0-30 range (675 samples)
  G. Combined Mega Ensemble with all innovations

Monitoring metrics:
  1. Overall CV RMSE
  2. Fold 2 RMSE
  3. y_true > 200 mean bias
  4. y_true > 150 RMSE
"""

from __future__ import annotations

import sys
import datetime
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")

LGBM_BASE = {
    "n_estimators": 400, "max_depth": 5, "num_leaves": 20,
    "learning_rate": 0.05, "min_child_samples": 20,
    "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "verbose": -1, "n_jobs": -1,
}

PP_BIN4 = [
    {"name": "emsc", "poly_order": 2},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 4},
    {"name": "standard_scaler"},
]


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


# ======================================================================
# A. MCR-ALS: Pure Component Decomposition (Gemini)
# ======================================================================

def mcr_als_features(X_train_raw, X_test_raw, y_train, n_components=2):
    """Decompose spectra into pure components using MCR-ALS.

    Returns concentration profiles as features.
    """
    try:
        from pymcr.mcr import McrAR
        from pymcr.constraints import ConstraintNonneg

        # Use PCA for initial estimates
        pca = PCA(n_components=n_components)
        C_init = pca.fit_transform(X_train_raw)
        # Make non-negative initial estimates
        C_init = np.abs(C_init)

        # Fit MCR-ALS on training data
        mcr = McrAR(max_iter=100, tol_increase=50,
                     c_constraints=[ConstraintNonneg()],
                     st_constraints=[ConstraintNonneg()])

        mcr.fit(np.abs(X_train_raw), C=C_init)

        # Get concentration profiles (features)
        C_train = mcr.C_opt_  # (n_train, n_components)
        ST = mcr.ST_opt_  # (n_components, n_wavelengths)

        # For test data: solve C_test = X_test @ pinv(ST)
        ST_pinv = np.linalg.pinv(ST)
        C_test = np.abs(X_test_raw) @ ST_pinv

        # Identify which component correlates most with moisture
        corrs = [np.corrcoef(C_train[:, i], y_train)[0, 1] for i in range(n_components)]
        print(f"    MCR-ALS component correlations with moisture: {[f'{c:.3f}' for c in corrs]}")

        # Return all concentration profiles + ratios
        feats_train = [C_train]
        feats_test = [C_test]

        if n_components >= 2:
            # Add ratio of water-like / wood-like component
            water_idx = np.argmax(np.abs(corrs))
            wood_idx = 1 - water_idx if n_components == 2 else np.argmin(np.abs(corrs))
            ratio_train = C_train[:, water_idx] / (C_train[:, wood_idx] + 1e-8)
            ratio_test = C_test[:, water_idx] / (C_test[:, wood_idx] + 1e-8)
            feats_train.append(ratio_train.reshape(-1, 1))
            feats_test.append(ratio_test.reshape(-1, 1))

        return np.hstack(feats_train), np.hstack(feats_test)

    except Exception as e:
        print(f"    MCR-ALS failed: {e}")
        return None, None


# ======================================================================
# B. Wavelet Multi-Resolution Features (Gemini)
# ======================================================================

def wavelet_features(X, wavelet='db4', level=4):
    """Apply DWT and return multi-resolution features.

    Returns concatenated approximation + detail coefficients at each level.
    """
    import pywt

    all_features = []
    for i in range(X.shape[0]):
        coeffs = pywt.wavedec(X[i], wavelet, level=level)
        # coeffs[0] = approximation at deepest level
        # coeffs[1:] = detail coefficients from deep to shallow

        # Use approximation (low-freq = baseline shape)
        # and detail at each level (high-freq = peaks)
        feat_parts = []
        for c in coeffs:
            feat_parts.append(c)
        all_features.append(np.concatenate(feat_parts))

    return np.array(all_features)


def wavelet_selective_features(X, wavelet='db4', level=4, keep_approx=True,
                                keep_details=None):
    """Selective wavelet features — choose which levels to keep."""
    import pywt

    if keep_details is None:
        keep_details = list(range(1, level + 1))

    all_features = []
    for i in range(X.shape[0]):
        coeffs = pywt.wavedec(X[i], wavelet, level=level)
        feat_parts = []
        if keep_approx:
            feat_parts.append(coeffs[0])  # Low-freq approximation
        for d_level in keep_details:
            if d_level <= len(coeffs) - 1:
                feat_parts.append(coeffs[d_level])  # Detail at this level
        all_features.append(np.concatenate(feat_parts))

    return np.array(all_features)


# ======================================================================
# C. Test-Reference EMSC (Gemini)
# ======================================================================

def test_ref_emsc(X_train, X_test, poly_order=2):
    """EMSC using test data mean as reference instead of train mean.

    This calibrates training data toward test domain.
    """
    ref = X_test.mean(axis=0)
    n_wl = X_train.shape[1]
    w = np.linspace(-1, 1, n_wl)

    cols = [ref, np.ones(n_wl)]
    for p in range(1, poly_order + 1):
        cols.append(w ** p)
    D = np.column_stack(cols)

    eps = 1e-10

    def apply_emsc(X):
        coefs, _, _, _ = np.linalg.lstsq(D, X.T, rcond=None)
        a1 = coefs[0]
        a1_safe = np.where(np.abs(a1) < eps, eps, a1)
        baseline = D[:, 1:] @ coefs[1:]
        return ((X.T - baseline) / a1_safe).T

    return apply_emsc(X_train), apply_emsc(X_test)


def combined_ref_emsc(X_train, X_test, poly_order=2, alpha=0.5):
    """EMSC using weighted average of train and test means as reference."""
    ref_train = X_train.mean(axis=0)
    ref_test = X_test.mean(axis=0)
    ref = alpha * ref_train + (1 - alpha) * ref_test

    n_wl = X_train.shape[1]
    w = np.linspace(-1, 1, n_wl)
    cols = [ref, np.ones(n_wl)]
    for p in range(1, poly_order + 1):
        cols.append(w ** p)
    D = np.column_stack(cols)
    eps = 1e-10

    def apply_emsc(X):
        coefs, _, _, _ = np.linalg.lstsq(D, X.T, rcond=None)
        a1 = coefs[0]
        a1_safe = np.where(np.abs(a1) < eps, eps, a1)
        baseline = D[:, 1:] @ coefs[1:]
        return ((X.T - baseline) / a1_safe).T

    return apply_emsc(X_train), apply_emsc(X_test)


# ======================================================================
# D. Tail Calibrator — Rule-based (ChatGPT idea, Claude implementation)
# ======================================================================

def tail_calibrate(oof, test_pred, y_true, thresholds=None):
    """Rule-based tail calibration.

    For samples predicted above threshold, apply linear correction
    based on observed bias in that region.
    """
    if thresholds is None:
        thresholds = [120, 150, 180]

    best_oof = oof.copy()
    best_test = test_pred.copy()
    best_score = rmse(y_true, oof)

    for t in thresholds:
        for scale in np.arange(1.01, 1.50, 0.01):
            oof_cal = oof.copy()
            mask = oof_cal > t
            oof_cal[mask] = t + (oof_cal[mask] - t) * scale
            s = rmse(y_true, oof_cal)
            if s < best_score:
                best_score = s
                best_oof = oof_cal.copy()
                test_cal = test_pred.copy()
                mask_t = test_cal > t
                test_cal[mask_t] = t + (test_cal[mask_t] - t) * scale
                best_test = test_cal.copy()

    return best_oof, best_test, best_score


# ======================================================================
# F. Low-moisture bias correction (Claude)
# ======================================================================

def low_bias_calibrate(oof, test_pred, y_true):
    """Fix systematic positive bias in low moisture range.

    The 0-30 range has 675 samples with bias=+2.87.
    Apply gentle shrinkage for low predictions.
    """
    best_oof = oof.copy()
    best_test = test_pred.copy()
    best_score = rmse(y_true, oof)

    for t in [20, 25, 30, 35, 40]:
        for shrink in np.arange(0.90, 1.00, 0.005):
            oof_cal = oof.copy()
            mask = oof_cal < t
            oof_cal[mask] = oof_cal[mask] * shrink
            s = rmse(y_true, oof_cal)
            if s < best_score:
                best_score = s
                best_oof = oof_cal.copy()
                test_cal = test_pred.copy()
                mask_t = test_cal < t
                test_cal[mask_t] = test_cal[mask_t] * shrink
                best_test = test_cal.copy()

    return best_oof, best_test, best_score


# ======================================================================
# E. Conditional Ensemble Gate Optimization (ChatGPT)
# ======================================================================

def conditional_ensemble(oofs_dict, tests_dict, y_true,
                         thresholds=None, n_trials=300):
    """Multi-threshold conditional ensemble.

    Different weight sets for different prediction regions.
    Uses sigmoid gating for smooth transitions.
    """
    names = list(oofs_dict.keys())
    oofs = np.column_stack([oofs_dict[n] for n in names])
    tests = np.column_stack([tests_dict[n] for n in names])
    n_models = len(names)

    if thresholds is None:
        thresholds = [120, 130, 140, 150, 160, 170]

    best_score = 999
    best_oof = None
    best_test = None
    best_config = None

    for threshold in thresholds:
        for width in [5, 10, 15, 20, 30]:
            # Sigmoid gate: smooth transition
            def sigmoid_gate(pred, t=threshold, w=width):
                return 1 / (1 + np.exp(-(pred - t) / w))

            # Optimize weights for low and high regions separately
            # Use a simple proxy: avg prediction to determine gate
            proxy_pred = oofs.mean(axis=1)
            gate = sigmoid_gate(proxy_pred)

            def obj(params):
                w_low = np.abs(params[:n_models])
                w_high = np.abs(params[n_models:])
                w_low = w_low / (w_low.sum() + 1e-8)
                w_high = w_high / (w_high.sum() + 1e-8)

                pred_low = (oofs * w_low).sum(axis=1)
                pred_high = (oofs * w_high).sum(axis=1)

                g = gate.reshape(-1)
                pred = (1 - g) * pred_low + g * pred_high
                return rmse(y_true, pred)

            for trial in range(n_trials):
                w0 = np.random.dirichlet(np.ones(n_models) * 2, size=2).ravel()
                res = minimize(obj, w0, method="Nelder-Mead",
                               options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10})

                if res.fun < best_score:
                    best_score = res.fun
                    w_low = np.abs(res.x[:n_models])
                    w_high = np.abs(res.x[n_models:])
                    w_low = w_low / (w_low.sum() + 1e-8)
                    w_high = w_high / (w_high.sum() + 1e-8)

                    pred_low = (oofs * w_low).sum(axis=1)
                    pred_high = (oofs * w_high).sum(axis=1)
                    g = gate.reshape(-1)
                    best_oof = (1 - g) * pred_low + g * pred_high

                    # Apply same to test
                    test_low = (tests * w_low).sum(axis=1)
                    test_high = (tests * w_high).sum(axis=1)
                    proxy_test = tests.mean(axis=1)
                    g_test = sigmoid_gate(proxy_test)
                    best_test = (1 - g_test) * test_low + g_test * test_high

                    best_config = {
                        "threshold": threshold, "width": width,
                        "w_low": w_low.tolist(), "w_high": w_high.tolist()
                    }

    return best_oof, best_test, best_score, best_config


# ======================================================================
# WDV augmentation (from Phase 23/24)
# ======================================================================

def generate_universal_wdv(X_tr, y_tr, groups_tr, n_aug=30,
                           extrap_factor=1.5, min_moisture=150,
                           dy_scale=0.3, dy_offset=30):
    species_deltas, species_dy = [], []
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


# ======================================================================
# WDV Projection feature (from Phase 25)
# ======================================================================

def compute_wdv_projection(X_tr, y_tr, groups_tr):
    """Compute WDV (Water Direction Vector) and return projection feature."""
    species_deltas, species_dy = [], []
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
    if len(species_deltas) < 3:
        return None
    species_deltas = np.array(species_deltas)
    pca = PCA(n_components=1)
    pca.fit(species_deltas)
    water_vec = pca.components_[0]
    if np.corrcoef(species_deltas @ water_vec, species_dy)[0, 1] < 0:
        water_vec = -water_vec
    return water_vec


# ======================================================================
# Core CV function — supports all new feature types
# ======================================================================

def cv_full(X_train, y_train, groups, X_test,
            preprocess=None, lgbm_params=None,
            # WDV augmentation
            n_aug=30, extrap=1.5, min_moisture=170,
            dy_scale=0.3, dy_offset=30,
            # PL params
            pl_w=0.5, pl_rounds=2,
            # Weight params
            sample_weight_fn=None,
            # Extra features to append
            extra_train_feats=None, extra_test_feats=None,
            # WDV projection
            use_wdv_proj=False,
            # Label
            label=""):
    """Core CV with augmentation and extra features."""
    if preprocess is None:
        preprocess = PP_BIN4
    params = {**(lgbm_params or LGBM_BASE)}
    gkf = GroupKFold(n_splits=5)
    test_preds_prev = None

    for pl_round in range(pl_rounds):
        oof = np.zeros(len(y_train))
        fold_rmses = []
        test_preds_folds = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            g_tr = groups[tr_idx]

            pipe = build_preprocess_pipeline(preprocess)
            pipe.fit(X_tr)

            X_tr_t = pipe.transform(X_tr)
            X_va_t = pipe.transform(X_va)
            X_test_t = pipe.transform(X_test)

            # WDV projection feature
            if use_wdv_proj:
                wdv = compute_wdv_projection(X_tr_t, y_tr, g_tr)
                if wdv is not None:
                    proj_tr = (X_tr_t @ wdv).reshape(-1, 1)
                    proj_va = (X_va_t @ wdv).reshape(-1, 1)
                    proj_test = (X_test_t @ wdv).reshape(-1, 1)
                    X_tr_t = np.hstack([X_tr_t, proj_tr])
                    X_va_t = np.hstack([X_va_t, proj_va])
                    X_test_t = np.hstack([X_test_t, proj_test])

            # Append extra features (MCR, wavelet, etc.)
            if extra_train_feats is not None and extra_test_feats is not None:
                X_tr_t = np.hstack([X_tr_t, extra_train_feats[tr_idx]])
                X_va_t = np.hstack([X_va_t, extra_train_feats[va_idx]])
                X_test_t = np.hstack([X_test_t, extra_test_feats])

            # WDV augmentation
            aug_X_list, aug_y_list = [], []
            if n_aug > 0:
                synth_X_raw, synth_y = generate_universal_wdv(
                    X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset)
                if len(synth_X_raw) > 0:
                    synth_X_t = pipe.transform(synth_X_raw)
                    # Add extra feats to augmented data too
                    if use_wdv_proj and wdv is not None:
                        proj_s = (synth_X_t @ wdv).reshape(-1, 1)
                        synth_X_t = np.hstack([synth_X_t, proj_s])
                    if extra_train_feats is not None:
                        # For augmented data, use nearest neighbor features
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=1).fit(X_tr)
                        _, nn_idx = nn.kneighbors(synth_X_raw)
                        aug_extra = extra_train_feats[tr_idx][nn_idx.ravel()]
                        synth_X_t = np.hstack([synth_X_t, aug_extra])
                    aug_X_list.append(synth_X_t)
                    aug_y_list.append(synth_y)

            if aug_X_list:
                X_aug = np.vstack([X_tr_t] + aug_X_list)
                y_aug = np.concatenate([y_tr] + aug_y_list)
            else:
                X_aug = X_tr_t
                y_aug = y_tr

            # Sample weights
            sw_train = None
            if sample_weight_fn is not None:
                sw_base = sample_weight_fn(y_tr, g_tr)
                sw_parts = [sw_base]
                for ay in aug_y_list:
                    sw_parts.append(sample_weight_fn(ay, None))
                sw_train = np.concatenate(sw_parts) if aug_y_list else sw_base

            # Pseudo-labeling
            if pl_round == 0 and pl_w > 0:
                temp = create_model("lgbm", params)
                if sw_train is not None:
                    temp.fit(X_aug, y_aug, sample_weight=sw_train)
                else:
                    temp.fit(X_aug, y_aug)
                pl_pred = temp.predict(X_test_t)
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, pl_pred])
                w = np.ones(len(y_final))
                if sw_train is not None:
                    w[:len(sw_train)] = sw_train
                w[-len(pl_pred):] *= pl_w
                model = create_model("lgbm", params)
                model.fit(X_final, y_final, sample_weight=w)
            elif pl_round > 0 and test_preds_prev is not None:
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, test_preds_prev])
                w = np.ones(len(y_final))
                if sw_train is not None:
                    w[:len(sw_train)] = sw_train
                w[-len(test_preds_prev):] *= pl_w
                model = create_model("lgbm", params)
                model.fit(X_final, y_final, sample_weight=w)
            else:
                model = create_model("lgbm", params)
                if sw_train is not None:
                    model.fit(X_aug, y_aug, sample_weight=sw_train)
                else:
                    model.fit(X_aug, y_aug)

            pred = model.predict(X_va_t).ravel()
            oof[va_idx] = pred
            fold_rmses.append(rmse(y_va, pred))
            test_preds_folds.append(model.predict(X_test_t).ravel())

        test_preds_prev = np.mean(test_preds_folds, axis=0)

    return oof, fold_rmses, test_preds_prev


# ======================================================================
# Residual analysis
# ======================================================================

def analyze_residuals(y_true, y_pred, label=""):
    bins = [0, 30, 60, 100, 150, 200, 300]
    residuals = y_pred - y_true

    overall = rmse(y_true, y_pred)

    # Fold 2 approx (use GroupKFold to find it)
    fold2_rmse = None

    # 200+ bias
    mask200 = y_true >= 200
    bias200 = residuals[mask200].mean() if mask200.sum() > 0 else 0

    # 150+ RMSE
    mask150 = y_true >= 150
    rmse150 = np.sqrt((residuals[mask150] ** 2).mean()) if mask150.sum() > 0 else 0

    print(f"  {label}")
    print(f"    Overall RMSE: {overall:.4f}")
    print(f"    200+ bias: {bias200:+.2f} (n={mask200.sum()})")
    print(f"    150+ RMSE: {rmse150:.2f} (n={mask150.sum()})")

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() == 0:
            continue
        r = residuals[mask]
        print(f"    {bins[i]:>5d}-{bins[i+1]:<5d} n={mask.sum():>4d} "
              f"RMSE={np.sqrt((r**2).mean()):>7.2f} bias={r.mean():>+7.2f}")

    return {"overall": overall, "bias200": bias200, "rmse150": rmse150}


# ======================================================================
# MAIN
# ======================================================================

def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 26: The Spectral Decoupling — 全AI連合攻撃")
    print("=" * 70)

    X_train, y_train, groups, X_test, test_ids = load_data()
    all_models = {}

    def log(name, oof, folds, tp):
        score = rmse(y_train, oof)
        fr = [round(f, 1) for f in folds]
        star = " ★★★" if score < 13.5 else (" ★★" if score < 14.0 else (" ★" if score < 14.5 else ""))
        print(f"  {score:.4f} {fr} {name}{star}")
        all_models[name] = {"oof": oof.copy(), "test": tp.copy(), "rmse": score}
        return score

    # ==================================================================
    # BASELINE: Reproduce Phase 25 best individual
    # ==================================================================
    print("\n=== BASELINE (Phase 25 best individual) ===")

    oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                          min_moisture=170, pl_w=0.5, pl_rounds=2)
    log("baseline", oof, f, tp)
    analyze_residuals(y_train, oof, "Baseline")

    # With WDV proj
    oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                          min_moisture=170, pl_w=0.5, pl_rounds=2,
                          use_wdv_proj=True)
    log("baseline_wdv_proj", oof, f, tp)

    # ==================================================================
    # A: MCR-ALS Pure Components (Gemini)
    # ==================================================================
    print("\n" + "=" * 70)
    print("A: MCR-ALS Pure Component Decomposition")
    print("=" * 70)

    for n_comp in [2, 3, 4]:
        print(f"\n  --- MCR-ALS with {n_comp} components ---")
        mcr_train, mcr_test = mcr_als_features(X_train, X_test, y_train, n_comp)
        if mcr_train is not None:
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  extra_train_feats=mcr_train, extra_test_feats=mcr_test,
                                  min_moisture=170, pl_w=0.5, pl_rounds=2)
            log(f"mcr_als_{n_comp}comp", oof, f, tp)

            # MCR + WDV proj
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  extra_train_feats=mcr_train, extra_test_feats=mcr_test,
                                  use_wdv_proj=True,
                                  min_moisture=170, pl_w=0.5, pl_rounds=2)
            log(f"mcr_als_{n_comp}comp_wdv", oof, f, tp)

    # ==================================================================
    # B: Wavelet Multi-Resolution Features (Gemini)
    # ==================================================================
    print("\n" + "=" * 70)
    print("B: Wavelet DWT Multi-Resolution Features")
    print("=" * 70)

    # B1: Full wavelet as preprocessing replacement
    for wavelet in ['db4', 'db6', 'sym4']:
        for level in [3, 4, 5]:
            print(f"\n  --- Wavelet {wavelet} level={level} ---")

            # Apply wavelet to raw EMSC-corrected spectra
            PP_EMSC_ONLY = [
                {"name": "emsc", "poly_order": 2},
            ]
            pipe_emsc = build_preprocess_pipeline(PP_EMSC_ONLY)
            pipe_emsc.fit(X_train)
            X_tr_emsc = pipe_emsc.transform(X_train)
            X_te_emsc = pipe_emsc.transform(X_test)

            # Wavelet features
            wl_tr = wavelet_features(X_tr_emsc, wavelet=wavelet, level=level)
            wl_te = wavelet_features(X_te_emsc, wavelet=wavelet, level=level)

            # Standardize
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            wl_tr = sc.fit_transform(wl_tr)
            wl_te = sc.transform(wl_te)

            # Train directly on wavelet features (no further SG/binning)
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  # Use simple EMSC pipeline (wavelet replaces SG+binning)
                                  preprocess=[{"name": "emsc", "poly_order": 2},
                                              {"name": "standard_scaler"}],
                                  extra_train_feats=wl_tr, extra_test_feats=wl_te,
                                  min_moisture=170, pl_w=0.5, pl_rounds=2)
            log(f"wavelet_{wavelet}_L{level}", oof, f, tp)

    # B2: Wavelet as supplementary features (add to standard pipeline)
    print("\n  --- Wavelet as supplementary features ---")
    PP_EMSC_ONLY = [{"name": "emsc", "poly_order": 2}]
    pipe_emsc = build_preprocess_pipeline(PP_EMSC_ONLY)
    pipe_emsc.fit(X_train)
    X_tr_emsc = pipe_emsc.transform(X_train)
    X_te_emsc = pipe_emsc.transform(X_test)

    for wavelet in ['db4']:
        for level in [4]:
            # Only keep low-freq approx (captures baseline shape)
            wl_tr = wavelet_selective_features(X_tr_emsc, wavelet=wavelet, level=level,
                                               keep_approx=True, keep_details=[])
            wl_te = wavelet_selective_features(X_te_emsc, wavelet=wavelet, level=level,
                                               keep_approx=True, keep_details=[])
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            wl_tr = sc.fit_transform(wl_tr)
            wl_te = sc.transform(wl_te)

            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  extra_train_feats=wl_tr, extra_test_feats=wl_te,
                                  min_moisture=170, pl_w=0.5, pl_rounds=2)
            log(f"wavelet_approx_{wavelet}_L{level}", oof, f, tp)

            # Only high-freq details
            wl_tr_d = wavelet_selective_features(X_tr_emsc, wavelet=wavelet, level=level,
                                                  keep_approx=False, keep_details=[1, 2])
            wl_te_d = wavelet_selective_features(X_te_emsc, wavelet=wavelet, level=level,
                                                  keep_approx=False, keep_details=[1, 2])
            sc2 = StandardScaler()
            wl_tr_d = sc2.fit_transform(wl_tr_d)
            wl_te_d = sc2.transform(wl_te_d)

            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  extra_train_feats=wl_tr_d, extra_test_feats=wl_te_d,
                                  min_moisture=170, pl_w=0.5, pl_rounds=2)
            log(f"wavelet_detail_{wavelet}_L{level}", oof, f, tp)

    # ==================================================================
    # C: Test-Reference EMSC (Gemini)
    # ==================================================================
    print("\n" + "=" * 70)
    print("C: Test-Reference EMSC Domain Adaptation")
    print("=" * 70)

    # C1: Pure test-reference EMSC
    X_tr_test_emsc, X_te_test_emsc = test_ref_emsc(X_train, X_test, poly_order=2)
    PP_SG_BIN = [
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 4},
        {"name": "standard_scaler"},
    ]
    oof, f, tp = cv_full(X_tr_test_emsc, y_train, groups, X_te_test_emsc,
                          preprocess=PP_SG_BIN,
                          min_moisture=170, pl_w=0.5, pl_rounds=2)
    log("test_ref_emsc", oof, f, tp)

    # C2: Combined reference (alpha blend)
    for alpha in [0.3, 0.5, 0.7]:
        X_tr_c, X_te_c = combined_ref_emsc(X_train, X_test, poly_order=2, alpha=alpha)
        oof, f, tp = cv_full(X_tr_c, y_train, groups, X_te_c,
                              preprocess=PP_SG_BIN,
                              min_moisture=170, pl_w=0.5, pl_rounds=2)
        log(f"combined_ref_emsc_a{alpha}", oof, f, tp)

    # ==================================================================
    # Diversity: Multiple LGBM HPs with best new features
    # ==================================================================
    print("\n" + "=" * 70)
    print("DIVERSITY: Multiple HP configs with best features")
    print("=" * 70)

    hp_variants = [
        ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
        ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01}),
        ("d3l10_n1500", {"max_depth": 3, "num_leaves": 10, "n_estimators": 1500, "learning_rate": 0.005}),
        ("d7l30_n600", {"max_depth": 7, "num_leaves": 30, "n_estimators": 600, "learning_rate": 0.03}),
        ("d4l15_n500", {"max_depth": 4, "num_leaves": 15, "n_estimators": 500, "learning_rate": 0.04}),
    ]

    for hp_name, hp_ov in hp_variants:
        params = {**LGBM_BASE, **hp_ov}
        # Standard
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params, min_moisture=170,
                              pl_w=0.5, pl_rounds=2)
        log(f"div_{hp_name}", oof, f, tp)

        # With WDV proj
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params, min_moisture=170,
                              pl_w=0.5, pl_rounds=2, use_wdv_proj=True)
        log(f"div_{hp_name}_wdv", oof, f, tp)

    # ==================================================================
    # LOAD PREVIOUS PHASE RESULTS
    # ==================================================================
    print("\n" + "=" * 70)
    print("LOADING PREVIOUS PHASE RESULTS")
    print("=" * 70)

    for phase_pattern in ["phase23_*", "phase24_*", "phase25_*"]:
        phase_dirs = sorted(Path("runs").glob(phase_pattern))
        for p_dir in phase_dirs:
            summary_path = p_dir / "summary.json"
            if not summary_path.exists():
                continue
            p_summary = json.loads(summary_path.read_text())
            loaded = 0
            for name, rmse_val in p_summary.get("all_results", {}).items():
                oof_path = p_dir / f"oof_{name}.npy"
                test_path = p_dir / f"test_{name}.npy"
                if oof_path.exists() and test_path.exists():
                    oof_data = np.load(oof_path)
                    test_data = np.load(test_path)
                    if (np.isfinite(oof_data).all() and np.isfinite(test_data).all()
                            and len(oof_data) == len(y_train)):
                        key = f"prev_{p_dir.name}_{name}"
                        all_models[key] = {"oof": oof_data, "test": test_data, "rmse": rmse_val}
                        loaded += 1
            print(f"  Loaded {loaded} models from {p_dir}")

    # ==================================================================
    # MEGA ENSEMBLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("MEGA ENSEMBLE (ALL MODELS)")
    print("=" * 70)

    # Filter valid models
    valid_names = []
    for n, d in all_models.items():
        if (np.isfinite(d["oof"]).all() and np.isfinite(d["test"]).all()
                and len(d["oof"]) == len(y_train)):
            valid_names.append(n)
        else:
            print(f"  SKIP: {n}")

    oofs = np.column_stack([all_models[n]["oof"] for n in valid_names])
    tests = np.column_stack([all_models[n]["test"] for n in valid_names])
    rmses_list = [all_models[n]["rmse"] for n in valid_names]

    print(f"\n  {len(valid_names)} valid models. Top 30:")
    ranked_idx = sorted(range(len(valid_names)), key=lambda i: rmses_list[i])
    for i in ranked_idx[:30]:
        r = rmses_list[i]
        star = " ★★★" if r < 13.5 else (" ★★" if r < 14.0 else (" ★" if r < 14.5 else ""))
        print(f"    {r:.4f}  {valid_names[i]}{star}")

    # Greedy forward selection
    print("\n  --- Greedy Forward Selection ---")
    selected = [ranked_idx[0]]
    for _ in range(min(60, len(ranked_idx) - 1)):
        cur_avg = oofs[:, selected].mean(axis=1)
        cur_s = rmse(y_train, cur_avg)
        best_s, best_i = cur_s, -1
        for i in ranked_idx:
            if i in selected:
                continue
            new_avg = (cur_avg * len(selected) + oofs[:, i]) / (len(selected) + 1)
            s = rmse(y_train, new_avg)
            if s < best_s - 0.001:
                best_s = s
                best_i = i
        if best_i >= 0:
            selected.append(best_i)
            if len(selected) <= 20 or len(selected) % 5 == 0:
                print(f"    +{len(selected)}: {valid_names[best_i][:55]:55s} ens={best_s:.4f}")
        else:
            break

    greedy_avg = oofs[:, selected].mean(axis=1)
    greedy_test = tests[:, selected].mean(axis=1)
    greedy_s = rmse(y_train, greedy_avg)
    print(f"  Greedy ({len(selected)} models): {greedy_s:.4f}")
    all_models["greedy_ens"] = {"oof": greedy_avg, "test": greedy_test, "rmse": greedy_s}

    # NM optimization
    print("\n  --- NM Weight Optimization (1000 trials) ---")
    sub_oofs = oofs[:, selected]
    sub_tests = tests[:, selected]
    ns = len(selected)

    def obj(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub_oofs * wn).sum(axis=1))

    best_opt = 999
    best_w = np.ones(ns) / ns
    for trial in range(1000):
        w0 = np.random.dirichlet(np.ones(ns) * 2)
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-10})
        if res.fun < best_opt:
            best_opt = res.fun
            w = np.abs(res.x)
            best_w = w / w.sum()

    opt_oof = (sub_oofs * best_w).sum(axis=1)
    opt_test = (sub_tests * best_w).sum(axis=1)
    print(f"  NM optimized: {best_opt:.4f}")
    for i, idx in enumerate(selected):
        if best_w[i] > 0.01:
            print(f"    {best_w[i]:.3f}  {valid_names[idx]}")
    all_models["nm_opt"] = {"oof": opt_oof, "test": opt_test, "rmse": best_opt}

    # ==================================================================
    # D: TAIL CALIBRATION (ChatGPT / Claude)
    # ==================================================================
    print("\n" + "=" * 70)
    print("D: TAIL + LOW-BIAS CALIBRATION")
    print("=" * 70)

    # Apply tail calibration to top candidates
    top_cands = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:10]
    for base_name, base_data in top_cands:
        if "tailcal" in base_name or "lowcal" in base_name:
            continue

        # Tail calibration (high region)
        oof_tc, test_tc, score_tc = tail_calibrate(
            base_data["oof"], base_data["test"], y_train,
            thresholds=[100, 120, 140, 150, 160, 180])
        if score_tc < base_data["rmse"] - 0.005:
            name_tc = f"tailcal_{base_name}"
            all_models[name_tc] = {"oof": oof_tc, "test": test_tc, "rmse": score_tc}
            print(f"  Tail cal: {base_data['rmse']:.4f} → {score_tc:.4f} ({base_name})")

        # Low bias correction
        oof_lc, test_lc, score_lc = low_bias_calibrate(
            base_data["oof"], base_data["test"], y_train)
        if score_lc < base_data["rmse"] - 0.005:
            name_lc = f"lowcal_{base_name}"
            all_models[name_lc] = {"oof": oof_lc, "test": test_lc, "rmse": score_lc}
            print(f"  Low cal:  {base_data['rmse']:.4f} → {score_lc:.4f} ({base_name})")

        # Both together
        oof_both, test_both, score_both = tail_calibrate(
            oof_lc if score_lc < base_data["rmse"] else base_data["oof"],
            test_lc if score_lc < base_data["rmse"] else base_data["test"],
            y_train, thresholds=[100, 120, 140, 150, 160, 180])
        if score_both < min(score_tc, score_lc, base_data["rmse"]) - 0.005:
            name_b = f"bothcal_{base_name}"
            all_models[name_b] = {"oof": oof_both, "test": test_both, "rmse": score_both}
            print(f"  Both cal: {base_data['rmse']:.4f} → {score_both:.4f} ({base_name})")

    # ==================================================================
    # E: CONDITIONAL ENSEMBLE (ChatGPT gate optimization)
    # ==================================================================
    print("\n" + "=" * 70)
    print("E: CONDITIONAL ENSEMBLE GATE OPTIMIZATION")
    print("=" * 70)

    # Select top diverse models for conditional ensemble
    top_for_cond = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:20]
    cond_oofs = {n: d["oof"] for n, d in top_for_cond}
    cond_tests = {n: d["test"] for n, d in top_for_cond}

    ce_oof, ce_test, ce_score, ce_config = conditional_ensemble(
        cond_oofs, cond_tests, y_train,
        thresholds=[110, 120, 130, 140, 150, 160],
        n_trials=200)

    if ce_oof is not None:
        print(f"  Conditional ensemble: {ce_score:.4f}")
        print(f"  Config: threshold={ce_config['threshold']}, width={ce_config['width']}")
        all_models["cond_ensemble"] = {"oof": ce_oof, "test": ce_test, "rmse": ce_score}
        analyze_residuals(y_train, ce_oof, "Conditional Ensemble")

        # Tail cal on conditional ensemble
        oof_tc, test_tc, score_tc = tail_calibrate(ce_oof, ce_test, y_train)
        if score_tc < ce_score - 0.005:
            all_models["cond_ens_tailcal"] = {"oof": oof_tc, "test": test_tc, "rmse": score_tc}
            print(f"  + Tail cal: {ce_score:.4f} → {score_tc:.4f}")

    # ==================================================================
    # FINAL RE-ENSEMBLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("FINAL RE-ENSEMBLE WITH ALL MODELS")
    print("=" * 70)

    valid_names2 = [n for n in all_models
                    if np.isfinite(all_models[n]["oof"]).all()
                    and np.isfinite(all_models[n]["test"]).all()
                    and len(all_models[n]["oof"]) == len(y_train)]
    oofs2 = np.column_stack([all_models[n]["oof"] for n in valid_names2])
    tests2 = np.column_stack([all_models[n]["test"] for n in valid_names2])
    rmses2 = [all_models[n]["rmse"] for n in valid_names2]

    print(f"  Total valid models: {len(valid_names2)}")

    # Final greedy
    order2 = sorted(range(len(valid_names2)), key=lambda i: rmses2[i])
    selected2 = [order2[0]]
    for _ in range(min(60, len(order2) - 1)):
        cur_avg = oofs2[:, selected2].mean(axis=1)
        cur_s = rmse(y_train, cur_avg)
        best_s, best_i = cur_s, -1
        for i in order2:
            if i in selected2:
                continue
            new_avg = (cur_avg * len(selected2) + oofs2[:, i]) / (len(selected2) + 1)
            s = rmse(y_train, new_avg)
            if s < best_s - 0.001:
                best_s = s
                best_i = i
        if best_i >= 0:
            selected2.append(best_i)
            if len(selected2) <= 15 or len(selected2) % 5 == 0:
                print(f"    +{len(selected2)}: {valid_names2[best_i][:55]:55s} ens={best_s:.4f}")
        else:
            break

    greedy_avg2 = oofs2[:, selected2].mean(axis=1)
    greedy_test2 = tests2[:, selected2].mean(axis=1)
    greedy_s2 = rmse(y_train, greedy_avg2)
    print(f"  Final greedy ({len(selected2)} models): {greedy_s2:.4f}")

    # Final NM (1000 trials)
    sub_oofs2 = oofs2[:, selected2]
    sub_tests2 = tests2[:, selected2]
    ns2 = len(selected2)

    def obj2(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub_oofs2 * wn).sum(axis=1))

    best_opt2 = 999
    best_w2 = np.ones(ns2) / ns2
    for trial in range(1000):
        w0 = np.random.dirichlet(np.ones(ns2) * 2)
        res = minimize(obj2, w0, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-10})
        if res.fun < best_opt2:
            best_opt2 = res.fun
            w = np.abs(res.x)
            best_w2 = w / w.sum()

    opt_oof2 = (sub_oofs2 * best_w2).sum(axis=1)
    opt_test2 = (sub_tests2 * best_w2).sum(axis=1)
    print(f"  Final NM optimized: {best_opt2:.4f}")
    for i, idx in enumerate(selected2):
        if best_w2[i] > 0.01:
            print(f"    {best_w2[i]:.3f}  {valid_names2[idx]}")

    all_models["final_nm_opt"] = {"oof": opt_oof2, "test": opt_test2, "rmse": best_opt2}
    all_models["final_greedy"] = {"oof": greedy_avg2, "test": greedy_test2, "rmse": greedy_s2}

    # Final tail + low calibration on the best
    for cand_name in ["final_nm_opt", "final_greedy", "cond_ensemble"]:
        if cand_name not in all_models:
            continue
        d = all_models[cand_name]
        oof_tc, test_tc, score_tc = tail_calibrate(d["oof"], d["test"], y_train)
        if score_tc < d["rmse"] - 0.005:
            all_models[f"tailcal_{cand_name}"] = {"oof": oof_tc, "test": test_tc, "rmse": score_tc}
            print(f"  Tail cal {cand_name}: {d['rmse']:.4f} → {score_tc:.4f}")

        oof_lc, test_lc, score_lc = low_bias_calibrate(d["oof"], d["test"], y_train)
        if score_lc < d["rmse"] - 0.005:
            all_models[f"lowcal_{cand_name}"] = {"oof": oof_lc, "test": test_lc, "rmse": score_lc}
            print(f"  Low cal {cand_name}: {d['rmse']:.4f} → {score_lc:.4f}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 26 FINAL SUMMARY")
    print("=" * 70)

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:50]:
        star = " ★★★" if data["rmse"] < 13.5 else (" ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else ""))
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 25 best: 13.61")
    improvement = 13.61 - best_data["rmse"]
    print(f"  Improvement: {improvement:+.4f}")

    analyze_residuals(y_train, best_data["oof"], f"BEST ({best_name})")

    # Save submissions
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        sub = pd.DataFrame({
            "sample number": test_ids.values,
            "含水率": data["test"]
        })
        path = submissions_dir / f"submission_phase26_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    # Save artifacts
    oof_dir = Path("runs") / f"phase26_{ts}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked},
        "phase": "26",
        "description": "Spectral Decoupling — 全AI連合攻撃",
    }
    with open(oof_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, data in all_models.items():
        np.save(oof_dir / f"oof_{name}.npy", data["oof"])
        np.save(oof_dir / f"test_{name}.npy", data["test"])

    print(f"\n  Artifacts: {oof_dir}")
    print(f"  Total models: {len(all_models)}")
    print("=" * 70)
    print("PHASE 26 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
