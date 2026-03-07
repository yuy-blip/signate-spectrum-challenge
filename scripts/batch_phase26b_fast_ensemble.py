#!/usr/bin/env python
"""Phase 26b: Fast Ensemble — use Phase 26a individual models + Phase 23 models.

Phase 26a results (individual models):
  14.00  wavelet_detail_db4_L4 ★ (NEW)
  14.21  div_n800lr02
  14.26  div_n800lr02_wdv
  14.30  div_d4l15_n500_wdv
  14.33  div_d4l15_n500
  14.32  div_n1000lr01
  14.37  baseline_wdv_proj
  14.42  mcr_als_4comp
  14.45  mcr_als_2comp
  14.47  baseline

Strategy:
  1. Load all Phase 23 + Phase 26a individual model OOFs
  2. Greedy forward selection + NM optimization
  3. Lightweight Conditional Ensemble (top 8 models, few configs)
  4. Piecewise calibration (ChatGPT idea: low + high correction)
  5. Tail + Low-bias calibration
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
from sklearn.preprocessing import StandardScaler
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
# Wavelet features (from Phase 26a — the winner)
# ======================================================================

def wavelet_selective_features(X, wavelet='db4', level=4, keep_approx=True,
                                keep_details=None):
    import pywt
    if keep_details is None:
        keep_details = list(range(1, level + 1))
    all_features = []
    for i in range(X.shape[0]):
        coeffs = pywt.wavedec(X[i], wavelet, level=level)
        feat_parts = []
        if keep_approx:
            feat_parts.append(coeffs[0])
        for d_level in keep_details:
            if d_level <= len(coeffs) - 1:
                feat_parts.append(coeffs[d_level])
        all_features.append(np.concatenate(feat_parts))
    return np.array(all_features)


# ======================================================================
# MCR-ALS features
# ======================================================================

def mcr_als_features(X_train_raw, X_test_raw, y_train, n_components=2):
    try:
        from pymcr.mcr import McrAR
        from pymcr.constraints import ConstraintNonneg
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        C_init = np.abs(pca.fit_transform(X_train_raw))
        mcr = McrAR(max_iter=100, tol_increase=50,
                     c_constraints=[ConstraintNonneg()],
                     st_constraints=[ConstraintNonneg()])
        mcr.fit(np.abs(X_train_raw), C=C_init)
        C_train = mcr.C_opt_
        ST = mcr.ST_opt_
        ST_pinv = np.linalg.pinv(ST)
        C_test = np.abs(X_test_raw) @ ST_pinv
        if n_components >= 2:
            corrs = [np.corrcoef(C_train[:, i], y_train)[0, 1] for i in range(n_components)]
            water_idx = np.argmax(np.abs(corrs))
            wood_idx = 1 - water_idx if n_components == 2 else np.argmin(np.abs(corrs))
            ratio_train = C_train[:, water_idx] / (C_train[:, wood_idx] + 1e-8)
            ratio_test = C_test[:, water_idx] / (C_test[:, wood_idx] + 1e-8)
            return np.hstack([C_train, ratio_train.reshape(-1, 1)]), np.hstack([C_test, ratio_test.reshape(-1, 1)])
        return C_train, C_test
    except Exception as e:
        print(f"    MCR-ALS failed: {e}")
        return None, None


# ======================================================================
# WDV projection
# ======================================================================

def compute_wdv_projection(X_tr, y_tr, groups_tr):
    from sklearn.decomposition import PCA
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
# WDV augmentation
# ======================================================================

def generate_universal_wdv(X_tr, y_tr, groups_tr, n_aug=30,
                           extrap_factor=1.5, min_moisture=150,
                           dy_scale=0.3, dy_offset=30):
    from sklearn.decomposition import PCA
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
# CV function with extra features
# ======================================================================

def cv_full(X_train, y_train, groups, X_test,
            preprocess=None, lgbm_params=None,
            n_aug=30, extrap=1.5, min_moisture=170,
            dy_scale=0.3, dy_offset=30,
            pl_w=0.5, pl_rounds=2,
            sample_weight_fn=None,
            extra_train_feats=None, extra_test_feats=None,
            use_wdv_proj=False):
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

            if use_wdv_proj:
                wdv = compute_wdv_projection(X_tr_t, y_tr, g_tr)
                if wdv is not None:
                    X_tr_t = np.hstack([X_tr_t, (X_tr_t @ wdv).reshape(-1, 1)])
                    X_va_t = np.hstack([X_va_t, (X_va_t @ wdv).reshape(-1, 1)])
                    X_test_t = np.hstack([X_test_t, (X_test_t @ wdv).reshape(-1, 1)])

            if extra_train_feats is not None and extra_test_feats is not None:
                X_tr_t = np.hstack([X_tr_t, extra_train_feats[tr_idx]])
                X_va_t = np.hstack([X_va_t, extra_train_feats[va_idx]])
                X_test_t = np.hstack([X_test_t, extra_test_feats])

            # WDV augmentation
            aug_X, aug_y = [], []
            if n_aug > 0:
                sX, sy = generate_universal_wdv(X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset)
                if len(sX) > 0:
                    sX_t = pipe.transform(sX)
                    if use_wdv_proj and wdv is not None:
                        sX_t = np.hstack([sX_t, (sX_t @ wdv).reshape(-1, 1)])
                    if extra_train_feats is not None:
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=1).fit(X_tr)
                        _, nn_idx = nn.kneighbors(sX)
                        sX_t = np.hstack([sX_t, extra_train_feats[tr_idx][nn_idx.ravel()]])
                    aug_X.append(sX_t)
                    aug_y.append(sy)

            X_all = np.vstack([X_tr_t] + aug_X) if aug_X else X_tr_t
            y_all = np.concatenate([y_tr] + aug_y) if aug_y else y_tr

            sw = None
            if sample_weight_fn is not None:
                sw_parts = [sample_weight_fn(y_tr, g_tr)]
                for ay in aug_y:
                    sw_parts.append(sample_weight_fn(ay, None))
                sw = np.concatenate(sw_parts) if aug_y else sw_parts[0]

            if pl_round == 0 and pl_w > 0:
                temp = create_model("lgbm", params)
                temp.fit(X_all, y_all, sample_weight=sw) if sw is not None else temp.fit(X_all, y_all)
                pl_pred = temp.predict(X_test_t)
                X_f = np.vstack([X_all, X_test_t])
                y_f = np.concatenate([y_all, pl_pred])
                w = np.ones(len(y_f))
                if sw is not None:
                    w[:len(sw)] = sw
                w[-len(pl_pred):] *= pl_w
                model = create_model("lgbm", params)
                model.fit(X_f, y_f, sample_weight=w)
            elif pl_round > 0 and test_preds_prev is not None:
                X_f = np.vstack([X_all, X_test_t])
                y_f = np.concatenate([y_all, test_preds_prev])
                w = np.ones(len(y_f))
                if sw is not None:
                    w[:len(sw)] = sw
                w[-len(test_preds_prev):] *= pl_w
                model = create_model("lgbm", params)
                model.fit(X_f, y_f, sample_weight=w)
            else:
                model = create_model("lgbm", params)
                model.fit(X_all, y_all, sample_weight=sw) if sw is not None else model.fit(X_all, y_all)

            oof[va_idx] = model.predict(X_va_t).ravel()
            fold_rmses.append(rmse(y_va, oof[va_idx]))
            test_preds_folds.append(model.predict(X_test_t).ravel())

        test_preds_prev = np.mean(test_preds_folds, axis=0)

    return oof, fold_rmses, test_preds_prev


# ======================================================================
# Piecewise Calibration (ChatGPT + Claude combined idea)
# ======================================================================

def piecewise_calibrate(oof, test_pred, y_true):
    """3-zone piecewise linear calibration.

    Low zone: y' = a_low * y + b_low   (for y < t_low)
    Mid zone: unchanged
    High zone: y' = t_high + s_high * (y - t_high)  (for y > t_high)
    """
    best_oof = oof.copy()
    best_test = test_pred.copy()
    best_score = rmse(y_true, oof)

    for t_low in [20, 25, 30, 35]:
        for t_high in [100, 120, 140, 150, 160]:
            for a_low in np.arange(0.88, 1.01, 0.01):
                for s_high in np.arange(1.01, 1.50, 0.02):
                    oof_cal = oof.copy()
                    # Low zone
                    mask_low = oof_cal < t_low
                    oof_cal[mask_low] = oof_cal[mask_low] * a_low
                    # High zone
                    mask_high = oof_cal > t_high
                    oof_cal[mask_high] = t_high + (oof_cal[mask_high] - t_high) * s_high

                    s = rmse(y_true, oof_cal)
                    if s < best_score - 0.001:
                        best_score = s
                        best_oof = oof_cal.copy()

                        test_cal = test_pred.copy()
                        test_cal[test_cal < t_low] *= a_low
                        mask_ht = test_cal > t_high
                        test_cal[mask_ht] = t_high + (test_cal[mask_ht] - t_high) * s_high
                        best_test = test_cal.copy()

    return best_oof, best_test, best_score


# ======================================================================
# Conditional Ensemble (lightweight)
# ======================================================================

def conditional_ensemble_light(oofs_dict, tests_dict, y_true, n_trials=200):
    """Lightweight conditional ensemble with hard gate."""
    names = list(oofs_dict.keys())
    n = len(names)
    oofs = np.column_stack([oofs_dict[n_] for n_ in names])
    tests = np.column_stack([tests_dict[n_] for n_ in names])

    best_score = 999
    best_oof = None
    best_test = None

    # Simple approach: optimize for different threshold splits
    for threshold in [120, 140, 150, 160]:
        for width in [10, 20]:
            proxy = oofs.mean(axis=1)

            def sigmoid_gate(p, t=threshold, w=width):
                return 1 / (1 + np.exp(-(p - t) / w))

            gate = sigmoid_gate(proxy)

            def obj(params):
                w_lo = np.abs(params[:n])
                w_hi = np.abs(params[n:])
                w_lo = w_lo / (w_lo.sum() + 1e-8)
                w_hi = w_hi / (w_hi.sum() + 1e-8)
                p_lo = (oofs * w_lo).sum(axis=1)
                p_hi = (oofs * w_hi).sum(axis=1)
                pred = (1 - gate) * p_lo + gate * p_hi
                return rmse(y_true, pred)

            for _ in range(n_trials):
                w0 = np.random.dirichlet(np.ones(n) * 2, size=2).ravel()
                res = minimize(obj, w0, method="Nelder-Mead",
                               options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9})
                if res.fun < best_score:
                    best_score = res.fun
                    w_lo = np.abs(res.x[:n])
                    w_hi = np.abs(res.x[n:])
                    w_lo = w_lo / (w_lo.sum() + 1e-8)
                    w_hi = w_hi / (w_hi.sum() + 1e-8)
                    p_lo = (oofs * w_lo).sum(axis=1)
                    p_hi = (oofs * w_hi).sum(axis=1)
                    best_oof = (1 - gate) * p_lo + gate * p_hi

                    proxy_t = tests.mean(axis=1)
                    g_t = sigmoid_gate(proxy_t)
                    t_lo = (tests * w_lo).sum(axis=1)
                    t_hi = (tests * w_hi).sum(axis=1)
                    best_test = (1 - g_t) * t_lo + g_t * t_hi

    return best_oof, best_test, best_score


# ======================================================================
# Residual analysis
# ======================================================================

def analyze_residuals(y_true, y_pred, label=""):
    bins = [0, 30, 60, 100, 150, 200, 300]
    residuals = y_pred - y_true
    overall = rmse(y_true, y_pred)
    mask200 = y_true >= 200
    bias200 = residuals[mask200].mean() if mask200.sum() > 0 else 0
    mask150 = y_true >= 150
    rmse150 = np.sqrt((residuals[mask150] ** 2).mean()) if mask150.sum() > 0 else 0

    print(f"  {label}")
    print(f"    Overall RMSE: {overall:.4f} | 200+ bias: {bias200:+.2f} | 150+ RMSE: {rmse150:.2f}")
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() == 0:
            continue
        r = residuals[mask]
        print(f"    {bins[i]:>5d}-{bins[i+1]:<5d} n={mask.sum():>4d} RMSE={np.sqrt((r**2).mean()):>7.2f} bias={r.mean():>+7.2f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 26b: Fast Ensemble + Piecewise Calibration")
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
    # STEP 1: Build individual models (including Phase 26a winners)
    # ==================================================================
    print("\n=== STEP 1: Individual Models ===")

    # A: Wavelet detail features (Phase 26a winner — 14.00)
    print("\n  --- Wavelet Detail (Phase 26a winner) ---")
    PP_EMSC = [{"name": "emsc", "poly_order": 2}]
    pipe_emsc = build_preprocess_pipeline(PP_EMSC)
    pipe_emsc.fit(X_train)
    X_tr_emsc = pipe_emsc.transform(X_train)
    X_te_emsc = pipe_emsc.transform(X_test)

    for wavelet in ['db4', 'db6']:
        for level in [3, 4, 5]:
            for details in [[1, 2], [1, 2, 3], [2, 3]]:
                det_str = "_".join(map(str, details))
                wl_tr = wavelet_selective_features(X_tr_emsc, wavelet=wavelet, level=level,
                                                    keep_approx=False, keep_details=details)
                wl_te = wavelet_selective_features(X_te_emsc, wavelet=wavelet, level=level,
                                                    keep_approx=False, keep_details=details)
                sc = StandardScaler()
                wl_tr = sc.fit_transform(wl_tr)
                wl_te = sc.transform(wl_te)

                oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                      extra_train_feats=wl_tr, extra_test_feats=wl_te,
                                      min_moisture=170, pl_w=0.5, pl_rounds=2)
                log(f"wl_{wavelet}_L{level}_d{det_str}", oof, f, tp)

    # Also with approx+details combined
    for wavelet in ['db4']:
        for level in [4]:
            wl_tr = wavelet_selective_features(X_tr_emsc, wavelet=wavelet, level=level,
                                                keep_approx=True, keep_details=[1, 2])
            wl_te = wavelet_selective_features(X_te_emsc, wavelet=wavelet, level=level,
                                                keep_approx=True, keep_details=[1, 2])
            sc = StandardScaler()
            wl_tr = sc.fit_transform(wl_tr)
            wl_te = sc.transform(wl_te)
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  extra_train_feats=wl_tr, extra_test_feats=wl_te,
                                  min_moisture=170, pl_w=0.5, pl_rounds=2)
            log(f"wl_{wavelet}_L{level}_approx_d12", oof, f, tp)

    # B: MCR-ALS features
    print("\n  --- MCR-ALS ---")
    for n_comp in [2, 4]:
        mcr_tr, mcr_te = mcr_als_features(X_train, X_test, y_train, n_comp)
        if mcr_tr is not None:
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  extra_train_feats=mcr_tr, extra_test_feats=mcr_te,
                                  min_moisture=170, pl_w=0.5, pl_rounds=2)
            log(f"mcr_{n_comp}", oof, f, tp)

    # C: Standard diversity models
    print("\n  --- Diversity HP ---")
    hp_variants = [
        ("base", {}),
        ("wdv", {"__wdv__": True}),
        ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
        ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01}),
        ("d3l10", {"max_depth": 3, "num_leaves": 10, "n_estimators": 1500, "learning_rate": 0.005}),
        ("d7l30", {"max_depth": 7, "num_leaves": 30, "n_estimators": 600, "learning_rate": 0.03}),
        ("d4l15", {"max_depth": 4, "num_leaves": 15, "n_estimators": 500, "learning_rate": 0.04}),
    ]

    for hp_name, hp_ov in hp_variants:
        use_wdv = hp_ov.pop("__wdv__", False)
        params = {**LGBM_BASE, **hp_ov}
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params, min_moisture=170,
                              pl_w=0.5, pl_rounds=2, use_wdv_proj=use_wdv)
        log(f"div_{hp_name}", oof, f, tp)

    # D: Wavelet + diversity HPs
    print("\n  --- Best Wavelet + HP diversity ---")
    # Use db4 L4 details [1,2] (the winner from Phase 26a)
    wl_tr_best = wavelet_selective_features(X_tr_emsc, wavelet='db4', level=4,
                                             keep_approx=False, keep_details=[1, 2])
    wl_te_best = wavelet_selective_features(X_te_emsc, wavelet='db4', level=4,
                                             keep_approx=False, keep_details=[1, 2])
    sc_best = StandardScaler()
    wl_tr_best = sc_best.fit_transform(wl_tr_best)
    wl_te_best = sc_best.transform(wl_te_best)

    for hp_name, hp_ov in [
        ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
        ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01}),
        ("d3l10", {"max_depth": 3, "num_leaves": 10, "n_estimators": 1500, "learning_rate": 0.005}),
    ]:
        params = {**LGBM_BASE, **hp_ov}
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params,
                              extra_train_feats=wl_tr_best, extra_test_feats=wl_te_best,
                              min_moisture=170, pl_w=0.5, pl_rounds=2)
        log(f"wl_d12_{hp_name}", oof, f, tp)

    # E: Wavelet + MCR combined
    print("\n  --- Wavelet + MCR combined ---")
    mcr_tr2, mcr_te2 = mcr_als_features(X_train, X_test, y_train, 4)
    if mcr_tr2 is not None:
        combo_tr = np.hstack([wl_tr_best, mcr_tr2])
        combo_te = np.hstack([wl_te_best, mcr_te2])
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              extra_train_feats=combo_tr, extra_test_feats=combo_te,
                              min_moisture=170, pl_w=0.5, pl_rounds=2)
        log("wl_mcr_combo", oof, f, tp)

    # ==================================================================
    # STEP 2: Load previous phase models
    # ==================================================================
    print("\n=== STEP 2: Load Previous Phase Models ===")

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
            print(f"  Loaded {loaded} from {p_dir}")

    # ==================================================================
    # STEP 3: Greedy + NM Ensemble
    # ==================================================================
    print("\n=== STEP 3: Greedy + NM Ensemble ===")

    valid_names = [n for n in all_models
                   if np.isfinite(all_models[n]["oof"]).all()
                   and np.isfinite(all_models[n]["test"]).all()
                   and len(all_models[n]["oof"]) == len(y_train)]
    oofs = np.column_stack([all_models[n]["oof"] for n in valid_names])
    tests = np.column_stack([all_models[n]["test"] for n in valid_names])
    rmses_list = [all_models[n]["rmse"] for n in valid_names]

    print(f"  {len(valid_names)} valid models")
    ranked_idx = sorted(range(len(valid_names)), key=lambda i: rmses_list[i])
    for i in ranked_idx[:20]:
        print(f"    {rmses_list[i]:.4f}  {valid_names[i]}")

    # Greedy
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
            if len(selected) <= 15 or len(selected) % 5 == 0:
                print(f"    +{len(selected)}: {valid_names[best_i][:55]:55s} {best_s:.4f}")
        else:
            break

    greedy_avg = oofs[:, selected].mean(axis=1)
    greedy_test = tests[:, selected].mean(axis=1)
    greedy_s = rmse(y_train, greedy_avg)
    print(f"  Greedy ({len(selected)} models): {greedy_s:.4f}")
    all_models["greedy_ens"] = {"oof": greedy_avg, "test": greedy_test, "rmse": greedy_s}

    # NM (1000 trials)
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
    # STEP 4: Conditional Ensemble (lightweight)
    # ==================================================================
    print("\n=== STEP 4: Conditional Ensemble ===")

    # Use top 8 models for conditional ensemble
    top8 = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:8]
    ce_oofs = {n: d["oof"] for n, d in top8}
    ce_tests = {n: d["test"] for n, d in top8}

    ce_oof, ce_test, ce_score = conditional_ensemble_light(
        ce_oofs, ce_tests, y_train, n_trials=200)

    if ce_oof is not None:
        print(f"  Conditional ensemble (8 models): {ce_score:.4f}")
        all_models["cond_ens_8"] = {"oof": ce_oof, "test": ce_test, "rmse": ce_score}

    # Also try top 5
    top5 = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:5]
    ce_oofs5 = {n: d["oof"] for n, d in top5}
    ce_tests5 = {n: d["test"] for n, d in top5}
    ce_oof5, ce_test5, ce_score5 = conditional_ensemble_light(
        ce_oofs5, ce_tests5, y_train, n_trials=300)
    if ce_oof5 is not None:
        print(f"  Conditional ensemble (5 models): {ce_score5:.4f}")
        all_models["cond_ens_5"] = {"oof": ce_oof5, "test": ce_test5, "rmse": ce_score5}

    # ==================================================================
    # STEP 5: Piecewise Calibration (ChatGPT + Claude)
    # ==================================================================
    print("\n=== STEP 5: Piecewise Calibration ===")

    top_cands = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:15]
    for base_name, base_data in top_cands:
        if "pwcal" in base_name:
            continue
        oof_pw, test_pw, score_pw = piecewise_calibrate(
            base_data["oof"], base_data["test"], y_train)
        if score_pw < base_data["rmse"] - 0.005:
            all_models[f"pwcal_{base_name}"] = {"oof": oof_pw, "test": test_pw, "rmse": score_pw}
            print(f"  PW cal: {base_data['rmse']:.4f} → {score_pw:.4f} ({base_name})")

    # ==================================================================
    # STEP 6: Final re-ensemble with calibrated models
    # ==================================================================
    print("\n=== STEP 6: Final Re-Ensemble ===")

    valid_names2 = [n for n in all_models
                    if np.isfinite(all_models[n]["oof"]).all()
                    and np.isfinite(all_models[n]["test"]).all()
                    and len(all_models[n]["oof"]) == len(y_train)]
    oofs2 = np.column_stack([all_models[n]["oof"] for n in valid_names2])
    tests2 = np.column_stack([all_models[n]["test"] for n in valid_names2])
    rmses2 = [all_models[n]["rmse"] for n in valid_names2]

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
            if len(selected2) <= 10 or len(selected2) % 5 == 0:
                print(f"    +{len(selected2)}: {valid_names2[best_i][:55]:55s} {best_s:.4f}")
        else:
            break

    greedy_s2 = rmse(y_train, oofs2[:, selected2].mean(axis=1))
    print(f"  Final greedy ({len(selected2)} models): {greedy_s2:.4f}")

    # Final NM
    sub2 = oofs2[:, selected2]
    sub_t2 = tests2[:, selected2]
    ns2 = len(selected2)

    def obj2(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub2 * wn).sum(axis=1))

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

    final_oof = (sub2 * best_w2).sum(axis=1)
    final_test = (sub_t2 * best_w2).sum(axis=1)
    print(f"  Final NM: {best_opt2:.4f}")
    for i, idx in enumerate(selected2):
        if best_w2[i] > 0.01:
            print(f"    {best_w2[i]:.3f}  {valid_names2[idx]}")

    all_models["final_nm"] = {"oof": final_oof, "test": final_test, "rmse": best_opt2}
    all_models["final_greedy"] = {"oof": oofs2[:, selected2].mean(axis=1),
                                   "test": tests2[:, selected2].mean(axis=1),
                                   "rmse": greedy_s2}

    # Final piecewise on best
    for cname in ["final_nm", "final_greedy", "cond_ens_8", "cond_ens_5"]:
        if cname not in all_models:
            continue
        d = all_models[cname]
        oof_pw, test_pw, score_pw = piecewise_calibrate(d["oof"], d["test"], y_train)
        if score_pw < d["rmse"] - 0.005:
            all_models[f"pwcal_{cname}"] = {"oof": oof_pw, "test": test_pw, "rmse": score_pw}
            print(f"  Final PW: {d['rmse']:.4f} → {score_pw:.4f} ({cname})")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 26b FINAL SUMMARY")
    print("=" * 70)

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:40]:
        star = " ★★★" if data["rmse"] < 13.5 else (" ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else ""))
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 25 best: 13.61")
    improvement = 13.61 - best_data["rmse"]
    print(f"  Improvement: {improvement:+.4f}")

    analyze_residuals(y_train, best_data["oof"], f"BEST ({best_name})")

    # Save
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        sub = pd.DataFrame({
            "sample number": test_ids.values,
            "含水率": data["test"]
        })
        path = submissions_dir / f"submission_phase26b_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    oof_dir = Path("runs") / f"phase26b_{ts}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked},
        "phase": "26b",
        "description": "Fast Ensemble + Piecewise Calibration",
    }
    with open(oof_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, data in all_models.items():
        np.save(oof_dir / f"oof_{name}.npy", data["oof"])
        np.save(oof_dir / f"test_{name}.npy", data["test"])

    print(f"\n  Artifacts: {oof_dir}")
    print(f"  Total models: {len(all_models)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
