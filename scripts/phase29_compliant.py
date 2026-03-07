#!/usr/bin/env python3
"""Phase 29: Rule-compliant pipeline (NO pseudo-labeling).

Uses ONLY training data. No test data features used in any way.
Combines:
1. EMSC + SG1 + Binning(4) (proven best preprocessing)
2. Universal WDV (training data only, deterministic)
3. Model diversity (LightGBM, XGBoost, CatBoost, SVR, PLS, Ridge, KNN, RF, ET)
4. Moisture Weighting
5. Wavelet features
6. Water band interval models
7. NNLS ensemble with diversity
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.decomposition import PCA
from scipy.optimize import nnls
import lightgbm as lgb
import xgboost as xgb
import warnings, json, sys
from datetime import datetime
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("CatBoost not available")

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("PyWavelets not available")

# =============================================================================
# Data
# =============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
train = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
test = pd.read_csv(DATA_DIR / "test.csv", encoding="cp932")

feat_cols = [c for c in train.columns
             if c not in ['sample number', 'species number', '樹種', '含水率']]
X = train[feat_cols].values.astype(np.float64)
y = train['含水率'].values.astype(np.float64)
species = train['species number'].values
wavelengths = np.array([float(c) for c in feat_cols])

feat_cols_test = [c for c in test.columns
                  if c not in ['sample number', 'species number', '樹種']]
X_test = test[feat_cols_test].values.astype(np.float64)

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Target: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}")

# Water band masks
def get_range_mask(lo, hi):
    return (wavelengths >= lo) & (wavelengths <= hi)

WATER_5200 = get_range_mask(4950, 5300)
WATER_7000 = get_range_mask(6400, 7100)
WATER_INTERVALS = WATER_5200 | WATER_7000

# =============================================================================
# Universal WDV (training data only — rule compliant)
# =============================================================================
def generate_universal_wdv(X_tr, y_tr, sp_tr, n_aug=30, extrap_factor=1.5,
                           min_moisture=170, dy_scale=0.3, dy_offset=30):
    """Generate synthetic high-moisture samples using Universal Water Vector.
    Uses ONLY training data. Fully deterministic."""
    species_deltas = []
    species_dy = []

    for s in np.unique(sp_tr):
        mask = sp_tr == s
        X_sp, y_sp = X_tr[mask], y_tr[mask]
        med = np.median(y_sp)
        hi = y_sp > med
        lo = ~hi
        if hi.sum() < 3 or lo.sum() < 3:
            continue
        delta = X_sp[hi].mean(0) - X_sp[lo].mean(0)
        dy = y_sp[hi].mean() - y_sp[lo].mean()
        if abs(dy) < 1e-10:
            continue
        species_deltas.append(delta / dy)
        species_dy.append(dy)

    if len(species_deltas) < 2:
        return np.zeros((0, X_tr.shape[1])), np.zeros(0)

    delta_mat = np.array(species_deltas)
    dy_arr = np.array(species_dy)

    if len(species_deltas) >= 3:
        pca = PCA(n_components=1)
        pca.fit(delta_mat)
        water_vec = pca.components_[0]
        if np.corrcoef(delta_mat @ water_vec, dy_arr)[0, 1] < 0:
            water_vec = -water_vec
    else:
        water_vec = delta_mat.mean(0)
        water_vec /= np.linalg.norm(water_vec) + 1e-10

    # Scale calibration
    proj = X_tr @ water_vec
    coef = np.polyfit(proj, y_tr, 1)
    scale = coef[0]

    # Generate synthetic samples
    synth_X_list, synth_y_list = [], []
    for s in np.unique(sp_tr):
        mask = sp_tr == s
        X_sp, y_sp = X_tr[mask], y_tr[mask]
        hi_idx = y_sp >= min_moisture
        if hi_idx.sum() == 0:
            continue
        for i in np.where(hi_idx)[0]:
            target_dy = extrap_factor * (y_sp[i] * dy_scale + dy_offset)
            step = target_dy / (scale + 1e-8)
            synth_X_list.append(X_sp[i] + step * water_vec)
            synth_y_list.append(y_sp[i] + target_dy)

    if len(synth_X_list) == 0:
        return np.zeros((0, X_tr.shape[1])), np.zeros(0)

    synth_X = np.array(synth_X_list)
    synth_y = np.array(synth_y_list)

    # Deterministic downsampling
    if len(synth_y) > n_aug:
        idx = np.linspace(0, len(synth_y) - 1, n_aug).astype(int)
        synth_X = synth_X[idx]
        synth_y = synth_y[idx]

    return synth_X, synth_y

# =============================================================================
# Preprocessing
# =============================================================================
def emsc_transform(X_ref, X_in, poly=2):
    n_wl = X_ref.shape[0]
    w = np.linspace(-1, 1, n_wl)
    cols = [X_ref, np.ones(n_wl)]
    for p in range(1, poly + 1):
        cols.append(w ** p)
    D = np.column_stack(cols)
    c, _, _, _ = np.linalg.lstsq(D, X_in.T, rcond=None)
    a1 = c[0]
    a1[np.abs(a1) < 1e-10] = 1e-10
    bl = D[:, 1:] @ c[1:]
    return ((X_in.T - bl) / a1).T


def preprocess(Xtr, Xte, sg_w=7, bs=4, poly=2, sg_d=1, water_only=False):
    """Standard preprocessing: EMSC + SG + Binning + Scale."""
    if water_only:
        Xtr = Xtr[:, WATER_INTERVALS]
        Xte = Xte[:, WATER_INTERVALS]

    ref = Xtr.mean(axis=0)
    Xtr = emsc_transform(ref, Xtr, poly)
    Xte = emsc_transform(ref, Xte, poly)

    if sg_d > 0:
        Xtr = savgol_filter(Xtr, sg_w, min(2, sg_w - 1), deriv=sg_d, axis=1)
        Xte = savgol_filter(Xte, sg_w, min(2, sg_w - 1), deriv=sg_d, axis=1)

    if bs > 1:
        def _bin(X):
            n, p = X.shape
            nf = p // bs
            parts = []
            if nf > 0:
                parts.append(X[:, :nf * bs].reshape(n, nf, bs).mean(2))
            if p % bs > 0:
                parts.append(X[:, nf * bs:].mean(1, keepdims=True))
            return np.hstack(parts)
        Xtr = _bin(Xtr)
        Xte = _bin(Xte)

    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def preprocess_wavelet(Xtr, Xte, wavelet='db4', level=4, details=(1, 2)):
    """EMSC + Wavelet DWT features."""
    ref = Xtr.mean(axis=0)
    Xtr = emsc_transform(ref, Xtr, 2)
    Xte = emsc_transform(ref, Xte, 2)

    def _dwt_features(X):
        feats = []
        for i in range(X.shape[0]):
            coeffs = pywt.wavedec(X[i], wavelet, level=level)
            row = []
            for d in details:
                if d < len(coeffs):
                    row.append(coeffs[d])
            feats.append(np.concatenate(row))
        return np.array(feats)

    Xtr = _dwt_features(Xtr)
    Xte = _dwt_features(Xte)

    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


# =============================================================================
# CV helpers
# =============================================================================
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def sb_rmse(y_true, y_pred, sp):
    scores = []
    for s in np.unique(sp):
        m = sp == s
        scores.append(rmse(y_true[m], y_pred[m]))
    return np.mean(scores)

GKF = GroupKFold(n_splits=13)


def cv_score(model, pp_fn, pp_kwargs, use_wdv=False, wdv_kwargs=None,
             mw_power=0, mw_base=1.0, label=""):
    """GroupKFold CV with optional UWV augmentation and moisture weighting."""
    oof = np.zeros(len(y))
    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx], species[tr_idx]
        X_va = X[va_idx].copy()

        # UWV augmentation (training data only)
        if use_wdv and wdv_kwargs:
            synth_X, synth_y = generate_universal_wdv(
                X_tr, y_tr, sp_tr, **wdv_kwargs)
            if len(synth_y) > 0:
                X_tr_full = np.vstack([X_tr, synth_X])
                y_tr_full = np.concatenate([y_tr, synth_y])
                sp_tr_full = np.concatenate([sp_tr,
                    np.full(len(synth_y), -1)])  # synthetic = no species
            else:
                X_tr_full, y_tr_full, sp_tr_full = X_tr, y_tr, sp_tr
        else:
            X_tr_full, y_tr_full, sp_tr_full = X_tr, y_tr, sp_tr

        # Preprocess
        Xtr_t, Xva_t = pp_fn(X_tr_full, X_va, **pp_kwargs)

        # Moisture weighting
        sample_weight = None
        if mw_power > 0:
            sample_weight = mw_base + (y_tr_full / y_tr_full.max()) ** mw_power

        m = clone(model)
        fit_kwargs = {}
        if isinstance(m, lgb.LGBMRegressor):
            fit_kwargs['eval_set'] = [(Xva_t, y[va_idx])]
            fit_kwargs['callbacks'] = [lgb.early_stopping(50, verbose=False)]
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
        elif isinstance(m, xgb.XGBRegressor):
            fit_kwargs['eval_set'] = [(Xva_t, y[va_idx])]
            fit_kwargs['verbose'] = False
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
        elif HAS_CB and isinstance(m, CatBoostRegressor):
            fit_kwargs['eval_set'] = (Xva_t, y[va_idx])
            fit_kwargs['verbose'] = False
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight

        m.fit(Xtr_t, y_tr_full, **fit_kwargs)
        oof[va_idx] = np.clip(m.predict(Xva_t).ravel(), 0, 500)

    score = rmse(y, oof)
    sbs = sb_rmse(y, oof, species)
    if label:
        print(f"  {label}: RMSE={score:.4f}, SB={sbs:.4f}")
    return oof, score, sbs


def cv_with_test(model, pp_fn, pp_kwargs, use_wdv=False, wdv_kwargs=None,
                 mw_power=0, mw_base=1.0):
    """CV + test prediction."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx], species[tr_idx]
        X_va = X[va_idx].copy()

        if use_wdv and wdv_kwargs:
            synth_X, synth_y = generate_universal_wdv(
                X_tr, y_tr, sp_tr, **wdv_kwargs)
            if len(synth_y) > 0:
                X_tr_full = np.vstack([X_tr, synth_X])
                y_tr_full = np.concatenate([y_tr, synth_y])
            else:
                X_tr_full, y_tr_full = X_tr, y_tr
        else:
            X_tr_full, y_tr_full = X_tr, y_tr

        Xtr_t, Xva_t = pp_fn(X_tr_full, X_va, **pp_kwargs)
        _, Xte_t = pp_fn(X_tr_full, X_test.copy(), **pp_kwargs)

        sample_weight = None
        if mw_power > 0:
            sample_weight = mw_base + (y_tr_full / y_tr_full.max()) ** mw_power

        m = clone(model)
        fit_kwargs = {}
        if isinstance(m, lgb.LGBMRegressor):
            fit_kwargs['eval_set'] = [(Xva_t, y[va_idx])]
            fit_kwargs['callbacks'] = [lgb.early_stopping(50, verbose=False)]
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
        elif isinstance(m, xgb.XGBRegressor):
            fit_kwargs['eval_set'] = [(Xva_t, y[va_idx])]
            fit_kwargs['verbose'] = False
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
        elif HAS_CB and isinstance(m, CatBoostRegressor):
            fit_kwargs['eval_set'] = (Xva_t, y[va_idx])
            fit_kwargs['verbose'] = False
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight

        m.fit(Xtr_t, y_tr_full, **fit_kwargs)
        oof[va_idx] = np.clip(m.predict(Xva_t).ravel(), 0, 500)
        test_pred += np.clip(m.predict(Xte_t).ravel(), 0, 500)
    test_pred /= 13
    return oof, test_pred, rmse(y, oof)


# =============================================================================
# PHASE 1: Screening (no WDV, no MW — pure baseline with diversity)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Pure Baseline Screening (no WDV, no MW)")
print("=" * 60)

PP_STANDARD = {"sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1}

MODELS = {}
# LightGBM variants
for d in [3, 4, 5]:
    for lr in [0.02, 0.05, 0.1]:
        for seed in [42, 0, 123]:
            MODELS[f"lgbm_d{d}_lr{lr}_s{seed}"] = lgb.LGBMRegressor(
                n_estimators=2000, max_depth=d, num_leaves=min(2**d-1, 31),
                learning_rate=lr, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=5.0,
                verbose=-1, n_jobs=-1, random_state=seed)

# XGBoost
for d in [3, 4]:
    for lr in [0.03, 0.05, 0.1]:
        MODELS[f"xgb_d{d}_lr{lr}"] = xgb.XGBRegressor(
            n_estimators=2000, max_depth=d, learning_rate=lr,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, verbosity=0, n_jobs=-1)

# CatBoost
if HAS_CB:
    for d in [4, 6]:
        for lr in [0.03, 0.05, 0.1]:
            MODELS[f"cat_d{d}_lr{lr}"] = CatBoostRegressor(
                iterations=2000, depth=d, learning_rate=lr,
                l2_leaf_reg=5.0, random_seed=42, verbose=0)

# PLS
for nc in [5, 8, 12, 15, 20, 28]:
    MODELS[f"pls_{nc}"] = PLSRegression(n_components=nc, max_iter=1000)

# Ridge
for a in [1.0, 10.0, 100.0]:
    MODELS[f"ridge_{a}"] = Ridge(alpha=a)

# KNN
for k in [5, 10, 20]:
    MODELS[f"knn_{k}"] = KNeighborsRegressor(n_neighbors=k, weights="distance")

# RF / ET
MODELS["rf_500"] = RandomForestRegressor(
    n_estimators=500, max_features=0.5, min_samples_leaf=5,
    n_jobs=-1, random_state=42)
MODELS["et_500"] = ExtraTreesRegressor(
    n_estimators=500, max_features=0.5, min_samples_leaf=5,
    n_jobs=-1, random_state=42)

# Huber
MODELS["huber"] = HuberRegressor(epsilon=1.35, max_iter=1000)

# Quantile LGBM
MODELS["lgbm_q75"] = lgb.LGBMRegressor(
    n_estimators=2000, max_depth=4, num_leaves=15,
    learning_rate=0.05, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
    objective='quantile', alpha=0.75,
    verbose=-1, n_jobs=-1, random_state=42)
MODELS["lgbm_q25"] = lgb.LGBMRegressor(
    n_estimators=2000, max_depth=4, num_leaves=15,
    learning_rate=0.05, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
    objective='quantile', alpha=0.25,
    verbose=-1, n_jobs=-1, random_state=42)

print(f"Models: {len(MODELS)}")

# Screen all models with standard PP
all_results = []
done = 0
total = len(MODELS)

print("\n--- Standard PP (emsc_sg1_w7_b4) ---")
for mname, model in MODELS.items():
    done += 1
    try:
        oof, score, sbs = cv_score(model, preprocess, PP_STANDARD, label="" if done % 10 != 0 else mname)
        all_results.append({
            "pipe": "standard", "model": mname, "rmse": score, "sb_rmse": sbs,
            "oof": oof, "pp_fn": preprocess, "pp_kwargs": PP_STANDARD,
            "use_wdv": False, "wdv_kwargs": None, "mw_power": 0, "mw_base": 1.0,
        })
        if done % 10 == 0 or score < 18.0:
            print(f"  [{done}/{total}] {mname}: RMSE={score:.4f}, SB={sbs:.4f}")
    except Exception as e:
        if done % 20 == 0:
            print(f"  [{done}/{total}] FAILED {mname}: {e}")

# Water band only models
print("\n--- Water Band Only ---")
PP_WATER = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1, "water_only": True}
for mname in ["lgbm_d3_lr0.1_s42", "lgbm_d4_lr0.05_s0", "pls_12", "ridge_10.0", "knn_10"]:
    if mname not in MODELS:
        continue
    try:
        oof, score, sbs = cv_score(MODELS[mname], preprocess, PP_WATER, label=f"H2O_{mname}")
        all_results.append({
            "pipe": "water_only", "model": mname, "rmse": score, "sb_rmse": sbs,
            "oof": oof, "pp_fn": preprocess, "pp_kwargs": PP_WATER,
            "use_wdv": False, "wdv_kwargs": None, "mw_power": 0, "mw_base": 1.0,
        })
    except Exception as e:
        print(f"  FAILED H2O_{mname}: {e}")

# Wavelet features
if HAS_PYWT:
    print("\n--- Wavelet Features ---")
    for mname in ["lgbm_d3_lr0.1_s42", "lgbm_d4_lr0.05_s0", "lgbm_d3_lr0.05_s123"]:
        try:
            oof, score, sbs = cv_score(
                MODELS[mname], preprocess_wavelet,
                {"wavelet": "db4", "level": 4, "details": (1, 2)},
                label=f"wav_{mname}")
            all_results.append({
                "pipe": "wavelet", "model": mname, "rmse": score, "sb_rmse": sbs,
                "oof": oof, "pp_fn": preprocess_wavelet,
                "pp_kwargs": {"wavelet": "db4", "level": 4, "details": (1, 2)},
                "use_wdv": False, "wdv_kwargs": None, "mw_power": 0, "mw_base": 1.0,
            })
        except Exception as e:
            print(f"  FAILED wav_{mname}: {e}")

# =============================================================================
# PHASE 2: UWV augmentation (training data only)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: UWV Augmentation (rule compliant)")
print("=" * 60)

WDV_CONFIGS = [
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 150},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 50, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 30, "extrap_factor": 2.0, "min_moisture": 170},
]

# Test with best model
best_lgbm = "lgbm_d3_lr0.1_s42"
for wdv_kw in WDV_CONFIGS:
    wdv_name = f"uwv_n{wdv_kw['n_aug']}_f{wdv_kw['extrap_factor']}_m{wdv_kw['min_moisture']}"
    oof, score, sbs = cv_score(
        MODELS[best_lgbm], preprocess, PP_STANDARD,
        use_wdv=True, wdv_kwargs=wdv_kw,
        label=wdv_name)
    all_results.append({
        "pipe": wdv_name, "model": best_lgbm, "rmse": score, "sb_rmse": sbs,
        "oof": oof, "pp_fn": preprocess, "pp_kwargs": PP_STANDARD,
        "use_wdv": True, "wdv_kwargs": wdv_kw, "mw_power": 0, "mw_base": 1.0,
    })

# Best UWV config with multiple models
best_wdv = {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170}
print("\n--- Best UWV with diverse models ---")
for mname in ["lgbm_d3_lr0.1_s42", "lgbm_d4_lr0.05_s0", "lgbm_d3_lr0.02_s123",
              "lgbm_d5_lr0.1_s42", "lgbm_d3_lr0.1_s0", "lgbm_d3_lr0.1_s123",
              "lgbm_d4_lr0.1_s42", "lgbm_d3_lr0.05_s42",
              "xgb_d3_lr0.05", "xgb_d4_lr0.05"]:
    if mname not in MODELS:
        continue
    try:
        oof, score, sbs = cv_score(
            MODELS[mname], preprocess, PP_STANDARD,
            use_wdv=True, wdv_kwargs=best_wdv,
            label=f"uwv_{mname}")
        all_results.append({
            "pipe": "uwv_best", "model": mname, "rmse": score, "sb_rmse": sbs,
            "oof": oof, "pp_fn": preprocess, "pp_kwargs": PP_STANDARD,
            "use_wdv": True, "wdv_kwargs": best_wdv, "mw_power": 0, "mw_base": 1.0,
        })
    except Exception as e:
        print(f"  FAILED uwv_{mname}: {e}")

# =============================================================================
# PHASE 3: Moisture Weighting
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: Moisture Weighting")
print("=" * 60)

MW_CONFIGS = [
    (1.0, 0.5), (1.5, 0.5), (2.0, 0.5), (1.0, 0.3),
]

for mw_p, mw_b in MW_CONFIGS:
    # Without UWV
    oof, score, sbs = cv_score(
        MODELS[best_lgbm], preprocess, PP_STANDARD,
        mw_power=mw_p, mw_base=mw_b,
        label=f"mw_p{mw_p}_b{mw_b}")
    all_results.append({
        "pipe": f"mw_p{mw_p}_b{mw_b}", "model": best_lgbm,
        "rmse": score, "sb_rmse": sbs, "oof": oof,
        "pp_fn": preprocess, "pp_kwargs": PP_STANDARD,
        "use_wdv": False, "wdv_kwargs": None,
        "mw_power": mw_p, "mw_base": mw_b,
    })

    # With UWV
    oof, score, sbs = cv_score(
        MODELS[best_lgbm], preprocess, PP_STANDARD,
        use_wdv=True, wdv_kwargs=best_wdv,
        mw_power=mw_p, mw_base=mw_b,
        label=f"uwv+mw_p{mw_p}_b{mw_b}")
    all_results.append({
        "pipe": f"uwv+mw_p{mw_p}_b{mw_b}", "model": best_lgbm,
        "rmse": score, "sb_rmse": sbs, "oof": oof,
        "pp_fn": preprocess, "pp_kwargs": PP_STANDARD,
        "use_wdv": True, "wdv_kwargs": best_wdv,
        "mw_power": mw_p, "mw_base": mw_b,
    })

# =============================================================================
# PHASE 4: Sort and select top models for test prediction
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 4: Results Summary")
print("=" * 60)

all_results.sort(key=lambda x: x["rmse"])

print("\nTOP 50:")
for i, r in enumerate(all_results[:50]):
    print(f"{i+1:3d}. RMSE={r['rmse']:7.4f}  SB={r['sb_rmse']:7.4f}  {r['pipe']:25s}  {r['model']}")

# Model diversity in top 30
print("\nModel diversity (top 30):")
for mtype, cnt in Counter(r["model"].split("_")[0] for r in all_results[:30]).most_common():
    print(f"  {mtype}: {cnt}")

# =============================================================================
# PHASE 5: Test Predictions with diversity
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 5: Test Predictions")
print("=" * 60)

N_TOP = 40
selected = []
seen = set()
type_count = Counter()
MAX_PER_TYPE = 15

for r in all_results:
    mtype = r["model"].split("_")[0]
    key = (r["pipe"], r["model"])
    if key in seen or type_count[mtype] >= MAX_PER_TYPE:
        continue
    seen.add(key)
    type_count[mtype] += 1
    selected.append(r)
    if len(selected) >= N_TOP:
        break

print(f"Selected {len(selected)} models:")
for mtype, cnt in Counter(r["model"].split("_")[0] for r in selected).most_common():
    print(f"  {mtype}: {cnt}")

top_results = []
for i, r in enumerate(selected):
    mname = r["model"]
    model = MODELS[mname]
    print(f"  [{i+1}/{len(selected)}] {r['pipe']} + {mname} (CV={r['rmse']:.4f})")
    oof, test_pred, score = cv_with_test(
        model, r["pp_fn"], r["pp_kwargs"],
        use_wdv=r["use_wdv"], wdv_kwargs=r["wdv_kwargs"],
        mw_power=r["mw_power"], mw_base=r["mw_base"])
    top_results.append({
        "pipe": r["pipe"], "model": mname, "rmse": score,
        "oof": oof, "test_pred": test_pred,
    })

# =============================================================================
# PHASE 6: Ensemble
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 6: Ensemble")
print("=" * 60)

oof_matrix = np.column_stack([r["oof"] for r in top_results])
test_matrix = np.column_stack([r["test_pred"] for r in top_results])

for n in [5, 10, 15, 20, 25, 30, 35, 40]:
    if n <= len(top_results):
        avg = oof_matrix[:, :n].mean(1)
        print(f"Top {n:2d} avg: RMSE={rmse(y, avg):.4f}")

# NNLS
w, _ = nnls(oof_matrix, y)
if w.sum() > 0:
    w /= w.sum()
else:
    w = np.ones(len(top_results)) / len(top_results)
nnls_oof = oof_matrix @ w
nnls_test = test_matrix @ w
print(f"NNLS: RMSE={rmse(y, nnls_oof):.4f}, SB={sb_rmse(y, nnls_oof, species):.4f}")

print("\nNNLS weights (>0.5%):")
for i in np.argsort(-w)[:20]:
    if w[i] > 0.005:
        print(f"  {w[i]:.3f}  {top_results[i]['pipe']} + {top_results[i]['model']}")

# Ridge stack
oof_stack = np.zeros(len(y))
test_stack_parts = []
for tr_idx, va_idx in GKF.split(X, y, species):
    meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
    meta.fit(oof_matrix[tr_idx], y[tr_idx])
    oof_stack[va_idx] = meta.predict(oof_matrix[va_idx])
    test_stack_parts.append(meta.predict(test_matrix))
oof_stack = np.clip(oof_stack, 0, 500)
test_stack = np.clip(np.mean(test_stack_parts, axis=0), 0, 500)
print(f"Ridge stack: RMSE={rmse(y, oof_stack):.4f}")

# =============================================================================
# Save
# =============================================================================
print("\n" + "=" * 60)
print("Saving Submissions (RULE COMPLIANT)")
print("=" * 60)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_dir = Path(__file__).resolve().parent.parent / "submissions"
sub_dir.mkdir(exist_ok=True)

submissions = {
    "best_single": np.clip(top_results[0]["test_pred"], 0, 500),
    "top5_avg": np.clip(test_matrix[:, :5].mean(1), 0, 500),
    "top10_avg": np.clip(test_matrix[:, :10].mean(1), 0, 500),
    "top20_avg": np.clip(test_matrix[:, :20].mean(1), 0, 500),
    "nnls": np.clip(nnls_test, 0, 500),
    "ridge_stack": test_stack,
}

for name, preds in submissions.items():
    fname = f"submission_p29_{name}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(sub_dir / fname, header=False, index=False)
    print(f"  {fname}")

# Save results
results_dir = Path(__file__).resolve().parent.parent / "runs"
results_dir.mkdir(exist_ok=True)
summary = {
    "rule_compliant": True,
    "no_pseudo_labeling": True,
    "no_test_data_usage": True,
    "top_models": [{"pipe": r["pipe"], "model": r["model"], "rmse": r["rmse"]}
                   for r in top_results],
    "ensemble": {
        "best_single": float(top_results[0]["rmse"]),
        "top5_avg": float(rmse(y, oof_matrix[:, :5].mean(1))),
        "nnls": float(rmse(y, nnls_oof)),
        "ridge_stack": float(rmse(y, oof_stack)),
    },
    "nnls_weights": w.tolist(),
}
with open(results_dir / "phase29_compliant_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("FINAL SUMMARY (RULE COMPLIANT — NO PL)")
print("=" * 60)
print(f"Best single:  {top_results[0]['rmse']:.4f}  ({top_results[0]['pipe']} + {top_results[0]['model']})")
print(f"NNLS blend:   {rmse(y, nnls_oof):.4f}")
print(f"Ridge stack:  {rmse(y, oof_stack):.4f}")
print(f"\nPrevious best (PL-free): 17.29")
print(f"Target: < 17.29")
