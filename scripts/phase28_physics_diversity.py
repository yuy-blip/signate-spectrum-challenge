#!/usr/bin/env python3
"""Phase 28: Physics-informed + Model diversity + Expert insights.

Based on NIR spectroscopy domain expertise (Gemini + ChatGPT):
1. Model diversity: CatBoost, SVR, ElasticNet (not just LightGBM)
2. Water band interval models (7100-6400, 5300-4950 cm⁻¹)
3. Multi-window SG derivatives
4. Piecewise regression (≤30, 30-100, >100% FSP zones)
5. SNV/MSC as alternative scatter corrections
6. Physics features: water band integrals + ratios
7. Species-balanced evaluation
8. Quantile regression branches for high moisture
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import (Ridge, ElasticNet, HuberRegressor, RidgeCV,
                                  LogisticRegression)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.base import clone
from scipy.optimize import nnls
import lightgbm as lgb
import xgboost as xgb
import warnings, json
from datetime import datetime
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not available")

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
test_ids = test['sample number'].values

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Target: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}, std={y.std():.1f}")

# =============================================================================
# Wavelength band definitions (from NIR domain knowledge)
# =============================================================================
def get_band_mask(center, half_w=150):
    return (wavelengths >= center - half_w) & (wavelengths <= center + half_w)

def get_range_mask(lo, hi):
    return (wavelengths >= lo) & (wavelengths <= hi)

# Water bands
WATER_5200 = get_range_mask(4950, 5300)    # O-H combination
WATER_7000 = get_range_mask(6400, 7100)    # O-H first overtone
WATER_8400 = get_band_mask(8400, 100)      # ~1190nm water band
REF_8800 = get_band_mask(8800, 150)        # reference (no water)

# Water-only interval for dedicated models
WATER_INTERVALS = WATER_5200 | WATER_7000
WATER_INTERVALS_PLUS = WATER_5200 | WATER_7000 | WATER_8400

n_water = WATER_INTERVALS.sum()
n_water_plus = WATER_INTERVALS_PLUS.sum()
print(f"Water interval features: {n_water} (core), {n_water_plus} (extended)")
print(f"  5200 band: {WATER_5200.sum()} pts [{wavelengths[WATER_5200][0]:.0f}-{wavelengths[WATER_5200][-1]:.0f}]")
print(f"  7000 band: {WATER_7000.sum()} pts [{wavelengths[WATER_7000][0]:.0f}-{wavelengths[WATER_7000][-1]:.0f}]")

# =============================================================================
# Physics feature extraction
# =============================================================================
def extract_physics_features(X_in):
    """Extract physics-informed features from raw spectra."""
    feats = []
    # Mean absorbance in water bands
    feats.append(X_in[:, WATER_5200].mean(axis=1, keepdims=True))
    feats.append(X_in[:, WATER_7000].mean(axis=1, keepdims=True))
    if WATER_8400.any():
        feats.append(X_in[:, WATER_8400].mean(axis=1, keepdims=True))
    # Reference band
    if REF_8800.any():
        ref = np.clip(X_in[:, REF_8800].mean(axis=1, keepdims=True), 1e-10, None)
        feats.append(X_in[:, WATER_5200].mean(axis=1, keepdims=True) / ref)
        feats.append(X_in[:, WATER_7000].mean(axis=1, keepdims=True) / ref)
    # Peak depth in water bands
    feats.append(X_in[:, WATER_5200].max(axis=1, keepdims=True) -
                 X_in[:, WATER_5200].min(axis=1, keepdims=True))
    feats.append(X_in[:, WATER_7000].max(axis=1, keepdims=True) -
                 X_in[:, WATER_7000].min(axis=1, keepdims=True))
    # Integral (sum) of water bands
    feats.append(X_in[:, WATER_5200].sum(axis=1, keepdims=True))
    feats.append(X_in[:, WATER_7000].sum(axis=1, keepdims=True))
    return np.hstack(feats)

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

def snv_transform(X_in):
    means = X_in.mean(axis=1, keepdims=True)
    stds = np.clip(X_in.std(axis=1, keepdims=True), 1e-10, None)
    return (X_in - means) / stds

def msc_transform(X_ref, X_in):
    """Multiplicative Scatter Correction."""
    out = np.zeros_like(X_in)
    for i in range(X_in.shape[0]):
        coef = np.polyfit(X_ref, X_in[i], 1)
        out[i] = (X_in[i] - coef[1]) / max(coef[0], 1e-10)
    return out

def preprocess_v2(Xtr, Xte, scatter="emsc", sg_w=9, bs=4, poly=2, sg_d=1,
                  add_physics=False, multi_sg=False, water_only=False, **_kw):
    """Enhanced preprocessing."""
    # Physics features (from raw spectra before correction)
    phys_tr = extract_physics_features(Xtr) if add_physics else None
    phys_te = extract_physics_features(Xte) if add_physics else None

    # Water band selection (before scatter correction on full spectrum)
    if water_only:
        mask = WATER_INTERVALS_PLUS
        Xtr = Xtr[:, mask]
        Xte = Xte[:, mask]

    # Scatter correction
    if scatter == "emsc":
        ref = Xtr.mean(axis=0)
        Xtr = emsc_transform(ref, Xtr, poly)
        Xte = emsc_transform(ref, Xte, poly)
    elif scatter == "snv":
        Xtr = snv_transform(Xtr)
        Xte = snv_transform(Xte)
    elif scatter == "msc":
        ref = Xtr.mean(axis=0)
        Xtr = msc_transform(ref, Xtr)
        Xte = msc_transform(ref, Xte)

    # SG derivative
    if multi_sg and sg_d > 0:
        parts_tr, parts_te = [], []
        for w in [7, 11, 21]:
            if w > Xtr.shape[1]:
                continue
            parts_tr.append(savgol_filter(Xtr, w, min(2, w - 1), deriv=sg_d, axis=1))
            parts_te.append(savgol_filter(Xte, w, min(2, w - 1), deriv=sg_d, axis=1))
        Xtr = np.hstack(parts_tr)
        Xte = np.hstack(parts_te)
    elif sg_d > 0:
        Xtr = savgol_filter(Xtr, sg_w, min(2, sg_w - 1), deriv=sg_d, axis=1)
        Xte = savgol_filter(Xte, sg_w, min(2, sg_w - 1), deriv=sg_d, axis=1)

    # Binning
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

    if phys_tr is not None:
        Xtr = np.hstack([Xtr, phys_tr])
        Xte = np.hstack([Xte, phys_te])

    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)

# =============================================================================
# CV helpers
# =============================================================================
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def species_balanced_rmse(y_true, y_pred, sp):
    """Per-species RMSE averaged (not weighted by n)."""
    scores = []
    for s in np.unique(sp):
        mask = sp == s
        scores.append(rmse(y_true[mask], y_pred[mask]))
    return np.mean(scores)

GKF = GroupKFold(n_splits=13)

def cv_score(model, pp_kwargs, label=""):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in GKF.split(X, y, species):
        Xtr_t, Xva_t = preprocess_v2(X[tr_idx].copy(), X[va_idx].copy(), **pp_kwargs)
        m = clone(model)
        if isinstance(m, lgb.LGBMRegressor):
            m.fit(Xtr_t, y[tr_idx], eval_set=[(Xva_t, y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        elif isinstance(m, xgb.XGBRegressor):
            m.fit(Xtr_t, y[tr_idx], eval_set=[(Xva_t, y[va_idx])], verbose=False)
        elif HAS_CATBOOST and isinstance(m, CatBoostRegressor):
            m.fit(Xtr_t, y[tr_idx], eval_set=(Xva_t, y[va_idx]), verbose=False)
        else:
            m.fit(Xtr_t, y[tr_idx])
        oof[va_idx] = np.clip(m.predict(Xva_t).ravel(), 0, 500)
    score = rmse(y, oof)
    sb_score = species_balanced_rmse(y, oof, species)
    if label:
        print(f"  {label}: RMSE={score:.4f}, SB-RMSE={sb_score:.4f}")
    return oof, score

def cv_with_test(model, pp_kwargs):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        Xtr_t, Xva_t = preprocess_v2(X[tr_idx].copy(), X[va_idx].copy(), **pp_kwargs)
        _, Xte_t = preprocess_v2(X[tr_idx].copy(), X_test.copy(), **pp_kwargs)
        m = clone(model)
        if isinstance(m, lgb.LGBMRegressor):
            m.fit(Xtr_t, y[tr_idx], eval_set=[(Xva_t, y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        elif isinstance(m, xgb.XGBRegressor):
            m.fit(Xtr_t, y[tr_idx], eval_set=[(Xva_t, y[va_idx])], verbose=False)
        elif HAS_CATBOOST and isinstance(m, CatBoostRegressor):
            m.fit(Xtr_t, y[tr_idx], eval_set=(Xva_t, y[va_idx]), verbose=False)
        else:
            m.fit(Xtr_t, y[tr_idx])
        oof[va_idx] = np.clip(m.predict(Xva_t).ravel(), 0, 500)
        test_pred += np.clip(m.predict(Xte_t).ravel(), 0, 500)
    test_pred /= 13
    return oof, test_pred, rmse(y, oof)

# =============================================================================
# PHASE 0: Residual Analysis
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 0: Residual Analysis of Baseline")
print("=" * 60)

pp_base = {"scatter": "emsc", "sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1}
base_lgbm = lgb.LGBMRegressor(
    n_estimators=2000, max_depth=3, num_leaves=7,
    learning_rate=0.1, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1, random_state=42)
oof_base, score_base = cv_score(base_lgbm, pp_base, "Baseline LightGBM")
residuals = y - oof_base

for lo, hi, name in [(0, 30, "Bound (<30)"), (30, 100, "Free low (30-100)"),
                      (100, 500, "Free high (>100)")]:
    mask = (y >= lo) & (y < hi)
    if mask.sum() == 0: continue
    r = residuals[mask]
    print(f"  {name}: n={mask.sum():4d}, RMSE={np.sqrt((r**2).mean()):.2f}, "
          f"bias={r.mean():+.2f}, |max|={np.abs(r).max():.1f}")

print("\nPer-species:")
for s in sorted(np.unique(species)):
    mask = species == s
    r = residuals[mask]
    print(f"  Sp{s:2d}: RMSE={np.sqrt((r**2).mean()):6.2f}, bias={r.mean():+6.1f}, "
          f"n={mask.sum():3d}, y_range=[{y[mask].min():.0f}-{y[mask].max():.0f}]")

# =============================================================================
# PHASE 1: Strategic PP + Model Screening
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Preprocessing + Model Screening")
print("=" * 60)

PP_CONFIGS = {
    # Proven best configs
    "emsc_sg1_w7_b4":    {"scatter": "emsc", "sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1},
    "emsc_sg1_w9_b4":    {"scatter": "emsc", "sg_w": 9, "bs": 4, "poly": 2, "sg_d": 1},
    "emsc_sg1_w7_b8":    {"scatter": "emsc", "sg_w": 7, "bs": 8, "poly": 2, "sg_d": 1},
    "emsc_sg1_w11_b4":   {"scatter": "emsc", "sg_w": 11, "bs": 4, "poly": 2, "sg_d": 1},
    "emsc_sg1_w7_b16":   {"scatter": "emsc", "sg_w": 7, "bs": 16, "poly": 2, "sg_d": 1},
    # NEW: SNV
    "snv_sg1_w7_b4":     {"scatter": "snv", "sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1},
    "snv_sg1_w9_b4":     {"scatter": "snv", "sg_w": 9, "bs": 4, "poly": 2, "sg_d": 1},
    # NEW: MSC
    "msc_sg1_w7_b4":     {"scatter": "msc", "sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1},
    # NEW: Physics features
    "emsc_sg1_w7_b4_ph": {"scatter": "emsc", "sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1, "add_physics": True},
    "emsc_sg1_w9_b4_ph": {"scatter": "emsc", "sg_w": 9, "bs": 4, "poly": 2, "sg_d": 1, "add_physics": True},
    # NEW: Multi-window SG
    "emsc_multisg_b4":   {"scatter": "emsc", "sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1, "multi_sg": True},
    "emsc_multisg_b8":   {"scatter": "emsc", "sg_w": 7, "bs": 8, "poly": 2, "sg_d": 1, "multi_sg": True},
    # NEW: 2nd derivative (wider windows to reduce noise)
    "emsc_sg2_w11_b4":   {"scatter": "emsc", "sg_w": 11, "bs": 4, "poly": 2, "sg_d": 2},
    "emsc_sg2_w15_b4":   {"scatter": "emsc", "sg_w": 15, "bs": 4, "poly": 2, "sg_d": 2},
    "emsc_sg2_w21_b4":   {"scatter": "emsc", "sg_w": 21, "bs": 4, "poly": 2, "sg_d": 2},
    # NEW: No derivative
    "emsc_sg0_b4":       {"scatter": "emsc", "sg_w": 7, "bs": 4, "poly": 2, "sg_d": 0},
    "snv_sg0_b4":        {"scatter": "snv", "sg_w": 7, "bs": 4, "poly": 2, "sg_d": 0},
    # NEW: Water-only interval models
    "emsc_sg1_w7_b2_H2O": {"scatter": "emsc", "sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1, "water_only": True},
    "snv_sg1_w7_b2_H2O":  {"scatter": "snv", "sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1, "water_only": True},
    "emsc_sg0_b2_H2O":    {"scatter": "emsc", "sg_w": 7, "bs": 2, "poly": 2, "sg_d": 0, "water_only": True},
}

# Model pool
MODELS = {}

# LightGBM
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
if HAS_CATBOOST:
    for d in [4, 6]:
        for lr in [0.03, 0.05, 0.1]:
            MODELS[f"cat_d{d}_lr{lr}"] = CatBoostRegressor(
                iterations=2000, depth=d, learning_rate=lr,
                l2_leaf_reg=5.0, random_seed=42, verbose=0)

# SVR (great for spectral data per literature)
for C in [1, 10, 100]:
    for eps in [0.1, 1.0]:
        MODELS[f"svr_C{C}_e{eps}"] = SVR(kernel='rbf', C=C, epsilon=eps, gamma='scale')

# PLS (5-35 per ChatGPT recommendation)
for nc in [5, 8, 10, 12, 15, 20, 28, 35]:
    MODELS[f"pls_{nc}"] = PLSRegression(n_components=nc, max_iter=1000)

# Ridge
for a in [0.1, 1.0, 10.0, 100.0]:
    MODELS[f"ridge_{a}"] = Ridge(alpha=a)

# ElasticNet
for a in [0.1, 1.0]:
    for r in [0.3, 0.5, 0.7]:
        MODELS[f"enet_a{a}_r{r}"] = ElasticNet(alpha=a, l1_ratio=r, max_iter=5000)

# KNN
for k in [5, 10, 15, 20]:
    MODELS[f"knn_{k}"] = KNeighborsRegressor(n_neighbors=k, weights="distance")

# RF / ET
MODELS["rf_500"] = RandomForestRegressor(
    n_estimators=500, max_features=0.5, min_samples_leaf=5, n_jobs=-1, random_state=42)
MODELS["et_500"] = ExtraTreesRegressor(
    n_estimators=500, max_features=0.5, min_samples_leaf=5, n_jobs=-1, random_state=42)

# Huber
MODELS["huber"] = HuberRegressor(epsilon=1.35, max_iter=1000)

# LightGBM quantile (upper) for high moisture correction
MODELS["lgbm_q75"] = lgb.LGBMRegressor(
    n_estimators=2000, max_depth=4, num_leaves=15,
    learning_rate=0.05, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0,
    objective='quantile', alpha=0.75,
    verbose=-1, n_jobs=-1, random_state=42)
MODELS["lgbm_q25"] = lgb.LGBMRegressor(
    n_estimators=2000, max_depth=4, num_leaves=15,
    learning_rate=0.05, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0,
    objective='quantile', alpha=0.25,
    verbose=-1, n_jobs=-1, random_state=42)

# GradientBoosting (sklearn, different from LightGBM)
MODELS["gbr_d3"] = GradientBoostingRegressor(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    min_samples_leaf=10, subsample=0.8, random_state=42)

print(f"PP configs: {len(PP_CONFIGS)}, Models: {len(MODELS)}")

# --- Step 1: Screen PP configs with proven best model ---
print("\n--- PP Config Screening ---")
pp_scores = {}
for pp_name, pp_kwargs in PP_CONFIGS.items():
    try:
        oof, score = cv_score(MODELS["lgbm_d3_lr0.1_s42"], pp_kwargs)
        sb = species_balanced_rmse(y, oof, species)
        pp_scores[pp_name] = {"score": score, "sb_score": sb, "oof": oof, "pp_kwargs": pp_kwargs}
        print(f"  {pp_name:28s}: RMSE={score:.4f}, SB-RMSE={sb:.4f}")
    except Exception as e:
        print(f"  {pp_name:28s}: FAILED - {e}")

pp_ranked = sorted(pp_scores.items(), key=lambda x: x[1]["score"])
print("\nPP Config ranking (by RMSE):")
for i, (name, info) in enumerate(pp_ranked[:10]):
    print(f"  {i+1:2d}. RMSE={info['score']:.4f}  SB={info['sb_score']:.4f}  {name}")

# --- Step 2: Full model screening on top PP configs ---
print("\n--- Full Model Screening ---")
# Use top 5 PP configs (proven + best new ones)
top_pp_names = [n for n, _ in pp_ranked[:5]]
# Also always include water-only and physics variants if they exist
for extra in ["emsc_sg1_w7_b2_H2O", "snv_sg1_w7_b2_H2O", "emsc_sg1_w7_b4_ph",
              "emsc_multisg_b4", "emsc_sg2_w15_b4"]:
    if extra in pp_scores and extra not in top_pp_names:
        top_pp_names.append(extra)

all_results = []
done = 0
total = len(top_pp_names) * len(MODELS)

for pp_name in top_pp_names:
    pp_kwargs = pp_scores[pp_name]["pp_kwargs"]
    print(f"\n--- {pp_name} (RMSE={pp_scores[pp_name]['score']:.4f} with ref model) ---")
    for mname, model in MODELS.items():
        done += 1
        try:
            oof, score = cv_score(model, pp_kwargs)
            all_results.append({
                "pipe": pp_name, "model": mname, "rmse": score,
                "sb_rmse": species_balanced_rmse(y, oof, species),
                "pp_kwargs": pp_kwargs, "oof": oof,
            })
            if done % 30 == 0 or score < 18.0:
                print(f"  [{done}/{total}] {mname}: {score:.4f}")
        except Exception as e:
            if done % 80 == 0:
                print(f"  [{done}/{total}] FAILED {mname}: {e}")

# Add remaining PP results with reference model only
for pp_name in [n for n, _ in pp_ranked if n not in top_pp_names]:
    info = pp_scores[pp_name]
    all_results.append({
        "pipe": pp_name, "model": "lgbm_d3_lr0.1_s42", "rmse": info["score"],
        "sb_rmse": info["sb_score"],
        "pp_kwargs": info["pp_kwargs"], "oof": info["oof"],
    })

all_results.sort(key=lambda x: x["rmse"])

print("\n" + "=" * 60)
print("TOP 50 CV RESULTS")
print("=" * 60)
for i, r in enumerate(all_results[:50]):
    print(f"{i+1:3d}. RMSE={r['rmse']:7.4f}  SB={r['sb_rmse']:7.4f}  {r['pipe']:28s}  {r['model']}")

# Model diversity check
print("\nModel diversity in top 30:")
for mtype, count in Counter(r["model"].split("_")[0] for r in all_results[:30]).most_common():
    print(f"  {mtype}: {count}")

# =============================================================================
# PHASE 2: Piecewise Regression (≤30, 30-100, >100)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: Piecewise Regression")
print("=" * 60)

def piecewise_cv(model_lo, model_mid, model_hi, pp_kwargs, boundaries=(30, 100)):
    """3-zone piecewise: bound water, free water low, free water high."""
    b1, b2 = boundaries
    oof = np.zeros(len(y))

    for tr_idx, va_idx in GKF.split(X, y, species):
        Xtr_t, Xva_t = preprocess_v2(X[tr_idx].copy(), X[va_idx].copy(), **pp_kwargs)

        # Classifier for zone assignment
        y_zone = np.zeros(len(tr_idx), dtype=int)
        y_zone[y[tr_idx] > b1] = 1
        y_zone[y[tr_idx] > b2] = 2

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial')
        clf.fit(Xtr_t, y_zone)
        va_zone = clf.predict(Xva_t)

        # Train zone-specific models
        for zone, model_z, clip_lo, clip_hi in [
            (0, model_lo, 0, b1 + 15),
            (1, model_mid, b1 - 15, b2 + 30),
            (2, model_hi, b2 - 30, 500)
        ]:
            tr_mask = y_zone == zone
            va_mask = va_zone == zone
            if tr_mask.sum() < 5 or not va_mask.any():
                continue
            m = clone(model_z)
            m.fit(Xtr_t[tr_mask], y[tr_idx][tr_mask])
            oof[va_idx[va_mask]] = np.clip(m.predict(Xva_t[va_mask]).ravel(), clip_lo, clip_hi)

    return oof, rmse(y, oof)

# Test piecewise with best PP
best_pp = pp_ranked[0][1]["pp_kwargs"]
pp_name_best = pp_ranked[0][0]

model_lo = Ridge(alpha=10.0)
model_mid = lgb.LGBMRegressor(
    n_estimators=1500, max_depth=3, num_leaves=7, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1, random_state=42)
model_hi = lgb.LGBMRegressor(
    n_estimators=1500, max_depth=4, num_leaves=15, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1, random_state=42)

oof_pw, score_pw = piecewise_cv(model_lo, model_mid, model_hi, best_pp, (30, 100))
print(f"Piecewise (30/100): {score_pw:.4f}")

# Also try 2-zone (FSP only)
oof_pw2, score_pw2 = piecewise_cv(
    Ridge(alpha=10.0),
    lgb.LGBMRegressor(n_estimators=1500, max_depth=3, num_leaves=7,
                       learning_rate=0.05, min_child_samples=5,
                       subsample=0.8, colsample_bytree=0.8,
                       verbose=-1, n_jobs=-1, random_state=42),
    lgb.LGBMRegressor(n_estimators=1500, max_depth=4, num_leaves=15,
                       learning_rate=0.05, min_child_samples=5,
                       subsample=0.8, colsample_bytree=0.8,
                       verbose=-1, n_jobs=-1, random_state=42),
    best_pp, (30, 300))  # effectively 2-zone
print(f"Piecewise (30 only): {score_pw2:.4f}")
print(f"Baseline single:     {score_base:.4f}")

# =============================================================================
# PHASE 3: Test Predictions for Top N (diverse)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: Test Predictions (diversity-aware)")
print("=" * 60)

N_TOP = 40
selected = []
seen = set()
# Ensure model type diversity: limit per type
type_count = Counter()
MAX_PER_TYPE = 15

for r in all_results:
    mtype = r["model"].split("_")[0]
    combo = (r["pipe"], r["model"])
    if combo in seen:
        continue
    if type_count[mtype] >= MAX_PER_TYPE:
        continue
    seen.add(combo)
    type_count[mtype] += 1
    selected.append(r)
    if len(selected) >= N_TOP:
        break

print(f"Selected {len(selected)} models:")
for mtype, count in Counter(r["model"].split("_")[0] for r in selected).most_common():
    print(f"  {mtype}: {count}")

top_results = []
for i, r in enumerate(selected):
    mname = r["model"]
    model = MODELS[mname]
    print(f"  [{i+1}/{len(selected)}] {r['pipe']} + {mname} (CV={r['rmse']:.4f})")
    oof, test_pred, score = cv_with_test(model, r["pp_kwargs"])
    top_results.append({
        "pipe": r["pipe"], "model": mname, "rmse": score,
        "oof": oof, "test_pred": test_pred,
    })

# =============================================================================
# PHASE 4: Ensemble
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 4: Ensemble")
print("=" * 60)

oof_matrix = np.column_stack([r["oof"] for r in top_results])
test_matrix = np.column_stack([r["test_pred"] for r in top_results])

# Simple averages
for n in [5, 10, 15, 20, 25, 30, 35, 40]:
    if n <= len(top_results):
        avg = oof_matrix[:, :n].mean(1)
        print(f"Top {n:2d} avg: RMSE={rmse(y, avg):.4f}, SB={species_balanced_rmse(y, avg, species):.4f}")

# NNLS
w, _ = nnls(oof_matrix, y)
if w.sum() > 0:
    w /= w.sum()
else:
    w = np.ones(len(top_results)) / len(top_results)
nnls_oof = oof_matrix @ w
nnls_test = test_matrix @ w
print(f"NNLS:    RMSE={rmse(y, nnls_oof):.4f}, SB={species_balanced_rmse(y, nnls_oof, species):.4f}")

print("\nNNLS weights (>0.5%):")
for i in np.argsort(-w)[:20]:
    if w[i] > 0.005:
        print(f"  {w[i]:.3f}  {top_results[i]['pipe']} + {top_results[i]['model']}")

# Ridge stacking
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

# NNLS on species-reweighted target (to reduce species bias)
# Weight each sample inversely proportional to species count
sp_weights = np.zeros(len(y))
for s in np.unique(species):
    mask = species == s
    sp_weights[mask] = 1.0 / mask.sum()
sp_weights /= sp_weights.sum() / len(y)  # normalize to mean=1

w_sb, _ = nnls(oof_matrix * sp_weights[:, None], y * sp_weights)
if w_sb.sum() > 0:
    w_sb /= w_sb.sum()
    nnls_sb_oof = oof_matrix @ w_sb
    nnls_sb_test = test_matrix @ w_sb
    print(f"NNLS-SB: RMSE={rmse(y, nnls_sb_oof):.4f}, SB={species_balanced_rmse(y, nnls_sb_oof, species):.4f}")

# =============================================================================
# Save submissions
# =============================================================================
print("\n" + "=" * 60)
print("Saving Submissions")
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
if w_sb.sum() > 0:
    submissions["nnls_sb"] = np.clip(nnls_sb_test, 0, 500)

for name, preds in submissions.items():
    if preds is None: continue
    fname = f"submission_p28_{name}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(sub_dir / fname, header=False, index=False)
    print(f"  {fname}")

# Save results
results_dir = Path(__file__).resolve().parent.parent / "runs"
results_dir.mkdir(exist_ok=True)
summary = {
    "pp_ranking": [{"name": n, "rmse": i["score"], "sb_rmse": i["sb_score"]}
                   for n, i in pp_ranked],
    "top_models": [{"pipe": r["pipe"], "model": r["model"], "rmse": r["rmse"]}
                   for r in top_results],
    "ensemble": {
        "best_single": float(top_results[0]["rmse"]),
        "top5_avg": float(rmse(y, oof_matrix[:, :5].mean(1))),
        "top10_avg": float(rmse(y, oof_matrix[:, :10].mean(1))),
        "nnls": float(rmse(y, nnls_oof)),
        "ridge_stack": float(rmse(y, oof_stack)),
    },
    "piecewise": {"3zone": float(score_pw), "2zone": float(score_pw2)},
    "baseline": float(score_base),
    "nnls_weights": w.tolist(),
}
with open(results_dir / "phase28_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Baseline (single LGBM):   {score_base:.4f}")
print(f"Best single (this run):   {top_results[0]['rmse']:.4f}  ({top_results[0]['pipe']} + {top_results[0]['model']})")
best_avg = min(rmse(y, oof_matrix[:, :n].mean(1)) for n in range(2, min(len(top_results)+1, 41)))
print(f"Best simple avg:          {best_avg:.4f}")
print(f"NNLS blend:               {rmse(y, nnls_oof):.4f}")
print(f"Ridge stacking:           {rmse(y, oof_stack):.4f}")
print(f"Piecewise 3-zone:         {score_pw:.4f}")
print(f"Piecewise 2-zone:         {score_pw2:.4f}")
print(f"\n>>> Previous best: NNLS 17.29 <<<")
