#!/usr/bin/env python3
"""Phase 30: NIR Domain Feature Engineering + Stacking + Conditional Ensemble.

Based on Gemini/ChatGPT domain expertise:
1. Water band ratios and integrals (5200/7000 cm⁻¹)
2. Peak position/width features from water absorption bands
3. Spectral shape descriptors (curvature, slope, asymmetry)
4. FSP-based conditional ensemble (bound vs free water)
5. Stacking meta-learner (L2 Ridge on L1 OOF predictions)
6. Species-aware features (interaction with spectral features)

Rule compliant: NO pseudo-labeling, NO test data for training.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapezoid
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
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
print(f"Wavelengths: [{wavelengths.min():.0f}, {wavelengths.max():.0f}]")

# =============================================================================
# Preprocessing (proven best: EMSC + SG1 + Bin4)
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


def standard_preprocess(Xtr, Xte, sg_w=7, bs=4, poly=2, sg_d=1):
    """Standard: EMSC + SG1(w=7) + Bin(4) + Scale."""
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


# =============================================================================
# NIR Domain Feature Engineering (Gemini/ChatGPT inspired)
# =============================================================================
def get_range_mask(lo, hi):
    return (wavelengths >= lo) & (wavelengths <= hi)

# Water band regions
WATER_5200 = get_range_mask(4950, 5300)  # O-H combination band
WATER_7000 = get_range_mask(6400, 7100)  # O-H first overtone
CH_BAND = get_range_mask(5600, 6100)     # C-H stretching (wood matrix)
FULL_NIR = get_range_mask(4000, 10000)

print(f"Water 5200 band: {WATER_5200.sum()} pts")
print(f"Water 7000 band: {WATER_7000.sum()} pts")
print(f"C-H band: {CH_BAND.sum()} pts")


def extract_domain_features(X_raw, wl=wavelengths):
    """Extract NIR domain-specific features from raw spectra.

    Features inspired by NIR spectroscopy domain knowledge:
    1. Water band integrals (area under absorption bands)
    2. Band ratios (water/matrix ratios)
    3. Peak positions and widths
    4. Spectral shape descriptors
    5. Derivative features at key wavelengths
    """
    n = X_raw.shape[0]
    feats = {}

    # --- 1. Band integrals (area under curve in key regions) ---
    wl_5200 = wl[WATER_5200]
    wl_7000 = wl[WATER_7000]
    wl_ch = wl[CH_BAND]

    for i in range(n):
        if i == 0:
            # Initialize feature arrays
            feats['integral_5200'] = np.zeros(n)
            feats['integral_7000'] = np.zeros(n)
            feats['integral_ch'] = np.zeros(n)
            feats['ratio_5200_ch'] = np.zeros(n)
            feats['ratio_7000_ch'] = np.zeros(n)
            feats['ratio_5200_7000'] = np.zeros(n)
            feats['peak_depth_5200'] = np.zeros(n)
            feats['peak_depth_7000'] = np.zeros(n)
            feats['peak_pos_5200'] = np.zeros(n)
            feats['peak_pos_7000'] = np.zeros(n)
            feats['peak_width_5200'] = np.zeros(n)
            feats['peak_width_7000'] = np.zeros(n)
            feats['baseline_slope'] = np.zeros(n)
            feats['overall_absorbance'] = np.zeros(n)
            feats['spectral_std'] = np.zeros(n)
            feats['spectral_skew'] = np.zeros(n)
            feats['spectral_kurt'] = np.zeros(n)
            feats['curvature_5200'] = np.zeros(n)
            feats['curvature_7000'] = np.zeros(n)
            feats['diff_water_bands'] = np.zeros(n)

        spec = X_raw[i]

        # Band integrals
        if wl_5200.sum() > 1:
            feats['integral_5200'][i] = trapezoid(spec[WATER_5200], wl_5200)
        if wl_7000.sum() > 1:
            feats['integral_7000'][i] = trapezoid(spec[WATER_7000], wl_7000)
        if wl_ch.sum() > 1:
            feats['integral_ch'][i] = trapezoid(spec[CH_BAND], wl_ch)

        # Band ratios
        ch_int = abs(feats['integral_ch'][i]) + 1e-10
        feats['ratio_5200_ch'][i] = feats['integral_5200'][i] / ch_int
        feats['ratio_7000_ch'][i] = feats['integral_7000'][i] / ch_int
        int_7000 = abs(feats['integral_7000'][i]) + 1e-10
        feats['ratio_5200_7000'][i] = feats['integral_5200'][i] / int_7000

        # --- 2. Peak analysis in water bands ---
        # 5200 band: find deepest absorption
        spec_5200 = spec[WATER_5200]
        if len(spec_5200) > 5:
            # Peak depth (min value relative to edges)
            baseline = np.linspace(spec_5200[0], spec_5200[-1], len(spec_5200))
            corrected = spec_5200 - baseline
            min_idx = np.argmin(corrected)
            feats['peak_depth_5200'][i] = corrected[min_idx]
            feats['peak_pos_5200'][i] = wl_5200[min_idx] if len(wl_5200) > 0 else 0
            # Width at half depth
            half_depth = corrected[min_idx] / 2
            above_half = corrected > half_depth
            feats['peak_width_5200'][i] = above_half.sum() / len(above_half)

        # 7000 band
        spec_7000 = spec[WATER_7000]
        if len(spec_7000) > 5:
            baseline = np.linspace(spec_7000[0], spec_7000[-1], len(spec_7000))
            corrected = spec_7000 - baseline
            min_idx = np.argmin(corrected)
            feats['peak_depth_7000'][i] = corrected[min_idx]
            feats['peak_pos_7000'][i] = wl_7000[min_idx] if len(wl_7000) > 0 else 0
            above_half = corrected > corrected[min_idx] / 2
            feats['peak_width_7000'][i] = above_half.sum() / len(above_half)

        # --- 3. Spectral shape descriptors ---
        feats['baseline_slope'][i] = (spec[-1] - spec[0]) / (wl[-1] - wl[0] + 1e-10)
        feats['overall_absorbance'][i] = spec.mean()
        feats['spectral_std'][i] = spec.std()

        centered = spec - spec.mean()
        std = spec.std() + 1e-10
        feats['spectral_skew'][i] = np.mean((centered / std) ** 3)
        feats['spectral_kurt'][i] = np.mean((centered / std) ** 4) - 3

        # --- 4. Curvature at band centers ---
        # 2nd derivative magnitude at water band centers
        if WATER_5200.sum() > 10:
            c5200 = WATER_5200.sum() // 2
            idx_5200 = np.where(WATER_5200)[0][c5200]
            if idx_5200 > 0 and idx_5200 < len(spec) - 1:
                feats['curvature_5200'][i] = spec[idx_5200-1] - 2*spec[idx_5200] + spec[idx_5200+1]

        if WATER_7000.sum() > 10:
            c7000 = WATER_7000.sum() // 2
            idx_7000 = np.where(WATER_7000)[0][c7000]
            if idx_7000 > 0 and idx_7000 < len(spec) - 1:
                feats['curvature_7000'][i] = spec[idx_7000-1] - 2*spec[idx_7000] + spec[idx_7000+1]

        # Difference between water bands (bound vs free water proxy)
        feats['diff_water_bands'][i] = spec[WATER_5200].mean() - spec[WATER_7000].mean()

    return np.column_stack([feats[k] for k in sorted(feats.keys())])


def extract_derivative_features(X_raw, wl=wavelengths):
    """Features from 1st and 2nd derivatives at key wavelengths."""
    # SG derivatives
    d1 = savgol_filter(X_raw, 7, 2, deriv=1, axis=1)
    d2 = savgol_filter(X_raw, 11, 3, deriv=2, axis=1)

    # Key wavelength positions for water
    key_wl = [5150, 5200, 5250, 6850, 6900, 6950, 7000, 7050]
    feats = []
    feat_names = []

    for target_wl in key_wl:
        idx = np.argmin(np.abs(wl - target_wl))
        feats.append(d1[:, idx:idx+1])
        feats.append(d2[:, idx:idx+1])
        feat_names.extend([f'd1_{target_wl}', f'd2_{target_wl}'])

    # D1 zero crossings in water bands (peak positions)
    n_zc_5200 = np.zeros((X_raw.shape[0], 1))
    n_zc_7000 = np.zeros((X_raw.shape[0], 1))
    for i in range(X_raw.shape[0]):
        d1_5200 = d1[i, WATER_5200]
        d1_7000 = d1[i, WATER_7000]
        n_zc_5200[i] = np.sum(np.diff(np.sign(d1_5200)) != 0)
        n_zc_7000[i] = np.sum(np.diff(np.sign(d1_7000)) != 0)
    feats.extend([n_zc_5200, n_zc_7000])

    return np.hstack(feats)


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


def cv_score_feats(model, X_features, label=""):
    """CV with pre-computed features (already preprocessed)."""
    oof = np.zeros(len(y))
    for tr_idx, va_idx in GKF.split(X, y, species):
        Xtr = X_features[tr_idx]
        Xva = X_features[va_idx]

        # Scale
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xva = sc.transform(Xva)

        m = clone(model)
        fit_kwargs = {}
        if isinstance(m, lgb.LGBMRegressor):
            fit_kwargs['eval_set'] = [(Xva, y[va_idx])]
            fit_kwargs['callbacks'] = [lgb.early_stopping(50, verbose=False)]
        elif isinstance(m, xgb.XGBRegressor):
            fit_kwargs['eval_set'] = [(Xva, y[va_idx])]
            fit_kwargs['verbose'] = False
        elif HAS_CB and isinstance(m, CatBoostRegressor):
            fit_kwargs['eval_set'] = (Xva, y[va_idx])
            fit_kwargs['verbose'] = False

        m.fit(Xtr, y[tr_idx], **fit_kwargs)
        oof[va_idx] = np.clip(m.predict(Xva).ravel(), 0, 500)

    score = rmse(y, oof)
    sbs = sb_rmse(y, oof, species)
    if label:
        print(f"  {label}: RMSE={score:.4f}, SB={sbs:.4f}")
    return oof, score, sbs


def cv_with_test_feats(model, X_features, X_test_features):
    """CV + test prediction with pre-computed features."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(X_test_features.shape[0])
    for tr_idx, va_idx in GKF.split(X, y, species):
        Xtr = X_features[tr_idx]
        Xva = X_features[va_idx]

        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xva = sc.transform(Xva)
        Xte = sc.transform(X_test_features)

        m = clone(model)
        fit_kwargs = {}
        if isinstance(m, lgb.LGBMRegressor):
            fit_kwargs['eval_set'] = [(Xva, y[va_idx])]
            fit_kwargs['callbacks'] = [lgb.early_stopping(50, verbose=False)]
        elif isinstance(m, xgb.XGBRegressor):
            fit_kwargs['eval_set'] = [(Xva, y[va_idx])]
            fit_kwargs['verbose'] = False
        elif HAS_CB and isinstance(m, CatBoostRegressor):
            fit_kwargs['eval_set'] = (Xva, y[va_idx])
            fit_kwargs['verbose'] = False

        m.fit(Xtr, y[tr_idx], **fit_kwargs)
        oof[va_idx] = np.clip(m.predict(Xva).ravel(), 0, 500)
        test_pred += np.clip(m.predict(Xte).ravel(), 0, 500)
    test_pred /= 13
    return oof, test_pred, rmse(y, oof)


# =============================================================================
# PHASE 1: Feature Engineering
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Domain Feature Engineering")
print("=" * 60)

# Extract features before EMSC (on raw spectra for physical meaning)
print("Extracting domain features from raw spectra...")
domain_feats_train = extract_domain_features(X)
domain_feats_test = extract_domain_features(X_test)
print(f"  Domain features: {domain_feats_train.shape[1]}")

deriv_feats_train = extract_derivative_features(X)
deriv_feats_test = extract_derivative_features(X_test)
print(f"  Derivative features: {deriv_feats_train.shape[1]}")

# Standard preprocessed features
print("Computing standard preprocessed features...")
# We need fold-independent preprocessing for feature combination
# Use global EMSC reference (all training data)
ref = X.mean(axis=0)
X_emsc = emsc_transform(ref, X, 2)
X_test_emsc = emsc_transform(ref, X_test, 2)

X_sg1 = savgol_filter(X_emsc, 7, 2, deriv=1, axis=1)
X_test_sg1 = savgol_filter(X_test_emsc, 7, 2, deriv=1, axis=1)

def _bin(X, bs=4):
    n, p = X.shape
    nf = p // bs
    parts = []
    if nf > 0:
        parts.append(X[:, :nf * bs].reshape(n, nf, bs).mean(2))
    if p % bs > 0:
        parts.append(X[:, nf * bs:].mean(1, keepdims=True))
    return np.hstack(parts)

X_binned = _bin(X_sg1, 4)
X_test_binned = _bin(X_test_sg1, 4)
print(f"  Binned spectral features: {X_binned.shape[1]}")

# Feature combination strategies
FEATURE_SETS = {}

# A: Standard spectral only (baseline)
FEATURE_SETS['spectral_only'] = (X_binned, X_test_binned)

# B: Domain features only
FEATURE_SETS['domain_only'] = (domain_feats_train, domain_feats_test)

# C: Derivative features only
FEATURE_SETS['deriv_only'] = (deriv_feats_train, deriv_feats_test)

# D: Spectral + domain features
FEATURE_SETS['spectral+domain'] = (
    np.hstack([X_binned, domain_feats_train]),
    np.hstack([X_test_binned, domain_feats_test])
)

# E: Spectral + derivative features
FEATURE_SETS['spectral+deriv'] = (
    np.hstack([X_binned, deriv_feats_train]),
    np.hstack([X_test_binned, deriv_feats_test])
)

# F: Spectral + domain + derivative
FEATURE_SETS['spectral+domain+deriv'] = (
    np.hstack([X_binned, domain_feats_train, deriv_feats_train]),
    np.hstack([X_test_binned, domain_feats_test, deriv_feats_test])
)

# G: Water band spectral + domain
X_water_train = X_sg1[:, WATER_5200 | WATER_7000]
X_water_test = X_test_sg1[:, WATER_5200 | WATER_7000]
X_water_binned = _bin(X_water_train, 2)
X_water_test_binned = _bin(X_water_test, 2)
FEATURE_SETS['water+domain'] = (
    np.hstack([X_water_binned, domain_feats_train]),
    np.hstack([X_water_test_binned, domain_feats_test])
)

# H: PCA-reduced spectral + domain
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_binned)
X_test_pca = pca.transform(X_test_binned)
FEATURE_SETS['pca50+domain'] = (
    np.hstack([X_pca, domain_feats_train]),
    np.hstack([X_test_pca, domain_feats_test])
)

print(f"\nFeature set sizes:")
for name, (Xtr, Xte) in FEATURE_SETS.items():
    print(f"  {name}: {Xtr.shape[1]} features")

# =============================================================================
# PHASE 2: Screen feature sets with reference model
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: Feature Set Screening")
print("=" * 60)

ref_model = lgb.LGBMRegressor(
    n_estimators=2000, max_depth=3, num_leaves=7,
    learning_rate=0.1, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0,
    verbose=-1, n_jobs=-1, random_state=42)

feat_results = {}
all_results = []

for fname, (Xtr, Xte) in FEATURE_SETS.items():
    oof, score, sbs = cv_score_feats(ref_model, Xtr, label=fname)
    feat_results[fname] = {'rmse': score, 'sb_rmse': sbs, 'oof': oof}
    all_results.append({
        'feat_set': fname, 'model': 'lgbm_ref', 'rmse': score, 'sb_rmse': sbs,
        'oof': oof, 'X_train': Xtr, 'X_test': Xte,
    })

# Rank feature sets
print("\nFeature set ranking:")
ranked = sorted(feat_results.items(), key=lambda x: x[1]['rmse'])
for i, (fname, r) in enumerate(ranked):
    print(f"  {i+1}. {fname}: RMSE={r['rmse']:.4f}, SB={r['sb_rmse']:.4f}")

# =============================================================================
# PHASE 3: Best feature sets × Model diversity
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: Model Diversity on Top Feature Sets")
print("=" * 60)

# Pick top 3 feature sets + baseline
top_feat_names = [r[0] for r in ranked[:3]]
if 'spectral_only' not in top_feat_names:
    top_feat_names.append('spectral_only')

MODELS = {
    'lgbm_d3_lr0.1_s42': lgb.LGBMRegressor(
        n_estimators=2000, max_depth=3, num_leaves=7,
        learning_rate=0.1, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=42),
    'lgbm_d3_lr0.02_s123': lgb.LGBMRegressor(
        n_estimators=2000, max_depth=3, num_leaves=7,
        learning_rate=0.02, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=123),
    'lgbm_d4_lr0.05_s0': lgb.LGBMRegressor(
        n_estimators=2000, max_depth=4, num_leaves=15,
        learning_rate=0.05, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=0),
    'lgbm_d5_lr0.05_s42': lgb.LGBMRegressor(
        n_estimators=2000, max_depth=5, num_leaves=31,
        learning_rate=0.05, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=42),
    'xgb_d3_lr0.05': xgb.XGBRegressor(
        n_estimators=2000, max_depth=3, learning_rate=0.05,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0, verbosity=0, n_jobs=-1),
    'xgb_d4_lr0.05': xgb.XGBRegressor(
        n_estimators=2000, max_depth=4, learning_rate=0.05,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0, verbosity=0, n_jobs=-1),
    'ridge_10': Ridge(alpha=10.0),
    'ridge_100': Ridge(alpha=100.0),
    'pls_12': PLSRegression(n_components=12, max_iter=1000),
    'pls_20': PLSRegression(n_components=20, max_iter=1000),
}

if HAS_CB:
    MODELS['cat_d4_lr0.05'] = CatBoostRegressor(
        iterations=2000, depth=4, learning_rate=0.05,
        l2_leaf_reg=5.0, random_seed=42, verbose=0)
    MODELS['cat_d6_lr0.05'] = CatBoostRegressor(
        iterations=2000, depth=6, learning_rate=0.05,
        l2_leaf_reg=5.0, random_seed=42, verbose=0)

print(f"Feature sets: {len(top_feat_names)}, Models: {len(MODELS)}")
total = len(top_feat_names) * len(MODELS)
done = 0

for fname in top_feat_names:
    Xtr, Xte = FEATURE_SETS[fname]
    print(f"\n--- {fname} ({Xtr.shape[1]} feats) ---")
    for mname, model in MODELS.items():
        done += 1
        try:
            oof, score, sbs = cv_score_feats(model, Xtr,
                label=f"{mname}" if (done % 5 == 0 or score < 17.5) else "")
            all_results.append({
                'feat_set': fname, 'model': mname, 'rmse': score, 'sb_rmse': sbs,
                'oof': oof, 'X_train': Xtr, 'X_test': Xte,
            })
            if score < 17.5:
                print(f"  *** [{done}/{total}] {mname}: RMSE={score:.4f}, SB={sbs:.4f}")
        except Exception as e:
            print(f"  [{done}/{total}] FAILED {mname}: {e}")

# =============================================================================
# PHASE 4: FSP-based Conditional Ensemble
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 4: FSP-based Conditional Ensemble")
print("=" * 60)

FSP = 30.0  # Fiber Saturation Point

# Train separate models for bound (<FSP) and free (>=FSP) water
best_feat = ranked[0][0]
best_Xtr, best_Xte = FEATURE_SETS[best_feat]
print(f"Using feature set: {best_feat}")

# Strategy 1: Two-model approach (bound + free water)
print("\n--- Two-model approach ---")
bound_mask = y < FSP
free_mask = y >= FSP
print(f"  Bound (<{FSP}): {bound_mask.sum()}, Free (>={FSP}): {free_mask.sum()}")

# Train a classifier to predict bound/free, then use specialized regressors
# Actually, use a single model's prediction as a soft gate
oof_conditional = np.zeros(len(y))
test_conditional = np.zeros(len(X_test))

for tr_idx, va_idx in GKF.split(X, y, species):
    Xtr_f = best_Xtr[tr_idx]
    Xva_f = best_Xtr[va_idx]

    sc = StandardScaler().fit(Xtr_f)
    Xtr_s = sc.transform(Xtr_f)
    Xva_s = sc.transform(Xva_f)
    Xte_s = sc.transform(best_Xte)

    y_tr = y[tr_idx]

    # Full model
    m_full = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=3, num_leaves=7,
        learning_rate=0.1, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=42)
    m_full.fit(Xtr_s, y_tr,
               eval_set=[(Xva_s, y[va_idx])],
               callbacks=[lgb.early_stopping(50, verbose=False)])

    # Bound water model (trained only on samples < 2*FSP)
    bound_tr = y_tr < 2 * FSP
    if bound_tr.sum() > 20:
        m_bound = lgb.LGBMRegressor(
            n_estimators=2000, max_depth=3, num_leaves=7,
            learning_rate=0.05, min_child_samples=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0,
            verbose=-1, n_jobs=-1, random_state=42)
        m_bound.fit(Xtr_s[bound_tr], y_tr[bound_tr],
                    eval_set=[(Xva_s, y[va_idx])],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

    # Free water model (trained only on samples >= FSP/2)
    free_tr = y_tr >= FSP / 2
    if free_tr.sum() > 20:
        m_free = lgb.LGBMRegressor(
            n_estimators=2000, max_depth=4, num_leaves=15,
            learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0,
            verbose=-1, n_jobs=-1, random_state=42)
        m_free.fit(Xtr_s[free_tr], y_tr[free_tr],
                   eval_set=[(Xva_s, y[va_idx])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])

    # Predictions
    pred_full = m_full.predict(Xva_s)
    pred_full_test = m_full.predict(Xte_s)

    # Soft gating based on full model prediction
    for va_i, pred_i in zip(va_idx, pred_full):
        if pred_i < FSP * 0.7:
            # Likely bound water — use bound specialist
            if bound_tr.sum() > 20:
                p_bound = m_bound.predict(Xva_s[va_idx == va_i])
                oof_conditional[va_i] = 0.6 * p_bound[0] + 0.4 * pred_i
            else:
                oof_conditional[va_i] = pred_i
        elif pred_i > FSP * 1.5:
            # Likely free water — use free specialist
            if free_tr.sum() > 20:
                p_free = m_free.predict(Xva_s[va_idx == va_i])
                oof_conditional[va_i] = 0.6 * p_free[0] + 0.4 * pred_i
            else:
                oof_conditional[va_i] = pred_i
        else:
            # Transition zone — blend all
            p_b = m_bound.predict(Xva_s[va_idx == va_i])[0] if bound_tr.sum() > 20 else pred_i
            p_f = m_free.predict(Xva_s[va_idx == va_i])[0] if free_tr.sum() > 20 else pred_i
            alpha = (pred_i - FSP * 0.7) / (FSP * 0.8)
            alpha = np.clip(alpha, 0, 1)
            oof_conditional[va_i] = (1-alpha) * p_b + alpha * p_f

    # Test predictions
    for ti in range(len(X_test)):
        pi = pred_full_test[ti]
        if pi < FSP * 0.7:
            if bound_tr.sum() > 20:
                test_conditional[ti] += 0.6 * m_bound.predict(Xte_s[ti:ti+1])[0] + 0.4 * pi
            else:
                test_conditional[ti] += pi
        elif pi > FSP * 1.5:
            if free_tr.sum() > 20:
                test_conditional[ti] += 0.6 * m_free.predict(Xte_s[ti:ti+1])[0] + 0.4 * pi
            else:
                test_conditional[ti] += pi
        else:
            p_b = m_bound.predict(Xte_s[ti:ti+1])[0] if bound_tr.sum() > 20 else pi
            p_f = m_free.predict(Xte_s[ti:ti+1])[0] if free_tr.sum() > 20 else pi
            alpha = (pi - FSP * 0.7) / (FSP * 0.8)
            alpha = np.clip(alpha, 0, 1)
            test_conditional[ti] += (1-alpha) * p_b + alpha * p_f

test_conditional /= 13

cond_rmse = rmse(y, np.clip(oof_conditional, 0, 500))
cond_sb = sb_rmse(y, np.clip(oof_conditional, 0, 500), species)
print(f"Conditional ensemble: RMSE={cond_rmse:.4f}, SB={cond_sb:.4f}")

all_results.append({
    'feat_set': f'conditional_{best_feat}', 'model': 'cond_ensemble',
    'rmse': cond_rmse, 'sb_rmse': cond_sb,
    'oof': np.clip(oof_conditional, 0, 500),
    'X_train': best_Xtr, 'X_test': best_Xte,
})

# Strategy 2: Log-transform for high-moisture
print("\n--- Log-transform for high moisture ---")
for tr_idx, va_idx in GKF.split(X, y, species):
    pass  # just getting fold structure

oof_log = np.zeros(len(y))
for tr_idx, va_idx in GKF.split(X, y, species):
    Xtr_f = best_Xtr[tr_idx]
    Xva_f = best_Xtr[va_idx]
    sc = StandardScaler().fit(Xtr_f)
    Xtr_s = sc.transform(Xtr_f)
    Xva_s = sc.transform(Xva_f)

    y_tr_log = np.log1p(y[tr_idx])
    m = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=3, num_leaves=7,
        learning_rate=0.1, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=42)
    m.fit(Xtr_s, y_tr_log,
          eval_set=[(Xva_s, np.log1p(y[va_idx]))],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_log[va_idx] = np.expm1(m.predict(Xva_s))

log_rmse = rmse(y, np.clip(oof_log, 0, 500))
log_sb = sb_rmse(y, np.clip(oof_log, 0, 500), species)
print(f"  Log1p target: RMSE={log_rmse:.4f}, SB={log_sb:.4f}")

all_results.append({
    'feat_set': f'log1p_{best_feat}', 'model': 'lgbm_log',
    'rmse': log_rmse, 'sb_rmse': log_sb,
    'oof': np.clip(oof_log, 0, 500),
    'X_train': best_Xtr, 'X_test': best_Xte,
})

# Strategy 3: Sqrt-transform
oof_sqrt = np.zeros(len(y))
for tr_idx, va_idx in GKF.split(X, y, species):
    Xtr_f = best_Xtr[tr_idx]
    Xva_f = best_Xtr[va_idx]
    sc = StandardScaler().fit(Xtr_f)
    Xtr_s = sc.transform(Xtr_f)
    Xva_s = sc.transform(Xva_f)

    y_tr_sqrt = np.sqrt(y[tr_idx])
    m = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=3, num_leaves=7,
        learning_rate=0.1, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=42)
    m.fit(Xtr_s, y_tr_sqrt,
          eval_set=[(Xva_s, np.sqrt(y[va_idx]))],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_sqrt[va_idx] = m.predict(Xva_s) ** 2

sqrt_rmse = rmse(y, np.clip(oof_sqrt, 0, 500))
sqrt_sb = sb_rmse(y, np.clip(oof_sqrt, 0, 500), species)
print(f"  Sqrt target: RMSE={sqrt_rmse:.4f}, SB={sqrt_sb:.4f}")

all_results.append({
    'feat_set': f'sqrt_{best_feat}', 'model': 'lgbm_sqrt',
    'rmse': sqrt_rmse, 'sb_rmse': sqrt_sb,
    'oof': np.clip(oof_sqrt, 0, 500),
    'X_train': best_Xtr, 'X_test': best_Xte,
})


# =============================================================================
# PHASE 5: Stacking Meta-Learner
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 5: Stacking Meta-Learner")
print("=" * 60)

# Collect diverse OOF predictions for stacking
# Use top results from different feature sets and models
all_results.sort(key=lambda x: x['rmse'])

# Select diverse base models for stacking
stack_candidates = []
seen_combos = set()
type_counts = Counter()
MAX_PER_MODEL = 5
MAX_PER_FEAT = 10

for r in all_results:
    key = (r['feat_set'], r['model'])
    mtype = r['model'].split('_')[0]
    if key in seen_combos:
        continue
    if type_counts[mtype] >= MAX_PER_MODEL:
        continue
    if type_counts[r['feat_set']] >= MAX_PER_FEAT:
        continue
    seen_combos.add(key)
    type_counts[mtype] += 1
    type_counts[r['feat_set']] += 1
    stack_candidates.append(r)
    if len(stack_candidates) >= 30:
        break

print(f"Selected {len(stack_candidates)} base models for stacking:")
for mtype, cnt in Counter(r['model'].split('_')[0] for r in stack_candidates).most_common():
    print(f"  {mtype}: {cnt}")
for fname, cnt in Counter(r['feat_set'] for r in stack_candidates).most_common():
    print(f"  {fname}: {cnt}")

# Build L2 features: OOF predictions from L1 models
L2_train = np.column_stack([r['oof'] for r in stack_candidates])
print(f"\nL2 feature matrix: {L2_train.shape}")

# Also include domain features as extra context
L2_train_plus = np.hstack([L2_train, domain_feats_train])
print(f"L2 + domain features: {L2_train_plus.shape}")

# Stacking with RidgeCV (nested CV to avoid leakage)
print("\n--- Stacking approaches ---")

# A: Pure Ridge stacking
oof_ridge_stack = np.zeros(len(y))
for tr_idx, va_idx in GKF.split(X, y, species):
    meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
    meta.fit(L2_train[tr_idx], y[tr_idx])
    oof_ridge_stack[va_idx] = meta.predict(L2_train[va_idx])
oof_ridge_stack = np.clip(oof_ridge_stack, 0, 500)
print(f"  Ridge stack (OOF only): RMSE={rmse(y, oof_ridge_stack):.4f}, SB={sb_rmse(y, oof_ridge_stack, species):.4f}")

# B: Ridge stacking with domain features
oof_ridge_stack_plus = np.zeros(len(y))
for tr_idx, va_idx in GKF.split(X, y, species):
    sc = StandardScaler().fit(L2_train_plus[tr_idx])
    meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
    meta.fit(sc.transform(L2_train_plus[tr_idx]), y[tr_idx])
    oof_ridge_stack_plus[va_idx] = meta.predict(sc.transform(L2_train_plus[va_idx]))
oof_ridge_stack_plus = np.clip(oof_ridge_stack_plus, 0, 500)
print(f"  Ridge stack (OOF+domain): RMSE={rmse(y, oof_ridge_stack_plus):.4f}, SB={sb_rmse(y, oof_ridge_stack_plus, species):.4f}")

# C: LightGBM stacking
oof_lgbm_stack = np.zeros(len(y))
for tr_idx, va_idx in GKF.split(X, y, species):
    sc = StandardScaler().fit(L2_train_plus[tr_idx])
    meta = lgb.LGBMRegressor(
        n_estimators=500, max_depth=2, num_leaves=4,
        learning_rate=0.05, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.5,
        reg_alpha=5.0, reg_lambda=10.0,
        verbose=-1, n_jobs=-1, random_state=42)
    meta.fit(sc.transform(L2_train_plus[tr_idx]), y[tr_idx],
             eval_set=[(sc.transform(L2_train_plus[va_idx]), y[va_idx])],
             callbacks=[lgb.early_stopping(30, verbose=False)])
    oof_lgbm_stack[va_idx] = meta.predict(sc.transform(L2_train_plus[va_idx]))
oof_lgbm_stack = np.clip(oof_lgbm_stack, 0, 500)
print(f"  LGBM stack (OOF+domain): RMSE={rmse(y, oof_lgbm_stack):.4f}, SB={sb_rmse(y, oof_lgbm_stack, species):.4f}")

# D: NNLS on all L1 models
w_nnls, _ = nnls(L2_train, y)
if w_nnls.sum() > 0:
    w_nnls /= w_nnls.sum()
    oof_nnls = L2_train @ w_nnls
else:
    oof_nnls = L2_train.mean(1)
print(f"  NNLS blend: RMSE={rmse(y, oof_nnls):.4f}, SB={sb_rmse(y, oof_nnls, species):.4f}")

print("\n  NNLS weights (>1%):")
for i in np.argsort(-w_nnls)[:15]:
    if w_nnls[i] > 0.01:
        print(f"    {w_nnls[i]:.3f}  {stack_candidates[i]['feat_set']} + {stack_candidates[i]['model']} (RMSE={stack_candidates[i]['rmse']:.4f})")


# =============================================================================
# PHASE 6: Test predictions + submission
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 6: Test Predictions")
print("=" * 60)

# Generate test predictions for top L1 models
N_TEST = min(20, len(stack_candidates))
test_preds_l1 = []

for i, r in enumerate(stack_candidates[:N_TEST]):
    Xtr = r['X_train']
    Xte = r['X_test']
    model_name = r['model']

    # Use MODELS dict first, then reconstruct
    model = MODELS.get(model_name)
    if model is None:
        # Try to reconstruct
        try:
            if model_name == 'lgbm_ref':
                model = lgb.LGBMRegressor(
                    n_estimators=2000, max_depth=3, num_leaves=7,
                    learning_rate=0.1, min_child_samples=10,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1.0, reg_lambda=5.0,
                    verbose=-1, n_jobs=-1, random_state=42)
            elif 'lgbm' in model_name:
                parts = model_name.split('_')
                d = int(parts[1][1:])
                lr = float(parts[2][2:])
                s = int(parts[3][1:]) if len(parts) > 3 else 42
                model = lgb.LGBMRegressor(
                    n_estimators=2000, max_depth=d, num_leaves=min(2**d-1, 31),
                    learning_rate=lr, min_child_samples=10,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1.0, reg_lambda=5.0,
                    verbose=-1, n_jobs=-1, random_state=s)
            elif 'xgb' in model_name:
                parts = model_name.split('_')
                d = int(parts[1][1:])
                lr = float(parts[2][2:])
                model = xgb.XGBRegressor(
                    n_estimators=2000, max_depth=d, learning_rate=lr,
                    min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1.0, reg_lambda=5.0, verbosity=0, n_jobs=-1)
            elif 'ridge' in model_name:
                a = float(model_name.split('_')[1])
                model = Ridge(alpha=a)
            elif 'pls' in model_name:
                nc = int(model_name.split('_')[1])
                model = PLSRegression(n_components=nc, max_iter=1000)
            elif 'cat' in model_name and HAS_CB:
                parts = model_name.split('_')
                d = int(parts[1][1:])
                lr = float(parts[2][2:])
                model = CatBoostRegressor(
                    iterations=2000, depth=d, learning_rate=lr,
                    l2_leaf_reg=5.0, random_seed=42, verbose=0)
            else:
                print(f"  Skipping unknown model: {model_name}")
                continue
        except Exception as e:
            print(f"  Skipping {model_name}: {e}")
            continue

    print(f"  [{i+1}/{N_TEST}] {r['feat_set']} + {model_name} (CV={r['rmse']:.4f})")
    oof, test_pred, score = cv_with_test_feats(model, Xtr, Xte)
    test_preds_l1.append({
        'oof': oof, 'test_pred': test_pred,
        'feat_set': r['feat_set'], 'model': model_name, 'rmse': score,
    })

# Build L2 test predictions
L2_test = np.column_stack([r['test_pred'] for r in test_preds_l1])
L2_train_final = np.column_stack([r['oof'] for r in test_preds_l1])

# NNLS ensemble
w_final, _ = nnls(L2_train_final, y)
if w_final.sum() > 0:
    w_final /= w_final.sum()
nnls_test = np.clip(L2_test @ w_final, 0, 500)
nnls_oof = L2_train_final @ w_final
print(f"\nNNLS ensemble: RMSE={rmse(y, nnls_oof):.4f}")

# Ridge stack
test_stack_parts = []
oof_stack_final = np.zeros(len(y))
for tr_idx, va_idx in GKF.split(X, y, species):
    meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
    meta.fit(L2_train_final[tr_idx], y[tr_idx])
    oof_stack_final[va_idx] = meta.predict(L2_train_final[va_idx])
    test_stack_parts.append(meta.predict(L2_test))
oof_stack_final = np.clip(oof_stack_final, 0, 500)
test_stack_final = np.clip(np.mean(test_stack_parts, axis=0), 0, 500)
print(f"Ridge stack: RMSE={rmse(y, oof_stack_final):.4f}")

# Top N averages
for n in [3, 5, 10, 15, 20]:
    if n <= len(test_preds_l1):
        avg_oof = np.column_stack([r['oof'] for r in test_preds_l1[:n]]).mean(1)
        print(f"Top {n:2d} avg: RMSE={rmse(y, avg_oof):.4f}")

# =============================================================================
# Save
# =============================================================================
print("\n" + "=" * 60)
print("Saving Results (RULE COMPLIANT)")
print("=" * 60)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_dir = Path(__file__).resolve().parent.parent / "submissions"
sub_dir.mkdir(exist_ok=True)

submissions = {
    "best_single": np.clip(test_preds_l1[0]['test_pred'], 0, 500),
    "top5_avg": np.clip(np.column_stack([r['test_pred'] for r in test_preds_l1[:5]]).mean(1), 0, 500),
    "nnls": nnls_test,
    "ridge_stack": test_stack_final,
}

for name, preds in submissions.items():
    fname = f"submission_p30_{name}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(sub_dir / fname, header=False, index=False)
    print(f"  {fname}")

# Save results summary
results_dir = Path(__file__).resolve().parent.parent / "runs"
results_dir.mkdir(exist_ok=True)
summary = {
    "rule_compliant": True,
    "phase": 30,
    "description": "NIR domain feature engineering + stacking + conditional ensemble",
    "feature_sets_tested": list(FEATURE_SETS.keys()),
    "feature_set_ranking": [
        {"name": r[0], "rmse": r[1]['rmse'], "sb_rmse": r[1]['sb_rmse']}
        for r in ranked
    ],
    "top_models": [
        {"feat_set": r['feat_set'], "model": r['model'], "rmse": r['rmse']}
        for r in all_results[:30]
    ],
    "conditional_ensemble": {"rmse": float(cond_rmse), "sb_rmse": float(cond_sb)},
    "log1p_transform": {"rmse": float(log_rmse), "sb_rmse": float(log_sb)},
    "sqrt_transform": {"rmse": float(sqrt_rmse), "sb_rmse": float(sqrt_sb)},
    "ensemble": {
        "nnls": float(rmse(y, nnls_oof)),
        "ridge_stack": float(rmse(y, oof_stack_final)),
        "best_single": float(test_preds_l1[0]['rmse']),
    },
    "nnls_weights": w_final.tolist(),
}
with open(results_dir / "phase30_feature_eng_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("FINAL SUMMARY (RULE COMPLIANT — Phase 30)")
print("=" * 60)
print(f"Best feature set: {ranked[0][0]} (RMSE={ranked[0][1]['rmse']:.4f})")
print(f"Best single model: {all_results[0]['rmse']:.4f} ({all_results[0]['feat_set']} + {all_results[0]['model']})")
print(f"Conditional ensemble: {cond_rmse:.4f}")
print(f"NNLS blend: {rmse(y, nnls_oof):.4f}")
print(f"Ridge stack: {rmse(y, oof_stack_final):.4f}")
print(f"\nPrevious best (PL-free): 17.29")
