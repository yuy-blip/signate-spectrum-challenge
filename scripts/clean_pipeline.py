#!/usr/bin/env python3
"""Rule-compliant pipeline: train-only, no test leakage, OSS tools only.

Key design:
- GroupKFold by species (simulates unseen-species generalization)
- All preprocessing fitted on training fold ONLY
- Multiple model types for ensemble diversity
- OOF-based calibration and blending
- No pseudo-labeling, no test-data usage in any step

Target: Predict wood moisture content from NIR spectra (10000-4000 cm⁻¹)
Challenge: Train/test species are completely disjoint
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Ridge,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Optional imports
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False


# =============================================================================
# Data Loading
# =============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_data():
    """Load train/test data. Returns X, y, species, X_test, test_ids."""
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="cp932")

    # Identify columns
    id_col = "sample number"
    target_col = "含水率"
    species_col = "species number"

    # Feature columns = float-named columns (wavenumbers)
    feat_cols = [c for c in train.columns if _is_float(c)]

    X = train[feat_cols].values.astype(np.float64)
    y = train[target_col].values.astype(np.float64)
    species = train[species_col].values

    X_test = test[feat_cols].values.astype(np.float64)
    test_ids = test[id_col].values

    print(f"Train: {X.shape}, Test: {X_test.shape}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f}, mean={y.mean():.1f}")
    print(f"Train species: {sorted(np.unique(species))}")
    print(f"Test species: {sorted(test[species_col].unique())}")

    return X, y, species, X_test, test_ids


def _is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# =============================================================================
# Preprocessing Transformers (all train-only, sklearn-compatible)
# =============================================================================
class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: row-wise normalization."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return (X - m) / s


class SG(BaseEstimator, TransformerMixin):
    """Savitzky-Golay filter."""
    def __init__(self, window=11, poly=2, deriv=0):
        self.window = window
        self.poly = poly
        self.deriv = deriv
    def fit(self, X, y=None): return self
    def transform(self, X):
        return savgol_filter(X, self.window, self.poly, deriv=self.deriv, axis=1)


class Binning(BaseEstimator, TransformerMixin):
    """Average adjacent spectral channels."""
    def __init__(self, size=4):
        self.size = size
    def fit(self, X, y=None): return self
    def transform(self, X):
        n, p = X.shape
        bs = self.size
        n_full = p // bs
        parts = []
        if n_full > 0:
            parts.append(X[:, :n_full*bs].reshape(n, n_full, bs).mean(axis=2))
        if p % bs > 0:
            parts.append(X[:, n_full*bs:].mean(axis=1, keepdims=True))
        return np.hstack(parts)


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction (reference from training mean)."""
    def __init__(self, eps=1e-10):
        self.eps = eps
    def fit(self, X, y=None):
        self.ref_ = X.mean(axis=0)
        return self
    def transform(self, X):
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            c = np.polyfit(self.ref_, X[i], 1)
            b = c[0] if abs(c[0]) > self.eps else self.eps
            out[i] = (X[i] - c[1]) / b
        return out


class EMSC(BaseEstimator, TransformerMixin):
    """Extended MSC with polynomial baseline."""
    def __init__(self, poly_order=2, eps=1e-10):
        self.poly_order = poly_order
        self.eps = eps
    def fit(self, X, y=None):
        self.ref_ = X.mean(axis=0)
        n_wl = X.shape[1]
        w = np.linspace(-1, 1, n_wl)
        cols = [self.ref_, np.ones(n_wl)]
        for p in range(1, self.poly_order + 1):
            cols.append(w ** p)
        self._D = np.column_stack(cols)
        return self
    def transform(self, X):
        D = self._D
        coefs, _, _, _ = np.linalg.lstsq(D, X.T, rcond=None)
        a1 = coefs[0]
        a1_safe = np.where(np.abs(a1) < self.eps, self.eps, a1)
        baseline = D[:, 1:] @ coefs[1:]
        return ((X.T - baseline) / a1_safe).T


class AreaNorm(BaseEstimator, TransformerMixin):
    """Normalize each spectrum by its total area."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        area = np.abs(X).sum(axis=1, keepdims=True)
        area[area == 0] = 1.0
        return X / area


# =============================================================================
# Preprocessing Pipeline Combos
# =============================================================================
def make_pipeline(name):
    """Return a list of (step_name, transformer) tuples."""
    pipelines = {
        # EMSC + SG 1st deriv + bin4
        "emsc_sg1_bin4": [
            ("emsc", EMSC(poly_order=2)),
            ("sg", SG(window=7, poly=2, deriv=1)),
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
        # EMSC + SG 2nd deriv + bin4
        "emsc_sg2_bin4": [
            ("emsc", EMSC(poly_order=2)),
            ("sg", SG(window=11, poly=2, deriv=2)),
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
        # SNV + SG 1st deriv + bin4
        "snv_sg1_bin4": [
            ("snv", SNV()),
            ("sg", SG(window=7, poly=2, deriv=1)),
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
        # SNV + SG 2nd deriv + bin8
        "snv_sg2_bin8": [
            ("snv", SNV()),
            ("sg", SG(window=15, poly=2, deriv=2)),
            ("bin", Binning(size=8)),
            ("scaler", StandardScaler()),
        ],
        # MSC + SG 1st deriv + bin4
        "msc_sg1_bin4": [
            ("msc", MSC()),
            ("sg", SG(window=7, poly=2, deriv=1)),
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
        # EMSC only + bin4 (no derivative)
        "emsc_bin4": [
            ("emsc", EMSC(poly_order=2)),
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
        # EMSC + SG smooth + bin8
        "emsc_smooth_bin8": [
            ("emsc", EMSC(poly_order=2)),
            ("sg", SG(window=15, poly=3, deriv=0)),
            ("bin", Binning(size=8)),
            ("scaler", StandardScaler()),
        ],
        # Area norm + SG 1st + bin4
        "area_sg1_bin4": [
            ("area", AreaNorm()),
            ("sg", SG(window=7, poly=2, deriv=1)),
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
        # EMSC(poly=3) + SG 1st + bin4
        "emsc3_sg1_bin4": [
            ("emsc", EMSC(poly_order=3)),
            ("sg", SG(window=7, poly=2, deriv=1)),
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
        # Raw + bin4 + scaler (minimal preprocessing)
        "raw_bin4": [
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
        # EMSC + SG1 + bin16 (very compressed)
        "emsc_sg1_bin16": [
            ("emsc", EMSC(poly_order=2)),
            ("sg", SG(window=7, poly=2, deriv=1)),
            ("bin", Binning(size=16)),
            ("scaler", StandardScaler()),
        ],
        # SNV + EMSC + SG1 + bin4
        "snv_emsc_sg1_bin4": [
            ("snv", SNV()),
            ("emsc", EMSC(poly_order=2)),
            ("sg", SG(window=7, poly=2, deriv=1)),
            ("bin", Binning(size=4)),
            ("scaler", StandardScaler()),
        ],
    }
    return pipelines[name]


def apply_pipeline(steps, X_train, X_val=None, X_test=None):
    """Fit pipeline on X_train, transform all. Returns (X_train_t, X_val_t, X_test_t)."""
    X_tr = X_train.copy()
    X_v = X_val.copy() if X_val is not None else None
    X_te = X_test.copy() if X_test is not None else None

    for name, step in steps:
        step.fit(X_tr)
        X_tr = step.transform(X_tr)
        if X_v is not None:
            X_v = step.transform(X_v)
        if X_te is not None:
            X_te = step.transform(X_te)

    return X_tr, X_v, X_te


# =============================================================================
# Model Definitions
# =============================================================================
def get_models():
    """Return dict of {name: model} for all models to try."""
    models = {}

    # PLS variants
    for nc in [10, 15, 20, 28, 40]:
        models[f"pls_nc{nc}"] = PLSRegression(n_components=nc, max_iter=1000)

    # Ridge variants
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        models[f"ridge_a{alpha}"] = Ridge(alpha=alpha)

    # ElasticNet
    for alpha in [0.01, 0.1]:
        for l1 in [0.1, 0.5, 0.9]:
            models[f"enet_a{alpha}_l{l1}"] = ElasticNet(
                alpha=alpha, l1_ratio=l1, max_iter=5000
            )

    # Huber
    for eps in [1.1, 1.35, 2.0]:
        models[f"huber_e{eps}"] = HuberRegressor(epsilon=eps, max_iter=1000)

    # KNN
    for k in [5, 10, 20]:
        models[f"knn_{k}"] = KNeighborsRegressor(n_neighbors=k, weights="distance")

    # LightGBM
    if HAS_LGB:
        for depth in [3, 4, 5, 6]:
            for lr in [0.03, 0.05, 0.1]:
                leaves = min(2**depth - 1, 31)
                models[f"lgbm_d{depth}_lr{lr}"] = lgb.LGBMRegressor(
                    n_estimators=2000,
                    max_depth=depth,
                    num_leaves=leaves,
                    learning_rate=lr,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    verbose=-1,
                    n_jobs=-1,
                )

        # Extra: DART
        models["lgbm_dart"] = lgb.LGBMRegressor(
            n_estimators=1500,
            max_depth=5,
            num_leaves=20,
            learning_rate=0.05,
            boosting_type="dart",
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            verbose=-1,
            n_jobs=-1,
        )

    # XGBoost
    if HAS_XGB:
        for depth in [3, 4, 5]:
            for lr in [0.03, 0.05, 0.1]:
                models[f"xgb_d{depth}_lr{lr}"] = xgb.XGBRegressor(
                    n_estimators=2000,
                    max_depth=depth,
                    learning_rate=lr,
                    min_child_weight=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    verbosity=0,
                    n_jobs=-1,
                )

    # CatBoost
    if HAS_CB:
        for depth in [4, 5, 6]:
            models[f"cb_d{depth}"] = CatBoostRegressor(
                iterations=2000,
                depth=depth,
                learning_rate=0.05,
                l2_leaf_reg=5.0,
                subsample=0.8,
                verbose=0,
            )

    # Random Forest / ExtraTrees
    for n_est in [300, 500]:
        models[f"rf_{n_est}"] = RandomForestRegressor(
            n_estimators=n_est, max_depth=None, min_samples_leaf=5,
            max_features=0.5, n_jobs=-1, random_state=42,
        )
        models[f"et_{n_est}"] = ExtraTreesRegressor(
            n_estimators=n_est, max_depth=None, min_samples_leaf=5,
            max_features=0.5, n_jobs=-1, random_state=42,
        )

    # GradientBoosting (sklearn)
    models["gb_d3"] = GradientBoostingRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8,
    )

    return models


def get_target_transforms():
    """Return dict of (forward, inverse) functions."""
    return {
        "none": (lambda y: y, lambda y: y),
        "log1p": (np.log1p, np.expm1),
        "sqrt": (np.sqrt, np.square),
    }


# =============================================================================
# Cross-Validation (GroupKFold by species)
# =============================================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def cross_validate_single(
    X, y, species, pipe_name, model_name, model, target_tx="none",
    n_folds=13, seeds=None,
):
    """Run GroupKFold CV for a single (pipeline, model) combo.

    Returns: oof_preds, fold_scores, test_preds_list
    """
    if seeds is None:
        seeds = [42]

    fwd, inv = get_target_transforms()[target_tx]
    y_tx = fwd(y)

    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y))
    fold_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, species)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y_tx[tr_idx]

        # Build and apply preprocessing (train-only fit)
        steps = make_pipeline(pipe_name)
        X_tr_t, X_va_t, _ = apply_pipeline(steps, X_tr, X_va)

        # Fit model
        m = clone(model)
        if hasattr(m, 'early_stopping_rounds') or 'lgb' in type(m).__module__:
            try:
                m.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_tx[va_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            except (TypeError, Exception):
                m.fit(X_tr_t, y_tr)
        elif 'xgb' in type(m).__module__:
            try:
                m.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_tx[va_idx])],
                      verbose=False)
            except (TypeError, Exception):
                m.fit(X_tr_t, y_tr)
        elif 'catboost' in type(m).__module__:
            try:
                m.fit(X_tr_t, y_tr, eval_set=(X_va_t, y_tx[va_idx]),
                      early_stopping_rounds=50, verbose=0)
            except (TypeError, Exception):
                m.fit(X_tr_t, y_tr)
        else:
            m.fit(X_tr_t, y_tr)

        # Predict
        pred = m.predict(X_va_t).ravel()
        pred = inv(pred)
        pred = np.clip(pred, 0, 500)
        oof[va_idx] = pred
        fold_scores.append(rmse(y[va_idx], pred))

    overall = rmse(y, oof)
    return oof, fold_scores, overall


def cross_validate_with_test(
    X, y, species, X_test, pipe_name, model, target_tx="none", n_folds=13,
):
    """Run GroupKFold CV and also produce test predictions."""
    fwd, inv = get_target_transforms()[target_tx]
    y_tx = fwd(y)

    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y))
    test_preds = np.zeros(X_test.shape[0])

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, species)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y_tx[tr_idx]

        steps = make_pipeline(pipe_name)
        X_tr_t, X_va_t, X_te_t = apply_pipeline(steps, X_tr, X_va, X_test)

        m = clone(model)
        # Fit with early stopping where possible
        if hasattr(m, 'n_estimators') and HAS_LGB and isinstance(m, lgb.LGBMRegressor):
            m.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_tx[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        elif HAS_XGB and isinstance(m, xgb.XGBRegressor):
            m.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_tx[va_idx])], verbose=False)
        elif HAS_CB and isinstance(m, CatBoostRegressor):
            m.fit(X_tr_t, y_tr, eval_set=(X_va_t, y_tx[va_idx]),
                  early_stopping_rounds=50, verbose=0)
        else:
            m.fit(X_tr_t, y_tr)

        pred_val = inv(m.predict(X_va_t).ravel())
        pred_test = inv(m.predict(X_te_t).ravel())

        oof[va_idx] = np.clip(pred_val, 0, 500)
        test_preds += np.clip(pred_test, 0, 500)

    test_preds /= n_folds
    overall = rmse(y, oof)
    return oof, test_preds, overall


# =============================================================================
# Ensemble: OOF-based blending (train-only)
# =============================================================================
def optimize_blend_weights(oof_matrix, y, method="ridge"):
    """Find optimal blend weights using OOF predictions (train data only).

    oof_matrix: (n_samples, n_models) - each column is OOF from one model
    """
    if method == "ridge":
        from sklearn.linear_model import RidgeCV
        meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        meta.fit(oof_matrix, y)
        return meta
    elif method == "nnls":
        from scipy.optimize import nnls
        weights, _ = nnls(oof_matrix, y)
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1]
        return weights
    else:
        # Simple average
        return None


# =============================================================================
# Phase 1: Screening (find top combos)
# =============================================================================
def phase1_screen(X, y, species):
    """Screen pipeline x model combos. Returns sorted results."""
    pipe_names = [
        "emsc_sg1_bin4", "emsc_sg2_bin4", "snv_sg1_bin4", "snv_sg2_bin8",
        "msc_sg1_bin4", "emsc_bin4", "emsc_smooth_bin8", "area_sg1_bin4",
        "emsc3_sg1_bin4", "raw_bin4", "emsc_sg1_bin16", "snv_emsc_sg1_bin4",
    ]

    # Quick models for screening
    quick_models = {}
    quick_models["pls_20"] = PLSRegression(n_components=20, max_iter=1000)
    quick_models["pls_28"] = PLSRegression(n_components=28, max_iter=1000)
    quick_models["ridge_10"] = Ridge(alpha=10.0)

    if HAS_LGB:
        quick_models["lgbm_d4"] = lgb.LGBMRegressor(
            n_estimators=1000, max_depth=4, num_leaves=15,
            learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1,
        )
        quick_models["lgbm_d5"] = lgb.LGBMRegressor(
            n_estimators=1000, max_depth=5, num_leaves=20,
            learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1,
        )

    if HAS_XGB:
        quick_models["xgb_d4"] = xgb.XGBRegressor(
            n_estimators=1000, max_depth=4, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, verbosity=0, n_jobs=-1,
        )

    target_txs = ["none", "log1p", "sqrt"]

    results = []
    total = len(pipe_names) * len(quick_models) * len(target_txs)
    done = 0

    for pipe_name in pipe_names:
        for model_name, model in quick_models.items():
            for ttx in target_txs:
                done += 1
                try:
                    oof, fold_scores, overall = cross_validate_single(
                        X, y, species, pipe_name, model_name, model, ttx
                    )
                    results.append({
                        "pipe": pipe_name,
                        "model": model_name,
                        "target_tx": ttx,
                        "rmse": overall,
                        "fold_std": np.std(fold_scores),
                        "fold_scores": fold_scores,
                    })
                    print(f"[{done}/{total}] {pipe_name} + {model_name} + {ttx}: RMSE={overall:.4f} (std={np.std(fold_scores):.2f})")
                except Exception as e:
                    print(f"[{done}/{total}] FAILED: {pipe_name} + {model_name} + {ttx}: {e}")

    results.sort(key=lambda x: x["rmse"])
    return results


# =============================================================================
# Phase 2: Deep optimization of top combos
# =============================================================================
def phase2_optimize(X, y, species, top_results, n_top=20):
    """Take top combos and optimize with more models/params."""
    # Get unique top pipelines + target transforms
    seen = set()
    top_configs = []
    for r in top_results[:n_top]:
        key = (r["pipe"], r["target_tx"])
        if key not in seen:
            seen.add(key)
            top_configs.append(key)

    all_models = get_models()
    results = []
    total = len(top_configs) * len(all_models)
    done = 0

    for pipe_name, ttx in top_configs:
        for model_name, model in all_models.items():
            done += 1
            try:
                oof, fold_scores, overall = cross_validate_single(
                    X, y, species, pipe_name, model_name, model, ttx
                )
                results.append({
                    "pipe": pipe_name,
                    "model": model_name,
                    "target_tx": ttx,
                    "rmse": overall,
                    "fold_std": np.std(fold_scores),
                    "oof": oof,
                })
                if done % 20 == 0 or overall < 17.0:
                    print(f"[{done}/{total}] {pipe_name}+{model_name}+{ttx}: {overall:.4f}")
            except Exception as e:
                if done % 50 == 0:
                    print(f"[{done}/{total}] FAILED: {pipe_name}+{model_name}+{ttx}: {e}")

    results.sort(key=lambda x: x["rmse"])
    return results


# =============================================================================
# Phase 3: Build Ensemble
# =============================================================================
def phase3_ensemble(X, y, species, X_test, test_ids, all_results, n_ensemble=30):
    """Build final ensemble from top N models.

    Uses OOF-based stacking with GroupKFold (train only).
    """
    top = all_results[:n_ensemble]
    print(f"\n=== Building ensemble from top {len(top)} models ===")
    for i, r in enumerate(top[:10]):
        print(f"  {i+1}. {r['pipe']}+{r['model']}+{r['target_tx']}: {r['rmse']:.4f}")

    # Collect OOF predictions
    oof_matrix = np.column_stack([r["oof"] for r in top])

    # Simple average baseline
    avg_pred = oof_matrix.mean(axis=1)
    print(f"\nSimple average RMSE: {rmse(y, avg_pred):.4f}")

    # Weighted average (optimize on OOF using GroupKFold)
    # Use nested CV to avoid overfitting the weights
    gkf = GroupKFold(n_splits=13)
    oof_blend = np.zeros(len(y))
    blend_models = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, species)):
        from sklearn.linear_model import RidgeCV
        meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        meta.fit(oof_matrix[tr_idx], y[tr_idx])
        oof_blend[va_idx] = meta.predict(oof_matrix[va_idx])
        blend_models.append(meta)

    oof_blend = np.clip(oof_blend, 0, 500)
    print(f"Ridge stacking RMSE: {rmse(y, oof_blend):.4f}")

    # NNLS weights (non-negative)
    from scipy.optimize import nnls
    w_nnls, _ = nnls(oof_matrix, y)
    if w_nnls.sum() > 0:
        w_nnls /= w_nnls.sum()
    else:
        w_nnls = np.ones(len(top)) / len(top)
    oof_nnls = oof_matrix @ w_nnls
    print(f"NNLS blend RMSE: {rmse(y, oof_nnls):.4f}")

    # Now generate test predictions
    print("\nGenerating test predictions for top models...")
    test_pred_matrix = np.zeros((X_test.shape[0], len(top)))

    for i, r in enumerate(top):
        pipe_name = r["pipe"]
        model_name = r["model"]
        ttx = r["target_tx"]
        model = get_models().get(model_name)
        if model is None:
            # Rebuild from quick models
            model = _rebuild_model(model_name)

        _, test_pred, _ = cross_validate_with_test(
            X, y, species, X_test, pipe_name, model, ttx
        )
        test_pred_matrix[:, i] = test_pred
        if (i + 1) % 5 == 0:
            print(f"  Generated {i+1}/{len(top)} test predictions")

    # Final test predictions using different blend strategies
    test_avg = test_pred_matrix.mean(axis=1)
    test_nnls = test_pred_matrix @ w_nnls

    # Ridge stacking test (average of fold meta-models)
    test_ridge = np.zeros(X_test.shape[0])
    for meta in blend_models:
        test_ridge += meta.predict(test_pred_matrix)
    test_ridge /= len(blend_models)
    test_ridge = np.clip(test_ridge, 0, 500)

    return {
        "oof_avg": avg_pred,
        "oof_nnls": oof_nnls,
        "oof_ridge": oof_blend,
        "test_avg": test_avg,
        "test_nnls": test_nnls,
        "test_ridge": test_ridge,
        "nnls_weights": w_nnls,
        "top_models": [(r["pipe"], r["model"], r["target_tx"], r["rmse"]) for r in top],
        "test_pred_matrix": test_pred_matrix,
    }


def _rebuild_model(name):
    """Rebuild a model by name."""
    all_m = get_models()
    if name in all_m:
        return all_m[name]
    # Fallback for quick screening models
    if name == "pls_20":
        return PLSRegression(n_components=20, max_iter=1000)
    if name == "pls_28":
        return PLSRegression(n_components=28, max_iter=1000)
    if name == "ridge_10":
        return Ridge(alpha=10.0)
    if name == "lgbm_d4":
        return lgb.LGBMRegressor(
            n_estimators=1000, max_depth=4, num_leaves=15,
            learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1,
        )
    if name == "lgbm_d5":
        return lgb.LGBMRegressor(
            n_estimators=1000, max_depth=5, num_leaves=20,
            learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=-1,
        )
    if name == "xgb_d4":
        return xgb.XGBRegressor(
            n_estimators=1000, max_depth=4, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, verbosity=0, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


# =============================================================================
# Save submission
# =============================================================================
def save_submission(test_ids, preds, suffix=""):
    sub_dir = Path(__file__).resolve().parent.parent / "submissions"
    sub_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"submission_clean_{suffix}_{ts}.csv" if suffix else f"submission_clean_{ts}.csv"
    df = pd.DataFrame({"含水率": np.clip(preds, 0, 500)})
    df.to_csv(sub_dir / fname, header=False, index=False)
    print(f"Saved: {sub_dir / fname}")
    return sub_dir / fname


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("CLEAN PIPELINE: Rule-compliant, train-only, no test leakage")
    print("=" * 60)

    X, y, species, X_test, test_ids = load_data()

    # Phase 1: Screen
    print("\n" + "=" * 60)
    print("PHASE 1: Screening pipeline x model combos")
    print("=" * 60)
    screen_results = phase1_screen(X, y, species)

    # Save screening results
    out_dir = Path(__file__).resolve().parent.parent / "runs" / "clean_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "screening_results.json", "w") as f:
        json.dump(
            [{"pipe": r["pipe"], "model": r["model"], "target_tx": r["target_tx"],
              "rmse": r["rmse"], "fold_std": r["fold_std"]}
             for r in screen_results],
            f, indent=2,
        )

    print(f"\n=== Top 20 Screening Results ===")
    for i, r in enumerate(screen_results[:20]):
        print(f"  {i+1:2d}. {r['pipe']:20s} + {r['model']:12s} + {r['target_tx']:5s}: RMSE={r['rmse']:.4f}")

    # Phase 2: Deep optimization
    print("\n" + "=" * 60)
    print("PHASE 2: Deep optimization of top combos")
    print("=" * 60)
    opt_results = phase2_optimize(X, y, species, screen_results, n_top=8)

    print(f"\n=== Top 30 Optimized Results ===")
    for i, r in enumerate(opt_results[:30]):
        print(f"  {i+1:2d}. {r['pipe']:20s} + {r['model']:20s} + {r['target_tx']:5s}: RMSE={r['rmse']:.4f}")

    # Phase 3: Ensemble
    print("\n" + "=" * 60)
    print("PHASE 3: Building ensemble")
    print("=" * 60)
    ensemble = phase3_ensemble(X, y, species, X_test, test_ids, opt_results, n_ensemble=25)

    # Save submissions
    save_submission(test_ids, ensemble["test_avg"], "avg")
    save_submission(test_ids, ensemble["test_nnls"], "nnls")
    save_submission(test_ids, ensemble["test_ridge"], "ridge")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"OOF Simple Avg RMSE: {rmse(y, ensemble['oof_avg']):.4f}")
    print(f"OOF NNLS Blend RMSE: {rmse(y, ensemble['oof_nnls']):.4f}")
    print(f"OOF Ridge Stack RMSE: {rmse(y, ensemble['oof_ridge']):.4f}")

    # Save all results
    with open(out_dir / "final_results.json", "w") as f:
        json.dump({
            "oof_avg_rmse": rmse(y, ensemble["oof_avg"]),
            "oof_nnls_rmse": rmse(y, ensemble["oof_nnls"]),
            "oof_ridge_rmse": rmse(y, ensemble["oof_ridge"]),
            "nnls_weights": ensemble["nnls_weights"].tolist(),
            "top_models": ensemble["top_models"],
        }, f, indent=2)

    print("\nDone! All submissions saved to submissions/")


if __name__ == "__main__":
    main()
