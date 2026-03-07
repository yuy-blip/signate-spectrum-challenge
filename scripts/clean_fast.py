#!/usr/bin/env python3
"""Fast clean pipeline: CV-only first, test pred for top models only.

Rule-compliant: train-only, no test leakage, GroupKFold by species.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, HuberRegressor, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
from scipy.optimize import nnls
import lightgbm as lgb
import xgboost as xgb
import warnings, json, sys
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

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

feat_cols_test = [c for c in test.columns
                  if c not in ['sample number', 'species number', '樹種']]
X_test = test[feat_cols_test].values.astype(np.float64)
test_ids = test['sample number'].values

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Target: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}")

# =============================================================================
# Preprocessing (all fitted on training fold only)
# =============================================================================
def emsc_transform(X_ref, X_in, poly=2):
    """EMSC: Extended Multiplicative Scatter Correction."""
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


def preprocess(Xtr, Xte, sg_w=9, bs=4, poly=2, sg_d=1):
    """EMSC + SG derivative + binning + scaling. Fit on Xtr only."""
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
# CV helper
# =============================================================================
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


GKF = GroupKFold(n_splits=13)


def cv_score(model, pp_kwargs, label=""):
    """Run GroupKFold CV, return (oof, score, fold_scores)."""
    oof = np.zeros(len(y))
    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(GKF.split(X, y, species)):
        Xtr_t, Xva_t = preprocess(X[tr_idx], X[va_idx], **pp_kwargs)
        m = clone(model)

        if isinstance(m, lgb.LGBMRegressor):
            m.fit(Xtr_t, y[tr_idx],
                  eval_set=[(Xva_t, y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        elif isinstance(m, xgb.XGBRegressor):
            m.fit(Xtr_t, y[tr_idx],
                  eval_set=[(Xva_t, y[va_idx])], verbose=False)
        else:
            m.fit(Xtr_t, y[tr_idx])

        pred = np.clip(m.predict(Xva_t).ravel(), 0, 500)
        oof[va_idx] = pred
        fold_scores.append(rmse(y[va_idx], pred))

    score = rmse(y, oof)
    if label:
        print(f"  {label}: {score:.4f}  (folds: {', '.join(f'{s:.1f}' for s in fold_scores)})")
    return oof, score, fold_scores


def cv_with_test(model, pp_kwargs):
    """CV + test prediction."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        Xtr_t, Xva_t = preprocess(X[tr_idx], X[va_idx], **pp_kwargs)
        _, Xte_t = preprocess(X[tr_idx], X_test, **pp_kwargs)
        m = clone(model)
        if isinstance(m, lgb.LGBMRegressor):
            m.fit(Xtr_t, y[tr_idx], eval_set=[(Xva_t, y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        elif isinstance(m, xgb.XGBRegressor):
            m.fit(Xtr_t, y[tr_idx], eval_set=[(Xva_t, y[va_idx])], verbose=False)
        else:
            m.fit(Xtr_t, y[tr_idx])
        oof[va_idx] = np.clip(m.predict(Xva_t).ravel(), 0, 500)
        test_pred += np.clip(m.predict(Xte_t).ravel(), 0, 500)
    test_pred /= 13
    return oof, test_pred, rmse(y, oof)


# =============================================================================
# Phase 1: Fast CV screening (no test pred)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Fast CV Screening")
print("=" * 60)

PP_CONFIGS = [
    {"sg_w": 9, "bs": 4, "poly": 2, "sg_d": 1},
    {"sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1},
    {"sg_w": 7, "bs": 8, "poly": 2, "sg_d": 1},
    {"sg_w": 7, "bs": 16, "poly": 2, "sg_d": 1},
    {"sg_w": 11, "bs": 4, "poly": 2, "sg_d": 1},
    {"sg_w": 7, "bs": 4, "poly": 3, "sg_d": 1},
    {"sg_w": 9, "bs": 8, "poly": 2, "sg_d": 1},
]

MODELS = {}
# LightGBM - main workhorse
for d in [3, 4, 5]:
    for lr in [0.02, 0.05, 0.1]:
        leaves = min(2**d - 1, 31)
        for seed in [42, 0, 123]:
            MODELS[f"lgbm_d{d}_lr{lr}_s{seed}"] = lgb.LGBMRegressor(
                n_estimators=2000, max_depth=d, num_leaves=leaves,
                learning_rate=lr, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=5.0,
                verbose=-1, n_jobs=-1, random_state=seed,
            )

# XGBoost
for d in [3, 4]:
    for lr in [0.03, 0.05]:
        MODELS[f"xgb_d{d}_lr{lr}"] = xgb.XGBRegressor(
            n_estimators=2000, max_depth=d, learning_rate=lr,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, verbosity=0, n_jobs=-1,
        )

# PLS
for nc in [15, 20, 28]:
    MODELS[f"pls_{nc}"] = PLSRegression(n_components=nc, max_iter=1000)

# Ridge
for a in [1.0, 10.0, 100.0]:
    MODELS[f"ridge_{a}"] = Ridge(alpha=a)

# KNN
for k in [5, 10, 20]:
    MODELS[f"knn_{k}"] = KNeighborsRegressor(n_neighbors=k, weights="distance")

# RF / ET (fast)
MODELS["rf_300"] = RandomForestRegressor(
    n_estimators=300, max_features=0.5, min_samples_leaf=5, n_jobs=-1, random_state=42)
MODELS["et_300"] = ExtraTreesRegressor(
    n_estimators=300, max_features=0.5, min_samples_leaf=5, n_jobs=-1, random_state=42)

# Huber
MODELS["huber"] = HuberRegressor(epsilon=1.35, max_iter=1000)

all_results = []
total = len(PP_CONFIGS) * len(MODELS)
done = 0

for pp_kwargs in PP_CONFIGS:
    pp_name = f"emsc{pp_kwargs['poly']}_sg{pp_kwargs['sg_d']}_w{pp_kwargs['sg_w']}_b{pp_kwargs['bs']}"
    print(f"\n--- {pp_name} ---")

    for mname, model in MODELS.items():
        done += 1
        try:
            oof, score, folds = cv_score(model, pp_kwargs)
            all_results.append({
                "pipe": pp_name, "model": mname, "rmse": score,
                "pp_kwargs": pp_kwargs, "oof": oof,
            })
            if done % 20 == 0 or score < 18.5:
                print(f"  [{done}/{total}] {mname}: {score:.4f}")
        except Exception as e:
            print(f"  [{done}/{total}] FAILED {mname}: {e}")

all_results.sort(key=lambda x: x["rmse"])
print("\n" + "=" * 60)
print("TOP 40 CV RESULTS")
print("=" * 60)
for i, r in enumerate(all_results[:40]):
    print(f"{i+1:3d}. {r['rmse']:8.4f}  {r['pipe']:22s}  {r['model']}")

# =============================================================================
# Phase 2: Generate test predictions for top N
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: Test Predictions for Top Models")
print("=" * 60)

N_TOP = 30
top_results = []

for i, r in enumerate(all_results[:N_TOP]):
    pp_kwargs = r["pp_kwargs"]
    mname = r["model"]
    model = MODELS[mname]

    print(f"  [{i+1}/{N_TOP}] Generating test pred: {r['pipe']} + {mname} (CV={r['rmse']:.4f})")
    oof, test_pred, score = cv_with_test(model, pp_kwargs)
    top_results.append({
        "pipe": r["pipe"], "model": mname, "rmse": score,
        "oof": oof, "test_pred": test_pred,
    })

# =============================================================================
# Phase 3: Ensemble
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: Ensemble Building")
print("=" * 60)

oof_matrix = np.column_stack([r["oof"] for r in top_results])
test_matrix = np.column_stack([r["test_pred"] for r in top_results])

# Simple average
for n in [5, 10, 15, 20, 25, 30]:
    if n <= len(top_results):
        avg_oof = oof_matrix[:, :n].mean(1)
        print(f"Top {n:2d} avg: {rmse(y, avg_oof):.4f}")

# NNLS
w, _ = nnls(oof_matrix, y)
if w.sum() > 0:
    w /= w.sum()
else:
    w = np.ones(N_TOP) / N_TOP
nnls_oof = oof_matrix @ w
nnls_test = test_matrix @ w
print(f"NNLS blend:  {rmse(y, nnls_oof):.4f}")

# Print NNLS weights for top contributors
print("\nNNLS weight distribution:")
for i in np.argsort(-w)[:10]:
    if w[i] > 0.01:
        print(f"  {w[i]:.3f}  {top_results[i]['pipe']} + {top_results[i]['model']}")

# Ridge stacking (nested GroupKFold)
oof_stack = np.zeros(len(y))
test_stack_parts = []
for tr_idx, va_idx in GKF.split(X, y, species):
    meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    meta.fit(oof_matrix[tr_idx], y[tr_idx])
    oof_stack[va_idx] = meta.predict(oof_matrix[va_idx])
    test_stack_parts.append(meta.predict(test_matrix))
oof_stack = np.clip(oof_stack, 0, 500)
test_stack = np.clip(np.mean(test_stack_parts, axis=0), 0, 500)
print(f"Ridge stack: {rmse(y, oof_stack):.4f}")

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
    "top30_avg": np.clip(test_matrix[:, :30].mean(1), 0, 500) if len(top_results) >= 30 else None,
    "nnls": np.clip(nnls_test, 0, 500),
    "ridge_stack": test_stack,
}

for name, preds in submissions.items():
    if preds is None:
        continue
    fname = f"submission_clean_{name}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(sub_dir / fname, header=False, index=False)
    print(f"  {fname}")

# Save results summary
results_dir = Path(__file__).resolve().parent.parent / "runs"
results_dir.mkdir(exist_ok=True)
with open(results_dir / "clean_final_results.json", "w") as f:
    json.dump({
        "top_models": [
            {"pipe": r["pipe"], "model": r["model"], "rmse": r["rmse"]}
            for r in top_results
        ],
        "ensemble_scores": {
            "best_single": float(top_results[0]["rmse"]),
            "top5_avg": float(rmse(y, oof_matrix[:, :5].mean(1))),
            "top10_avg": float(rmse(y, oof_matrix[:, :10].mean(1))),
            "nnls": float(rmse(y, nnls_oof)),
            "ridge_stack": float(rmse(y, oof_stack)),
        },
        "nnls_weights": w.tolist(),
    }, f, indent=2)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Best single model:  {top_results[0]['rmse']:.4f}  ({top_results[0]['pipe']} + {top_results[0]['model']})")
print(f"Best ensemble:      {min(rmse(y, oof_matrix[:,:n].mean(1)) for n in range(2, min(31, len(top_results)+1))):.4f}")
print(f"NNLS blend:         {rmse(y, nnls_oof):.4f}")
print(f"Ridge stacking:     {rmse(y, oof_stack):.4f}")
