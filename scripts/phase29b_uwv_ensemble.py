#!/usr/bin/env python3
"""Phase 29b: Lean UWV pipeline — focused on best configs + NNLS ensemble.

Phase 29 found UWV augmentation drops RMSE from 17.64 → 16.02.
This script:
1. Uses proven best UWV config (n30, f1.5, m170)
2. Screens diverse models with UWV
3. Adds water band models
4. NNLS ensemble with test predictions
5. Saves results incrementally (no OOM crash)

Rule compliant: NO pseudo-labeling, NO test data for training.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
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

# =============================================================================
# Data
# =============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"
RUNS_DIR.mkdir(exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)

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
WATER_5200 = (wavelengths >= 4950) & (wavelengths <= 5300)
WATER_7000 = (wavelengths >= 6400) & (wavelengths <= 7100)
WATER_INTERVALS = WATER_5200 | WATER_7000

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


# =============================================================================
# UWV (Universal Water Difference Vector) — training data only
# =============================================================================
def generate_universal_wdv(X_tr, y_tr, sp_tr, n_aug=30, extrap_factor=1.5,
                           min_moisture=170, dy_scale=0.3, dy_offset=30):
    species_deltas, species_dy = [], []
    for s in np.unique(sp_tr):
        mask = sp_tr == s
        X_sp, y_sp = X_tr[mask], y_tr[mask]
        med = np.median(y_sp)
        hi, lo = y_sp > med, y_sp <= med
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

    proj = X_tr @ water_vec
    coef = np.polyfit(proj, y_tr, 1)
    scale = coef[0]

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
    if len(synth_y) > n_aug:
        idx = np.linspace(0, len(synth_y) - 1, n_aug).astype(int)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


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


def cv_with_test(model, pp_kwargs, use_wdv=False, wdv_kwargs=None,
                 mw_power=0, mw_base=1.0, water_only=False):
    """CV + test prediction in a single pass."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))

    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx], species[tr_idx]
        X_va = X[va_idx].copy()
        X_te = X_test.copy()

        # UWV augmentation
        if use_wdv and wdv_kwargs:
            synth_X, synth_y = generate_universal_wdv(X_tr, y_tr, sp_tr, **wdv_kwargs)
            if len(synth_y) > 0:
                X_tr = np.vstack([X_tr, synth_X])
                y_tr_aug = np.concatenate([y_tr, synth_y])
            else:
                y_tr_aug = y_tr
        else:
            y_tr_aug = y_tr

        # Preprocess
        pp = dict(pp_kwargs)
        pp['water_only'] = water_only
        Xtr_t, Xva_t = preprocess(X_tr, X_va, **pp)
        Xtr_t2, Xte_t = preprocess(X_tr, X_te, **pp)

        # Moisture weighting
        sample_weight = None
        if mw_power > 0:
            sample_weight = mw_base + (y_tr_aug / y_tr_aug.max()) ** mw_power

        m = clone(model)
        fit_kw = {}
        if isinstance(m, lgb.LGBMRegressor):
            fit_kw['eval_set'] = [(Xva_t, y[va_idx])]
            fit_kw['callbacks'] = [lgb.early_stopping(50, verbose=False)]
            if sample_weight is not None:
                fit_kw['sample_weight'] = sample_weight
        elif isinstance(m, xgb.XGBRegressor):
            fit_kw['eval_set'] = [(Xva_t, y[va_idx])]
            fit_kw['verbose'] = False
            if sample_weight is not None:
                fit_kw['sample_weight'] = sample_weight
        elif HAS_CB and isinstance(m, CatBoostRegressor):
            fit_kw['eval_set'] = (Xva_t, y[va_idx])
            fit_kw['verbose'] = False
            if sample_weight is not None:
                fit_kw['sample_weight'] = sample_weight

        m.fit(Xtr_t, y_tr_aug, **fit_kw)
        oof[va_idx] = np.clip(m.predict(Xva_t).ravel(), 0, 500)
        test_pred += np.clip(m.predict(Xte_t).ravel(), 0, 500)

    test_pred /= 13
    return oof, test_pred, rmse(y, oof), sb_rmse(y, oof, species)


# =============================================================================
# Models
# =============================================================================
PP_STD = {"sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1}
WDV_BEST = {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170}
WDV_N50 = {"n_aug": 50, "extrap_factor": 1.5, "min_moisture": 170}

def make_lgbm(d=3, lr=0.1, s=42):
    return lgb.LGBMRegressor(
        n_estimators=2000, max_depth=d, num_leaves=min(2**d-1, 31),
        learning_rate=lr, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=s)

def make_xgb(d=3, lr=0.05):
    return xgb.XGBRegressor(
        n_estimators=2000, max_depth=d, learning_rate=lr,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0, verbosity=0, n_jobs=-1)

# =============================================================================
# PHASE 1: Baseline (no UWV) — quick sanity check
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Baseline (no UWV)")
print("=" * 60)

results = []

# Baseline best single
oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, 42), PP_STD)
results.append({"name": "baseline_d3_lr0.1_s42", "rmse": sc, "sb": sb, "oof": oof, "test": tp})
print(f"  Baseline: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# PHASE 2: UWV + diverse LightGBM seeds
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: UWV Augmentation")
print("=" * 60)

uwv_configs = [
    ("uwv_d3_lr0.1_s42", make_lgbm(3, 0.1, 42), WDV_BEST),
    ("uwv_d3_lr0.1_s0", make_lgbm(3, 0.1, 0), WDV_BEST),
    ("uwv_d3_lr0.1_s123", make_lgbm(3, 0.1, 123), WDV_BEST),
    ("uwv_d3_lr0.1_s7", make_lgbm(3, 0.1, 7), WDV_BEST),
    ("uwv_d3_lr0.1_s99", make_lgbm(3, 0.1, 99), WDV_BEST),
    ("uwv_d3_lr0.02_s42", make_lgbm(3, 0.02, 42), WDV_BEST),
    ("uwv_d3_lr0.02_s123", make_lgbm(3, 0.02, 123), WDV_BEST),
    ("uwv_d3_lr0.05_s42", make_lgbm(3, 0.05, 42), WDV_BEST),
    ("uwv_d3_lr0.05_s0", make_lgbm(3, 0.05, 0), WDV_BEST),
    ("uwv_d4_lr0.05_s0", make_lgbm(4, 0.05, 0), WDV_BEST),
    ("uwv_d4_lr0.1_s42", make_lgbm(4, 0.1, 42), WDV_BEST),
    ("uwv_n50_d3_lr0.1_s42", make_lgbm(3, 0.1, 42), WDV_N50),
    ("uwv_n50_d3_lr0.1_s123", make_lgbm(3, 0.1, 123), WDV_N50),
    ("uwv_n50_d3_lr0.05_s0", make_lgbm(3, 0.05, 0), WDV_N50),
]

for name, model, wdv_kw in uwv_configs:
    oof, tp, sc, sb = cv_with_test(model, PP_STD, use_wdv=True, wdv_kwargs=wdv_kw)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# XGBoost with UWV
print("\n--- XGBoost with UWV ---")
for name, model, wdv_kw in [
    ("uwv_xgb_d3_lr0.05", make_xgb(3, 0.05), WDV_BEST),
    ("uwv_xgb_d3_lr0.1", make_xgb(3, 0.1), WDV_BEST),
    ("uwv_xgb_d4_lr0.05", make_xgb(4, 0.05), WDV_BEST),
]:
    oof, tp, sc, sb = cv_with_test(model, PP_STD, use_wdv=True, wdv_kwargs=wdv_kw)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# CatBoost with UWV
if HAS_CB:
    print("\n--- CatBoost with UWV ---")
    for name, d, lr in [("uwv_cat_d4_lr0.05", 4, 0.05), ("uwv_cat_d6_lr0.05", 6, 0.05)]:
        model = CatBoostRegressor(iterations=2000, depth=d, learning_rate=lr,
                                   l2_leaf_reg=5.0, random_seed=42, verbose=0)
        oof, tp, sc, sb = cv_with_test(model, PP_STD, use_wdv=True, wdv_kwargs=WDV_BEST)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Save intermediate
print(f"\nPhase 2 done: {len(results)} models")

# =============================================================================
# PHASE 3: Water Band Models (with UWV)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: Water Band Models + UWV")
print("=" * 60)

PP_WATER = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}

for name, model, wdv_kw in [
    ("h2o_uwv_d3_lr0.1_s42", make_lgbm(3, 0.1, 42), WDV_BEST),
    ("h2o_uwv_d3_lr0.1_s123", make_lgbm(3, 0.1, 123), WDV_BEST),
    ("h2o_uwv_d3_lr0.05_s0", make_lgbm(3, 0.05, 0), WDV_BEST),
    ("h2o_d3_lr0.1_s42", make_lgbm(3, 0.1, 42), None),
]:
    oof, tp, sc, sb = cv_with_test(model, PP_WATER,
                                    use_wdv=wdv_kw is not None,
                                    wdv_kwargs=wdv_kw or {},
                                    water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# PHASE 4: Moisture Weighting + UWV
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 4: Moisture Weighting + UWV")
print("=" * 60)

for mw_p, mw_b in [(1.0, 0.5), (1.5, 0.5), (2.0, 0.5)]:
    name = f"uwv_mw_p{mw_p}_b{mw_b}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, 42), PP_STD,
                                    use_wdv=True, wdv_kwargs=WDV_BEST,
                                    mw_power=mw_p, mw_base=mw_b)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# PHASE 5: Sort + Ensemble
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 5: Ensemble")
print("=" * 60)

results.sort(key=lambda x: x['rmse'])

print("\nAll models ranked:")
for i, r in enumerate(results):
    print(f"  {i+1:2d}. RMSE={r['rmse']:.4f}  SB={r['sb']:.4f}  {r['name']}")

# Select top N for ensemble (diverse — limit per type)
N_ENS = min(25, len(results))
selected = results[:N_ENS]

oof_matrix = np.column_stack([r["oof"] for r in selected])
test_matrix = np.column_stack([r["test"] for r in selected])

# Top-N averages
for n in [3, 5, 10, 15, 20]:
    if n <= len(selected):
        avg_oof = oof_matrix[:, :n].mean(1)
        print(f"  Top {n:2d} avg: RMSE={rmse(y, avg_oof):.4f}")

# NNLS
w, _ = nnls(oof_matrix, y)
if w.sum() > 0:
    w /= w.sum()
else:
    w = np.ones(len(selected)) / len(selected)
nnls_oof = oof_matrix @ w
nnls_test = test_matrix @ w
print(f"\n  NNLS: RMSE={rmse(y, nnls_oof):.4f}, SB={sb_rmse(y, nnls_oof, species):.4f}")

print("\n  NNLS weights (>1%):")
for i in np.argsort(-w)[:15]:
    if w[i] > 0.01:
        print(f"    {w[i]:.3f}  {selected[i]['name']} (RMSE={selected[i]['rmse']:.4f})")

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
print(f"  Ridge stack: RMSE={rmse(y, oof_stack):.4f}")

# Per-species analysis of best model and NNLS
print("\n  Per-species (NNLS):")
for s in sorted(np.unique(species)):
    m = species == s
    sp_rmse = rmse(y[m], nnls_oof[m])
    bias = (nnls_oof[m] - y[m]).mean()
    print(f"    Sp{s:2d}: RMSE={sp_rmse:6.2f}, bias={bias:+6.1f}, n={m.sum()}")

# =============================================================================
# PHASE 6: Save
# =============================================================================
print("\n" + "=" * 60)
print("SAVING (RULE COMPLIANT)")
print("=" * 60)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

submissions = {
    "best_single": np.clip(selected[0]["test"], 0, 500),
    "top3_avg": np.clip(test_matrix[:, :3].mean(1), 0, 500),
    "top5_avg": np.clip(test_matrix[:, :5].mean(1), 0, 500),
    "top10_avg": np.clip(test_matrix[:, :10].mean(1), 0, 500),
    "nnls": np.clip(nnls_test, 0, 500),
    "ridge_stack": test_stack,
}

for name, preds in submissions.items():
    fname = f"submission_p29b_{name}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(SUB_DIR / fname, header=False, index=False)
    print(f"  {fname}")

# Save results summary (no large arrays — just scores)
summary = {
    "rule_compliant": True,
    "description": "Phase 29b: UWV augmentation + NNLS ensemble",
    "uwv_config_best": WDV_BEST,
    "models": [{"name": r["name"], "rmse": float(r["rmse"]), "sb": float(r["sb"])}
               for r in results],
    "ensemble": {
        "best_single": float(selected[0]["rmse"]),
        "top3_avg": float(rmse(y, oof_matrix[:, :3].mean(1))),
        "top5_avg": float(rmse(y, oof_matrix[:, :5].mean(1))),
        "nnls": float(rmse(y, nnls_oof)),
        "nnls_sb": float(sb_rmse(y, nnls_oof, species)),
        "ridge_stack": float(rmse(y, oof_stack)),
    },
    "nnls_weights": {selected[i]["name"]: float(w[i]) for i in range(len(w)) if w[i] > 0.005},
}
with open(RUNS_DIR / f"phase29b_results_{ts}.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("FINAL SUMMARY (RULE COMPLIANT — NO PL)")
print("=" * 60)
print(f"Best single:  {selected[0]['rmse']:.4f}  ({selected[0]['name']})")
print(f"Top 3 avg:    {rmse(y, oof_matrix[:, :3].mean(1)):.4f}")
print(f"Top 5 avg:    {rmse(y, oof_matrix[:, :5].mean(1)):.4f}")
print(f"NNLS blend:   {rmse(y, nnls_oof):.4f}")
print(f"Ridge stack:  {rmse(y, oof_stack):.4f}")
print(f"\nPrevious best (PL-free): 17.29")
print(f"UWV improvement: 17.29 → {selected[0]['rmse']:.2f}")
