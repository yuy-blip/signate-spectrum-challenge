#!/usr/bin/env python3
"""Phase 36: PLS + Stacking + Extended Diversity.

Phase 35 hit 13.85 with MLP diversity. Next strategies:
1. PLS regression — classic NIR method, dimensionality reduction + regression
2. Stacking — use OOF predictions as features for 2nd-level model
3. More MLP variants — tanh activation, different learning rates
4. Gaussian Process on water bands
5. Combine all into mega NNLS

Rule compliant: NO pseudo-labeling, NO test data for training.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from scipy.optimize import nnls
import lightgbm as lgb
import warnings, json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

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

WATER_5200 = (wavelengths >= 4950) & (wavelengths <= 5300)
WATER_7000 = (wavelengths >= 6400) & (wavelengths <= 7100)
WATER_INTERVALS = WATER_5200 | WATER_7000


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


def generate_uwv(X_tr, y_tr, sp_tr, n_aug=20, extrap_factor=1.5,
                  min_moisture=170):
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
    pca = PCA(n_components=1)
    pca.fit(delta_mat)
    water_vec = pca.components_[0]
    if np.corrcoef(delta_mat @ water_vec, dy_arr)[0, 1] < 0:
        water_vec = -water_vec
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
            target_dy = extrap_factor * (y_sp[i] * 0.3 + 30)
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


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def sb_rmse(y_true, y_pred, sp):
    return np.mean([rmse(y_true[sp == s], y_pred[sp == s]) for s in np.unique(sp)])

GKF = GroupKFold(n_splits=13)

def make_lgbm(d=3, lr=0.1, s=42, leaves=None, mc=10):
    if leaves is None:
        leaves = min(2**d - 1, 31)
    return lgb.LGBMRegressor(
        n_estimators=2000, max_depth=d, num_leaves=leaves,
        learning_rate=lr, min_child_samples=mc,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=s)

PP_STD = {"sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1}
PP_WATER = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}
UWV_BEST = {"n_aug": 20, "extrap_factor": 1.5, "min_moisture": 170}


def cv_lgbm(model, pp_kwargs, uwv_kwargs=None, water_only=False):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx].copy(), species[tr_idx]
        X_va, X_te = X[va_idx].copy(), X_test.copy()
        if uwv_kwargs:
            sX, sy = generate_uwv(X_tr, y_tr, sp_tr, **uwv_kwargs)
            if len(sy) > 0:
                X_tr = np.vstack([X_tr, sX])
                y_tr = np.concatenate([y_tr, sy])
        pp = dict(pp_kwargs)
        pp['water_only'] = water_only
        Xtr_t, Xva_t = preprocess(X_tr, X_va, **pp)
        _, Xte_t = preprocess(X_tr, X_te, **pp)
        m = clone(model)
        m.fit(Xtr_t, y_tr,
              eval_set=[(Xva_t, y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_idx] = np.clip(m.predict(Xva_t), 0, 500)
        test_pred += np.clip(m.predict(Xte_t), 0, 500)
    test_pred /= 13
    return oof, test_pred, rmse(y, oof), sb_rmse(y, oof, species)


def cv_sklearn(model, pp_kwargs, uwv_kwargs=None, water_only=False,
               pca_dim=None):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx].copy(), species[tr_idx]
        X_va, X_te = X[va_idx].copy(), X_test.copy()
        if uwv_kwargs:
            sX, sy = generate_uwv(X_tr, y_tr, sp_tr, **uwv_kwargs)
            if len(sy) > 0:
                X_tr = np.vstack([X_tr, sX])
                y_tr = np.concatenate([y_tr, sy])
        pp = dict(pp_kwargs)
        pp['water_only'] = water_only
        Xtr_t, Xva_t = preprocess(X_tr, X_va, **pp)
        _, Xte_t = preprocess(X_tr, X_te, **pp)
        if pca_dim is not None and pca_dim < Xtr_t.shape[1]:
            pca = PCA(n_components=pca_dim)
            Xtr_t = pca.fit_transform(Xtr_t)
            Xva_t = pca.transform(Xva_t)
            Xte_t = pca.transform(Xte_t)
        m = clone(model)
        m.fit(Xtr_t, y_tr)
        oof[va_idx] = np.clip(m.predict(Xva_t), 0, 500)
        test_pred += np.clip(m.predict(Xte_t), 0, 500)
    test_pred /= 13
    return oof, test_pred, rmse(y, oof), sb_rmse(y, oof, species)


def cv_pls(n_comp, pp_kwargs, uwv_kwargs=None, water_only=False):
    """CV for PLS regression."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx].copy(), species[tr_idx]
        X_va, X_te = X[va_idx].copy(), X_test.copy()
        if uwv_kwargs:
            sX, sy = generate_uwv(X_tr, y_tr, sp_tr, **uwv_kwargs)
            if len(sy) > 0:
                X_tr = np.vstack([X_tr, sX])
                y_tr = np.concatenate([y_tr, sy])
        pp = dict(pp_kwargs)
        pp['water_only'] = water_only
        Xtr_t, Xva_t = preprocess(X_tr, X_va, **pp)
        _, Xte_t = preprocess(X_tr, X_te, **pp)
        m = PLSRegression(n_components=min(n_comp, Xtr_t.shape[1]))
        m.fit(Xtr_t, y_tr)
        oof[va_idx] = np.clip(m.predict(Xva_t).ravel(), 0, 500)
        test_pred += np.clip(m.predict(Xte_t).ravel(), 0, 500)
    test_pred /= 13
    return oof, test_pred, rmse(y, oof), sb_rmse(y, oof, species)


def per_sp(y_true, y_pred, sp):
    for s in sorted(np.unique(sp)):
        m = sp == s
        print(f"    Sp{s:2d}: RMSE={rmse(y_true[m], y_pred[m]):6.2f}, "
              f"bias={(y_pred[m]-y_true[m]).mean():+6.1f}, n={m.sum()}")


results = []

# =============================================================================
# GROUP A: PLS Regression (classic NIR)
# =============================================================================
print("=" * 60)
print("GROUP A: PLS Regression")
print("=" * 60)

# PLS on full spectrum + UWV
for nc in [2, 3, 5, 7, 10, 15, 20, 30, 50]:
    name = f"pls_nc{nc}_uwv"
    oof, tp, sc, sb = cv_pls(nc, PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# PLS on water bands
for nc in [2, 3, 5, 7, 10, 15]:
    name = f"pls_h2o_nc{nc}"
    oof, tp, sc, sb = cv_pls(nc, PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# PLS no UWV
for nc in [5, 10, 20]:
    name = f"pls_noUWV_nc{nc}"
    oof, tp, sc, sb = cv_pls(nc, PP_STD)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# PLS on water bands + UWV
for nc in [3, 5, 7, 10]:
    name = f"pls_h2o_uwv_nc{nc}"
    oof, tp, sc, sb = cv_pls(nc, PP_WATER, UWV_BEST, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP B: MLP with different activations/learning rates
# =============================================================================
print("\n" + "=" * 60)
print("GROUP B: MLP Variants (tanh, learning rate)")
print("=" * 60)

# tanh activation on water bands
for hidden, s in [((128, 64), 42), ((128, 64), 123), ((64, 32, 16), 42),
                   ((64, 32, 16), 123), ((256, 128), 42), ((128, 64, 32), 42)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_h2o_tanh_{hstr}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=10.0, activation='tanh',
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Different learning rates
for lr_init, hidden, s in [(0.01, (128, 64), 42), (0.001, (128, 64), 42),
                            (0.01, (64, 32, 16), 42), (0.001, (64, 32, 16), 42),
                            (0.01, (128, 64), 123)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_h2o_lr{lr_init}_{hstr}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=10.0,
                     learning_rate_init=lr_init,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# MLP with UWV on water bands (more seeds — Phase 35 showed these get weight)
for hidden, s in [((64, 32, 16), 42), ((64, 32, 16), 123), ((64, 32, 16), 0),
                   ((128, 64), 42), ((128, 64), 7), ((128, 64), 99),
                   ((128, 64, 32), 42), ((128, 64, 32), 123)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_h2o_uwv_{hstr}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=10.0,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, UWV_BEST, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP C: LGBM Core (reproduce Phase 35 NNLS winners)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP C: LGBM Core Models")
print("=" * 60)

# h2o models
for s in [555, 42, 123, 0, 7, 13, 77, 99]:
    name = f"lgbm_h2o_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv20 l12
for s in [123, 42, 0, 7, 77]:
    name = f"lgbm_uwv20l12_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s, leaves=12), PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv10
uwv10 = {"n_aug": 10, "extrap_factor": 1.5, "min_moisture": 170}
for s in [42, 123, 0, 7]:
    name = f"lgbm_uwv10_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_STD, uwv10)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv b2
pp_b2 = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}
for s in [42, 123]:
    name = f"lgbm_uwv_b2_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), pp_b2, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv20 standard
for s in [13, 77, 200, 314, 555, 999]:
    name = f"lgbm_uwv20_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# =============================================================================
# GROUP D: MLP h2o Phase 35 winners (reproduce for combined NNLS)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP D: MLP H2O Core (Phase 35 winners)")
print("=" * 60)

# Phase 35 NNLS winners
for hidden, s in [((64, 32, 16), 123), ((128, 64, 32), 123),
                   ((128, 64), 0), ((128, 64), 42), ((128, 64), 555),
                   ((256, 128), 123), ((128, 64, 32), 42)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_h2o_{hstr}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=10.0,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# Phase 35 UWV MLPs
for s in [0, 123]:
    name = f"mlp_h2o_uwv_128x64_core_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=(128, 64), alpha=10.0,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, UWV_BEST, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# Ridge h2o UWV
name = "ridge_h2o_uwv_a5"
oof, tp, sc, sb = cv_sklearn(Ridge(alpha=5.0), PP_WATER, UWV_BEST, water_only=True)
results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
print(f"  {name}: RMSE={sc:.4f}")

# =============================================================================
# MEGA ENSEMBLE
# =============================================================================
print("\n" + "=" * 60)
print("MEGA ENSEMBLE")
print("=" * 60)

results.sort(key=lambda x: x['rmse'])
N = len(results)

print(f"\nTotal models: {N}")
print("\nTop 40:")
for i, r in enumerate(results[:min(40, N)]):
    print(f"  {i+1:2d}. RMSE={r['rmse']:.4f}  SB={r['sb']:.4f}  {r['name']}")

oof_matrix = np.column_stack([r["oof"] for r in results])
test_matrix = np.column_stack([r["test"] for r in results])

for n in [3, 5, 10, 20, 30]:
    if n <= N:
        print(f"  Top {n:2d} avg: RMSE={rmse(y, oof_matrix[:, :n].mean(1)):.4f}")

# NNLS all
w_all, _ = nnls(oof_matrix, y)
if w_all.sum() > 0:
    w_all /= w_all.sum()
nnls_oof = oof_matrix @ w_all
nnls_test = test_matrix @ w_all
nnls_all_rmse = rmse(y, nnls_oof)
nnls_all_sb = sb_rmse(y, nnls_oof, species)
print(f"\n  NNLS (all {N}): RMSE={nnls_all_rmse:.4f}, SB={nnls_all_sb:.4f}")

print("\n  NNLS weights (>0.5%):")
for i in np.argsort(-w_all)[:25]:
    if w_all[i] > 0.005:
        print(f"    {w_all[i]:.3f}  {results[i]['name']} (RMSE={results[i]['rmse']:.4f})")

print(f"\n  Per-species (NNLS all):")
per_sp(y, nnls_oof, species)

# NNLS top 30
for tn in [30, 50]:
    tn = min(tn, N)
    wn, _ = nnls(oof_matrix[:, :tn], y)
    if wn.sum() > 0:
        wn /= wn.sum()
    nn_oof = oof_matrix[:, :tn] @ wn
    nn_rmse = rmse(y, nn_oof)
    print(f"  NNLS (top {tn}): RMSE={nn_rmse:.4f}")

# =============================================================================
# STACKING: Use OOF predictions as 2nd-level features
# =============================================================================
print("\n" + "=" * 60)
print("STACKING (2nd level)")
print("=" * 60)

# Select top 20 diverse models for stacking features
top20_idx = list(range(min(20, N)))
stack_oof = oof_matrix[:, top20_idx]
stack_test = test_matrix[:, top20_idx]

# 2nd level with Ridge (different alphas)
for alpha in [0.1, 1.0, 10.0, 100.0]:
    stack_pred_oof = np.zeros(len(y))
    stack_pred_test = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        m = Ridge(alpha=alpha)
        m.fit(stack_oof[tr_idx], y[tr_idx])
        stack_pred_oof[va_idx] = np.clip(m.predict(stack_oof[va_idx]), 0, 500)
        stack_pred_test += np.clip(m.predict(stack_test), 0, 500)
    stack_pred_test /= 13
    sc = rmse(y, stack_pred_oof)
    sb = sb_rmse(y, stack_pred_oof, species)
    name = f"stack_ridge_a{alpha}"
    results.append({"name": name, "rmse": sc, "sb": sb,
                    "oof": stack_pred_oof, "test": stack_pred_test})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# 2nd level with LGBM
for d, lr in [(2, 0.05), (3, 0.05), (2, 0.1)]:
    stack_pred_oof = np.zeros(len(y))
    stack_pred_test = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        m = lgb.LGBMRegressor(
            n_estimators=500, max_depth=d, num_leaves=min(2**d - 1, 8),
            learning_rate=lr, min_child_samples=15,
            reg_alpha=2.0, reg_lambda=10.0,
            verbose=-1, n_jobs=-1, random_state=42)
        m.fit(stack_oof[tr_idx], y[tr_idx],
              eval_set=[(stack_oof[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        stack_pred_oof[va_idx] = np.clip(m.predict(stack_oof[va_idx]), 0, 500)
        stack_pred_test += np.clip(m.predict(stack_test), 0, 500)
    stack_pred_test /= 13
    sc = rmse(y, stack_pred_oof)
    sb = sb_rmse(y, stack_pred_oof, species)
    name = f"stack_lgbm_d{d}_lr{lr}"
    results.append({"name": name, "rmse": sc, "sb": sb,
                    "oof": stack_pred_oof, "test": stack_pred_test})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# FINAL NNLS (all including stacking)
# =============================================================================
print("\n" + "=" * 60)
print("FINAL NNLS (with stacking)")
print("=" * 60)

results.sort(key=lambda x: x['rmse'])
N = len(results)
oof_matrix = np.column_stack([r["oof"] for r in results])
test_matrix = np.column_stack([r["test"] for r in results])

w_final, _ = nnls(oof_matrix, y)
if w_final.sum() > 0:
    w_final /= w_final.sum()
final_oof = oof_matrix @ w_final
final_test = test_matrix @ w_final
final_rmse = rmse(y, final_oof)
final_sb = sb_rmse(y, final_oof, species)
print(f"  NNLS final ({N}): RMSE={final_rmse:.4f}, SB={final_sb:.4f}")

print("\n  NNLS final weights (>0.5%):")
for i in np.argsort(-w_final)[:25]:
    if w_final[i] > 0.005:
        print(f"    {w_final[i]:.3f}  {results[i]['name']} (RMSE={results[i]['rmse']:.4f})")

print(f"\n  Per-species (final):")
per_sp(y, final_oof, species)

best_rmse = min(final_rmse, nnls_all_rmse)
best_test_out = final_test if final_rmse <= nnls_all_rmse else nnls_test

# =============================================================================
# SAVE
# =============================================================================
print("\n" + "=" * 60)
print("SAVING")
print("=" * 60)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

submissions = {
    "best_single": np.clip(results[0]["test"], 0, 500),
    "top5_avg": np.clip(test_matrix[:, :5].mean(1), 0, 500),
    "nnls_all": np.clip(nnls_test, 0, 500),
    "nnls_final": np.clip(best_test_out, 0, 500),
}

for sname, preds in submissions.items():
    fname = f"submission_p36_{sname}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(SUB_DIR / fname, header=False, index=False)
    print(f"  {fname}")

summary = {
    "rule_compliant": True,
    "phase": 36,
    "total_models": N,
    "models": [{"name": r["name"], "rmse": float(r["rmse"]), "sb": float(r["sb"])}
               for r in results],
    "ensemble": {
        "best_single": float(results[0]["rmse"]),
        "nnls_all": float(nnls_all_rmse),
        "nnls_final": float(final_rmse),
    },
    "nnls_weights": {results[i]["name"]: float(w_final[i])
                     for i in range(len(w_final)) if w_final[i] > 0.005},
}
with open(RUNS_DIR / f"phase36_results_{ts}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFINAL: Best single={results[0]['rmse']:.4f}, NNLS best={best_rmse:.4f}")
print(f"Phase 35 best: 13.85")
print(f"Improvement: {13.85 - best_rmse:+.4f}")
