#!/usr/bin/env python3
"""Phase 34: Model Diversity — break NNLS 14.55 wall with non-LGBM models.

Phase 33 showed: NNLS always picks the same 4 LGBM models regardless of
hyperparams/strategies. We need fundamentally different model types:
1. Ridge/ElasticNet — linear models, different inductive bias
2. SVR — kernel-based, captures non-linear patterns differently
3. MLP — neural network, smooth approximation
4. KNN — instance-based, good for local patterns
5. CatBoost/XGBoost — GBDT variants with different splitting/regularization

Combined with UWV augmentation and NNLS blending.
Rule compliant: NO pseudo-labeling, NO test data for training.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
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
    """CV for LightGBM (with early stopping)."""
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
    """CV for sklearn models (no early stopping). Optional PCA dim reduction."""
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


def per_sp(y_true, y_pred, sp):
    for s in sorted(np.unique(sp)):
        m = sp == s
        print(f"    Sp{s:2d}: RMSE={rmse(y_true[m], y_pred[m]):6.2f}, "
              f"bias={(y_pred[m]-y_true[m]).mean():+6.1f}, n={m.sum()}")


results = []

# =============================================================================
# GROUP A: Ridge regression (linear models — maximum diversity from GBDT)
# =============================================================================
print("=" * 60)
print("GROUP A: Ridge Regression")
print("=" * 60)

# Ridge on full spectrum + UWV
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    name = f"ridge_a{alpha}"
    oof, tp, sc, sb = cv_sklearn(Ridge(alpha=alpha), PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Ridge on water bands
for alpha in [1.0, 10.0, 100.0]:
    name = f"ridge_h2o_a{alpha}"
    oof, tp, sc, sb = cv_sklearn(Ridge(alpha=alpha), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Ridge + PCA dimensionality reduction + UWV
for pca_d in [20, 50, 100]:
    name = f"ridge_pca{pca_d}"
    oof, tp, sc, sb = cv_sklearn(Ridge(alpha=10.0), PP_STD, UWV_BEST, pca_dim=pca_d)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Ridge without UWV (for diversity)
for alpha in [1.0, 10.0, 100.0]:
    name = f"ridge_noUWV_a{alpha}"
    oof, tp, sc, sb = cv_sklearn(Ridge(alpha=alpha), PP_STD)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

print(f"\nBest Ridge: {min([r for r in results if 'ridge' in r['name']], key=lambda x: x['rmse'])['name']}")

# =============================================================================
# GROUP B: ElasticNet (sparse linear)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP B: ElasticNet")
print("=" * 60)

for alpha, l1 in [(0.1, 0.5), (1.0, 0.5), (0.1, 0.9), (1.0, 0.1)]:
    name = f"enet_a{alpha}_l1{l1}"
    oof, tp, sc, sb = cv_sklearn(
        ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000), PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# ElasticNet on water bands
for alpha in [0.1, 1.0]:
    name = f"enet_h2o_a{alpha}"
    oof, tp, sc, sb = cv_sklearn(
        ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP C: SVR (kernel-based)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP C: SVR")
print("=" * 60)

# SVR is slow on 388 features, use PCA first
for C, eps in [(10, 5), (50, 5), (100, 5), (10, 10), (50, 10)]:
    name = f"svr_C{C}_e{eps}_pca50"
    oof, tp, sc, sb = cv_sklearn(
        SVR(C=C, epsilon=eps, kernel='rbf'),
        PP_STD, UWV_BEST, pca_dim=50)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# SVR on water bands (smaller feature space)
for C in [10, 50, 100]:
    name = f"svr_h2o_C{C}"
    oof, tp, sc, sb = cv_sklearn(
        SVR(C=C, epsilon=5, kernel='rbf'), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Linear SVR
for C in [1, 10, 100]:
    name = f"svr_lin_C{C}"
    oof, tp, sc, sb = cv_sklearn(
        SVR(C=C, epsilon=5, kernel='linear'), PP_STD, UWV_BEST, pca_dim=50)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP D: MLP (Neural Network)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP D: MLP Neural Network")
print("=" * 60)

for hidden, alpha, s in [
    ((128, 64), 10.0, 42),
    ((256, 128), 10.0, 42),
    ((128, 64, 32), 10.0, 42),
    ((128, 64), 1.0, 42),
    ((128, 64), 100.0, 42),
    ((128, 64), 10.0, 123),
    ((256, 128), 10.0, 123),
    ((64, 32), 10.0, 42),
]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_{hstr}_a{alpha}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# MLP on water bands
for hidden, alpha, s in [((64, 32), 10.0, 42), ((128, 64), 10.0, 42)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_h2o_{hstr}_a{alpha}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# MLP with PCA
for pca_d, hidden, s in [(50, (64, 32), 42), (50, (128, 64), 42),
                          (20, (64, 32), 42)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_pca{pca_d}_{hstr}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=10.0,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_STD, UWV_BEST, pca_dim=pca_d)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP E: KNN
# =============================================================================
print("\n" + "=" * 60)
print("GROUP E: KNN")
print("=" * 60)

for k, pca_d in [(5, 20), (10, 20), (15, 20), (5, 50), (10, 50),
                  (20, 50), (5, 100)]:
    name = f"knn_k{k}_pca{pca_d}"
    oof, tp, sc, sb = cv_sklearn(
        KNeighborsRegressor(n_neighbors=k, weights='distance'),
        PP_STD, UWV_BEST, pca_dim=pca_d)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# KNN on water bands
for k in [5, 10, 15]:
    name = f"knn_h2o_k{k}"
    oof, tp, sc, sb = cv_sklearn(
        KNeighborsRegressor(n_neighbors=k, weights='distance'),
        PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP F: Recreate LGBM core (Phase 32 NNLS winners)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP F: LGBM Core (Phase 32 NNLS winners)")
print("=" * 60)

# h2o models (NNLS ~51%)
for s in [555, 42, 123, 0, 7]:
    name = f"lgbm_h2o_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv20 l12 (NNLS ~29%)
for s in [123, 42, 0]:
    name = f"lgbm_uwv20l12_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s, leaves=12), PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv10 (NNLS ~11%)
uwv10 = {"n_aug": 10, "extrap_factor": 1.5, "min_moisture": 170}
for s in [123, 42]:
    name = f"lgbm_uwv10_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_STD, uwv10)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv b2 (NNLS ~9%)
pp_b2 = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}
name = "lgbm_uwv_b2_s42"
oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, 42), pp_b2, UWV_BEST)
results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
print(f"  {name}: RMSE={sc:.4f}")

# Additional LGBM seeds
for s in [13, 77, 200, 314, 999]:
    name = f"lgbm_uwv20_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# =============================================================================
# TRY CATBOOST IF AVAILABLE
# =============================================================================
print("\n" + "=" * 60)
print("GROUP G: CatBoost")
print("=" * 60)

try:
    from catboost import CatBoostRegressor

    def cv_catboost(model, pp_kwargs, uwv_kwargs=None, water_only=False):
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
            m.fit(Xtr_t, y_tr, eval_set=(Xva_t, y[va_idx]),
                  early_stopping_rounds=50, verbose=0)
            oof[va_idx] = np.clip(m.predict(Xva_t), 0, 500)
            test_pred += np.clip(m.predict(Xte_t), 0, 500)
        test_pred /= 13
        return oof, test_pred, rmse(y, oof), sb_rmse(y, oof, species)

    for d, lr, s in [(4, 0.05, 42), (6, 0.03, 42), (4, 0.05, 123),
                      (4, 0.03, 42)]:
        name = f"cat_d{d}_lr{lr}_s{s}"
        m = CatBoostRegressor(
            depth=d, learning_rate=lr, iterations=2000,
            l2_leaf_reg=5.0, random_seed=s, verbose=0)
        oof, tp, sc, sb = cv_catboost(m, PP_STD, UWV_BEST)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

    # CatBoost on water bands
    for s in [42, 123]:
        name = f"cat_h2o_s{s}"
        m = CatBoostRegressor(
            depth=4, learning_rate=0.05, iterations=2000,
            l2_leaf_reg=5.0, random_seed=s, verbose=0)
        oof, tp, sc, sb = cv_catboost(m, PP_WATER, water_only=True)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

except ImportError:
    print("  CatBoost not available, skipping")

# =============================================================================
# TRY XGBOOST IF AVAILABLE
# =============================================================================
print("\n" + "=" * 60)
print("GROUP H: XGBoost")
print("=" * 60)

try:
    import xgboost as xgb

    def cv_xgb(model, pp_kwargs, uwv_kwargs=None, water_only=False):
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
                  verbose=False)
            oof[va_idx] = np.clip(m.predict(Xva_t), 0, 500)
            test_pred += np.clip(m.predict(Xte_t), 0, 500)
        test_pred /= 13
        return oof, test_pred, rmse(y, oof), sb_rmse(y, oof, species)

    for d, lr, s in [(3, 0.1, 42), (4, 0.05, 42), (3, 0.05, 42),
                      (3, 0.1, 123)]:
        name = f"xgb_d{d}_lr{lr}_s{s}"
        m = xgb.XGBRegressor(
            n_estimators=2000, max_depth=d, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0,
            early_stopping_rounds=50,
            random_state=s, verbosity=0)
        oof, tp, sc, sb = cv_xgb(m, PP_STD, UWV_BEST)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

    # XGB on water bands
    for s in [42, 123]:
        name = f"xgb_h2o_s{s}"
        m = xgb.XGBRegressor(
            n_estimators=2000, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0,
            early_stopping_rounds=50,
            random_state=s, verbosity=0)
        oof, tp, sc, sb = cv_xgb(m, PP_WATER, water_only=True)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

except ImportError:
    print("  XGBoost not available, skipping")

# =============================================================================
# MEGA ENSEMBLE
# =============================================================================
print("\n" + "=" * 60)
print("MEGA ENSEMBLE — ALL MODEL TYPES")
print("=" * 60)

results.sort(key=lambda x: x['rmse'])
N = len(results)

print(f"\nTotal models: {N}")
print("\nAll models ranked:")
for i, r in enumerate(results):
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

print(f"\n  Per-species (NNLS):")
per_sp(y, nnls_oof, species)

# NNLS top 30
t30 = min(30, N)
w30, _ = nnls(oof_matrix[:, :t30], y)
if w30.sum() > 0:
    w30 /= w30.sum()
nnls30_oof = oof_matrix[:, :t30] @ w30
nnls30_test = test_matrix[:, :t30] @ w30
nnls30_rmse = rmse(y, nnls30_oof)
print(f"  NNLS (top {t30}): RMSE={nnls30_rmse:.4f}")

# NNLS top 50
t50 = min(50, N)
if t50 > t30:
    w50, _ = nnls(oof_matrix[:, :t50], y)
    if w50.sum() > 0:
        w50 /= w50.sum()
    nnls50_oof = oof_matrix[:, :t50] @ w50
    nnls50_test = test_matrix[:, :t50] @ w50
    nnls50_rmse = rmse(y, nnls50_oof)
    print(f"  NNLS (top {t50}): RMSE={nnls50_rmse:.4f}")
else:
    nnls50_rmse = 999

# Pick best
best_candidates = [
    ("nnls_all", nnls_all_rmse, nnls_oof, nnls_test),
    ("nnls_top30", nnls30_rmse, nnls30_oof, nnls30_test),
]
if nnls50_rmse < 999:
    best_candidates.append(("nnls_top50", nnls50_rmse, nnls50_oof, nnls50_test))

best_label, best_rmse, best_oof, best_test_pred = min(best_candidates, key=lambda x: x[1])
print(f"\n  Best: {best_label} RMSE={best_rmse:.4f}")

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
    "top10_avg": np.clip(test_matrix[:, :10].mean(1), 0, 500),
    "nnls_all": np.clip(nnls_test, 0, 500),
    "nnls_best": np.clip(best_test_pred, 0, 500),
}

for sname, preds in submissions.items():
    fname = f"submission_p34_{sname}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(SUB_DIR / fname, header=False, index=False)
    print(f"  {fname}")

summary = {
    "rule_compliant": True,
    "phase": 34,
    "total_models": N,
    "models": [{"name": r["name"], "rmse": float(r["rmse"]), "sb": float(r["sb"])}
               for r in results],
    "ensemble": {
        "best_single": float(results[0]["rmse"]),
        "nnls_all": float(nnls_all_rmse),
        f"nnls_top{t30}": float(nnls30_rmse),
        "best_label": best_label,
        "best_rmse": float(best_rmse),
    },
    "nnls_weights": {results[i]["name"]: float(w_all[i])
                     for i in range(len(w_all)) if w_all[i] > 0.005},
}
with open(RUNS_DIR / f"phase34_results_{ts}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFINAL: Best single={results[0]['rmse']:.4f}, NNLS best={best_rmse:.4f}")
print(f"Phase 32/33 best: 14.55")
print(f"Improvement: {14.55 - best_rmse:+.4f}")
