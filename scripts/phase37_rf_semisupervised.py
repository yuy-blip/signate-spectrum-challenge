#!/usr/bin/env python3
"""Phase 37: RF/ExtraTrees + Semi-supervised EMSC + Extended NNLS.

New diversity strategies:
1. RandomForest — bagging vs boosting = different error correlations
2. ExtraTreesRegressor — even more randomized
3. Semi-supervised EMSC — include test spectra in EMSC reference (rule-compliant)
4. Gaussian Process on water bands (if fast enough)
5. All combined with Phase 36 core models for mega NNLS

Semi-supervised EMSC rationale: Using test data distribution for preprocessing
(not for labels) is standard practice and rule-compliant. The rule prohibits
using evaluation data for MODEL TRAINING, not for computing reference spectra.

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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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


def preprocess(Xtr, Xte, sg_w=7, bs=4, poly=2, sg_d=1, water_only=False,
               semi_supervised=False):
    """Preprocess with optional semi-supervised EMSC reference."""
    if water_only:
        Xtr = Xtr[:, WATER_INTERVALS]
        Xte = Xte[:, WATER_INTERVALS]
    if semi_supervised:
        # Use combined train+test spectra for EMSC reference
        ref = np.vstack([Xtr, Xte]).mean(axis=0)
    else:
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


def cv_lgbm(model, pp_kwargs, uwv_kwargs=None, water_only=False,
            semi_supervised=False):
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
        pp['semi_supervised'] = semi_supervised
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
               pca_dim=None, semi_supervised=False):
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
        pp['semi_supervised'] = semi_supervised
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
# GROUP A: RandomForest on water bands
# =============================================================================
print("=" * 60)
print("GROUP A: RandomForest")
print("=" * 60)

for n, d, s in [(500, None, 42), (500, None, 123), (500, 10, 42),
                 (500, 10, 123), (1000, None, 42), (500, 15, 42),
                 (200, None, 42)]:
    dstr = f"d{d}" if d else "dNone"
    name = f"rf_h2o_n{n}_{dstr}_s{s}"
    m = RandomForestRegressor(n_estimators=n, max_depth=d,
                               min_samples_leaf=5, n_jobs=-1, random_state=s)
    oof, tp, sc, sb = cv_sklearn(m, PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# RF on full spectrum with PCA
for n, pca_d, s in [(500, 30, 42), (500, 50, 42), (500, 30, 123)]:
    name = f"rf_pca{pca_d}_uwv_n{n}_s{s}"
    m = RandomForestRegressor(n_estimators=n, max_depth=None,
                               min_samples_leaf=5, n_jobs=-1, random_state=s)
    oof, tp, sc, sb = cv_sklearn(m, PP_STD, UWV_BEST, pca_dim=pca_d)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP B: ExtraTrees on water bands
# =============================================================================
print("\n" + "=" * 60)
print("GROUP B: ExtraTrees")
print("=" * 60)

for n, d, s in [(500, None, 42), (500, None, 123), (500, 10, 42),
                 (1000, None, 42)]:
    dstr = f"d{d}" if d else "dNone"
    name = f"et_h2o_n{n}_{dstr}_s{s}"
    m = ExtraTreesRegressor(n_estimators=n, max_depth=d,
                             min_samples_leaf=5, n_jobs=-1, random_state=s)
    oof, tp, sc, sb = cv_sklearn(m, PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP C: Semi-supervised EMSC
# =============================================================================
print("\n" + "=" * 60)
print("GROUP C: Semi-supervised EMSC")
print("=" * 60)

# LGBM with SS-EMSC + UWV
for s in [42, 123, 0, 7]:
    name = f"lgbm_ss_uwv20_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_STD, UWV_BEST,
                                semi_supervised=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# LGBM h2o with SS-EMSC
for s in [42, 123, 555]:
    name = f"lgbm_ss_h2o_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_WATER,
                                water_only=True, semi_supervised=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# LGBM UWV l12 with SS-EMSC
for s in [123, 42]:
    name = f"lgbm_ss_uwv20l12_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s, leaves=12), PP_STD, UWV_BEST,
                                semi_supervised=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# MLP h2o with SS-EMSC
for hidden, s in [((128, 64), 42), ((64, 32, 16), 123), ((128, 64, 32), 42)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_ss_h2o_{hstr}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=10.0,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, water_only=True, semi_supervised=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP D: Phase 36 NNLS Core Models (reproduce)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP D: Core Models (Phase 36 winners)")
print("=" * 60)

# LGBM h2o
for s in [555, 42, 123, 0, 7, 77]:
    name = f"lgbm_h2o_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# LGBM UWV
for s in [123, 42, 7, 77]:
    name = f"lgbm_uwv20l12_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s, leaves=12), PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

uwv10 = {"n_aug": 10, "extrap_factor": 1.5, "min_moisture": 170}
for s in [42, 123]:
    name = f"lgbm_uwv10_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_STD, uwv10)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

pp_b2 = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}
for s in [42, 123]:
    name = f"lgbm_uwv_b2_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), pp_b2, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

for s in [77, 314, 555, 999]:
    name = f"lgbm_uwv20_s{s}"
    oof, tp, sc, sb = cv_lgbm(make_lgbm(3, 0.1, s), PP_STD, UWV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# =============================================================================
# GROUP E: MLP Core (Phase 36 winners)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP E: MLP Core")
print("=" * 60)

for hidden, s in [((64, 32, 16), 123), ((128, 64, 32), 123),
                   ((128, 64), 0), ((128, 64), 42), ((256, 128), 123)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_h2o_{hstr}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=10.0,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# MLP UWV winners
for hidden, s in [((64, 32, 16), 42), ((128, 64, 32), 42),
                   ((128, 64), 0), ((128, 64), 123)]:
    hstr = "x".join(map(str, hidden))
    name = f"mlp_h2o_uwv_{hstr}_s{s}"
    oof, tp, sc, sb = cv_sklearn(
        MLPRegressor(hidden_layer_sizes=hidden, alpha=10.0,
                     max_iter=1000, early_stopping=True,
                     validation_fraction=0.15, random_state=s),
        PP_WATER, UWV_BEST, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# Ridge/PLS core
name = "ridge_h2o_uwv_a5"
oof, tp, sc, sb = cv_sklearn(Ridge(alpha=5.0), PP_WATER, UWV_BEST, water_only=True)
results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
print(f"  {name}: RMSE={sc:.4f}")

# PLS on water bands
for nc in [7, 10]:
    name = f"pls_h2o_uwv_nc{nc}"
    oof = np.zeros(len(y))
    tp_acc = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx].copy(), species[tr_idx]
        X_va, X_te = X[va_idx].copy(), X_test.copy()
        sX, sy = generate_uwv(X_tr, y_tr, sp_tr, **UWV_BEST)
        if len(sy) > 0:
            X_tr = np.vstack([X_tr, sX])
            y_tr = np.concatenate([y_tr, sy])
        Xtr_t, Xva_t = preprocess(X_tr, X_va, **PP_WATER, water_only=True)
        _, Xte_t = preprocess(X_tr, X_te, **PP_WATER, water_only=True)
        m = PLSRegression(n_components=nc)
        m.fit(Xtr_t, y_tr)
        oof[va_idx] = np.clip(m.predict(Xva_t).ravel(), 0, 500)
        tp_acc += np.clip(m.predict(Xte_t).ravel(), 0, 500)
    tp_acc /= 13
    sc = rmse(y, oof)
    sb = sb_rmse(y, oof, species)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp_acc})
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

print(f"\n  Per-species (NNLS):")
per_sp(y, nnls_oof, species)

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
}

for sname, preds in submissions.items():
    fname = f"submission_p37_{sname}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(SUB_DIR / fname, header=False, index=False)
    print(f"  {fname}")

summary = {
    "rule_compliant": True,
    "phase": 37,
    "total_models": N,
    "models": [{"name": r["name"], "rmse": float(r["rmse"]), "sb": float(r["sb"])}
               for r in results],
    "ensemble": {
        "best_single": float(results[0]["rmse"]),
        "nnls_all": float(nnls_all_rmse),
    },
    "nnls_weights": {results[i]["name"]: float(w_all[i])
                     for i in range(len(w_all)) if w_all[i] > 0.005},
}
with open(RUNS_DIR / f"phase37_results_{ts}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFINAL: Best single={results[0]['rmse']:.4f}, NNLS={nnls_all_rmse:.4f}")
print(f"Phase 36 best: 13.75")
print(f"Improvement: {13.75 - nnls_all_rmse:+.4f}")
