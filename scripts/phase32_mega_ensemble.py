#!/usr/bin/env python3
"""Phase 32: Mega Ensemble — maximize NNLS diversity.

Phase 31 showed: NNLS consistently picks ~50% UWV + ~50% water band.
This phase generates a massive pool of diverse candidates:
1. Many UWV parameter combos with many seeds
2. Many water band model seeds
3. Different UWV PCA strategies (1D, 2D, species-weighted)
4. PP variants × UWV × seeds
5. Let NNLS find the optimal combination from 100+ models

Rule compliant: NO pseudo-labeling, NO test data for training.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
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
                  min_moisture=170, dy_scale=0.3, dy_offset=30, n_pca=1):
    """UWV with configurable PCA dimensions."""
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

    if len(species_deltas) >= max(3, n_pca + 1):
        pca = PCA(n_components=min(n_pca, len(species_deltas)))
        pca.fit(delta_mat)
        # Use first component as water vector
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


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def sb_rmse(y_true, y_pred, sp):
    return np.mean([rmse(y_true[sp == s], y_pred[sp == s]) for s in np.unique(sp)])

GKF = GroupKFold(n_splits=13)


def cv_with_test(model, pp_kwargs, uwv_kwargs=None, water_only=False):
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

results = []

# =============================================================================
# GROUP A: UWV full-spectrum models (many seeds × configs)
# =============================================================================
print("=" * 60)
print("GROUP A: UWV Full-Spectrum Models")
print("=" * 60)

# Best UWV configs from Phase 31
uwv_configs = [
    {"n_aug": 10, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 20, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 20, "extrap_factor": 1.5, "min_moisture": 130},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 130},
    {"n_aug": 20, "extrap_factor": 1.2, "min_moisture": 170},
    {"n_aug": 30, "extrap_factor": 1.2, "min_moisture": 150},
]

seeds = [0, 7, 13, 42, 77, 99, 123, 200, 314, 555, 999]
model_configs = [
    ("d3l7", make_lgbm(3, 0.1, 42)),
    ("d3l12", make_lgbm(3, 0.1, 42, leaves=12)),
    ("d3l7_mc20", make_lgbm(3, 0.1, 42, mc=20)),
]

total_a = len(uwv_configs) * len(seeds)
done = 0

# Core: best config × many seeds (d3l7, the standard)
uwv_best = {"n_aug": 20, "extrap_factor": 1.5, "min_moisture": 170}
for s in seeds:
    done += 1
    name = f"uwv20_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_STD, uwv_best)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  [{done}] {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Second best: n10
uwv_n10 = {"n_aug": 10, "extrap_factor": 1.5, "min_moisture": 170}
for s in [0, 42, 123, 7, 314]:
    done += 1
    name = f"uwv10_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_STD, uwv_n10)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  [{done}] {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# n30 with m130
uwv_m130 = {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 130}
for s in [123, 42, 0, 7]:
    done += 1
    name = f"uwv30m130_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_STD, uwv_m130)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  [{done}] {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Leaves=12 (Phase 31 finding)
for s in [123, 42, 0, 7, 99]:
    done += 1
    name = f"uwv20_l12_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s, leaves=12), PP_STD, uwv_best)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  [{done}] {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# mc=20
for s in [123, 42, 0]:
    done += 1
    name = f"uwv20_mc20_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s, mc=20), PP_STD, uwv_best)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  [{done}] {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Lower lr with UWV
for s in [123, 42]:
    done += 1
    name = f"uwv20_lr02_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.02, s), PP_STD, uwv_best)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  [{done}] {name}: RMSE={sc:.4f}, SB={sb:.4f}")

print(f"\nGroup A: {len(results)} models")

# =============================================================================
# GROUP B: Water band models (many seeds)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP B: Water Band Models")
print("=" * 60)

h2o_seeds = [0, 7, 13, 42, 77, 99, 123, 200, 314, 555, 999]
for s in h2o_seeds:
    name = f"h2o_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# H2O with different lr
for s in [42, 123]:
    name = f"h2o_lr05_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.05, s), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# H2O with UWV
for s in [42, 123, 0]:
    name = f"h2o_uwv_s{s}"
    oof, tp, sc, sb = cv_with_test(
        make_lgbm(3, 0.1, s), PP_WATER,
        uwv_kwargs=uwv_best, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

print(f"\nGroup B: {len(results) - done - 3} water models")

# =============================================================================
# GROUP C: No-UWV baseline models (for diversity)
# =============================================================================
print("\n" + "=" * 60)
print("GROUP C: No-UWV Baseline Models")
print("=" * 60)

for s in [42, 123, 0, 7, 99]:
    name = f"base_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_STD)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# GROUP D: PP variant + UWV
# =============================================================================
print("\n" + "=" * 60)
print("GROUP D: PP Variants + UWV")
print("=" * 60)

pp_w9 = {"sg_w": 9, "bs": 4, "poly": 2, "sg_d": 1}
for s in [123, 42, 0]:
    name = f"uwv_w9_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), pp_w9, uwv_best)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

pp_b2 = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}
for s in [123, 42]:
    name = f"uwv_b2full_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), pp_b2, uwv_best)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# ENSEMBLE
# =============================================================================
print("\n" + "=" * 60)
print("MEGA ENSEMBLE")
print("=" * 60)

results.sort(key=lambda x: x['rmse'])
N = len(results)

print(f"\nTotal models: {N}")
print("\nTop 30:")
for i, r in enumerate(results[:30]):
    print(f"  {i+1:2d}. RMSE={r['rmse']:.4f}  SB={r['sb']:.4f}  {r['name']}")

oof_matrix = np.column_stack([r["oof"] for r in results])
test_matrix = np.column_stack([r["test"] for r in results])

for n in [3, 5, 10, 20, 30, 50]:
    if n <= N:
        print(f"  Top {n:2d} avg: RMSE={rmse(y, oof_matrix[:, :n].mean(1)):.4f}")

# NNLS all
w_all, _ = nnls(oof_matrix, y)
if w_all.sum() > 0:
    w_all /= w_all.sum()
nnls_oof = oof_matrix @ w_all
nnls_test = test_matrix @ w_all
print(f"\n  NNLS (all {N}): RMSE={rmse(y, nnls_oof):.4f}, SB={sb_rmse(y, nnls_oof, species):.4f}")

# NNLS top 40
t40 = min(40, N)
w40, _ = nnls(oof_matrix[:, :t40], y)
if w40.sum() > 0:
    w40 /= w40.sum()
nnls40_oof = oof_matrix[:, :t40] @ w40
nnls40_test = test_matrix[:, :t40] @ w40
print(f"  NNLS (top {t40}): RMSE={rmse(y, nnls40_oof):.4f}")

print("\n  NNLS weights (all, >0.5%):")
for i in np.argsort(-w_all)[:20]:
    if w_all[i] > 0.005:
        print(f"    {w_all[i]:.3f}  {results[i]['name']} (RMSE={results[i]['rmse']:.4f})")

# Per-species
best_oof = nnls_oof if rmse(y, nnls_oof) < rmse(y, nnls40_oof) else nnls40_oof
best_test = nnls_test if rmse(y, nnls_oof) < rmse(y, nnls40_oof) else nnls40_test
print(f"\n  Per-species (best NNLS):")
for s in sorted(np.unique(species)):
    m = species == s
    print(f"    Sp{s:2d}: RMSE={rmse(y[m], best_oof[m]):6.2f}, bias={(best_oof[m]-y[m]).mean():+6.1f}, n={m.sum()}")

# =============================================================================
# SAVE
# =============================================================================
print("\n" + "=" * 60)
print("SAVING")
print("=" * 60)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

best_nnls_rmse = min(rmse(y, nnls_oof), rmse(y, nnls40_oof))
best_nnls_test_out = best_test

submissions = {
    "best_single": np.clip(results[0]["test"], 0, 500),
    "top5_avg": np.clip(test_matrix[:, :5].mean(1), 0, 500),
    "top10_avg": np.clip(test_matrix[:, :10].mean(1), 0, 500),
    "top20_avg": np.clip(test_matrix[:, :20].mean(1), 0, 500),
    "nnls_all": np.clip(nnls_test, 0, 500),
    "nnls_best": np.clip(best_nnls_test_out, 0, 500),
}

for name, preds in submissions.items():
    fname = f"submission_p32_{name}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(SUB_DIR / fname, header=False, index=False)
    print(f"  {fname}")

summary = {
    "rule_compliant": True,
    "phase": 32,
    "total_models": N,
    "models": [{"name": r["name"], "rmse": float(r["rmse"]), "sb": float(r["sb"])}
               for r in results],
    "ensemble": {
        "best_single": float(results[0]["rmse"]),
        "nnls_all": float(rmse(y, nnls_oof)),
        "nnls_top40": float(rmse(y, nnls40_oof)),
    },
    "nnls_weights": {results[i]["name"]: float(w_all[i])
                     for i in range(len(w_all)) if w_all[i] > 0.005},
}
with open(RUNS_DIR / f"phase32_results_{ts}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFINAL: Best single={results[0]['rmse']:.4f}, NNLS={best_nnls_rmse:.4f}")
print(f"Phase 31 best: 14.89")
