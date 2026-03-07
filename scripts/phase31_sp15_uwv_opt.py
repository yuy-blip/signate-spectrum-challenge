#!/usr/bin/env python3
"""Phase 31: Species 15/11 targeting + UWV parameter optimization + diverse ensemble.

Phase 29b found:
- NNLS RMSE 15.06 with UWV + water band specialist
- Sp15 RMSE 36.25 (bias -20.3) — largest error source
- Sp11 RMSE 22.88 (bias +16.7) — second largest
- Only 3 models get NNLS weight (need more diversity)

This phase:
1. UWV parameter grid search (extrap_factor, min_moisture, n_aug)
2. Separate UWV configs for different moisture zones
3. Target transformation + UWV (log1p, sqrt)
4. PP variants + UWV (different binning, window sizes)
5. Larger diverse ensemble with more NNLS candidates

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
# Core functions (same as Phase 29b)
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
        wl_mask = WATER_INTERVALS
        Xtr = Xtr[:, wl_mask]
        Xte = Xte[:, wl_mask]
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
                 water_only=False, target_transform=None):
    """CV + test in one pass. target_transform: None, 'log1p', 'sqrt'."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))

    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx].copy(), species[tr_idx]
        X_va = X[va_idx].copy()
        X_te = X_test.copy()

        if use_wdv and wdv_kwargs:
            synth_X, synth_y = generate_universal_wdv(X_tr, y_tr, sp_tr, **wdv_kwargs)
            if len(synth_y) > 0:
                X_tr = np.vstack([X_tr, synth_X])
                y_tr = np.concatenate([y_tr, synth_y])

        pp = dict(pp_kwargs)
        pp['water_only'] = water_only
        Xtr_t, Xva_t = preprocess(X_tr, X_va, **pp)
        _, Xte_t = preprocess(X_tr, X_te, **pp)

        # Target transform
        if target_transform == 'log1p':
            y_fit = np.log1p(np.clip(y_tr, 0, None))
        elif target_transform == 'sqrt':
            y_fit = np.sqrt(np.clip(y_tr, 0, None))
        else:
            y_fit = y_tr

        m = clone(model)
        va_y = y[va_idx]
        if target_transform == 'log1p':
            va_y_t = np.log1p(va_y)
        elif target_transform == 'sqrt':
            va_y_t = np.sqrt(va_y)
        else:
            va_y_t = va_y

        m.fit(Xtr_t, y_fit,
              eval_set=[(Xva_t, va_y_t)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

        pred_va = m.predict(Xva_t)
        pred_te = m.predict(Xte_t)

        if target_transform == 'log1p':
            pred_va = np.expm1(pred_va)
            pred_te = np.expm1(pred_te)
        elif target_transform == 'sqrt':
            pred_va = pred_va ** 2
            pred_te = pred_te ** 2

        oof[va_idx] = np.clip(pred_va, 0, 500)
        test_pred += np.clip(pred_te, 0, 500)

    test_pred /= 13
    return oof, test_pred, rmse(y, oof), sb_rmse(y, oof, species)


def make_lgbm(d=3, lr=0.1, s=42, leaves=None, min_child=10):
    if leaves is None:
        leaves = min(2**d - 1, 31)
    return lgb.LGBMRegressor(
        n_estimators=2000, max_depth=d, num_leaves=leaves,
        learning_rate=lr, min_child_samples=min_child,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0,
        verbose=-1, n_jobs=-1, random_state=s)


PP_STD = {"sg_w": 7, "bs": 4, "poly": 2, "sg_d": 1}
PP_WATER = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}

results = []

# =============================================================================
# PHASE 1: UWV Parameter Grid Search
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: UWV Parameter Grid Search")
print("=" * 60)

ref_model = make_lgbm(3, 0.1, 123)  # best seed from Phase 29b

uwv_grid = [
    # Vary extrap_factor
    {"n_aug": 30, "extrap_factor": 1.0, "min_moisture": 170},
    {"n_aug": 30, "extrap_factor": 1.2, "min_moisture": 170},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170},  # Phase 29b best
    {"n_aug": 30, "extrap_factor": 1.8, "min_moisture": 170},
    {"n_aug": 30, "extrap_factor": 2.5, "min_moisture": 170},
    # Vary min_moisture
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 100},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 130},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 150},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 200},
    # Vary n_aug
    {"n_aug": 10, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 20, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 50, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 80, "extrap_factor": 1.5, "min_moisture": 170},
    {"n_aug": 100, "extrap_factor": 1.5, "min_moisture": 170},
    # Vary dy_scale/dy_offset
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170, "dy_scale": 0.2, "dy_offset": 20},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170, "dy_scale": 0.4, "dy_offset": 40},
    {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170, "dy_scale": 0.5, "dy_offset": 50},
    # Optimal combos
    {"n_aug": 40, "extrap_factor": 1.3, "min_moisture": 150},
    {"n_aug": 50, "extrap_factor": 1.2, "min_moisture": 150},
    {"n_aug": 40, "extrap_factor": 1.5, "min_moisture": 150},
]

for wdv_kw in uwv_grid:
    name = f"uwv_n{wdv_kw['n_aug']}_f{wdv_kw['extrap_factor']}_m{wdv_kw['min_moisture']}"
    if 'dy_scale' in wdv_kw:
        name += f"_ds{wdv_kw['dy_scale']}_do{wdv_kw['dy_offset']}"
    oof, tp, sc, sb = cv_with_test(ref_model, PP_STD, use_wdv=True, wdv_kwargs=wdv_kw)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# PHASE 2: Target Transform + UWV
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: Target Transform + UWV")
print("=" * 60)

WDV_BEST = {"n_aug": 30, "extrap_factor": 1.5, "min_moisture": 170}

for tt in ['log1p', 'sqrt']:
    for seed in [42, 123, 0]:
        name = f"uwv_{tt}_s{seed}"
        oof, tp, sc, sb = cv_with_test(
            make_lgbm(3, 0.1, seed), PP_STD,
            use_wdv=True, wdv_kwargs=WDV_BEST,
            target_transform=tt)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# PHASE 3: PP Variants + UWV (different binning/windows)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: PP Variants + UWV")
print("=" * 60)

pp_variants = [
    ("b2", {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}),
    ("b8", {"sg_w": 7, "bs": 8, "poly": 2, "sg_d": 1}),
    ("w9", {"sg_w": 9, "bs": 4, "poly": 2, "sg_d": 1}),
    ("w11", {"sg_w": 11, "bs": 4, "poly": 2, "sg_d": 1}),
    ("w5", {"sg_w": 5, "bs": 4, "poly": 2, "sg_d": 1}),
    ("p3", {"sg_w": 7, "bs": 4, "poly": 3, "sg_d": 1}),
]

for pp_name, pp_kw in pp_variants:
    for seed in [123, 42]:
        name = f"uwv_{pp_name}_s{seed}"
        oof, tp, sc, sb = cv_with_test(
            make_lgbm(3, 0.1, seed), pp_kw,
            use_wdv=True, wdv_kwargs=WDV_BEST)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# PHASE 4: Diverse model configs + UWV (for ensemble diversity)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 4: Diverse Configs + UWV")
print("=" * 60)

diverse_configs = [
    # Vary regularization
    ("uwv_reg_a0.1_l1", make_lgbm(3, 0.1, 123), {"reg_alpha": 0.1, "reg_lambda": 1.0}),
    ("uwv_reg_a5_l10", make_lgbm(3, 0.1, 123), {"reg_alpha": 5.0, "reg_lambda": 10.0}),
    ("uwv_reg_a10_l20", make_lgbm(3, 0.1, 123), {"reg_alpha": 10.0, "reg_lambda": 20.0}),
    # Vary subsample
    ("uwv_ss0.6_cs0.6", make_lgbm(3, 0.1, 123), {"subsample": 0.6, "colsample_bytree": 0.6}),
    ("uwv_ss0.9_cs0.9", make_lgbm(3, 0.1, 123), {"subsample": 0.9, "colsample_bytree": 0.9}),
    # Vary min_child
    ("uwv_mc5", make_lgbm(3, 0.1, 123, min_child=5), {}),
    ("uwv_mc20", make_lgbm(3, 0.1, 123, min_child=20), {}),
    ("uwv_mc30", make_lgbm(3, 0.1, 123, min_child=30), {}),
    # Vary leaves
    ("uwv_l4", make_lgbm(3, 0.1, 123, leaves=4), {}),
    ("uwv_l12", make_lgbm(3, 0.1, 123, leaves=12), {}),
    ("uwv_l16", make_lgbm(3, 0.1, 123, leaves=16), {}),
    # More seeds
    ("uwv_s5", make_lgbm(3, 0.1, 5), {}),
    ("uwv_s13", make_lgbm(3, 0.1, 13), {}),
    ("uwv_s31", make_lgbm(3, 0.1, 31), {}),
    ("uwv_s77", make_lgbm(3, 0.1, 77), {}),
    ("uwv_s200", make_lgbm(3, 0.1, 200), {}),
    ("uwv_s314", make_lgbm(3, 0.1, 314), {}),
    ("uwv_s555", make_lgbm(3, 0.1, 555), {}),
    ("uwv_s999", make_lgbm(3, 0.1, 999), {}),
]

for name, model, overrides in diverse_configs:
    # Apply overrides
    for k, v in overrides.items():
        model.set_params(**{k: v})
    oof, tp, sc, sb = cv_with_test(model, PP_STD, use_wdv=True, wdv_kwargs=WDV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# PHASE 5: Water band models (for ensemble diversity)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 5: Water Band Models")
print("=" * 60)

for seed in [42, 0, 123, 7, 99]:
    # Water band only, no UWV
    name = f"h2o_s{seed}"
    oof, tp, sc, sb = cv_with_test(
        make_lgbm(3, 0.1, seed), PP_WATER, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

    # Water band + UWV
    name = f"h2o_uwv_s{seed}"
    oof, tp, sc, sb = cv_with_test(
        make_lgbm(3, 0.1, seed), PP_WATER,
        use_wdv=True, wdv_kwargs=WDV_BEST, water_only=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Phase 29b top models (reproduce for ensemble)
print("\n--- Phase 29b top models ---")
for seed in [123, 7, 0, 42]:
    name = f"uwv_repr_s{seed}"
    oof, tp, sc, sb = cv_with_test(
        make_lgbm(3, 0.1, seed), PP_STD, use_wdv=True, wdv_kwargs=WDV_BEST)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# PHASE 6: MEGA Ensemble
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 6: MEGA Ensemble")
print("=" * 60)

results.sort(key=lambda x: x['rmse'])

print(f"\nTotal models: {len(results)}")
print("\nTop 30:")
for i, r in enumerate(results[:30]):
    print(f"  {i+1:2d}. RMSE={r['rmse']:.4f}  SB={r['sb']:.4f}  {r['name']}")

# Use all models for NNLS (let NNLS find the best combination)
N = len(results)
oof_matrix = np.column_stack([r["oof"] for r in results])
test_matrix = np.column_stack([r["test"] for r in results])

# Top-N averages
for n in [3, 5, 10, 15, 20, 30]:
    if n <= N:
        avg_oof = oof_matrix[:, :n].mean(1)
        print(f"  Top {n:2d} avg: RMSE={rmse(y, avg_oof):.4f}")

# NNLS with all models
w_all, _ = nnls(oof_matrix, y)
if w_all.sum() > 0:
    w_all /= w_all.sum()
nnls_oof = oof_matrix @ w_all
nnls_test = test_matrix @ w_all
print(f"\n  NNLS (all {N}): RMSE={rmse(y, nnls_oof):.4f}, SB={sb_rmse(y, nnls_oof, species):.4f}")

# NNLS with top 30 only
top30_oof = oof_matrix[:, :30]
top30_test = test_matrix[:, :30]
w30, _ = nnls(top30_oof, y)
if w30.sum() > 0:
    w30 /= w30.sum()
nnls30_oof = top30_oof @ w30
nnls30_test = top30_test @ w30
print(f"  NNLS (top 30): RMSE={rmse(y, nnls30_oof):.4f}, SB={sb_rmse(y, nnls30_oof, species):.4f}")

# NNLS with top 50
if N >= 50:
    top50_oof = oof_matrix[:, :50]
    top50_test = test_matrix[:, :50]
    w50, _ = nnls(top50_oof, y)
    if w50.sum() > 0:
        w50 /= w50.sum()
    nnls50_oof = top50_oof @ w50
    nnls50_test = top50_test @ w50
    print(f"  NNLS (top 50): RMSE={rmse(y, nnls50_oof):.4f}, SB={sb_rmse(y, nnls50_oof, species):.4f}")

# Print NNLS weights
print("\n  NNLS weights (all, >0.5%):")
for i in np.argsort(-w_all)[:20]:
    if w_all[i] > 0.005:
        print(f"    {w_all[i]:.3f}  {results[i]['name']} (RMSE={results[i]['rmse']:.4f})")

# Per-species analysis
best_w = w_all
best_oof = nnls_oof
best_test = nnls_test
best_label = "NNLS_all"

# If top30 is better, use that
if rmse(y, nnls30_oof) < rmse(y, nnls_oof):
    best_w = w30
    best_oof = nnls30_oof
    best_test = nnls30_test
    best_label = "NNLS_top30"
    print(f"\n  → Using {best_label} (better)")

print(f"\n  Per-species ({best_label}):")
for s in sorted(np.unique(species)):
    m = species == s
    sp_rmse_val = rmse(y[m], best_oof[m])
    bias = (best_oof[m] - y[m]).mean()
    print(f"    Sp{s:2d}: RMSE={sp_rmse_val:6.2f}, bias={bias:+6.1f}, n={m.sum()}")

# =============================================================================
# PHASE 7: Save
# =============================================================================
print("\n" + "=" * 60)
print("SAVING (RULE COMPLIANT)")
print("=" * 60)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Choose the best ensemble method
submissions = {
    "best_single": np.clip(results[0]["test"], 0, 500),
    "top5_avg": np.clip(test_matrix[:, :5].mean(1), 0, 500),
    "top10_avg": np.clip(test_matrix[:, :10].mean(1), 0, 500),
    "nnls_all": np.clip(nnls_test, 0, 500),
    "nnls_top30": np.clip(nnls30_test, 0, 500),
}

for name, preds in submissions.items():
    fname = f"submission_p31_{name}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(SUB_DIR / fname, header=False, index=False)
    print(f"  {fname}")

# Save results
summary = {
    "rule_compliant": True,
    "phase": 31,
    "description": "UWV parameter optimization + diverse ensemble",
    "models": [{"name": r["name"], "rmse": float(r["rmse"]), "sb": float(r["sb"])}
               for r in results],
    "ensemble": {
        "best_single": float(results[0]["rmse"]),
        "nnls_all": float(rmse(y, nnls_oof)),
        "nnls_all_sb": float(sb_rmse(y, nnls_oof, species)),
        "nnls_top30": float(rmse(y, nnls30_oof)),
        "nnls_top30_sb": float(sb_rmse(y, nnls30_oof, species)),
    },
    "nnls_weights_all": {results[i]["name"]: float(w_all[i])
                         for i in range(len(w_all)) if w_all[i] > 0.005},
}
with open(RUNS_DIR / f"phase31_results_{ts}.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("FINAL SUMMARY (RULE COMPLIANT)")
print("=" * 60)
print(f"Best single:   {results[0]['rmse']:.4f} ({results[0]['name']})")
print(f"NNLS (all):    {rmse(y, nnls_oof):.4f}")
print(f"NNLS (top30):  {rmse(y, nnls30_oof):.4f}")
print(f"\nPhase 29b best: 15.06")
print(f"Target: < 15.06")
