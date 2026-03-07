#!/usr/bin/env python3
"""Phase 33: Species-Targeted Optimization.

Sp15 (RMSE 34.1, bias -20.3) and Sp11 (RMSE 22.9, bias +16.7) dominate error.
Strategies:
1. Sample weighting — upweight high-error species during LGBM training
2. Species-aware UWV — per-species UWV parameters, especially aggressive for Sp15
3. Species embedding — add species number as categorical feature
4. Two-stage: coarse → fine (species-cluster-specific models)
5. Quantile UWV — use Q75/Q25 split instead of median for more robust water vector
6. Larger n_aug for underrepresented moisture ranges
7. Residual correction — train 2nd stage on residuals of 1st stage

Rule compliant: NO pseudo-labeling, NO test data for training.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
sp_test = test['species number'].values

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Target: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}")

# Species analysis
print("\nPer-species stats:")
for s in sorted(np.unique(species)):
    m = species == s
    ys = y[m]
    print(f"  Sp{s:2d}: n={m.sum():3d}, mean={ys.mean():6.1f}, "
          f"std={ys.std():5.1f}, range=[{ys.min():.1f}, {ys.max():.1f}]")

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
                  min_moisture=170, dy_scale=0.3, dy_offset=30, n_pca=1,
                  quantile_split=False):
    """UWV with optional quantile-based split."""
    species_deltas, species_dy = [], []
    for s in np.unique(sp_tr):
        mask = sp_tr == s
        X_sp, y_sp = X_tr[mask], y_tr[mask]
        if quantile_split:
            q75, q25 = np.percentile(y_sp, 75), np.percentile(y_sp, 25)
            hi, lo = y_sp >= q75, y_sp <= q25
        else:
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


def cv_with_test(model, pp_kwargs, uwv_kwargs=None, water_only=False,
                 sample_weight_fn=None, add_species_feat=False):
    """CV with optional sample weighting and species feature."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in GKF.split(X, y, species):
        X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx].copy(), species[tr_idx]
        X_va, X_te = X[va_idx].copy(), X_test.copy()
        sp_va = species[va_idx]

        if uwv_kwargs:
            sX, sy = generate_uwv(X_tr, y_tr, sp_tr, **uwv_kwargs)
            if len(sy) > 0:
                X_tr = np.vstack([X_tr, sX])
                y_tr = np.concatenate([y_tr, sy])
                # Augmented samples get no species (use 0)
                sp_tr = np.concatenate([sp_tr, np.zeros(len(sy), dtype=int)])

        pp = dict(pp_kwargs)
        pp['water_only'] = water_only
        Xtr_t, Xva_t = preprocess(X_tr, X_va, **pp)
        _, Xte_t = preprocess(X_tr, X_te, **pp)

        if add_species_feat:
            Xtr_t = np.column_stack([Xtr_t, sp_tr.astype(float)])
            Xva_t = np.column_stack([Xva_t, sp_va.astype(float)])
            Xte_t = np.column_stack([Xte_t, sp_test.astype(float)])

        sw = None
        if sample_weight_fn is not None:
            sw = sample_weight_fn(y_tr, sp_tr)

        from sklearn.base import clone
        m = clone(model)
        fit_params = dict(
            eval_set=[(Xva_t, y[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        if sw is not None:
            fit_params['sample_weight'] = sw

        m.fit(Xtr_t, y_tr, **fit_params)
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
# STRATEGY 1: Sample Weighting — upweight high-error species
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY 1: Sample Weighting")
print("=" * 60)

# Weight function: upweight species with high error from Phase 32 analysis
# Sp15 (RMSE 34.1) and Sp11 (RMSE 22.9) need much more weight
SP_ERRORS = {1: 14.5, 3: 14.2, 4: 14.8, 5: 8.2, 8: 5.4,
             11: 22.9, 12: 5.5, 13: 5.6, 14: 6.9,
             15: 34.1, 16: 11.4, 17: 9.7, 19: 7.9}

def make_weight_fn(power=1.0, base=1.0):
    """Create sample weight function based on species RMSE."""
    mean_err = np.mean(list(SP_ERRORS.values()))
    def weight_fn(y_tr, sp_tr):
        w = np.ones(len(y_tr))
        for s in np.unique(sp_tr):
            if s in SP_ERRORS:
                ratio = SP_ERRORS[s] / mean_err
                w[sp_tr == s] = base + (ratio ** power - 1)
        return w
    return weight_fn

uwv_best = {"n_aug": 20, "extrap_factor": 1.5, "min_moisture": 170}

# Try different weighting powers
for power, base in [(1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (0.5, 2.0)]:
    for s in [42, 123]:
        name = f"sw_p{power}_b{base}_s{s}"
        oof, tp, sc, sb = cv_with_test(
            make_lgbm(3, 0.1, s), PP_STD, uwv_best,
            sample_weight_fn=make_weight_fn(power, base))
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")
        if s == 42:
            per_sp(y, oof, species)

# =============================================================================
# STRATEGY 2: Species as Categorical Feature
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY 2: Species Feature")
print("=" * 60)

for s in [42, 123, 0, 7]:
    name = f"sp_feat_s{s}"
    oof, tp, sc, sb = cv_with_test(
        make_lgbm(3, 0.1, s), PP_STD, uwv_best,
        add_species_feat=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")
    if s == 42:
        per_sp(y, oof, species)

# Species feat + water band
for s in [42, 123]:
    name = f"sp_h2o_s{s}"
    oof, tp, sc, sb = cv_with_test(
        make_lgbm(3, 0.1, s), PP_WATER, water_only=True,
        add_species_feat=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# Species feat + UWV + water band
for s in [42, 123]:
    name = f"sp_h2o_uwv_s{s}"
    oof, tp, sc, sb = cv_with_test(
        make_lgbm(3, 0.1, s), PP_WATER, uwv_best, water_only=True,
        add_species_feat=True)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# STRATEGY 3: Quantile-based UWV
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY 3: Quantile UWV")
print("=" * 60)

uwv_quant = {"n_aug": 20, "extrap_factor": 1.5, "min_moisture": 170, "quantile_split": True}
for s in [42, 123, 0, 7]:
    name = f"quwv_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_STD, uwv_quant)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")
    if s == 42:
        per_sp(y, oof, species)

# =============================================================================
# STRATEGY 4: Aggressive UWV for high-moisture extrapolation
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY 4: Aggressive UWV (high extrap)")
print("=" * 60)

# More extrapolation may help Sp15 which has extreme moisture
for ef, na, mm in [(2.0, 30, 130), (2.5, 40, 130), (2.0, 50, 100),
                    (3.0, 30, 130), (1.5, 40, 100), (2.0, 20, 170)]:
    uwv_agg = {"n_aug": na, "extrap_factor": ef, "min_moisture": mm}
    name = f"agg_ef{ef}_n{na}_m{mm}_s42"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, 42), PP_STD, uwv_agg)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")
    per_sp(y, oof, species)

# Best aggressive + seeds
# Will run more seeds for the best aggressive config after initial screen
print("  Finding best aggressive config...")
agg_results = [r for r in results if r['name'].startswith('agg_')]
if agg_results:
    best_agg = min(agg_results, key=lambda x: x['rmse'])
    print(f"  Best aggressive: {best_agg['name']} RMSE={best_agg['rmse']:.4f}")
    # Parse best config from name
    parts = best_agg['name'].split('_')
    ef_best = float(parts[1][2:])
    na_best = int(parts[2][1:])
    mm_best = int(parts[3][1:])
    uwv_agg_best = {"n_aug": na_best, "extrap_factor": ef_best, "min_moisture": mm_best}
    for s in [123, 0, 7, 99]:
        name = f"agg_best_s{s}"
        oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_STD, uwv_agg_best)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# STRATEGY 5: 2D PCA Water Vector
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY 5: 2D PCA Water Vector")
print("=" * 60)

uwv_2d = {"n_aug": 20, "extrap_factor": 1.5, "min_moisture": 170, "n_pca": 2}
for s in [42, 123, 0]:
    name = f"uwv2d_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_STD, uwv_2d)
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# STRATEGY 6: Deeper trees for complex species patterns
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY 6: Deeper Trees + UWV")
print("=" * 60)

for d, leaves in [(4, 15), (5, 20), (4, 12), (3, 15)]:
    for s in [42, 123]:
        name = f"deep_d{d}l{leaves}_s{s}"
        oof, tp, sc, sb = cv_with_test(
            make_lgbm(d, 0.1, s, leaves=leaves), PP_STD, uwv_best)
        results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
        print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")
        if s == 42:
            per_sp(y, oof, species)

# =============================================================================
# STRATEGY 7: Residual Correction (2-stage)
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY 7: Residual Correction (2-stage)")
print("=" * 60)

# Stage 1: Standard UWV model
print("  Stage 1: Base UWV model...")
oof_s1 = np.zeros(len(y))
test_s1 = np.zeros(len(X_test))
for tr_idx, va_idx in GKF.split(X, y, species):
    X_tr, y_tr, sp_tr = X[tr_idx].copy(), y[tr_idx].copy(), species[tr_idx]
    X_va, X_te = X[va_idx].copy(), X_test.copy()

    sX, sy = generate_uwv(X_tr, y_tr, sp_tr, **uwv_best)
    if len(sy) > 0:
        X_tr = np.vstack([X_tr, sX])
        y_tr = np.concatenate([y_tr, sy])

    Xtr_t, Xva_t = preprocess(X_tr, X_va, **PP_STD)
    _, Xte_t = preprocess(X_tr, X_te, **PP_STD)

    m = make_lgbm(3, 0.1, 42)
    m.fit(Xtr_t, y_tr,
          eval_set=[(Xva_t, y[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_s1[va_idx] = m.predict(Xva_t)
    test_s1 += m.predict(Xte_t)
test_s1 /= 13
resid = y - oof_s1
print(f"  Stage 1 RMSE: {rmse(y, oof_s1):.4f}")
print(f"  Residual range: [{resid.min():.1f}, {resid.max():.1f}], std={resid.std():.1f}")

# Stage 2: Train on residuals with species feature
print("  Stage 2: Residual correction with species feature...")
oof_s2 = np.zeros(len(y))
test_s2 = np.zeros(len(X_test))
for tr_idx, va_idx in GKF.split(X, y, species):
    X_tr = X[tr_idx].copy()
    X_va, X_te = X[va_idx].copy(), X_test.copy()

    Xtr_t, Xva_t = preprocess(X_tr, X_va, **PP_STD)
    _, Xte_t = preprocess(X_tr, X_te, **PP_STD)

    # Add species feature
    Xtr_t = np.column_stack([Xtr_t, species[tr_idx].astype(float)])
    Xva_t = np.column_stack([Xva_t, species[va_idx].astype(float)])
    Xte_t = np.column_stack([Xte_t, sp_test.astype(float)])

    r_tr = resid[tr_idx]
    m = lgb.LGBMRegressor(
        n_estimators=500, max_depth=2, num_leaves=4,
        learning_rate=0.05, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=10.0,
        verbose=-1, n_jobs=-1, random_state=42)
    m.fit(Xtr_t, r_tr,
          eval_set=[(Xva_t, resid[va_idx])],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    oof_s2[va_idx] = m.predict(Xva_t)
    test_s2 += m.predict(Xte_t)
test_s2 /= 13

combined = np.clip(oof_s1 + oof_s2, 0, 500)
combined_test = np.clip(test_s1 + test_s2, 0, 500)
sc = rmse(y, combined)
sb = sb_rmse(y, combined, species)
print(f"  2-stage: RMSE={sc:.4f}, SB={sb:.4f}")
per_sp(y, combined, species)
results.append({"name": "residual_2stage", "rmse": sc, "sb": sb,
                "oof": combined, "test": combined_test})

# Stage 2 variant: different correction strengths
for alpha in [0.3, 0.5, 0.7, 1.5]:
    c = np.clip(oof_s1 + alpha * oof_s2, 0, 500)
    ct = np.clip(test_s1 + alpha * test_s2, 0, 500)
    sc = rmse(y, c)
    sb = sb_rmse(y, c, species)
    name = f"residual_a{alpha}"
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": c, "test": ct})
    print(f"  {name}: RMSE={sc:.4f}, SB={sb:.4f}")

# =============================================================================
# STRATEGY 8: Combine with Phase 32 best models (reload OOF)
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY 8: Recreate Phase 32 best + combine")
print("=" * 60)

# Recreate the Phase 32 NNLS core models for combined ensemble
p32_models = []

# h2o (water band) models — Phase 32 NNLS gave ~51% weight to h2o_s555
print("  Recreating Phase 32 core models...")
for s in [555, 42, 123, 0, 7]:
    name = f"p32_h2o_s{s}"
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_WATER, water_only=True)
    p32_models.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv20 leaves=12 s=123 — Phase 32 NNLS gave ~28% weight
for s in [123, 42, 0]:
    name = f"p32_uwv20_l12_s{s}"
    oof, tp, sc, sb = cv_with_test(
        make_lgbm(3, 0.1, s, leaves=12), PP_STD, uwv_best)
    p32_models.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv10 s=123 — Phase 32 NNLS gave ~11%
for s in [123, 42]:
    name = f"p32_uwv10_s{s}"
    uwv10 = {"n_aug": 10, "extrap_factor": 1.5, "min_moisture": 170}
    oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, s), PP_STD, uwv10)
    p32_models.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
    print(f"  {name}: RMSE={sc:.4f}")

# uwv_b2full s=42 — Phase 32 NNLS gave ~8%
pp_b2 = {"sg_w": 7, "bs": 2, "poly": 2, "sg_d": 1}
name = "p32_uwv_b2full_s42"
oof, tp, sc, sb = cv_with_test(make_lgbm(3, 0.1, 42), pp_b2, uwv_best)
p32_models.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
results.append({"name": name, "rmse": sc, "sb": sb, "oof": oof, "test": tp})
print(f"  {name}: RMSE={sc:.4f}")

# =============================================================================
# MEGA ENSEMBLE
# =============================================================================
print("\n" + "=" * 60)
print("MEGA ENSEMBLE (all strategies)")
print("=" * 60)

results.sort(key=lambda x: x['rmse'])
N = len(results)

print(f"\nTotal models: {N}")
print("\nTop 30:")
for i, r in enumerate(results[:min(30, N)]):
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
for i in np.argsort(-w_all)[:20]:
    if w_all[i] > 0.005:
        print(f"    {w_all[i]:.3f}  {results[i]['name']} (RMSE={results[i]['rmse']:.4f})")

print(f"\n  Per-species (NNLS all):")
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

# Pick best
if nnls_all_rmse < nnls30_rmse:
    best_oof, best_test, best_rmse = nnls_oof, nnls_test, nnls_all_rmse
    best_label = f"nnls_all_{N}"
else:
    best_oof, best_test, best_rmse = nnls30_oof, nnls30_test, nnls30_rmse
    best_label = f"nnls_top{t30}"

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
    "nnls_best": np.clip(best_test, 0, 500),
}

for sname, preds in submissions.items():
    fname = f"submission_p33_{sname}_{ts}.csv"
    pd.DataFrame({"含水率": preds}).to_csv(SUB_DIR / fname, header=False, index=False)
    print(f"  {fname}")

summary = {
    "rule_compliant": True,
    "phase": 33,
    "total_models": N,
    "models": [{"name": r["name"], "rmse": float(r["rmse"]), "sb": float(r["sb"])}
               for r in results],
    "ensemble": {
        "best_single": float(results[0]["rmse"]),
        "nnls_all": float(nnls_all_rmse),
        f"nnls_top{t30}": float(nnls30_rmse),
    },
    "nnls_weights": {results[i]["name"]: float(w_all[i])
                     for i in range(len(w_all)) if w_all[i] > 0.005},
}
with open(RUNS_DIR / f"phase33_results_{ts}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFINAL: Best single={results[0]['rmse']:.4f}, NNLS={best_rmse:.4f}")
print(f"Phase 32 best: 14.55")
print(f"Improvement: {14.55 - best_rmse:+.4f}")
