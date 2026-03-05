#!/usr/bin/env python
"""Phase 19: Breakthrough strategies from Gemini + ChatGPT consultation.

Core ideas:
A. LOSO CV (13-fold leave-one-species-out) — stable evaluation
B. Deterministic WDV (no randomness, fixed matching)
C. Water absorption proxy → extrapolation teacher (break PL ceiling)
D. Linear/Huber base + LGBM residual (extrapolation architecture)
E. CARS-inspired variable selection (species-invariant features)
F. Combined: best of above
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")

BEST_PREPROCESS = [
    {"name": "emsc", "poly_order": 2},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 8},
    {"name": "standard_scaler"},
]

LGBM_PARAMS = {
    "n_estimators": 400, "max_depth": 5, "num_leaves": 20,
    "learning_rate": 0.05, "min_child_samples": 20,
    "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "verbose": -1, "n_jobs": -1,
}


def load_data():
    cfg = Config(
        train_file="train.csv", test_file="test.csv",
        id_col="sample number", target_col="含水率",
        group_col="species number",
    )
    X_train, y_train, ids = load_train(cfg, DATA_DIR)
    X_test, test_ids = load_test(cfg, DATA_DIR)
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    groups = df["species number"].values
    return X_train, y_train, groups, X_test, test_ids


def get_water_band_indices(wavenumbers):
    """Get indices of water absorption bands in NIR.

    Key water bands in NIR:
    - ~5150 cm⁻¹ (1940nm): O-H stretch + bend combination
    - ~6900 cm⁻¹ (1450nm): O-H stretch first overtone
    - ~8300 cm⁻¹ (1200nm): O-H stretch second overtone
    """
    water_bands = []
    for center, half_width in [(5150, 200), (6900, 300), (8300, 200)]:
        mask = (wavenumbers >= center - half_width) & (wavenumbers <= center + half_width)
        water_bands.extend(np.where(mask)[0])
    return sorted(set(water_bands))


def compute_water_proxy(X, wavenumbers):
    """Compute water-related spectral features as moisture proxy.

    Uses absorption at water bands relative to baseline.
    """
    proxies = []

    # Band around 5150 cm⁻¹ (strongest water band)
    b1_mask = (wavenumbers >= 4950) & (wavenumbers <= 5350)
    if b1_mask.sum() > 0:
        proxies.append(X[:, b1_mask].mean(axis=1))

    # Band around 6900 cm⁻¹
    b2_mask = (wavenumbers >= 6600) & (wavenumbers <= 7200)
    if b2_mask.sum() > 0:
        proxies.append(X[:, b2_mask].mean(axis=1))

    # Band around 8300 cm⁻¹
    b3_mask = (wavenumbers >= 8100) & (wavenumbers <= 8500)
    if b3_mask.sum() > 0:
        proxies.append(X[:, b3_mask].mean(axis=1))

    # Ratio: water band / reference band
    ref_mask = (wavenumbers >= 7500) & (wavenumbers <= 7800)
    if ref_mask.sum() > 0 and b2_mask.sum() > 0:
        ref = X[:, ref_mask].mean(axis=1) + 1e-8
        water = X[:, b2_mask].mean(axis=1)
        proxies.append(water / ref)

    # Overall area under water bands
    all_water = b1_mask | b2_mask | b3_mask
    if all_water.sum() > 0:
        proxies.append(X[:, all_water].sum(axis=1))

    return np.column_stack(proxies) if proxies else None


def generate_deterministic_wdv(X_tr, y_tr, groups_tr, n_per_species=10,
                               extrap_factor=1.5, min_moisture=150):
    """Deterministic WDV: fixed matching, no randomness.

    For each species with high-moisture samples:
    - Sort by moisture
    - Match top-K high with bottom-K low (by index order)
    - Generate extrapolations at fixed factors
    """
    synth_X, synth_y = [], []

    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue

        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 1:
            continue

        sorted_idx = np.argsort(y_sp)
        high_idx = sorted_idx[high_mask[sorted_idx]]
        low_idx = sorted_idx[:min(5, len(sorted_idx) // 2)]

        if len(low_idx) == 0:
            continue

        # Fixed matching: pair high[i] with low[i % len(low)]
        count = 0
        for i, hi in enumerate(high_idx):
            lo = low_idx[i % len(low_idx)]
            dy = y_sp[hi] - y_sp[lo]
            if dy < 10:
                continue
            dx = X_sp[hi] - X_sp[lo]

            # Generate at multiple fixed factors
            for f in [extrap_factor * 0.8, extrap_factor, extrap_factor * 1.2]:
                synth_X.append(X_sp[hi] + f * dx)
                synth_y.append(y_sp[hi] + f * dy)
                count += 1
                if count >= n_per_species:
                    break
            if count >= n_per_species:
                break

    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    return np.array(synth_X), np.array(synth_y)


def generate_universal_wdv(X_tr, y_tr, groups_tr, n_aug=30,
                           extrap_factor=1.5, min_moisture=150):
    """Universal Water Vector: PCA on group-mean differences.

    1. For each species, compute mean(high) - mean(low) spectrum
    2. PCA on these species-level differences → PC1 = universal water vector
    3. Apply this single direction for augmentation
    """
    species_deltas = []
    species_dy = []

    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 5:
            continue

        # Split into high and low groups
        median_y = np.median(y_sp)
        high = X_sp[y_sp >= median_y]
        low = X_sp[y_sp < median_y]

        if len(high) < 2 or len(low) < 2:
            continue

        delta = high.mean(axis=0) - low.mean(axis=0)
        dy = y_sp[y_sp >= median_y].mean() - y_sp[y_sp < median_y].mean()

        if dy > 5:
            species_deltas.append(delta / dy)  # normalize per unit moisture
            species_dy.append(dy)

    if len(species_deltas) < 2:
        return np.empty((0, X_tr.shape[1])), np.empty(0)

    species_deltas = np.array(species_deltas)

    # Universal water vector = mean of species deltas (or PC1)
    if len(species_deltas) >= 3:
        pca = PCA(n_components=1)
        pca.fit(species_deltas)
        water_vec = pca.components_[0]
        # Ensure positive correlation with moisture (flip if needed)
        if np.corrcoef(species_deltas @ water_vec, species_dy)[0, 1] < 0:
            water_vec = -water_vec
    else:
        water_vec = species_deltas.mean(axis=0)
        water_vec /= np.linalg.norm(water_vec) + 1e-8

    # Scale: how much does 1 unit of this vector change moisture?
    # Use regression: y = a * (X @ water_vec) + b
    proj = X_tr @ water_vec
    from numpy.polynomial.polynomial import polyfit
    coeffs = polyfit(proj, y_tr, 1)
    scale = coeffs[1]  # slope

    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 1:
            continue

        for hi_idx in np.where(high_mask)[0]:
            target_dy = extrap_factor * (y_sp[hi_idx] * 0.3 + 30)
            step = target_dy / (scale + 1e-8)
            synth_X.append(X_sp[hi_idx] + step * water_vec)
            synth_y.append(y_sp[hi_idx] + target_dy)

    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        # Take evenly spaced samples (deterministic)
        idx = np.linspace(0, len(synth_X) - 1, n_aug, dtype=int)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


def cv_loso(X_train, y_train, groups, X_test, wdv_func=None, wdv_kwargs=None,
            pl_w=0.0, lgbm_ov=None, preprocess=None, residual_model=None):
    """Leave-One-Species-Out CV (13-fold)."""
    if preprocess is None:
        preprocess = BEST_PREPROCESS
    params = {**LGBM_PARAMS}
    if lgbm_ov:
        params.update(lgbm_ov)

    logo = LeaveOneGroupOut()
    oof = np.zeros(len(y_train))
    species_rmses = {}
    test_preds = []

    for tr_idx, va_idx in logo.split(X_train, y_train, groups):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups[tr_idx]
        va_species = groups[va_idx][0]

        pipe = build_preprocess_pipeline(preprocess)
        pipe.fit(X_tr)

        # WDV augmentation
        if wdv_func is not None:
            kw = wdv_kwargs or {}
            synth_X, synth_y = wdv_func(X_tr, y_tr, g_tr, **kw)
        else:
            synth_X, synth_y = np.empty((0, X_tr.shape[1])), np.empty(0)

        X_tr_t = pipe.transform(X_tr)
        X_va_t = pipe.transform(X_va)
        X_test_t = pipe.transform(X_test)

        if len(synth_X) > 0:
            synth_X_t = pipe.transform(synth_X)
            X_tr_aug = np.vstack([X_tr_t, synth_X_t])
            y_tr_aug = np.concatenate([y_tr, synth_y])
        else:
            X_tr_aug = X_tr_t
            y_tr_aug = y_tr

        # Residual learning
        if residual_model == "huber":
            base = HuberRegressor(max_iter=200)
            base.fit(X_tr_aug, y_tr_aug)
            residuals = y_tr_aug - base.predict(X_tr_aug)
            model = create_model("lgbm", params)
            model.fit(X_tr_aug, residuals)
            pred = base.predict(X_va_t) + model.predict(X_va_t)
            tp = base.predict(X_test_t) + model.predict(X_test_t)
        elif residual_model == "ridge":
            base = Ridge(alpha=1.0)
            base.fit(X_tr_aug, y_tr_aug)
            residuals = y_tr_aug - base.predict(X_tr_aug)
            model = create_model("lgbm", params)
            model.fit(X_tr_aug, residuals)
            pred = base.predict(X_va_t) + model.predict(X_va_t)
            tp = base.predict(X_test_t) + model.predict(X_test_t)
        else:
            # Standard: PL then train
            if pl_w > 0:
                temp = create_model("lgbm", params)
                temp.fit(X_tr_aug, y_tr_aug)
                pl = temp.predict(X_test_t)
                X_tr_aug = np.vstack([X_tr_aug, X_test_t])
                y_tr_aug = np.concatenate([y_tr_aug, pl])
                w = np.ones(len(y_tr_aug))
                w[-len(pl):] = pl_w
                model = create_model("lgbm", params)
                model.fit(X_tr_aug, y_tr_aug, sample_weight=w)
            else:
                model = create_model("lgbm", params)
                model.fit(X_tr_aug, y_tr_aug)
            pred = model.predict(X_va_t)
            tp = model.predict(X_test_t)

        oof[va_idx] = pred
        sp_rmse = rmse(y_va, pred)
        species_rmses[va_species] = sp_rmse
        test_preds.append(tp)

    return oof, species_rmses, np.mean(test_preds, axis=0)


def cv_gkf5(X_train, y_train, groups, X_test, wdv_func=None, wdv_kwargs=None,
             pl_w=0.0, lgbm_ov=None, preprocess=None, residual_model=None,
             proxy_teacher=False, wavenumbers=None):
    """Standard 5-fold GroupKFold CV with all improvements."""
    if preprocess is None:
        preprocess = BEST_PREPROCESS
    params = {**LGBM_PARAMS}
    if lgbm_ov:
        params.update(lgbm_ov)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_train))
    fold_rmses = []
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups[tr_idx]

        pipe = build_preprocess_pipeline(preprocess)
        pipe.fit(X_tr)

        # WDV
        if wdv_func is not None:
            kw = wdv_kwargs or {}
            synth_X, synth_y = wdv_func(X_tr, y_tr, g_tr, **kw)
        else:
            synth_X, synth_y = np.empty((0, X_tr.shape[1])), np.empty(0)

        X_tr_t = pipe.transform(X_tr)
        X_va_t = pipe.transform(X_va)
        X_test_t = pipe.transform(X_test)

        if len(synth_X) > 0:
            synth_X_t = pipe.transform(synth_X)
            X_tr_aug = np.vstack([X_tr_t, synth_X_t])
            y_tr_aug = np.concatenate([y_tr, synth_y])
        else:
            X_tr_aug = X_tr_t
            y_tr_aug = y_tr

        # Residual learning
        if residual_model == "huber":
            base = HuberRegressor(max_iter=200)
            base.fit(X_tr_aug, y_tr_aug)
            residuals = y_tr_aug - base.predict(X_tr_aug)
            model = create_model("lgbm", params)
            model.fit(X_tr_aug, residuals)
            pred = base.predict(X_va_t) + model.predict(X_va_t)
            tp = base.predict(X_test_t) + model.predict(X_test_t)
        elif residual_model == "ridge":
            base = Ridge(alpha=1.0)
            base.fit(X_tr_aug, y_tr_aug)
            residuals = y_tr_aug - base.predict(X_tr_aug)
            model = create_model("lgbm", params)
            model.fit(X_tr_aug, residuals)
            pred = base.predict(X_va_t) + model.predict(X_va_t)
            tp = base.predict(X_test_t) + model.predict(X_test_t)
        else:
            # Proxy teacher for PL
            if proxy_teacher and wavenumbers is not None:
                proxy_tr = compute_water_proxy(X_tr, wavenumbers)
                proxy_test = compute_water_proxy(X_test, wavenumbers)
                if proxy_tr is not None:
                    # Fit isotonic/polynomial on proxy → moisture
                    from sklearn.linear_model import LinearRegression
                    poly = PolynomialFeatures(degree=2)
                    proxy_poly = poly.fit_transform(proxy_tr)
                    lr = LinearRegression()
                    lr.fit(proxy_poly, y_tr)
                    pl_proxy = lr.predict(poly.transform(proxy_test))
                    # Blend: use proxy PL for high predictions, LGBM PL for low
                    temp = create_model("lgbm", params)
                    temp.fit(X_tr_aug, y_tr_aug)
                    pl_lgbm = temp.predict(X_test_t)
                    # Use proxy where LGBM saturates (>190)
                    pl = pl_lgbm.copy()
                    high_mask = pl_lgbm > 190
                    if high_mask.sum() > 0:
                        pl[high_mask] = 0.5 * pl_lgbm[high_mask] + 0.5 * pl_proxy[high_mask]

                    X_tr_aug = np.vstack([X_tr_aug, X_test_t])
                    y_tr_aug = np.concatenate([y_tr_aug, pl])
                    w = np.ones(len(y_tr_aug))
                    w[-len(pl):] = pl_w if pl_w > 0 else 0.5
                    model = create_model("lgbm", params)
                    model.fit(X_tr_aug, y_tr_aug, sample_weight=w)
                else:
                    model = create_model("lgbm", params)
                    model.fit(X_tr_aug, y_tr_aug)
            elif pl_w > 0:
                temp = create_model("lgbm", params)
                temp.fit(X_tr_aug, y_tr_aug)
                pl = temp.predict(X_test_t)
                X_tr_aug = np.vstack([X_tr_aug, X_test_t])
                y_tr_aug = np.concatenate([y_tr_aug, pl])
                w = np.ones(len(y_tr_aug))
                w[-len(pl):] = pl_w
                model = create_model("lgbm", params)
                model.fit(X_tr_aug, y_tr_aug, sample_weight=w)
            else:
                model = create_model("lgbm", params)
                model.fit(X_tr_aug, y_tr_aug)

            pred = model.predict(X_va_t)
            tp = model.predict(X_test_t)

        oof[va_idx] = pred
        fold_rmses.append(rmse(y_va, pred))
        test_preds.append(tp)

    return oof, fold_rmses, np.mean(test_preds, axis=0)


def main():
    np.random.seed(42)
    X_train, y_train, groups, X_test, test_ids = load_data()

    # Wavenumbers (4000 to 9994 cm⁻¹)
    wavenumbers = np.linspace(4000, 9994, X_train.shape[1])

    all_results = []
    all_oofs = {}

    def log(name, score, folds_or_sp, oof=None, is_loso=False):
        if is_loso:
            sp_str = {k: round(v, 1) for k, v in sorted(folds_or_sp.items())}
            m = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
            print(f"  {name}: RMSE={score:.4f}  species={sp_str}{m}")
        else:
            fr = [round(x, 1) for x in folds_or_sp]
            m = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
            print(f"  {name}: RMSE={score:.4f}  folds={fr}{m}")
        all_results.append((name, score))
        if oof is not None:
            all_oofs[name] = oof

    # ============================================================
    # Section A: LOSO CV — baseline and with improvements
    # ============================================================
    print("\n=== Section A: LOSO CV (13-fold leave-one-species-out) ===")

    # A1: LOSO baseline (no WDV, no PL)
    print("\n  --- A1: LOSO baselines ---")
    oof, sp_rmse, _ = cv_loso(X_train, y_train, groups, X_test)
    score = rmse(y_train, oof)
    log("loso_baseline", score, sp_rmse, oof, is_loso=True)

    # A2: LOSO with PL
    for pw in [0.3, 0.5, 0.7]:
        oof, sp_rmse, _ = cv_loso(X_train, y_train, groups, X_test, pl_w=pw)
        score = rmse(y_train, oof)
        log(f"loso_pl{pw}", score, sp_rmse, oof, is_loso=True)

    # A3: LOSO with deterministic WDV
    print("\n  --- A3: LOSO + deterministic WDV ---")
    for n_sp in [5, 10, 15]:
        for ef in [1.0, 1.5, 2.0]:
            oof, sp_rmse, _ = cv_loso(
                X_train, y_train, groups, X_test,
                wdv_func=generate_deterministic_wdv,
                wdv_kwargs={"n_per_species": n_sp, "extrap_factor": ef, "min_moisture": 150}
            )
            score = rmse(y_train, oof)
            log(f"loso_dwdv_n{n_sp}_f{ef}", score, sp_rmse, oof, is_loso=True)

    # A4: LOSO with universal WDV
    print("\n  --- A4: LOSO + universal WDV ---")
    for n_aug in [20, 30, 50]:
        for ef in [1.0, 1.5, 2.0]:
            oof, sp_rmse, _ = cv_loso(
                X_train, y_train, groups, X_test,
                wdv_func=generate_universal_wdv,
                wdv_kwargs={"n_aug": n_aug, "extrap_factor": ef, "min_moisture": 150}
            )
            score = rmse(y_train, oof)
            log(f"loso_uwdv_n{n_aug}_f{ef}", score, sp_rmse, oof, is_loso=True)

    # A5: LOSO with det WDV + PL
    print("\n  --- A5: LOSO + det WDV + PL ---")
    for pw in [0.3, 0.5]:
        oof, sp_rmse, _ = cv_loso(
            X_train, y_train, groups, X_test,
            wdv_func=generate_deterministic_wdv,
            wdv_kwargs={"n_per_species": 10, "extrap_factor": 1.5, "min_moisture": 150},
            pl_w=pw
        )
        score = rmse(y_train, oof)
        log(f"loso_dwdv_pl{pw}", score, sp_rmse, oof, is_loso=True)

    # ============================================================
    # Section B: 5-fold GKF with deterministic WDV
    # ============================================================
    print("\n=== Section B: 5-fold GKF with deterministic WDV ===")

    for n_sp in [5, 10, 15, 20]:
        for ef in [1.0, 1.2, 1.5, 2.0]:
            oof, folds, _ = cv_gkf5(
                X_train, y_train, groups, X_test,
                wdv_func=generate_deterministic_wdv,
                wdv_kwargs={"n_per_species": n_sp, "extrap_factor": ef, "min_moisture": 150}
            )
            score = rmse(y_train, oof)
            log(f"gkf_dwdv_n{n_sp}_f{ef}", score, folds, oof)

    # With PL
    print("\n  --- B2: det WDV + PL ---")
    for n_sp in [10, 15]:
        for ef in [1.0, 1.5]:
            for pw in [0.3, 0.5]:
                oof, folds, _ = cv_gkf5(
                    X_train, y_train, groups, X_test,
                    wdv_func=generate_deterministic_wdv,
                    wdv_kwargs={"n_per_species": n_sp, "extrap_factor": ef, "min_moisture": 150},
                    pl_w=pw
                )
                score = rmse(y_train, oof)
                log(f"gkf_dwdv_n{n_sp}_f{ef}_pl{pw}", score, folds, oof)

    # ============================================================
    # Section C: Universal WDV (5-fold)
    # ============================================================
    print("\n=== Section C: Universal WDV (5-fold) ===")

    for n_aug in [20, 30, 40, 50]:
        for ef in [1.0, 1.5, 2.0]:
            oof, folds, _ = cv_gkf5(
                X_train, y_train, groups, X_test,
                wdv_func=generate_universal_wdv,
                wdv_kwargs={"n_aug": n_aug, "extrap_factor": ef, "min_moisture": 150}
            )
            score = rmse(y_train, oof)
            log(f"gkf_uwdv_n{n_aug}_f{ef}", score, folds, oof)

    # With PL
    print("\n  --- C2: universal WDV + PL ---")
    for n_aug in [30, 50]:
        for ef in [1.5]:
            for pw in [0.3, 0.5]:
                oof, folds, _ = cv_gkf5(
                    X_train, y_train, groups, X_test,
                    wdv_func=generate_universal_wdv,
                    wdv_kwargs={"n_aug": n_aug, "extrap_factor": ef, "min_moisture": 150},
                    pl_w=pw
                )
                score = rmse(y_train, oof)
                log(f"gkf_uwdv_n{n_aug}_f{ef}_pl{pw}", score, folds, oof)

    # ============================================================
    # Section D: Residual learning (Huber/Ridge base + LGBM)
    # ============================================================
    print("\n=== Section D: Residual learning ===")

    for base in ["huber", "ridge"]:
        # Baseline residual
        oof, folds, _ = cv_gkf5(
            X_train, y_train, groups, X_test, residual_model=base
        )
        score = rmse(y_train, oof)
        log(f"residual_{base}", score, folds, oof)

        # With det WDV
        oof, folds, _ = cv_gkf5(
            X_train, y_train, groups, X_test,
            wdv_func=generate_deterministic_wdv,
            wdv_kwargs={"n_per_species": 10, "extrap_factor": 1.5, "min_moisture": 150},
            residual_model=base
        )
        score = rmse(y_train, oof)
        log(f"residual_{base}_dwdv", score, folds, oof)

        # LOSO residual
        oof, sp_rmse, _ = cv_loso(
            X_train, y_train, groups, X_test, residual_model=base
        )
        score = rmse(y_train, oof)
        log(f"loso_residual_{base}", score, sp_rmse, oof, is_loso=True)

    # ============================================================
    # Section E: Water proxy teacher
    # ============================================================
    print("\n=== Section E: Water proxy teacher ===")

    # Compute and show proxy correlation
    proxy_all = compute_water_proxy(X_train, wavenumbers)
    if proxy_all is not None:
        print(f"  Proxy shape: {proxy_all.shape}")
        for i in range(proxy_all.shape[1]):
            corr = np.corrcoef(proxy_all[:, i], y_train)[0, 1]
            print(f"  Proxy feature {i}: corr={corr:.3f}")

    # Proxy teacher + LGBM
    for pw in [0.3, 0.5, 0.7]:
        oof, folds, _ = cv_gkf5(
            X_train, y_train, groups, X_test,
            proxy_teacher=True, wavenumbers=wavenumbers, pl_w=pw
        )
        score = rmse(y_train, oof)
        log(f"proxy_teacher_pl{pw}", score, folds, oof)

    # Proxy teacher + WDV
    for pw in [0.3, 0.5]:
        oof, folds, _ = cv_gkf5(
            X_train, y_train, groups, X_test,
            wdv_func=generate_deterministic_wdv,
            wdv_kwargs={"n_per_species": 10, "extrap_factor": 1.5, "min_moisture": 150},
            proxy_teacher=True, wavenumbers=wavenumbers, pl_w=pw
        )
        score = rmse(y_train, oof)
        log(f"proxy_teacher_dwdv_pl{pw}", score, folds, oof)

    # ============================================================
    # Section F: CARS-inspired variable selection
    # ============================================================
    print("\n=== Section F: Water band feature selection ===")

    water_idx = get_water_band_indices(wavenumbers)
    print(f"  Water band indices: {len(water_idx)} out of {X_train.shape[1]}")

    # Only water bands
    if water_idx:
        # Modify preprocessing: use water-band-only features
        water_preprocess = [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "standard_scaler"},
        ]

        X_train_water = X_train[:, water_idx]
        X_test_water = X_test[:, water_idx]

        # Simple test: water bands only with LGBM
        gkf = GroupKFold(n_splits=5)
        oof = np.zeros(len(y_train))
        fold_rmses = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train_water, y_train, groups)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_water[tr_idx])
            X_va = scaler.transform(X_train_water[va_idx])
            model = create_model("lgbm", LGBM_PARAMS)
            model.fit(X_tr, y_train[tr_idx])
            oof[va_idx] = model.predict(X_va)
            fold_rmses.append(rmse(y_train[va_idx], oof[va_idx]))
        score = rmse(y_train, oof)
        log("water_bands_only", score, fold_rmses, oof)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 19 FINAL SUMMARY")
    print("=" * 70)

    all_results.sort(key=lambda x: x[1])
    for name, score in all_results[:40]:
        m = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"  {score:.4f}  {name}{m}")

    if all_results:
        print(f"\nBEST: {all_results[0][1]:.4f} ({all_results[0][0]})")
    print(f"Phase 15c best: 15.10")


if __name__ == "__main__":
    main()
