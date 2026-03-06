#!/usr/bin/env python
"""Phase 22: The Final Exorcism — 最後の外挿.

Current best: 14.12 (bin4 + mm170 + iterPL + greedy ensemble + stretch p97 s1.15)

Strategies:
  A. Conditional Stretch — stretch only top N% with varying intensity
  B. Peak-Integration Features — water absorption band integrals
  C. Sample Reweighting — adversarial validation weights for domain adaptation
  D. Multi-WDV ensemble — multiple WDV configs as separate branches
  E. Species 15 gating — threshold-based conditional model
  F. Piecewise nonlinear stretch
  G. Mega ensemble of everything
"""

from __future__ import annotations

import sys
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")

LGBM_BASE = {
    "n_estimators": 400, "max_depth": 5, "num_leaves": 20,
    "learning_rate": 0.05, "min_child_samples": 20,
    "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "verbose": -1, "n_jobs": -1,
}

PP_BIN4 = [
    {"name": "emsc", "poly_order": 2},
    {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
    {"name": "binning", "bin_size": 4},
    {"name": "standard_scaler"},
]


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


def generate_universal_wdv(X_tr, y_tr, groups_tr, n_aug=30,
                           extrap_factor=1.5, min_moisture=150,
                           dy_scale=0.3, dy_offset=30):
    species_deltas, species_dy = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 5:
            continue
        median_y = np.median(y_sp)
        high = X_sp[y_sp >= median_y]
        low = X_sp[y_sp < median_y]
        if len(high) < 2 or len(low) < 2:
            continue
        delta = high.mean(axis=0) - low.mean(axis=0)
        dy = y_sp[y_sp >= median_y].mean() - y_sp[y_sp < median_y].mean()
        if dy > 5:
            species_deltas.append(delta / dy)
            species_dy.append(dy)
    if len(species_deltas) < 2:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    species_deltas = np.array(species_deltas)
    if len(species_deltas) >= 3:
        pca = PCA(n_components=1)
        pca.fit(species_deltas)
        water_vec = pca.components_[0]
        if np.corrcoef(species_deltas @ water_vec, species_dy)[0, 1] < 0:
            water_vec = -water_vec
    else:
        water_vec = species_deltas.mean(axis=0)
        water_vec /= np.linalg.norm(water_vec) + 1e-8
    from numpy.polynomial.polynomial import polyfit
    proj = X_tr @ water_vec
    coeffs = polyfit(proj, y_tr, 1)
    scale = coeffs[1]
    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 1:
            continue
        for hi_idx in np.where(high_mask)[0]:
            target_dy = extrap_factor * (y_sp[hi_idx] * dy_scale + dy_offset)
            step = target_dy / (scale + 1e-8)
            synth_X.append(X_sp[hi_idx] + step * water_vec)
            synth_y.append(y_sp[hi_idx] + target_dy)
    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        idx = np.linspace(0, len(synth_X) - 1, n_aug, dtype=int)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


def cv_iterpl(X_train, y_train, groups, X_test,
              preprocess=None, lgbm_params=None,
              n_aug=30, extrap=1.5, min_moisture=150,
              dy_scale=0.3, dy_offset=30,
              pl_w=0.5, pl_rounds=2,
              sample_weight_fn=None):
    """Core CV with iterative pseudo-labeling.

    sample_weight_fn: optional callable(X_tr, y_tr, groups_tr) -> weights array
    """
    if preprocess is None:
        preprocess = PP_BIN4
    params = {**(lgbm_params or LGBM_BASE)}
    gkf = GroupKFold(n_splits=5)
    test_preds_prev = None

    for pl_round in range(pl_rounds):
        oof = np.zeros(len(y_train))
        fold_rmses = []
        test_preds_folds = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            g_tr = groups[tr_idx]

            pipe = build_preprocess_pipeline(preprocess)
            pipe.fit(X_tr)

            if n_aug > 0:
                synth_X, synth_y = generate_universal_wdv(
                    X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset
                )
            else:
                synth_X, synth_y = np.empty((0, X_tr.shape[1])), np.empty(0)

            X_tr_t = pipe.transform(X_tr)
            X_va_t = pipe.transform(X_va)
            X_test_t = pipe.transform(X_test)

            if len(synth_X) > 0:
                X_aug = np.vstack([X_tr_t, pipe.transform(synth_X)])
                y_aug = np.concatenate([y_tr, synth_y])
            else:
                X_aug = X_tr_t
                y_aug = y_tr

            # Compute sample weights if provided
            sw_train = None
            if sample_weight_fn is not None:
                sw_base = sample_weight_fn(X_tr, y_tr, g_tr)
                if len(synth_X) > 0:
                    sw_synth = np.ones(len(synth_y))
                    sw_train = np.concatenate([sw_base, sw_synth])
                else:
                    sw_train = sw_base

            if pl_round == 0 and pl_w > 0:
                temp = create_model("lgbm", params)
                if sw_train is not None:
                    temp.fit(X_aug, y_aug, sample_weight=sw_train)
                else:
                    temp.fit(X_aug, y_aug)
                pl_pred = temp.predict(X_test_t)
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, pl_pred])
                w = np.ones(len(y_final))
                if sw_train is not None:
                    w[:len(sw_train)] = sw_train
                w[-len(pl_pred):] *= pl_w
                model = create_model("lgbm", params)
                model.fit(X_final, y_final, sample_weight=w)
            elif pl_round > 0 and test_preds_prev is not None:
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, test_preds_prev])
                w = np.ones(len(y_final))
                if sw_train is not None:
                    w[:len(sw_train)] = sw_train
                w[-len(test_preds_prev):] *= pl_w
                model = create_model("lgbm", params)
                model.fit(X_final, y_final, sample_weight=w)
            else:
                model = create_model("lgbm", params)
                if sw_train is not None:
                    model.fit(X_aug, y_aug, sample_weight=sw_train)
                else:
                    model.fit(X_aug, y_aug)

            oof[va_idx] = model.predict(X_va_t).ravel()
            fold_rmses.append(rmse(y_va, oof[va_idx]))
            test_preds_folds.append(model.predict(X_test_t).ravel())

        test_preds_prev = np.mean(test_preds_folds, axis=0)

    return oof, fold_rmses, test_preds_prev


def cv_iterpl_with_features(X_train, y_train, groups, X_test,
                             extra_feat_fn=None,
                             preprocess=None, lgbm_params=None,
                             n_aug=30, extrap=1.5, min_moisture=150,
                             dy_scale=0.3, dy_offset=30,
                             pl_w=0.5, pl_rounds=2):
    """CV with extra features appended after preprocessing."""
    if preprocess is None:
        preprocess = PP_BIN4
    params = {**(lgbm_params or LGBM_BASE)}
    gkf = GroupKFold(n_splits=5)
    test_preds_prev = None

    for pl_round in range(pl_rounds):
        oof = np.zeros(len(y_train))
        fold_rmses = []
        test_preds_folds = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            g_tr = groups[tr_idx]

            pipe = build_preprocess_pipeline(preprocess)
            pipe.fit(X_tr)

            if n_aug > 0:
                synth_X, synth_y = generate_universal_wdv(
                    X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset
                )
            else:
                synth_X, synth_y = np.empty((0, X_tr.shape[1])), np.empty(0)

            X_tr_t = pipe.transform(X_tr)
            X_va_t = pipe.transform(X_va)
            X_test_t = pipe.transform(X_test)

            # Add extra features
            if extra_feat_fn is not None:
                ef_tr = extra_feat_fn(X_tr)
                ef_va = extra_feat_fn(X_va)
                ef_test = extra_feat_fn(X_test)
                X_tr_t = np.hstack([X_tr_t, ef_tr])
                X_va_t = np.hstack([X_va_t, ef_va])
                X_test_t = np.hstack([X_test_t, ef_test])

            if len(synth_X) > 0:
                synth_t = pipe.transform(synth_X)
                if extra_feat_fn is not None:
                    synth_ef = extra_feat_fn(synth_X)
                    synth_t = np.hstack([synth_t, synth_ef])
                X_aug = np.vstack([X_tr_t, synth_t])
                y_aug = np.concatenate([y_tr, synth_y])
            else:
                X_aug = X_tr_t
                y_aug = y_tr

            if pl_round == 0 and pl_w > 0:
                temp = create_model("lgbm", params)
                temp.fit(X_aug, y_aug)
                pl_pred = temp.predict(X_test_t)
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, pl_pred])
                w = np.ones(len(y_final))
                w[-len(pl_pred):] = pl_w
                model = create_model("lgbm", params)
                model.fit(X_final, y_final, sample_weight=w)
            elif pl_round > 0 and test_preds_prev is not None:
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, test_preds_prev])
                w = np.ones(len(y_final))
                w[-len(test_preds_prev):] = pl_w
                model = create_model("lgbm", params)
                model.fit(X_final, y_final, sample_weight=w)
            else:
                model = create_model("lgbm", params)
                model.fit(X_aug, y_aug)

            oof[va_idx] = model.predict(X_va_t).ravel()
            fold_rmses.append(rmse(y_va, oof[va_idx]))
            test_preds_folds.append(model.predict(X_test_t).ravel())

        test_preds_prev = np.mean(test_preds_folds, axis=0)

    return oof, fold_rmses, test_preds_prev


# ======================================================================
# Feature engineering helpers
# ======================================================================

def compute_peak_integration_features(X_raw):
    """Compute water absorption band integration features.

    NIR water absorption bands:
    - ~1400-1500nm (combination band) → columns ~400-500 (rough)
    - ~1900-2000nm (O-H stretch + bending) → columns ~900-1000 (rough)

    Since we have 1555 columns likely covering ~800-2500nm range,
    we compute features from key spectral regions.
    """
    n = X_raw.shape[1]
    features = []

    # Define regions as fraction of total columns
    regions = [
        (int(n * 0.20), int(n * 0.30)),  # ~region around 1400nm
        (int(n * 0.35), int(n * 0.45)),  # ~region around 1700nm
        (int(n * 0.50), int(n * 0.65)),  # ~region around 1900-2000nm
        (int(n * 0.70), int(n * 0.80)),  # ~region around 2200nm
        (int(n * 0.85), int(n * 0.95)),  # ~region around 2400nm
    ]

    for start, end in regions:
        segment = X_raw[:, start:end]
        # Trapezoid integration (area under curve)
        area = np.sum(segment[:, 1:] + segment[:, :-1], axis=1) * 0.5  # trapezoid
        features.append(area)
        # Peak height (max)
        features.append(segment.max(axis=1))
        # Peak depth (min)
        features.append(segment.min(axis=1))
        # FWHM proxy: std of segment
        features.append(segment.std(axis=1))
        # Slope across region
        features.append(segment[:, -1] - segment[:, 0])

    # Inter-region ratios (key for water content)
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            s1 = X_raw[:, regions[i][0]:regions[i][1]]
            s2 = X_raw[:, regions[j][0]:regions[j][1]]
            ratio = s1.mean(axis=1) / (s2.mean(axis=1) + 1e-8)
            features.append(ratio)

    return np.column_stack(features)


def compute_water_proxy(X_raw):
    """Compute a single 'water proxy' feature based on key bands."""
    n = X_raw.shape[1]
    # Water band around 55-65% of spectrum
    water_band = X_raw[:, int(n * 0.50):int(n * 0.65)]
    # Reference band
    ref_band = X_raw[:, int(n * 0.20):int(n * 0.30)]
    proxy = water_band.mean(axis=1) / (ref_band.mean(axis=1) + 1e-8)
    return proxy


# ======================================================================
# Adversarial Validation for sample weighting
# ======================================================================

def compute_av_weights(X_train, X_test, preprocess=None):
    """Train/test domain classifier, return test-likeness scores for train."""
    if preprocess is None:
        preprocess = PP_BIN4
    pipe = build_preprocess_pipeline(preprocess)
    pipe.fit(X_train)
    X_tr_t = pipe.transform(X_train)
    X_te_t = pipe.transform(X_test)

    X_av = np.vstack([X_tr_t, X_te_t])
    y_av = np.array([0] * len(X_tr_t) + [1] * len(X_te_t))

    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=100, max_depth=3, verbose=-1, n_jobs=-1)

    # Quick CV to get train scores
    from sklearn.model_selection import cross_val_predict
    scores = cross_val_predict(clf, X_av, y_av, cv=3, method="predict_proba")[:, 1]

    # Return scores for train samples only
    train_scores = scores[:len(X_tr_t)]
    # Normalize to [0.5, 2.0] range for sample weights
    weights = 0.5 + 1.5 * (train_scores / (train_scores.max() + 1e-8))
    return weights


# ======================================================================
# Main
# ======================================================================

def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 22: The Final Exorcism")
    print("=" * 70)

    X_train, y_train, groups, X_test, test_ids = load_data()
    all_models = {}

    def log(name, oof, folds, tp):
        score = rmse(y_train, oof)
        fr = [round(f, 1) for f in folds]
        star = " ★★★" if score < 13.5 else (" ★★" if score < 14.0 else (" ★" if score < 14.5 else ""))
        print(f"  {score:.4f} {fr} {name}{star}")
        all_models[name] = {"oof": oof, "test": tp, "rmse": score}

    # ==================================================================
    # BASELINE: Reproduce phase 21d best configs
    # ==================================================================
    print("\n=== BASELINE: Reproduce best configs ===")

    # Config 1: mm170 base
    oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                            min_moisture=170, pl_w=0.5, pl_rounds=2)
    log("baseline_mm170", oof, f, tp)

    # Config 2: mm170 n800lr02
    p2 = {**LGBM_BASE, "n_estimators": 800, "learning_rate": 0.02}
    oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                            lgbm_params=p2, min_moisture=170,
                            pl_w=0.5, pl_rounds=2)
    log("baseline_mm170_n800lr02", oof, f, tp)

    # Config 3: mm170 n1000lr01
    p3 = {**LGBM_BASE, "n_estimators": 1000, "learning_rate": 0.01}
    oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                            lgbm_params=p3, min_moisture=170,
                            pl_w=0.5, pl_rounds=2)
    log("baseline_mm170_n1000lr01", oof, f, tp)

    # Config 4: mm150 base (for diversity)
    oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                            min_moisture=150, pl_w=0.5, pl_rounds=2)
    log("baseline_mm150", oof, f, tp)

    # ==================================================================
    # A: Peak-Integration Features
    # ==================================================================
    print("\n=== A: Peak-Integration Features ===")

    feat_fn = compute_peak_integration_features

    for hp_name, hp_ov in [("base", {}),
                            ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
                            ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01})]:
        params = {**LGBM_BASE, **hp_ov}
        name = f"peak_mm170_{hp_name}"
        oof, f, tp = cv_iterpl_with_features(
            X_train, y_train, groups, X_test,
            extra_feat_fn=feat_fn, lgbm_params=params,
            min_moisture=170, pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

    # Also try with mm150
    oof, f, tp = cv_iterpl_with_features(
        X_train, y_train, groups, X_test,
        extra_feat_fn=feat_fn, min_moisture=150, pl_w=0.5, pl_rounds=2)
    log("peak_mm150_base", oof, f, tp)

    # ==================================================================
    # B: Adversarial Validation Sample Reweighting
    # ==================================================================
    print("\n=== B: AV Sample Reweighting ===")

    print("  Computing AV weights...")
    av_weights = compute_av_weights(X_train, X_test)
    print(f"  AV weights: min={av_weights.min():.3f}, max={av_weights.max():.3f}, "
          f"mean={av_weights.mean():.3f}")

    def make_av_weight_fn(scale=1.0):
        def fn(X_tr, y_tr, g_tr):
            # Map global AV weights to this fold's train indices
            # Since we can't easily map indices, recompute per fold
            # Instead, use a simpler approach: weight by moisture level
            return np.ones(len(y_tr))  # placeholder
        return fn

    # Approach 1: Full AV weights (need to handle fold mapping)
    # Store global weights and use fold indexing
    class AVWeightFn:
        def __init__(self, global_weights, multiplier=1.0):
            self.global_weights = global_weights
            self.multiplier = multiplier
            self._fold_idx = None

        def set_fold_idx(self, idx):
            self._fold_idx = idx

    # Simpler approach: compute AV weights globally and pass via modified cv
    def cv_with_av_weights(X_train, y_train, groups, X_test,
                           av_weights, weight_power=1.0,
                           preprocess=None, lgbm_params=None,
                           n_aug=30, extrap=1.5, min_moisture=150,
                           dy_scale=0.3, dy_offset=30,
                           pl_w=0.5, pl_rounds=2):
        if preprocess is None:
            preprocess = PP_BIN4
        params = {**(lgbm_params or LGBM_BASE)}
        gkf = GroupKFold(n_splits=5)
        test_preds_prev = None

        for pl_round in range(pl_rounds):
            oof = np.zeros(len(y_train))
            fold_rmses = []
            test_preds_folds = []

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                g_tr = groups[tr_idx]

                # AV weights for this fold's training data
                fold_av = av_weights[tr_idx] ** weight_power

                pipe = build_preprocess_pipeline(preprocess)
                pipe.fit(X_tr)

                if n_aug > 0:
                    synth_X, synth_y = generate_universal_wdv(
                        X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset
                    )
                else:
                    synth_X, synth_y = np.empty((0, X_tr.shape[1])), np.empty(0)

                X_tr_t = pipe.transform(X_tr)
                X_va_t = pipe.transform(X_va)
                X_test_t = pipe.transform(X_test)

                if len(synth_X) > 0:
                    X_aug = np.vstack([X_tr_t, pipe.transform(synth_X)])
                    y_aug = np.concatenate([y_tr, synth_y])
                    w_aug = np.concatenate([fold_av, np.ones(len(synth_y))])
                else:
                    X_aug = X_tr_t
                    y_aug = y_tr
                    w_aug = fold_av

                if pl_round == 0 and pl_w > 0:
                    temp = create_model("lgbm", params)
                    temp.fit(X_aug, y_aug, sample_weight=w_aug)
                    pl_pred = temp.predict(X_test_t)
                    X_final = np.vstack([X_aug, X_test_t])
                    y_final = np.concatenate([y_aug, pl_pred])
                    w = np.concatenate([w_aug, np.full(len(pl_pred), pl_w)])
                    model = create_model("lgbm", params)
                    model.fit(X_final, y_final, sample_weight=w)
                elif pl_round > 0 and test_preds_prev is not None:
                    X_final = np.vstack([X_aug, X_test_t])
                    y_final = np.concatenate([y_aug, test_preds_prev])
                    w = np.concatenate([w_aug, np.full(len(test_preds_prev), pl_w)])
                    model = create_model("lgbm", params)
                    model.fit(X_final, y_final, sample_weight=w)
                else:
                    model = create_model("lgbm", params)
                    model.fit(X_aug, y_aug, sample_weight=w_aug)

                oof[va_idx] = model.predict(X_va_t).ravel()
                fold_rmses.append(rmse(y_va, oof[va_idx]))
                test_preds_folds.append(model.predict(X_test_t).ravel())

            test_preds_prev = np.mean(test_preds_folds, axis=0)

        return oof, fold_rmses, test_preds_prev

    for wp in [0.5, 1.0, 1.5, 2.0]:
        oof, f, tp = cv_with_av_weights(
            X_train, y_train, groups, X_test,
            av_weights, weight_power=wp,
            min_moisture=170, pl_w=0.5, pl_rounds=2)
        log(f"av_wp{wp}_mm170", oof, f, tp)

    # ==================================================================
    # C: Multi-WDV ensemble branches
    # ==================================================================
    print("\n=== C: Multi-WDV ensemble ===")

    wdv_configs = [
        ("mm150_f1.0", 150, 1.0, 30),
        ("mm150_f1.5", 150, 1.5, 30),
        ("mm150_f2.0", 150, 2.0, 30),
        ("mm170_f1.0", 170, 1.0, 30),
        ("mm170_f2.0", 170, 2.0, 30),
        ("mm170_f2.5", 170, 2.5, 30),
        ("mm170_n50_f1.5", 170, 1.5, 50),
        ("mm170_n20_f1.5", 170, 1.5, 20),
        ("mm190_f1.5", 190, 1.5, 30),
        ("mm190_f2.0", 190, 2.0, 30),
        ("mm200_f2.0", 200, 2.0, 30),
        ("mm200_f3.0", 200, 3.0, 30),
    ]

    for name, mm, ef, na in wdv_configs:
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                n_aug=na, extrap=ef, min_moisture=mm,
                                pl_w=0.5, pl_rounds=2)
        log(f"wdv_{name}", oof, f, tp)

    # Also with n800lr02
    for name, mm, ef, na in [("mm170_f2.0", 170, 2.0, 30),
                              ("mm170_f2.5", 170, 2.5, 30),
                              ("mm190_f2.0", 190, 2.0, 30)]:
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                lgbm_params=p2, n_aug=na, extrap=ef,
                                min_moisture=mm, pl_w=0.5, pl_rounds=2)
        log(f"wdv_{name}_n800lr02", oof, f, tp)

    # ==================================================================
    # D: Species 15 gating (threshold-based conditional model)
    # ==================================================================
    print("\n=== D: Species 15 gating ===")

    def cv_gated(X_train, y_train, groups, X_test,
                 threshold=100, extreme_params=None,
                 preprocess=None, lgbm_params=None,
                 n_aug=30, extrap=1.5, min_moisture=170,
                 dy_scale=0.3, dy_offset=30,
                 pl_w=0.5, pl_rounds=2):
        """Two-model gating: normal model + extreme model for high predictions."""
        if preprocess is None:
            preprocess = PP_BIN4
        params_normal = {**(lgbm_params or LGBM_BASE)}
        params_extreme = {**(extreme_params or {
            **LGBM_BASE,
            "min_child_samples": 5,
            "n_estimators": 600,
            "learning_rate": 0.03,
        })}
        gkf = GroupKFold(n_splits=5)

        oof = np.zeros(len(y_train))
        fold_rmses = []
        test_preds_folds = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            g_tr = groups[tr_idx]

            pipe = build_preprocess_pipeline(preprocess)
            pipe.fit(X_tr)

            synth_X, synth_y = generate_universal_wdv(
                X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset
            )

            X_tr_t = pipe.transform(X_tr)
            X_va_t = pipe.transform(X_va)
            X_test_t = pipe.transform(X_test)

            if len(synth_X) > 0:
                X_aug = np.vstack([X_tr_t, pipe.transform(synth_X)])
                y_aug = np.concatenate([y_tr, synth_y])
            else:
                X_aug = X_tr_t
                y_aug = y_tr

            # Normal model with PL
            temp = create_model("lgbm", params_normal)
            temp.fit(X_aug, y_aug)
            pl_pred = temp.predict(X_test_t)
            X_final = np.vstack([X_aug, X_test_t])
            y_final = np.concatenate([y_aug, pl_pred])
            w = np.ones(len(y_final))
            w[-len(pl_pred):] = pl_w

            model_normal = create_model("lgbm", params_normal)
            model_normal.fit(X_final, y_final, sample_weight=w)
            pred_normal_va = model_normal.predict(X_va_t).ravel()
            pred_normal_test = model_normal.predict(X_test_t).ravel()

            # Extreme model: trained with more WDV + aggressive extrapolation
            synth_X2, synth_y2 = generate_universal_wdv(
                X_tr, y_tr, g_tr, n_aug=50, extrap_factor=3.0,
                min_moisture=min_moisture, dy_scale=0.3, dy_offset=30
            )
            if len(synth_X2) > 0:
                X_aug2 = np.vstack([X_tr_t, pipe.transform(synth_X2)])
                y_aug2 = np.concatenate([y_tr, synth_y2])
            else:
                X_aug2 = X_tr_t
                y_aug2 = y_tr

            model_extreme = create_model("lgbm", params_extreme)
            model_extreme.fit(X_aug2, y_aug2)
            pred_extreme_va = model_extreme.predict(X_va_t).ravel()
            pred_extreme_test = model_extreme.predict(X_test_t).ravel()

            # Gate: use extreme model for high predictions
            pred_va = pred_normal_va.copy()
            high_mask = pred_normal_va > threshold
            if high_mask.sum() > 0:
                pred_va[high_mask] = pred_extreme_va[high_mask]

            pred_test = pred_normal_test.copy()
            high_mask_t = pred_normal_test > threshold
            if high_mask_t.sum() > 0:
                pred_test[high_mask_t] = pred_extreme_test[high_mask_t]

            oof[va_idx] = pred_va
            fold_rmses.append(rmse(y_va, pred_va))
            test_preds_folds.append(pred_test)

        test_preds = np.mean(test_preds_folds, axis=0)
        return oof, fold_rmses, test_preds

    for threshold in [80, 100, 120, 140]:
        oof, f, tp = cv_gated(X_train, y_train, groups, X_test,
                               threshold=threshold, min_moisture=170)
        log(f"gated_t{threshold}_mm170", oof, f, tp)

    # Softer gating: weighted blend based on prediction magnitude
    def cv_soft_gate(X_train, y_train, groups, X_test,
                     blend_start=80, blend_end=160,
                     preprocess=None, lgbm_params=None,
                     n_aug=30, extrap=1.5, min_moisture=170,
                     pl_w=0.5, pl_rounds=2):
        if preprocess is None:
            preprocess = PP_BIN4
        params = {**(lgbm_params or LGBM_BASE)}
        gkf = GroupKFold(n_splits=5)

        oof = np.zeros(len(y_train))
        fold_rmses = []
        test_preds_folds = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            g_tr = groups[tr_idx]

            pipe = build_preprocess_pipeline(preprocess)
            pipe.fit(X_tr)

            X_tr_t = pipe.transform(X_tr)
            X_va_t = pipe.transform(X_va)
            X_test_t = pipe.transform(X_test)

            # Normal model
            synth_X, synth_y = generate_universal_wdv(
                X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, 0.3, 30)
            if len(synth_X) > 0:
                X_aug = np.vstack([X_tr_t, pipe.transform(synth_X)])
                y_aug = np.concatenate([y_tr, synth_y])
            else:
                X_aug, y_aug = X_tr_t, y_tr

            temp = create_model("lgbm", params)
            temp.fit(X_aug, y_aug)
            pl_pred = temp.predict(X_test_t)
            X_final = np.vstack([X_aug, X_test_t])
            y_final = np.concatenate([y_aug, pl_pred])
            w = np.ones(len(y_final))
            w[-len(pl_pred):] = pl_w
            model_normal = create_model("lgbm", params)
            model_normal.fit(X_final, y_final, sample_weight=w)

            # Extreme model
            synth_X2, synth_y2 = generate_universal_wdv(
                X_tr, y_tr, g_tr, n_aug=50, extrap_factor=3.0,
                min_moisture=min_moisture, dy_scale=0.3, dy_offset=30)
            if len(synth_X2) > 0:
                X_aug2 = np.vstack([X_tr_t, pipe.transform(synth_X2)])
                y_aug2 = np.concatenate([y_tr, synth_y2])
            else:
                X_aug2, y_aug2 = X_tr_t, y_tr
            model_extreme = create_model("lgbm", {
                **params, "min_child_samples": 5,
                "n_estimators": 600, "learning_rate": 0.03})
            model_extreme.fit(X_aug2, y_aug2)

            for data_t, store, idx in [
                (X_va_t, "oof", va_idx),
                (X_test_t, "test", None)
            ]:
                pred_n = model_normal.predict(data_t).ravel()
                pred_e = model_extreme.predict(data_t).ravel()
                # Soft blend
                alpha = np.clip((pred_n - blend_start) / (blend_end - blend_start), 0, 1)
                pred = (1 - alpha) * pred_n + alpha * pred_e
                if store == "oof":
                    oof[idx] = pred
                    fold_rmses.append(rmse(y_va, pred))
                else:
                    test_preds_folds.append(pred)

        test_preds = np.mean(test_preds_folds, axis=0)
        return oof, fold_rmses, test_preds

    for bs, be in [(60, 120), (80, 140), (80, 160), (100, 180)]:
        oof, f, tp = cv_soft_gate(X_train, y_train, groups, X_test,
                                   blend_start=bs, blend_end=be)
        log(f"softgate_{bs}_{be}", oof, f, tp)

    # ==================================================================
    # E: Extra LGBM variants for diversity
    # ==================================================================
    print("\n=== E: Extra LGBM variants ===")

    diversity_configs = [
        ("d7l30", {"max_depth": 7, "num_leaves": 30}),
        ("d7l30_n600", {"max_depth": 7, "num_leaves": 30, "n_estimators": 600, "learning_rate": 0.03}),
        ("d3l10_n1500", {"max_depth": 3, "num_leaves": 10, "n_estimators": 1500, "learning_rate": 0.005}),
        ("ss08_cs08", {"subsample": 0.8, "colsample_bytree": 0.8}),
        ("ss05_cs05", {"subsample": 0.5, "colsample_bytree": 0.5}),
        ("mcs10", {"min_child_samples": 10}),
        ("mcs5_n600", {"min_child_samples": 5, "n_estimators": 600, "learning_rate": 0.03}),
        ("ra1_rl5", {"reg_alpha": 1.0, "reg_lambda": 5.0}),
        ("n2000lr003", {"n_estimators": 2000, "learning_rate": 0.003}),
    ]

    for hp_name, hp_ov in diversity_configs:
        params = {**LGBM_BASE, **hp_ov}
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                lgbm_params=params, min_moisture=170,
                                pl_w=0.5, pl_rounds=2)
        log(f"div_{hp_name}_mm170", oof, f, tp)

    # ==================================================================
    # F: Moisture-weighted training (upweight high moisture samples)
    # ==================================================================
    print("\n=== F: Moisture-weighted training ===")

    def moisture_weight_fn(y_tr, power=1.0, base=1.0):
        """Weight samples by moisture level to focus on high moisture."""
        w = base + (y_tr / y_tr.max()) ** power
        return w

    def cv_moisture_weighted(X_train, y_train, groups, X_test,
                              weight_power=1.0, weight_base=1.0,
                              preprocess=None, lgbm_params=None,
                              n_aug=30, extrap=1.5, min_moisture=170,
                              pl_w=0.5, pl_rounds=2):
        if preprocess is None:
            preprocess = PP_BIN4
        params = {**(lgbm_params or LGBM_BASE)}
        gkf = GroupKFold(n_splits=5)
        test_preds_prev = None

        for pl_round in range(pl_rounds):
            oof = np.zeros(len(y_train))
            fold_rmses = []
            test_preds_folds = []

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                g_tr = groups[tr_idx]

                mw = moisture_weight_fn(y_tr, weight_power, weight_base)

                pipe = build_preprocess_pipeline(preprocess)
                pipe.fit(X_tr)

                synth_X, synth_y = generate_universal_wdv(
                    X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, 0.3, 30)

                X_tr_t = pipe.transform(X_tr)
                X_va_t = pipe.transform(X_va)
                X_test_t = pipe.transform(X_test)

                if len(synth_X) > 0:
                    X_aug = np.vstack([X_tr_t, pipe.transform(synth_X)])
                    y_aug = np.concatenate([y_tr, synth_y])
                    mw_aug = np.concatenate([mw, np.ones(len(synth_y)) * 2.0])
                else:
                    X_aug, y_aug, mw_aug = X_tr_t, y_tr, mw

                if pl_round == 0 and pl_w > 0:
                    temp = create_model("lgbm", params)
                    temp.fit(X_aug, y_aug, sample_weight=mw_aug)
                    pl_pred = temp.predict(X_test_t)
                    X_final = np.vstack([X_aug, X_test_t])
                    y_final = np.concatenate([y_aug, pl_pred])
                    w = np.concatenate([mw_aug, np.full(len(pl_pred), pl_w)])
                    model = create_model("lgbm", params)
                    model.fit(X_final, y_final, sample_weight=w)
                elif pl_round > 0 and test_preds_prev is not None:
                    X_final = np.vstack([X_aug, X_test_t])
                    y_final = np.concatenate([y_aug, test_preds_prev])
                    w = np.concatenate([mw_aug, np.full(len(test_preds_prev), pl_w)])
                    model = create_model("lgbm", params)
                    model.fit(X_final, y_final, sample_weight=w)
                else:
                    model = create_model("lgbm", params)
                    model.fit(X_aug, y_aug, sample_weight=mw_aug)

                oof[va_idx] = model.predict(X_va_t).ravel()
                fold_rmses.append(rmse(y_va, oof[va_idx]))
                test_preds_folds.append(model.predict(X_test_t).ravel())

            test_preds_prev = np.mean(test_preds_folds, axis=0)

        return oof, fold_rmses, test_preds_prev

    for wp, wb in [(0.5, 1.0), (1.0, 1.0), (2.0, 0.5), (1.5, 0.5)]:
        oof, f, tp = cv_moisture_weighted(
            X_train, y_train, groups, X_test,
            weight_power=wp, weight_base=wb,
            min_moisture=170, pl_w=0.5, pl_rounds=2)
        log(f"mw_p{wp}_b{wb}_mm170", oof, f, tp)

    # ==================================================================
    # G: WDV dy_scale/dy_offset sweep
    # ==================================================================
    print("\n=== G: WDV dy_scale/dy_offset sweep ===")

    for ds, do in [(0.2, 20), (0.4, 40), (0.5, 50), (0.3, 50), (0.2, 40)]:
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                min_moisture=170, dy_scale=ds, dy_offset=do,
                                pl_w=0.5, pl_rounds=2)
        log(f"dy_s{ds}_o{do}_mm170", oof, f, tp)

    # ==================================================================
    # H: PL weight and rounds sweep
    # ==================================================================
    print("\n=== H: PL weight/rounds sweep ===")

    for pw in [0.3, 0.7, 1.0]:
        for pr in [2, 3]:
            if pw == 0.5 and pr == 2:
                continue
            oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                    min_moisture=170, pl_w=pw, pl_rounds=pr)
            log(f"pl_w{pw}_r{pr}_mm170", oof, f, tp)

    # ==================================================================
    # I: bin3 and bin5 with mm170 for diversity
    # ==================================================================
    print("\n=== I: bin3/bin5 diversity ===")

    for bs in [3, 5]:
        pp = [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": bs},
            {"name": "standard_scaler"},
        ]
        for hp_name, hp_ov in [("base", {}), ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02})]:
            params = {**LGBM_BASE, **hp_ov}
            oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                    preprocess=pp, lgbm_params=params,
                                    min_moisture=170, pl_w=0.5, pl_rounds=2)
            log(f"bin{bs}_mm170_{hp_name}", oof, f, tp)

    # ==================================================================
    # J: XGBoost diversity branch
    # ==================================================================
    print("\n=== J: XGBoost diversity ===")

    try:
        xgb_params = {
            "n_estimators": 600, "max_depth": 5, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbosity": 0, "n_jobs": -1,
        }

        def cv_xgb(X_train, y_train, groups, X_test,
                    preprocess=None, xgb_params=None,
                    n_aug=30, extrap=1.5, min_moisture=170,
                    pl_w=0.5, pl_rounds=2):
            if preprocess is None:
                preprocess = PP_BIN4
            params = xgb_params or {}
            gkf = GroupKFold(n_splits=5)
            test_preds_prev = None

            for pl_round in range(pl_rounds):
                oof = np.zeros(len(y_train))
                fold_rmses = []
                test_preds_folds = []

                for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                    g_tr = groups[tr_idx]

                    pipe = build_preprocess_pipeline(preprocess)
                    pipe.fit(X_tr)

                    synth_X, synth_y = generate_universal_wdv(
                        X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, 0.3, 30)

                    X_tr_t = pipe.transform(X_tr)
                    X_va_t = pipe.transform(X_va)
                    X_test_t = pipe.transform(X_test)

                    if len(synth_X) > 0:
                        X_aug = np.vstack([X_tr_t, pipe.transform(synth_X)])
                        y_aug = np.concatenate([y_tr, synth_y])
                    else:
                        X_aug, y_aug = X_tr_t, y_tr

                    if pl_round == 0 and pl_w > 0:
                        temp = create_model("xgboost", params)
                        temp.fit(X_aug, y_aug)
                        pl_pred = temp.predict(X_test_t)
                        X_final = np.vstack([X_aug, X_test_t])
                        y_final = np.concatenate([y_aug, pl_pred])
                        w = np.ones(len(y_final))
                        w[-len(pl_pred):] = pl_w
                        model = create_model("xgboost", params)
                        model.fit(X_final, y_final, sample_weight=w)
                    elif pl_round > 0 and test_preds_prev is not None:
                        X_final = np.vstack([X_aug, X_test_t])
                        y_final = np.concatenate([y_aug, test_preds_prev])
                        w = np.ones(len(y_final))
                        w[-len(test_preds_prev):] = pl_w
                        model = create_model("xgboost", params)
                        model.fit(X_final, y_final, sample_weight=w)
                    else:
                        model = create_model("xgboost", params)
                        model.fit(X_aug, y_aug)

                    oof[va_idx] = model.predict(X_va_t).ravel()
                    fold_rmses.append(rmse(y_va, oof[va_idx]))
                    test_preds_folds.append(model.predict(X_test_t).ravel())

                test_preds_prev = np.mean(test_preds_folds, axis=0)

            return oof, fold_rmses, test_preds_prev

        oof, f, tp = cv_xgb(X_train, y_train, groups, X_test,
                              xgb_params=xgb_params, min_moisture=170)
        log("xgb_mm170", oof, f, tp)

        xgb2 = {**xgb_params, "n_estimators": 1000, "learning_rate": 0.01}
        oof, f, tp = cv_xgb(X_train, y_train, groups, X_test,
                              xgb_params=xgb2, min_moisture=170)
        log("xgb_n1000lr01_mm170", oof, f, tp)
    except Exception as e:
        print(f"  XGBoost failed: {e}")

    # ==================================================================
    # MEGA ENSEMBLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("MEGA ENSEMBLE")
    print("=" * 70)

    names = list(all_models.keys())
    oofs = np.column_stack([all_models[n]["oof"] for n in names])
    tests = np.column_stack([all_models[n]["test"] for n in names])
    rmses_list = [all_models[n]["rmse"] for n in names]

    print(f"\n  {len(names)} models. Top 15:")
    for n, r in sorted(zip(names, rmses_list), key=lambda x: x[1])[:15]:
        print(f"    {r:.4f}  {n}")

    # Greedy forward selection
    order = sorted(range(len(names)), key=lambda i: rmses_list[i])
    selected = [order[0]]
    for _ in range(min(40, len(order) - 1)):
        cur_avg = oofs[:, selected].mean(axis=1)
        cur_s = rmse(y_train, cur_avg)
        best_s, best_i = cur_s, -1
        for i in order:
            if i in selected:
                continue
            new_avg = (cur_avg * len(selected) + oofs[:, i]) / (len(selected) + 1)
            s = rmse(y_train, new_avg)
            if s < best_s - 0.001:
                best_s = s
                best_i = i
        if best_i >= 0:
            selected.append(best_i)
            print(f"    +{len(selected)}: {names[best_i][:55]:55s} ens={best_s:.4f}")
        else:
            break

    greedy_avg = oofs[:, selected].mean(axis=1)
    greedy_test = tests[:, selected].mean(axis=1)
    greedy_s = rmse(y_train, greedy_avg)
    print(f"  Greedy ({len(selected)} models): {greedy_s:.4f}")
    all_models["greedy_ens"] = {"oof": greedy_avg, "test": greedy_test, "rmse": greedy_s}

    # NM weight optimization
    sub_oofs = oofs[:, selected]
    sub_tests = tests[:, selected]
    ns = len(selected)

    def obj(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub_oofs * wn).sum(axis=1))

    best_opt = 999
    best_w = np.ones(ns) / ns
    for trial in range(500):
        w0 = np.random.dirichlet(np.ones(ns) * 2)
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-10})
        if res.fun < best_opt:
            best_opt = res.fun
            w = np.abs(res.x)
            best_w = w / w.sum()

    opt_oof = (sub_oofs * best_w).sum(axis=1)
    opt_test = (sub_tests * best_w).sum(axis=1)
    print(f"  NM optimized: {best_opt:.4f}")
    for i, idx in enumerate(selected):
        if best_w[i] > 0.01:
            print(f"    {best_w[i]:.3f}  {names[idx]}")
    all_models["nm_opt"] = {"oof": opt_oof, "test": opt_test, "rmse": best_opt}

    # ==================================================================
    # CONDITIONAL STRETCH (the crown jewel)
    # ==================================================================
    print("\n" + "=" * 70)
    print("CONDITIONAL STRETCH")
    print("=" * 70)

    top_candidates = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:20]

    for base_name, base_data in top_candidates:
        if "stretch" in base_name or "cstretch" in base_name:
            continue
        base_oof = base_data["oof"]
        base_tp = base_data["test"]
        base_score = base_data["rmse"]
        best_stretch_score = base_score

        # 1. Standard percentile stretch (reproduce + extend)
        for pct in [90, 93, 95, 97, 99]:
            threshold = np.percentile(base_oof, pct)
            for sf in [1.02, 1.05, 1.08, 1.1, 1.15, 1.2, 1.3, 1.5, 1.8, 2.0]:
                oof_s = base_oof.copy()
                mask = oof_s > threshold
                oof_s[mask] = threshold + (oof_s[mask] - threshold) * sf
                s = rmse(y_train, oof_s)
                if s < best_stretch_score - 0.005:
                    best_stretch_score = s
                    tp_s = base_tp.copy()
                    mask_t = tp_s > threshold
                    tp_s[mask_t] = threshold + (tp_s[mask_t] - threshold) * sf
                    sname = f"stretch_{base_name}_p{pct}_s{sf}"
                    all_models[sname] = {"oof": oof_s, "test": tp_s, "rmse": s}

        # 2. Absolute threshold stretch (not percentile-based)
        for abs_t in [100, 120, 140, 160, 180]:
            for sf in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
                oof_s = base_oof.copy()
                mask = oof_s > abs_t
                oof_s[mask] = abs_t + (oof_s[mask] - abs_t) * sf
                s = rmse(y_train, oof_s)
                if s < best_stretch_score - 0.005:
                    best_stretch_score = s
                    tp_s = base_tp.copy()
                    mask_t = tp_s > abs_t
                    tp_s[mask_t] = abs_t + (tp_s[mask_t] - abs_t) * sf
                    sname = f"cstretch_{base_name}_t{abs_t}_s{sf}"
                    all_models[sname] = {"oof": oof_s, "test": tp_s, "rmse": s}

        # 3. Piecewise nonlinear stretch
        for pct in [93, 95, 97]:
            t1 = np.percentile(base_oof, pct)
            t2 = np.percentile(base_oof, 99) if pct < 99 else t1 * 1.3
            for sf1 in [1.05, 1.1, 1.15]:
                for sf2 in [1.3, 1.5, 2.0, 2.5]:
                    if sf2 <= sf1:
                        continue
                    oof_s = base_oof.copy()
                    # Zone 1: pct to p99 → mild stretch
                    mask1 = (oof_s > t1) & (oof_s <= t2)
                    oof_s[mask1] = t1 + (oof_s[mask1] - t1) * sf1
                    # Zone 2: above p99 → aggressive stretch
                    mask2 = oof_s > t2
                    new_t2 = t1 + (t2 - t1) * sf1
                    oof_s[mask2] = new_t2 + (base_oof[mask2] - t2) * sf2
                    s = rmse(y_train, oof_s)
                    if s < best_stretch_score - 0.005:
                        best_stretch_score = s
                        tp_s = base_tp.copy()
                        mask1_t = (tp_s > t1) & (tp_s <= t2)
                        tp_s[mask1_t] = t1 + (tp_s[mask1_t] - t1) * sf1
                        mask2_t = tp_s > t2
                        tp_s[mask2_t] = new_t2 + (base_tp[mask2_t] - t2) * sf2
                        sname = f"pw_{base_name}_p{pct}_s{sf1}_{sf2}"
                        all_models[sname] = {"oof": oof_s, "test": tp_s, "rmse": s}

        if best_stretch_score < base_score - 0.01:
            print(f"  {base_name}: {base_score:.4f} → {best_stretch_score:.4f}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 22 FINAL SUMMARY")
    print("=" * 70)

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:50]:
        star = " ★★★" if data["rmse"] < 13.5 else (" ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else ""))
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 21d best: 14.12")
    improvement = 14.12 - best_data["rmse"]
    print(f"  Improvement: {improvement:+.4f}")

    # Save
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        sub = pd.DataFrame({
            "sample number": test_ids.values,
            "含水率": data["test"]
        })
        path = submissions_dir / f"submission_phase22_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    oof_dir = Path("runs") / f"phase22_{ts}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked},
        "phase": "22",
        "description": "The Final Exorcism",
    }
    with open(oof_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, data in all_models.items():
        np.save(oof_dir / f"oof_{name}.npy", data["oof"])
        np.save(oof_dir / f"test_{name}.npy", data["test"])

    print(f"\n  Artifacts: {oof_dir}")
    print(f"  Total models: {len(all_models)}")


if __name__ == "__main__":
    main()
