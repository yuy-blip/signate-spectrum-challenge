#!/usr/bin/env python
"""Phase 23: The Weight of Truth — 全方位攻撃.

Current best: 13.87 (Phase 22 NM-optimized ensemble)

Strategies from Claude / ChatGPT / Gemini:
  A. Target transformation (log1p, sqrt, boxcox) — Claude本命
  B. Moisture weighting fine-grained grid (power/base/threshold) — ChatGPT本命
  C. Continuous weighting function — Gemini案
  D. Ultra slow learner + moisture weighting fusion
  E. Multi-factor WDV (multiple factors blended)
  F. Final mega ensemble with NM optimization
"""

from __future__ import annotations

import sys
import datetime
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

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


# ======================================================================
# Core CV function with all options
# ======================================================================

def cv_full(X_train, y_train, groups, X_test,
            preprocess=None, lgbm_params=None,
            n_aug=30, extrap=1.5, min_moisture=170,
            dy_scale=0.3, dy_offset=30,
            pl_w=0.5, pl_rounds=2,
            sample_weight_fn=None,
            target_transform=None,
            target_transform_lambda=0.5):
    """Core CV with all options: sample weights, target transform, etc."""
    if preprocess is None:
        preprocess = PP_BIN4
    params = {**(lgbm_params or LGBM_BASE)}
    gkf = GroupKFold(n_splits=5)
    test_preds_prev = None

    # Target transform functions
    def fwd(y):
        if target_transform == "log1p":
            return np.log1p(np.clip(y, 0, None))
        elif target_transform == "log":
            return np.log(np.clip(y, 1e-6, None))
        elif target_transform == "sqrt":
            return np.sqrt(np.clip(y, 0, None))
        elif target_transform == "boxcox":
            lam = target_transform_lambda
            if abs(lam) < 1e-6:
                return np.log(np.clip(y, 1e-6, None))
            return (np.clip(y, 1e-6, None) ** lam - 1) / lam
        return y

    def inv(y):
        if target_transform == "log1p":
            return np.expm1(y)
        elif target_transform == "log":
            return np.exp(y)
        elif target_transform == "sqrt":
            return np.clip(y, 0, None) ** 2
        elif target_transform == "boxcox":
            lam = target_transform_lambda
            if abs(lam) < 1e-6:
                return np.exp(y)
            return np.clip(lam * y + 1, 1e-6, None) ** (1 / lam)
        return y

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
                    X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset)
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

            # Compute sample weights
            sw_train = None
            if sample_weight_fn is not None:
                sw_base = sample_weight_fn(y_tr, g_tr)
                if len(synth_X) > 0:
                    sw_synth = sample_weight_fn(synth_y, None)
                    sw_train = np.concatenate([sw_base, sw_synth])
                else:
                    sw_train = sw_base

            # Apply target transform
            y_aug_t = fwd(y_aug)

            if pl_round == 0 and pl_w > 0:
                temp = create_model("lgbm", params)
                if sw_train is not None:
                    temp.fit(X_aug, y_aug_t, sample_weight=sw_train)
                else:
                    temp.fit(X_aug, y_aug_t)
                pl_pred_t = temp.predict(X_test_t)
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug_t, pl_pred_t])
                w = np.ones(len(y_final))
                if sw_train is not None:
                    w[:len(sw_train)] = sw_train
                w[-len(pl_pred_t):] *= pl_w
                model = create_model("lgbm", params)
                model.fit(X_final, y_final, sample_weight=w)
            elif pl_round > 0 and test_preds_prev is not None:
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug_t, fwd(test_preds_prev)])
                w = np.ones(len(y_final))
                if sw_train is not None:
                    w[:len(sw_train)] = sw_train
                w[-len(test_preds_prev):] *= pl_w
                model = create_model("lgbm", params)
                model.fit(X_final, y_final, sample_weight=w)
            else:
                model = create_model("lgbm", params)
                if sw_train is not None:
                    model.fit(X_aug, y_aug_t, sample_weight=sw_train)
                else:
                    model.fit(X_aug, y_aug_t)

            # Predict and inverse transform
            pred_t = model.predict(X_va_t).ravel()
            oof[va_idx] = inv(pred_t)
            fold_rmses.append(rmse(y_va, oof[va_idx]))

            test_pred_t = model.predict(X_test_t).ravel()
            test_preds_folds.append(inv(test_pred_t))

        test_preds_prev = np.mean(test_preds_folds, axis=0)

    return oof, fold_rmses, test_preds_prev


# ======================================================================
# Main
# ======================================================================

def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 23: The Weight of Truth — 全方位攻撃")
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
    # BASELINE: Reproduce Phase 22 best configs
    # ==================================================================
    print("\n=== BASELINE ===")

    oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                          min_moisture=170, pl_w=0.5, pl_rounds=2)
    log("baseline_mm170", oof, f, tp)

    p_n800 = {**LGBM_BASE, "n_estimators": 800, "learning_rate": 0.02}
    oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                          lgbm_params=p_n800, min_moisture=170)
    log("baseline_n800lr02", oof, f, tp)

    # Shallow tree (diversity)
    p_shallow = {**LGBM_BASE, "max_depth": 3, "num_leaves": 10,
                  "n_estimators": 1500, "learning_rate": 0.005}
    oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                          lgbm_params=p_shallow, min_moisture=170)
    log("baseline_shallow", oof, f, tp)

    # ==================================================================
    # A: TARGET TRANSFORMATION — Claude本命
    # ==================================================================
    print("\n=== A: TARGET TRANSFORMATION ===")

    for tt_name, tt, lam in [
        ("log1p", "log1p", None),
        ("log", "log", None),
        ("sqrt", "sqrt", None),
        ("boxcox_0.25", "boxcox", 0.25),
        ("boxcox_0.5", "boxcox", 0.5),
        ("boxcox_0.75", "boxcox", 0.75),
        ("boxcox_0.1", "boxcox", 0.1),
        ("boxcox_0.3", "boxcox", 0.3),
    ]:
        for hp_name, hp_ov in [("base", {}),
                                ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
                                ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01})]:
            params = {**LGBM_BASE, **hp_ov}
            name = f"tt_{tt_name}_{hp_name}"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  lgbm_params=params, min_moisture=170,
                                  target_transform=tt,
                                  target_transform_lambda=lam or 0.5)
            log(name, oof, f, tp)

    # Target transform + moisture weighting combo
    print("\n=== A2: Target Transform + Moisture Weighting ===")

    def mw_fn_factory(power, base, threshold=0):
        def fn(y, g):
            w = np.ones(len(y))
            if threshold > 0:
                mask = y > threshold
                w[mask] = base + ((y[mask] - threshold) / (y.max() - threshold + 1e-8)) ** power * (2.0 - base)
            else:
                w = base + (y / (y.max() + 1e-8)) ** power
            return w
        return fn

    for tt_name, tt, lam in [("log1p", "log1p", None), ("sqrt", "sqrt", None),
                              ("boxcox_0.25", "boxcox", 0.25), ("boxcox_0.5", "boxcox", 0.5)]:
        for wp, wb in [(1.5, 0.5), (2.0, 0.5)]:
            name = f"tt_{tt_name}_mw_p{wp}_b{wb}"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  min_moisture=170,
                                  target_transform=tt,
                                  target_transform_lambda=lam or 0.5,
                                  sample_weight_fn=mw_fn_factory(wp, wb))
            log(name, oof, f, tp)

    # ==================================================================
    # B: MOISTURE WEIGHTING FINE-GRAINED GRID — ChatGPT本命
    # ==================================================================
    print("\n=== B: MOISTURE WEIGHTING FINE-GRAINED GRID ===")

    # B1: power/base sweep (no threshold)
    for wp in [1.0, 1.3, 1.5, 1.7, 2.0, 2.2, 2.5]:
        for wb in [0.2, 0.3, 0.5, 0.7, 1.0]:
            name = f"mw_p{wp}_b{wb}"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  min_moisture=170,
                                  sample_weight_fn=mw_fn_factory(wp, wb))
            log(name, oof, f, tp)

    # B2: With threshold (ChatGPT's key insight)
    print("\n=== B2: MW with threshold ===")
    for threshold in [100, 120, 140, 160, 180]:
        for wp in [1.0, 1.5, 2.0, 2.5]:
            for wb in [0.3, 0.5, 0.7]:
                name = f"mw_t{threshold}_p{wp}_b{wb}"
                oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                      min_moisture=170,
                                      sample_weight_fn=mw_fn_factory(wp, wb, threshold))
                log(name, oof, f, tp)

    # B3: Best MW configs with n800lr02 and n1000lr01
    print("\n=== B3: Best MW + HP variants ===")
    # Collect top MW so far
    mw_models = {k: v for k, v in all_models.items() if k.startswith("mw_") and "tt_" not in k}
    if mw_models:
        top_mw = sorted(mw_models.items(), key=lambda x: x[1]["rmse"])[:5]
        print(f"  Top 5 MW configs:")
        for n, d in top_mw:
            print(f"    {d['rmse']:.4f} {n}")

        # Parse best configs and run with different HPs
        for mw_name, _ in top_mw:
            # Parse params from name
            parts = mw_name.split("_")
            has_threshold = "t" in mw_name and any(p.startswith("t") and p[1:].isdigit() for p in parts)

            # Extract params
            wp, wb, threshold = 1.5, 0.5, 0
            for p in parts:
                if p.startswith("p") and p[1:].replace(".", "").isdigit():
                    wp = float(p[1:])
                elif p.startswith("b") and p[1:].replace(".", "").isdigit():
                    wb = float(p[1:])
                elif p.startswith("t") and p[1:].isdigit():
                    threshold = int(p[1:])

            for hp_name, hp_ov in [("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
                                    ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01})]:
                params = {**LGBM_BASE, **hp_ov}
                name = f"{mw_name}_{hp_name}"
                oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                      lgbm_params=params, min_moisture=170,
                                      sample_weight_fn=mw_fn_factory(wp, wb, threshold))
                log(name, oof, f, tp)

    # ==================================================================
    # C: CONTINUOUS WEIGHTING FUNCTION — Gemini案
    # ==================================================================
    print("\n=== C: CONTINUOUS WEIGHTING ===")

    def continuous_weight_fn_factory(a, b, threshold):
        """weight = 1.0 + a * max(0, y - threshold)^b"""
        def fn(y, g):
            excess = np.clip(y - threshold, 0, None)
            w = 1.0 + a * excess ** b
            return w
        return fn

    for threshold in [100, 120, 140, 160]:
        for a in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            for b in [1.0, 1.5, 2.0]:
                name = f"cw_t{threshold}_a{a}_b{b}"
                oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                      min_moisture=170,
                                      sample_weight_fn=continuous_weight_fn_factory(a, b, threshold))
                log(name, oof, f, tp)

    # ==================================================================
    # D: ULTRA SLOW LEARNER + MOISTURE WEIGHTING FUSION
    # ==================================================================
    print("\n=== D: ULTRA SLOW + MW FUSION ===")

    ultra_slow_configs = [
        ("n2000lr003", {"n_estimators": 2000, "learning_rate": 0.003}),
        ("n3000lr002", {"n_estimators": 3000, "learning_rate": 0.002}),
        ("n1500lr005", {"n_estimators": 1500, "learning_rate": 0.005}),
    ]

    # Collect best MW/CW weight functions
    all_weight_models = {k: v for k, v in all_models.items()
                          if (k.startswith("mw_") or k.startswith("cw_")) and "tt_" not in k}
    best_weight_configs = sorted(all_weight_models.items(), key=lambda x: x[1]["rmse"])[:3]

    for us_name, us_hp in ultra_slow_configs:
        params = {**LGBM_BASE, **us_hp}

        # Without MW (baseline ultra slow)
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params, min_moisture=170)
        log(f"uslow_{us_name}", oof, f, tp)

        # With best MW configs
        for mw_config_name, _ in best_weight_configs:
            parts = mw_config_name.split("_")
            if mw_config_name.startswith("cw_"):
                # Parse continuous weight
                threshold, a, b = 140, 0.01, 1.5
                for p in parts:
                    if p.startswith("t") and p[1:].isdigit():
                        threshold = int(p[1:])
                    elif p.startswith("a"):
                        try: a = float(p[1:])
                        except: pass
                    elif p.startswith("b"):
                        try: b = float(p[1:])
                        except: pass
                wfn = continuous_weight_fn_factory(a, b, threshold)
            else:
                # Parse MW
                wp, wb, threshold = 1.5, 0.5, 0
                for p in parts:
                    if p.startswith("p") and p[1:].replace(".", "").isdigit():
                        wp = float(p[1:])
                    elif p.startswith("b") and p[1:].replace(".", "").isdigit():
                        wb = float(p[1:])
                    elif p.startswith("t") and p[1:].isdigit():
                        threshold = int(p[1:])
                wfn = mw_fn_factory(wp, wb, threshold)

            name = f"uslow_{us_name}_{mw_config_name}"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  lgbm_params=params, min_moisture=170,
                                  sample_weight_fn=wfn)
            log(name, oof, f, tp)

    # Ultra slow + target transform
    print("\n=== D2: Ultra Slow + Target Transform ===")
    for us_name, us_hp in ultra_slow_configs[:2]:
        params = {**LGBM_BASE, **us_hp}
        for tt_name, tt, lam in [("log1p", "log1p", None), ("sqrt", "sqrt", None)]:
            name = f"uslow_{us_name}_tt_{tt_name}"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  lgbm_params=params, min_moisture=170,
                                  target_transform=tt,
                                  target_transform_lambda=lam or 0.5)
            log(name, oof, f, tp)

    # ==================================================================
    # E: MULTI-FACTOR WDV — modified Gemini案
    # ==================================================================
    print("\n=== E: MULTI-FACTOR WDV ===")

    # Multiple WDV factor configs for diversity
    wdv_diversity = [
        ("wdv_f1.0_mm170", 1.0, 170, 30),
        ("wdv_f1.5_mm170", 1.5, 170, 30),
        ("wdv_f2.0_mm170", 2.0, 170, 30),
        ("wdv_f2.5_mm170", 2.5, 170, 30),
        ("wdv_f1.5_mm150", 1.5, 150, 30),
        ("wdv_f2.0_mm150", 2.0, 150, 30),
        ("wdv_f1.5_n50", 1.5, 170, 50),
        ("wdv_f2.0_n50", 2.0, 170, 50),
    ]

    for name, ef, mm, na in wdv_diversity:
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              n_aug=na, extrap=ef, min_moisture=mm)
        log(name, oof, f, tp)

    # WDV + best MW
    if best_weight_configs:
        best_wfn_name = best_weight_configs[0][0]
        parts = best_wfn_name.split("_")
        if best_wfn_name.startswith("cw_"):
            threshold, a, b = 140, 0.01, 1.5
            for p in parts:
                if p.startswith("t") and p[1:].isdigit(): threshold = int(p[1:])
                elif p.startswith("a"):
                    try: a = float(p[1:])
                    except: pass
                elif p.startswith("b"):
                    try: b = float(p[1:])
                    except: pass
            best_wfn = continuous_weight_fn_factory(a, b, threshold)
        else:
            wp, wb, threshold = 1.5, 0.5, 0
            for p in parts:
                if p.startswith("p") and p[1:].replace(".", "").isdigit(): wp = float(p[1:])
                elif p.startswith("b") and p[1:].replace(".", "").isdigit(): wb = float(p[1:])
                elif p.startswith("t") and p[1:].isdigit(): threshold = int(p[1:])
            best_wfn = mw_fn_factory(wp, wb, threshold)

        for wdv_name, ef, mm, na in [("f1.5_mm170", 1.5, 170, 30),
                                       ("f2.0_mm170", 2.0, 170, 30),
                                       ("f2.0_mm150", 2.0, 150, 30)]:
            name = f"wdv_{wdv_name}_bestmw"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  n_aug=na, extrap=ef, min_moisture=mm,
                                  sample_weight_fn=best_wfn)
            log(name, oof, f, tp)

    # ==================================================================
    # F: EXTRA DIVERSITY (d7, bin3, bin5)
    # ==================================================================
    print("\n=== F: EXTRA DIVERSITY ===")

    extra_div = [
        ("d7l30", {"max_depth": 7, "num_leaves": 30}),
        ("d7l30_n600lr03", {"max_depth": 7, "num_leaves": 30, "n_estimators": 600, "learning_rate": 0.03}),
        ("mcs10", {"min_child_samples": 10}),
        ("mcs5", {"min_child_samples": 5, "n_estimators": 600, "learning_rate": 0.03}),
        ("ss08cs08", {"subsample": 0.8, "colsample_bytree": 0.8}),
        ("ra1rl5", {"reg_alpha": 1.0, "reg_lambda": 5.0}),
    ]

    for hp_name, hp_ov in extra_div:
        params = {**LGBM_BASE, **hp_ov}
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params, min_moisture=170)
        log(f"div_{hp_name}", oof, f, tp)

    # bin3 and bin5
    for bs in [3, 5]:
        pp = [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": bs},
            {"name": "standard_scaler"},
        ]
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              preprocess=pp, min_moisture=170)
        log(f"bin{bs}_mm170", oof, f, tp)

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

    # Filter out broken models (NaN or inf)
    valid = []
    for i in range(len(names)):
        if np.isfinite(oofs[:, i]).all() and np.isfinite(tests[:, i]).all():
            valid.append(i)
        else:
            print(f"  SKIP (non-finite): {names[i]}")
    names = [names[i] for i in valid]
    oofs = oofs[:, valid]
    tests = tests[:, valid]
    rmses_list = [rmses_list[i] for i in valid]

    print(f"\n  {len(names)} valid models. Top 20:")
    for n, r in sorted(zip(names, rmses_list), key=lambda x: x[1])[:20]:
        star = " ★★★" if r < 13.5 else (" ★★" if r < 14.0 else (" ★" if r < 14.5 else ""))
        print(f"    {r:.4f}  {n}{star}")

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

    # NM weight optimization (500 trials)
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
    # CONDITIONAL STRETCH on top candidates
    # ==================================================================
    print("\n" + "=" * 70)
    print("CONDITIONAL STRETCH")
    print("=" * 70)

    top_candidates = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:15]

    for base_name, base_data in top_candidates:
        if "stretch" in base_name or "cstretch" in base_name or "pw_" in base_name:
            continue
        base_oof = base_data["oof"]
        base_tp = base_data["test"]
        base_score = base_data["rmse"]
        best_stretch_score = base_score

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

        # Piecewise nonlinear stretch
        for pct in [93, 95, 97]:
            t1 = np.percentile(base_oof, pct)
            t2 = np.percentile(base_oof, 99) if pct < 99 else t1 * 1.3
            for sf1 in [1.05, 1.1, 1.15]:
                for sf2 in [1.3, 1.5, 2.0, 2.5]:
                    if sf2 <= sf1:
                        continue
                    oof_s = base_oof.copy()
                    mask1 = (oof_s > t1) & (oof_s <= t2)
                    oof_s[mask1] = t1 + (oof_s[mask1] - t1) * sf1
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
    # FINAL RE-ENSEMBLE after stretch
    # ==================================================================
    print("\n" + "=" * 70)
    print("FINAL RE-ENSEMBLE (post-stretch)")
    print("=" * 70)

    # Rebuild with all models including stretch
    names2 = list(all_models.keys())
    valid2 = []
    for n in names2:
        d = all_models[n]
        if np.isfinite(d["oof"]).all() and np.isfinite(d["test"]).all():
            valid2.append(n)
    names2 = valid2
    oofs2 = np.column_stack([all_models[n]["oof"] for n in names2])
    tests2 = np.column_stack([all_models[n]["test"] for n in names2])
    rmses2 = [all_models[n]["rmse"] for n in names2]

    order2 = sorted(range(len(names2)), key=lambda i: rmses2[i])
    selected2 = [order2[0]]
    for _ in range(min(40, len(order2) - 1)):
        cur_avg = oofs2[:, selected2].mean(axis=1)
        cur_s = rmse(y_train, cur_avg)
        best_s, best_i = cur_s, -1
        for i in order2:
            if i in selected2:
                continue
            new_avg = (cur_avg * len(selected2) + oofs2[:, i]) / (len(selected2) + 1)
            s = rmse(y_train, new_avg)
            if s < best_s - 0.001:
                best_s = s
                best_i = i
        if best_i >= 0:
            selected2.append(best_i)
            print(f"    +{len(selected2)}: {names2[best_i][:55]:55s} ens={best_s:.4f}")
        else:
            break

    greedy_avg2 = oofs2[:, selected2].mean(axis=1)
    greedy_test2 = tests2[:, selected2].mean(axis=1)
    greedy_s2 = rmse(y_train, greedy_avg2)
    print(f"  Greedy ({len(selected2)} models): {greedy_s2:.4f}")
    all_models["greedy_ens_final"] = {"oof": greedy_avg2, "test": greedy_test2, "rmse": greedy_s2}

    # NM optimization on final ensemble
    sub_oofs2 = oofs2[:, selected2]
    sub_tests2 = tests2[:, selected2]
    ns2 = len(selected2)

    def obj2(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub_oofs2 * wn).sum(axis=1))

    best_opt2 = 999
    best_w2 = np.ones(ns2) / ns2
    for trial in range(500):
        w0 = np.random.dirichlet(np.ones(ns2) * 2)
        res = minimize(obj2, w0, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-10})
        if res.fun < best_opt2:
            best_opt2 = res.fun
            w = np.abs(res.x)
            best_w2 = w / w.sum()

    opt_oof2 = (sub_oofs2 * best_w2).sum(axis=1)
    opt_test2 = (sub_tests2 * best_w2).sum(axis=1)
    print(f"  NM optimized final: {best_opt2:.4f}")
    for i, idx in enumerate(selected2):
        if best_w2[i] > 0.01:
            print(f"    {best_w2[i]:.3f}  {names2[idx]}")
    all_models["nm_opt_final"] = {"oof": opt_oof2, "test": opt_test2, "rmse": best_opt2}

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 23 FINAL SUMMARY")
    print("=" * 70)

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:50]:
        star = " ★★★" if data["rmse"] < 13.5 else (" ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else ""))
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 22 best: 13.87")
    improvement = 13.87 - best_data["rmse"]
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
        path = submissions_dir / f"submission_phase23_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    oof_dir = Path("runs") / f"phase23_{ts}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked},
        "phase": "23",
        "description": "The Weight of Truth",
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
