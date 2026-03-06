#!/usr/bin/env python
"""Phase 21b: MEGA STRATEGY (Fixed iterative PL).

Key fix: Iterative PL must average test predictions ACROSS all folds
between rounds, not per-fold.

Three strategies:
  1. Stacking: Diverse base models → Ridge meta-learner
  2. Ceiling Breaking: Extreme WDV + post-processing stretch
  3. Species 15 Attack: High-moisture weighting + adaptive WDV

All OOF + test predictions collected → final mega-ensemble.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

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

LGBM_BASE = {
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


def cv_with_iterpl(X_train, y_train, groups, X_test,
                   preprocess=None, lgbm_params=None,
                   n_aug=30, extrap=1.5, min_moisture=150,
                   dy_scale=0.3, dy_offset=30,
                   pl_w=0.5, pl_rounds=2,
                   sample_weight_fn=None,
                   model_type="lgbm", model_params=None,
                   extra_wdv=None):
    """CV with CORRECT iterative PL: averaged test preds across folds between rounds."""
    if preprocess is None:
        preprocess = BEST_PREPROCESS
    if model_params is None:
        if model_type == "lgbm":
            model_params = {**(lgbm_params or LGBM_BASE)}
        else:
            model_params = lgbm_params or {}

    gkf = GroupKFold(n_splits=5)

    # Store test predictions per round (averaged across folds)
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

            # WDV augmentation
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

            # Extra WDV (e.g., extreme extrap for ceiling breaking)
            if extra_wdv is not None:
                sx, sy = generate_universal_wdv(
                    X_tr, y_tr, g_tr, **extra_wdv
                )
                if len(sx) > 0:
                    X_aug = np.vstack([X_aug, pipe.transform(sx)])
                    y_aug = np.concatenate([y_aug, sy])

            # Sample weights for base data
            if sample_weight_fn is not None:
                base_sw = sample_weight_fn(y_aug)
            else:
                base_sw = None

            if pl_round == 0 and pl_w > 0:
                # First PL round: train temp model, predict test, retrain with PL
                temp = create_model(model_type, model_params)
                if base_sw is not None:
                    temp.fit(X_aug, y_aug, sample_weight=base_sw)
                else:
                    temp.fit(X_aug, y_aug)
                pl_pred = temp.predict(X_test_t)

                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, pl_pred])
                w = np.ones(len(y_final))
                w[-len(pl_pred):] = pl_w
                if base_sw is not None:
                    w[:len(base_sw)] = base_sw

                model = create_model(model_type, model_params)
                model.fit(X_final, y_final, sample_weight=w)

            elif pl_round > 0 and test_preds_prev is not None:
                # Round 2+: use CROSS-FOLD AVERAGED test preds from previous round
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, test_preds_prev])
                w = np.ones(len(y_final))
                w[-len(test_preds_prev):] = pl_w
                if base_sw is not None:
                    w[:len(base_sw)] = base_sw

                model = create_model(model_type, model_params)
                model.fit(X_final, y_final, sample_weight=w)

            else:
                # No PL
                model = create_model(model_type, model_params)
                if base_sw is not None:
                    model.fit(X_aug, y_aug, sample_weight=base_sw)
                else:
                    model.fit(X_aug, y_aug)

            oof[va_idx] = model.predict(X_va_t).ravel()
            fold_rmses.append(rmse(y_va, oof[va_idx]))
            test_preds_folds.append(model.predict(X_test_t).ravel())

        # Average test predictions across folds for next PL round
        test_preds_prev = np.mean(test_preds_folds, axis=0)

    return oof, fold_rmses, test_preds_prev


def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 21b: MEGA STRATEGY (Fixed iterative PL)")
    print("=" * 70)

    X_train, y_train, groups, X_test, test_ids = load_data()
    print(f"Data: train={X_train.shape}, test={X_test.shape}")
    print(f"Target: [{y_train.min():.1f}, {y_train.max():.1f}], mean={y_train.mean():.1f}")

    all_models = {}

    def log(name, oof, folds, tp):
        score = rmse(y_train, oof)
        fr = [round(f, 1) for f in folds]
        star = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"    {score:.4f} {fr} {name}{star}")
        all_models[name] = {"oof": oof, "test": tp, "rmse": score}

    # ==================================================================
    # STRATEGY 1: DIVERSE BASE MODELS
    # ==================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 1: DIVERSE BASE MODELS")
    print("=" * 70)

    # 1. Best known: UW30 + iterPL2 pw0.5
    print("\n  --- Core models ---")
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2)
    log("core_uw30_pl05_r2", oof, f, tp)

    # 2. Different PL weights
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=30, extrap=1.5, pl_w=0.3, pl_rounds=2)
    log("core_uw30_pl03_r2", oof, f, tp)

    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=30, extrap=1.5, pl_w=0.7, pl_rounds=2)
    log("core_uw30_pl07_r2", oof, f, tp)

    # 3. 3-round PL
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=3)
    log("core_uw30_pl05_r3", oof, f, tp)

    # 4. Different WDV params
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=40, extrap=2.0, pl_w=0.5, pl_rounds=2)
    log("uw40_f2_pl05_r2", oof, f, tp)

    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=20, extrap=1.0, pl_w=0.5, pl_rounds=2)
    log("uw20_f1_pl05_r2", oof, f, tp)

    # 5. Binning(4) pipeline
    print("\n  --- Alternative pipelines ---")
    pp_bin4 = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 4},
        {"name": "standard_scaler"},
    ]
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 preprocess=pp_bin4, n_aug=30, extrap=1.5,
                                 pl_w=0.5, pl_rounds=2)
    log("bin4_uw30_pl05_r2", oof, f, tp)

    # 6. SG(9)
    pp_sg9 = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 9, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 preprocess=pp_sg9, n_aug=30, extrap=1.5,
                                 pl_w=0.5, pl_rounds=2)
    log("sg9_uw30_pl05_r2", oof, f, tp)

    # 7. Binning(16)
    pp_bin16 = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16},
        {"name": "standard_scaler"},
    ]
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 preprocess=pp_bin16, n_aug=30, extrap=1.5,
                                 pl_w=0.5, pl_rounds=2)
    log("bin16_uw30_pl05_r2", oof, f, tp)

    # 8. Different LGBM HPs
    print("\n  --- LGBM variants ---")
    deep = {**LGBM_BASE, "max_depth": 7, "num_leaves": 32,
            "n_estimators": 600, "learning_rate": 0.03}
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 lgbm_params=deep, n_aug=30, extrap=1.5,
                                 pl_w=0.5, pl_rounds=2)
    log("deep_d7l32_pl05_r2", oof, f, tp)

    shallow = {**LGBM_BASE, "max_depth": 4, "num_leaves": 15,
               "n_estimators": 800, "learning_rate": 0.02}
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 lgbm_params=shallow, n_aug=30, extrap=1.5,
                                 pl_w=0.5, pl_rounds=2)
    log("shallow_d4l15_pl05_r2", oof, f, tp)

    reg = {**LGBM_BASE, "reg_alpha": 0.5, "reg_lambda": 3.0, "min_child_samples": 30}
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 lgbm_params=reg, n_aug=30, extrap=1.5,
                                 pl_w=0.5, pl_rounds=2)
    log("reg_heavy_pl05_r2", oof, f, tp)

    fast = {**LGBM_BASE, "n_estimators": 200, "learning_rate": 0.1}
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 lgbm_params=fast, n_aug=30, extrap=1.5,
                                 pl_w=0.5, pl_rounds=2)
    log("fast_n200_pl05_r2", oof, f, tp)

    low_sub = {**LGBM_BASE, "subsample": 0.5, "colsample_bytree": 0.5}
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 lgbm_params=low_sub, n_aug=30, extrap=1.5,
                                 pl_w=0.5, pl_rounds=2)
    log("lowsub_pl05_r2", oof, f, tp)

    # 9. XGBoost
    print("\n  --- XGBoost ---")
    xgb_p = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbosity": 0, "n_jobs": -1,
    }
    try:
        oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                     model_type="xgb", model_params=xgb_p,
                                     n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2)
        log("xgb_uw30_pl05_r2", oof, f, tp)
    except Exception as e:
        print(f"    XGB failed: {e}")

    # 10. No WDV baseline (different error profile)
    print("\n  --- No WDV ---")
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=0, pl_w=0.5, pl_rounds=2)
    log("no_wdv_pl05_r2", oof, f, tp)

    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=0, pl_w=1.0, pl_rounds=2)
    log("no_wdv_pl10_r2", oof, f, tp)

    # ==================================================================
    # STRATEGY 2: CEILING BREAKING
    # ==================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 2: CEILING BREAKING")
    print("=" * 70)

    # Extreme WDV in stacking context
    print("\n  --- Extreme WDV models ---")
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=50, extrap=3.0, min_moisture=130,
                                 pl_w=0.5, pl_rounds=2)
    log("extreme_f3_pl05", oof, f, tp)

    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=50, extrap=2.5, min_moisture=120,
                                 pl_w=0.5, pl_rounds=2)
    log("extreme_f25_pl05", oof, f, tp)

    # Double WDV: normal + extreme
    print("\n  --- Double WDV: normal + extreme ---")
    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=30, extrap=1.5,
                                 extra_wdv={"n_aug": 30, "extrap_factor": 3.0,
                                            "min_moisture": 130, "dy_scale": 0.5,
                                            "dy_offset": 50},
                                 pl_w=0.5, pl_rounds=2)
    log("double_wdv_f15_f3", oof, f, tp)

    oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                 n_aug=30, extrap=1.5,
                                 extra_wdv={"n_aug": 20, "extrap_factor": 2.0,
                                            "min_moisture": 100, "dy_scale": 0.4,
                                            "dy_offset": 40},
                                 pl_w=0.5, pl_rounds=2)
    log("double_wdv_f15_f2", oof, f, tp)

    # ==================================================================
    # STRATEGY 3: SPECIES 15 ATTACK
    # ==================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 3: SPECIES 15 ATTACK")
    print("=" * 70)

    # High-moisture upweighting
    print("\n  --- High-moisture upweighting ---")
    for thresh, boost in [(100, 3.0), (120, 5.0), (80, 2.0), (150, 5.0)]:
        wfn = lambda y, t=thresh, b=boost: np.where(y > t, b, 1.0)
        name = f"upweight_t{thresh}_b{boost}"
        oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                     n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2,
                                     sample_weight_fn=wfn)
        log(name, oof, f, tp)

    # Broader WDV (lower min_moisture)
    print("\n  --- Broader WDV ---")
    for mm in [80, 100]:
        oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                     n_aug=40, extrap=1.5, min_moisture=mm,
                                     pl_w=0.5, pl_rounds=2)
        log(f"broad_mm{mm}_pl05", oof, f, tp)

    # Higher dy_scale (more aggressive extrapolation per sample)
    print("\n  --- Aggressive dy params ---")
    for ds, do in [(0.5, 50), (0.4, 40), (0.2, 20)]:
        oof, f, tp = cv_with_iterpl(X_train, y_train, groups, X_test,
                                     n_aug=30, extrap=1.5,
                                     dy_scale=ds, dy_offset=do,
                                     pl_w=0.5, pl_rounds=2)
        log(f"dy_s{ds}_o{do}_pl05", oof, f, tp)

    # ==================================================================
    # POST-PROCESSING: Stretch high predictions
    # ==================================================================
    print("\n" + "=" * 70)
    print("POST-PROCESSING: Prediction stretching")
    print("=" * 70)

    # Use the best model's OOF and test predictions
    best_base_name = min(all_models, key=lambda n: all_models[n]["rmse"])
    best_oof = all_models[best_base_name]["oof"]
    best_tp = all_models[best_base_name]["test"]
    best_score = all_models[best_base_name]["rmse"]
    print(f"  Base model: {best_base_name} RMSE={best_score:.4f}")

    best_stretch_score = best_score
    for pct in [75, 80, 85, 90, 95]:
        threshold = np.percentile(best_oof, pct)
        for stretch in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
            oof_s = best_oof.copy()
            mask = oof_s > threshold
            oof_s[mask] = threshold + (oof_s[mask] - threshold) * stretch
            s = rmse(y_train, oof_s)
            if s < best_stretch_score:
                best_stretch_score = s
                tp_s = best_tp.copy()
                mask_t = tp_s > threshold
                tp_s[mask_t] = threshold + (tp_s[mask_t] - threshold) * stretch
                name = f"stretch_p{pct}_s{stretch}"
                print(f"    {s:.4f} {name}")
                all_models[f"stretch_{best_base_name}_p{pct}_s{stretch}"] = {
                    "oof": oof_s, "test": tp_s, "rmse": s
                }

    # ==================================================================
    # MEGA ENSEMBLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("MEGA ENSEMBLE")
    print("=" * 70)

    names = list(all_models.keys())
    oofs = np.column_stack([all_models[n]["oof"] for n in names])
    tests = np.column_stack([all_models[n]["test"] for n in names])
    rmses = [all_models[n]["rmse"] for n in names]

    print(f"\n  {len(names)} models collected:")
    for n, r in sorted(zip(names, rmses), key=lambda x: x[1]):
        print(f"    {r:.4f}  {n}")

    gkf = GroupKFold(n_splits=5)
    results = {}

    # --- Simple average ---
    avg = oofs.mean(axis=1)
    avg_s = rmse(y_train, avg)
    print(f"\n  Simple average: {avg_s:.4f}")
    results["simple_avg"] = {"rmse": avg_s, "test": tests.mean(axis=1), "oof": avg}

    # --- Greedy selection ---
    print("\n  --- Greedy selection ---")
    order = sorted(range(len(names)), key=lambda i: rmses[i])
    selected = [order[0]]
    for _ in range(min(25, len(order) - 1)):
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
            print(f"    +{len(selected)}: {names[best_i][:45]:45s} ens={best_s:.4f}")
        else:
            break

    greedy_avg = oofs[:, selected].mean(axis=1)
    greedy_s = rmse(y_train, greedy_avg)
    greedy_test = tests[:, selected].mean(axis=1)
    print(f"  Greedy ({len(selected)} models): {greedy_s:.4f}")
    results["greedy_avg"] = {"rmse": greedy_s, "test": greedy_test, "oof": greedy_avg}

    # --- Ridge stacking on greedy-selected ---
    print("\n  --- Ridge stacking (greedy subset) ---")
    sub_oofs = oofs[:, selected]
    sub_tests = tests[:, selected]
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        oof_st = np.zeros(len(y_train))
        tp_st = []
        fs = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(sub_oofs, y_train, groups)):
            m = Ridge(alpha=alpha)
            m.fit(sub_oofs[tr_idx], y_train[tr_idx])
            oof_st[va_idx] = m.predict(sub_oofs[va_idx])
            tp_st.append(m.predict(sub_tests))
            fs.append(rmse(y_train[va_idx], oof_st[va_idx]))
        s = rmse(y_train, oof_st)
        print(f"    Ridge a={alpha:5.1f}: {s:.4f} folds={[round(x,1) for x in fs]}")
        results[f"greedy_ridge_a{alpha}"] = {
            "rmse": s, "test": np.mean(tp_st, axis=0), "oof": oof_st
        }

    # --- Ridge stacking on ALL models ---
    print("\n  --- Ridge stacking (all models) ---")
    for alpha in [10.0, 50.0, 100.0, 500.0]:
        oof_st = np.zeros(len(y_train))
        tp_st = []
        fs = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(oofs, y_train, groups)):
            m = Ridge(alpha=alpha)
            m.fit(oofs[tr_idx], y_train[tr_idx])
            oof_st[va_idx] = m.predict(oofs[va_idx])
            tp_st.append(m.predict(tests))
            fs.append(rmse(y_train[va_idx], oof_st[va_idx]))
        s = rmse(y_train, oof_st)
        print(f"    Ridge a={alpha:5.0f}: {s:.4f} folds={[round(x,1) for x in fs]}")
        results[f"all_ridge_a{alpha}"] = {
            "rmse": s, "test": np.mean(tp_st, axis=0), "oof": oof_st
        }

    # --- Nelder-Mead optimization ---
    print("\n  --- Nelder-Mead weight optimization ---")
    n = len(names)
    def obj(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (oofs * wn).sum(axis=1))

    best_opt = 999
    best_w = np.ones(n) / n
    for trial in range(200):
        w0 = np.random.dirichlet(np.ones(n) * 2)
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-9})
        if res.fun < best_opt:
            best_opt = res.fun
            w = np.abs(res.x)
            best_w = w / w.sum()

    opt_oof = (oofs * best_w).sum(axis=1)
    opt_test = (tests * best_w).sum(axis=1)
    print(f"    Optimized: {best_opt:.4f}")
    for idx in np.argsort(-best_w)[:8]:
        if best_w[idx] > 0.01:
            print(f"      {best_w[idx]:.3f}  {names[idx]}")
    results["nelder_mead"] = {"rmse": best_opt, "test": opt_test, "oof": opt_oof}

    # --- Nelder-Mead on greedy subset ---
    print("\n  --- Nelder-Mead on greedy subset ---")
    ns = len(selected)
    sub_names = [names[i] for i in selected]
    def obj_sub(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub_oofs * wn).sum(axis=1))

    best_opt_sub = 999
    best_w_sub = np.ones(ns) / ns
    for trial in range(200):
        w0 = np.random.dirichlet(np.ones(ns) * 2)
        res = minimize(obj_sub, w0, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-9})
        if res.fun < best_opt_sub:
            best_opt_sub = res.fun
            w = np.abs(res.x)
            best_w_sub = w / w.sum()

    opt_oof_sub = (sub_oofs * best_w_sub).sum(axis=1)
    opt_test_sub = (sub_tests * best_w_sub).sum(axis=1)
    print(f"    Optimized (greedy): {best_opt_sub:.4f}")
    for i, idx in enumerate(selected):
        if best_w_sub[i] > 0.01:
            print(f"      {best_w_sub[i]:.3f}  {names[idx]}")
    results["nelder_mead_greedy"] = {"rmse": best_opt_sub, "test": opt_test_sub, "oof": opt_oof_sub}

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 21b FINAL SUMMARY")
    print("=" * 70)

    all_final = {}
    for n, m in all_models.items():
        all_final[f"base:{n}"] = m
    for n, r in results.items():
        all_final[f"ens:{n}"] = r

    ranked = sorted(all_final.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:30]:
        star = " ★★" if data["rmse"] < 14.5 else (" ★" if data["rmse"] < 15.0 else "")
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 20 best: 15.63")

    # Save submissions
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)
    import datetime, json
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        sub = pd.DataFrame({
            "sample number": test_ids.values,
            "含水率": data["test"]
        })
        path = submissions_dir / f"submission_phase21b_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    # Save all artifacts
    oof_dir = Path("runs") / f"phase21b_{ts}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked},
    }
    with open(oof_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, data in all_models.items():
        np.save(oof_dir / f"oof_{name}.npy", data["oof"])
        np.save(oof_dir / f"test_{name}.npy", data["test"])

    print(f"\n  Artifacts: {oof_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
