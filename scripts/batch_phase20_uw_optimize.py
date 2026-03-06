#!/usr/bin/env python
"""Phase 20: Universal WDV deep optimization + PL combination.

Phase 19 key discovery:
- Universal WDV (gkf_uwdv_n30_f1.5) = 16.91 — deterministic, no seed dependency
- Universal WDV + PL (n50_f1.5_pl0.5) = 16.92

Goals:
A. Fine-tune universal WDV parameters
B. Optimize UW + PL combination (iterative PL, PL weights)
C. UW with LGBM tuning
D. UW in EMSC space (compute water vec post-preprocessing)
E. UW + targeted WDV combined
F. Multi-pipeline UW ensemble
G. Grand ensemble with optimized weights
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

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


def generate_universal_wdv(X_tr, y_tr, groups_tr, n_aug=30,
                           extrap_factor=1.5, min_moisture=150,
                           dy_scale=0.3, dy_offset=30):
    """Universal Water Vector: species-level PCA water direction."""
    species_deltas = []
    species_dy = []

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

    proj = X_tr @ water_vec
    from numpy.polynomial.polynomial import polyfit
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


def generate_universal_wdv_emsc(X_tr, y_tr, groups_tr, pipe, n_aug=30,
                                extrap_factor=1.5, min_moisture=150):
    """Universal WDV in preprocessed space."""
    X_tr_t = pipe.transform(X_tr)
    species_deltas = []
    species_dy = []

    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr_t[sp_mask], y_tr[sp_mask]
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
        return np.empty((0, X_tr_t.shape[1])), np.empty(0)

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

    proj = X_tr_t @ water_vec
    from numpy.polynomial.polynomial import polyfit
    coeffs = polyfit(proj, y_tr, 1)
    scale = coeffs[1]

    synth_X_t, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr_t[sp_mask], y_tr[sp_mask]
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 1:
            continue
        for hi_idx in np.where(high_mask)[0]:
            target_dy = extrap_factor * (y_sp[hi_idx] * 0.3 + 30)
            step = target_dy / (scale + 1e-8)
            synth_X_t.append(X_sp[hi_idx] + step * water_vec)
            synth_y.append(y_sp[hi_idx] + target_dy)

    if not synth_X_t:
        return np.empty((0, X_tr_t.shape[1])), np.empty(0)
    synth_X_t, synth_y = np.array(synth_X_t), np.array(synth_y)
    if len(synth_X_t) > n_aug:
        idx = np.linspace(0, len(synth_X_t) - 1, n_aug, dtype=int)
        synth_X_t, synth_y = synth_X_t[idx], synth_y[idx]
    return synth_X_t, synth_y


def generate_targeted_wdv(X_tr, y_tr, groups_tr, n_aug, extrap_factor, min_moisture=150):
    """Original targeted WDV."""
    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 2:
            continue
        sorted_idx = np.argsort(y_sp)
        high_idx = sorted_idx[high_mask[sorted_idx]][-5:]
        low_idx = sorted_idx[:5]
        for hi in high_idx:
            for lo in low_idx:
                dy = y_sp[hi] - y_sp[lo]
                if dy < 10:
                    continue
                dx = X_sp[hi] - X_sp[lo]
                synth_X.append(X_sp[hi] + extrap_factor * dx)
                synth_y.append(y_sp[hi] + extrap_factor * dy)
    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        idx = np.random.choice(len(synth_X), n_aug, replace=False)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


def cv_uwdv(X_train, y_train, groups, X_test, n_aug=30, extrap=1.5,
            min_moisture=150, dy_scale=0.3, dy_offset=30,
            pl_w=0.0, lgbm_ov=None, preprocess=None,
            emsc_space=False, also_targeted=False, tgt_n=30, tgt_f=1.5):
    """CV with Universal WDV."""
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

        if emsc_space:
            synth_X_t, synth_y = generate_universal_wdv_emsc(
                X_tr, y_tr, g_tr, pipe, n_aug, extrap, min_moisture
            )
            X_tr_t = pipe.transform(X_tr)
            X_va_t = pipe.transform(X_va)
            X_test_t = pipe.transform(X_test)

            if len(synth_X_t) > 0:
                X_tr_aug = np.vstack([X_tr_t, synth_X_t])
                y_tr_aug = np.concatenate([y_tr, synth_y])
            else:
                X_tr_aug = X_tr_t
                y_tr_aug = y_tr
        else:
            synth_X, synth_y = generate_universal_wdv(
                X_tr, y_tr, g_tr, n_aug, extrap, min_moisture,
                dy_scale, dy_offset
            )

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

        # Also add targeted WDV
        if also_targeted:
            tgt_X, tgt_y = generate_targeted_wdv(
                X_tr, y_tr, g_tr, tgt_n, tgt_f, min_moisture
            )
            if len(tgt_X) > 0:
                tgt_X_t = pipe.transform(tgt_X)
                X_tr_aug = np.vstack([X_tr_aug, tgt_X_t])
                y_tr_aug = np.concatenate([y_tr_aug, tgt_y])

        # PL
        if pl_w > 0:
            temp = create_model("lgbm", params)
            temp.fit(X_tr_aug, y_tr_aug)
            pl = temp.predict(X_test_t)
            X_tr_final = np.vstack([X_tr_aug, X_test_t])
            y_tr_final = np.concatenate([y_tr_aug, pl])
            w = np.ones(len(y_tr_final))
            w[-len(pl):] = pl_w
            model = create_model("lgbm", params)
            model.fit(X_tr_final, y_tr_final, sample_weight=w)
        else:
            model = create_model("lgbm", params)
            model.fit(X_tr_aug, y_tr_aug)

        oof[va_idx] = model.predict(X_va_t)
        fold_rmses.append(rmse(y_va, oof[va_idx]))
        test_preds.append(model.predict(X_test_t))

    return oof, fold_rmses, np.mean(test_preds, axis=0)


def main():
    np.random.seed(42)
    X_train, y_train, groups, X_test, test_ids = load_data()

    all_results = []
    all_oofs = {}

    def log(name, score, folds, oof=None):
        fr = [round(x, 1) for x in folds]
        m = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"  {name}: RMSE={score:.4f}  folds={fr}{m}")
        all_results.append((name, score, fr))
        if oof is not None:
            all_oofs[name] = oof

    # ============================================================
    # Section A: Fine-tune universal WDV
    # ============================================================
    print("\n=== Section A: UW fine-tuning ===")

    for n_aug in [15, 20, 25, 30, 35, 40]:
        for extrap in [1.0, 1.2, 1.3, 1.5, 1.7, 2.0]:
            name = f"uw_n{n_aug}_f{extrap}"
            oof, folds, _ = cv_uwdv(
                X_train, y_train, groups, X_test,
                n_aug=n_aug, extrap=extrap, min_moisture=150
            )
            score = rmse(y_train, oof)
            if score < 17.5:
                log(name, score, folds, oof)
            else:
                all_results.append((name, score, [round(x, 1) for x in folds]))

    # min_moisture sweep
    print("\n  --- min_moisture sweep ---")
    for mm in [100, 120, 130, 140, 150, 160, 170]:
        name = f"uw_mm{mm}"
        oof, folds, _ = cv_uwdv(
            X_train, y_train, groups, X_test,
            n_aug=30, extrap=1.5, min_moisture=mm
        )
        score = rmse(y_train, oof)
        log(name, score, folds, oof)

    # dy_scale and dy_offset
    print("\n  --- dy parameters ---")
    for ds in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for do in [10, 20, 30, 50]:
            name = f"uw_ds{ds}_do{do}"
            oof, folds, _ = cv_uwdv(
                X_train, y_train, groups, X_test,
                n_aug=30, extrap=1.5, dy_scale=ds, dy_offset=do
            )
            score = rmse(y_train, oof)
            if score < 17.0:
                log(name, score, folds, oof)
            else:
                all_results.append((name, score, []))

    # ============================================================
    # Section B: UW + PL optimization
    # ============================================================
    print("\n=== Section B: UW + PL ===")

    for n_aug in [20, 30, 40, 50]:
        for extrap in [1.0, 1.5, 2.0]:
            for pw in [0.3, 0.5, 0.7]:
                name = f"uw_n{n_aug}_f{extrap}_pl{pw}"
                oof, folds, _ = cv_uwdv(
                    X_train, y_train, groups, X_test,
                    n_aug=n_aug, extrap=extrap, pl_w=pw
                )
                score = rmse(y_train, oof)
                if score < 17.0:
                    log(name, score, folds, oof)
                else:
                    all_results.append((name, score, []))

    # Iterative PL
    print("\n  --- Iterative PL ---")
    for n_iter in [2, 3]:
        for pw in [0.3, 0.5]:
            name = f"uw_iterpl{n_iter}_pw{pw}"
            # Round 1
            oof, folds, tp = cv_uwdv(
                X_train, y_train, groups, X_test,
                n_aug=30, extrap=1.5, pl_w=pw
            )
            # Round 2+: use previous test preds
            for it in range(1, n_iter):
                gkf = GroupKFold(n_splits=5)
                oof2 = np.zeros(len(y_train))
                folds2 = []
                test_preds2 = []
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                    g_tr = groups[tr_idx]

                    pipe = build_preprocess_pipeline(BEST_PREPROCESS)
                    pipe.fit(X_tr)
                    synth_X, synth_y = generate_universal_wdv(
                        X_tr, y_tr, g_tr, 30, 1.5, 150
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

                    # Use previous round's test predictions as PL
                    X_aug = np.vstack([X_aug, X_test_t])
                    y_aug = np.concatenate([y_aug, tp])
                    w = np.ones(len(y_aug))
                    w[-len(tp):] = pw
                    model = create_model("lgbm", LGBM_PARAMS)
                    model.fit(X_aug, y_aug, sample_weight=w)
                    oof2[va_idx] = model.predict(X_va_t)
                    folds2.append(rmse(y_va, oof2[va_idx]))
                    test_preds2.append(model.predict(X_test_t))

                oof = oof2
                folds = folds2
                tp = np.mean(test_preds2, axis=0)

            score = rmse(y_train, oof)
            log(name, score, folds, oof)

    # ============================================================
    # Section C: LGBM tuning with UW
    # ============================================================
    print("\n=== Section C: LGBM tuning with UW ===")

    lgbm_tweaks = [
        {"min_child_samples": 10},
        {"min_child_samples": 15},
        {"min_child_samples": 25},
        {"n_estimators": 600, "learning_rate": 0.03},
        {"n_estimators": 800, "learning_rate": 0.02},
        {"num_leaves": 15, "max_depth": 4},
        {"num_leaves": 25, "max_depth": 6},
        {"subsample": 0.8, "colsample_bytree": 0.8},
        {"subsample": 0.6, "colsample_bytree": 0.6},
        {"reg_alpha": 0.05, "reg_lambda": 0.5},
        {"reg_alpha": 0.3, "reg_lambda": 2.0},
    ]

    for i, tw in enumerate(lgbm_tweaks):
        k = list(tw.keys())[0]
        name = f"uw_lgbm_{k}={tw[k]}"
        oof, folds, _ = cv_uwdv(
            X_train, y_train, groups, X_test,
            n_aug=30, extrap=1.5, lgbm_ov=tw
        )
        score = rmse(y_train, oof)
        log(name, score, folds, oof)

    # ============================================================
    # Section D: UW in EMSC space
    # ============================================================
    print("\n=== Section D: UW in EMSC space ===")

    for n_aug in [20, 30, 40]:
        for extrap in [1.0, 1.5, 2.0]:
            name = f"uw_emsc_n{n_aug}_f{extrap}"
            oof, folds, _ = cv_uwdv(
                X_train, y_train, groups, X_test,
                n_aug=n_aug, extrap=extrap, emsc_space=True
            )
            score = rmse(y_train, oof)
            log(name, score, folds, oof)

    # ============================================================
    # Section E: UW + targeted WDV combined
    # ============================================================
    print("\n=== Section E: UW + targeted WDV ===")

    for uw_n in [15, 20, 30]:
        for tgt_n in [20, 30, 50]:
            name = f"uw{uw_n}_tgt{tgt_n}"
            oof, folds, _ = cv_uwdv(
                X_train, y_train, groups, X_test,
                n_aug=uw_n, extrap=1.5,
                also_targeted=True, tgt_n=tgt_n, tgt_f=1.5
            )
            score = rmse(y_train, oof)
            log(name, score, folds, oof)

    # With PL
    for pw in [0.3, 0.5]:
        name = f"uw20_tgt30_pl{pw}"
        oof, folds, _ = cv_uwdv(
            X_train, y_train, groups, X_test,
            n_aug=20, extrap=1.5,
            also_targeted=True, tgt_n=30, tgt_f=1.5,
            pl_w=pw
        )
        score = rmse(y_train, oof)
        log(name, score, folds, oof)

    # ============================================================
    # Section F: Multi-pipeline ensemble
    # ============================================================
    print("\n=== Section F: Multi-pipeline UW ===")

    alt_preps = [
        ("sg9", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 9, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
        ("sg11", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 8},
            {"name": "standard_scaler"},
        ]),
        ("bin4", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 4},
            {"name": "standard_scaler"},
        ]),
        ("bin16", [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 16},
            {"name": "standard_scaler"},
        ]),
    ]

    for pp_name, pp in alt_preps:
        name = f"uw_{pp_name}"
        oof, folds, _ = cv_uwdv(
            X_train, y_train, groups, X_test,
            n_aug=30, extrap=1.5, preprocess=pp
        )
        score = rmse(y_train, oof)
        log(name, score, folds, oof)

    # ============================================================
    # Section G: Grand ensemble
    # ============================================================
    print("\n=== Section G: Grand ensemble ===")

    top = sorted([(rmse(y_train, o), n, o) for n, o in all_oofs.items()])
    print(f"  {len(top)} OOFs collected")
    for s, n, _ in top[:15]:
        print(f"    {s:.4f}  {n}")

    if len(top) >= 3:
        for n_top in [3, 5, 10, 15]:
            if len(top) >= n_top:
                avg = np.mean([o for _, _, o in top[:n_top]], axis=0)
                s = rmse(y_train, avg)
                print(f"  Grand avg top-{n_top}: RMSE={s:.4f}")
                all_results.append((f"grand_top{n_top}", s, []))

        try:
            from scipy.optimize import minimize
            n = min(15, len(top))
            mat = np.array([o for _, _, o in top[:n]])
            def obj(w):
                w = np.abs(w) / np.abs(w).sum()
                return rmse(y_train, (w[:, None] * mat).sum(axis=0))
            res = minimize(obj, np.ones(n)/n, method="Nelder-Mead",
                          options={"maxiter": 10000})
            w = np.abs(res.x) / np.abs(res.x).sum()
            opt = (w[:, None] * mat).sum(axis=0)
            opt_s = rmse(y_train, opt)
            print(f"  Grand optimized ({n}): RMSE={opt_s:.4f}")
            wn = sorted([(w[i], top[i][1]) for i in range(n)], reverse=True)
            for wi, ni in wn[:5]:
                print(f"    {wi:.3f}  {ni}")
            all_results.append(("grand_opt", opt_s, []))
        except Exception as e:
            print(f"  Grand opt FAILED: {e}")

    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 20 FINAL SUMMARY")
    print("=" * 70)

    all_results.sort(key=lambda x: x[1])
    for name, score, folds in all_results[:40]:
        m = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"  {score:.4f}  {folds}  {name}{m}")

    if all_results:
        print(f"\nBEST: {all_results[0][1]:.4f} ({all_results[0][0]})")
    print(f"Phase 19 best: 16.91 (gkf_uwdv_n30_f1.5)")


if __name__ == "__main__":
    main()
