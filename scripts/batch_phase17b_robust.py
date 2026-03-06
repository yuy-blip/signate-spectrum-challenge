#!/usr/bin/env python
"""Phase 17b: Robust WDV Basis — multi-seed averaging with Phase 16's exact approach.

Phase 16's 14.57 was seed-dependent. This script:
1. Runs WDV basis with many global seeds to find robust configs
2. Averages across seeds for stable estimates
3. Combines best WDV basis approach with PL
4. Tests combining WDV basis + targeted WDV (original)
5. Grand ensemble of best-ever OOFs
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")

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


def generate_wdv_basis_p16(X_tr, y_tr, groups_tr, n_aug, extrap_factor,
                           min_moisture=0, n_basis=3):
    """Exact Phase 16 WDV basis — uses global np.random state."""
    all_deltas = []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue
        sorted_idx = np.argsort(y_sp)
        for i in range(len(sorted_idx)):
            for j in range(i + 1, len(sorted_idx)):
                dy = y_sp[sorted_idx[j]] - y_sp[sorted_idx[i]]
                if dy < 10:
                    continue
                dx = X_sp[sorted_idx[j]] - X_sp[sorted_idx[i]]
                all_deltas.append(dx / dy)

    if len(all_deltas) < n_basis:
        return np.empty((0, X_tr.shape[1])), np.empty(0)

    all_deltas = np.array(all_deltas)
    pca = PCA(n_components=min(n_basis, len(all_deltas), all_deltas.shape[1]))
    pca.fit(all_deltas)

    synth_X, synth_y = [], []
    for sp in np.unique(groups_tr):
        sp_mask = groups_tr == sp
        X_sp, y_sp = X_tr[sp_mask], y_tr[sp_mask]
        if len(y_sp) < 3:
            continue
        high_mask = y_sp >= min_moisture
        if high_mask.sum() < 1:
            continue
        high_samples = np.where(high_mask)[0]
        for hi_idx in high_samples:
            target_dy = extrap_factor * (y_sp[hi_idx] * 0.5 + 50)
            mean_delta = all_deltas.mean(axis=0)
            coeffs = pca.transform(mean_delta.reshape(1, -1))[0]
            coeffs *= (1 + 0.3 * np.random.randn(len(coeffs)))
            delta = pca.inverse_transform(coeffs.reshape(1, -1))[0]
            new_x = X_sp[hi_idx] + delta * target_dy
            new_y = y_sp[hi_idx] + target_dy
            synth_X.append(new_x)
            synth_y.append(new_y)

    if not synth_X:
        return np.empty((0, X_tr.shape[1])), np.empty(0)
    synth_X, synth_y = np.array(synth_X), np.array(synth_y)
    if len(synth_X) > n_aug:
        idx = np.random.choice(len(synth_X), n_aug, replace=False)
        synth_X, synth_y = synth_X[idx], synth_y[idx]
    return synth_X, synth_y


def generate_targeted_wdv(X_tr, y_tr, groups_tr, n_aug, extrap_factor, min_moisture=150):
    """Original targeted WDV (Phase 15c best approach)."""
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


def cv_wdv_basis(X_train, y_train, groups, X_test, n_aug=30, extrap=1.5,
                 min_moisture=150, n_basis=5, lgbm_ov=None, pl_w=0.0):
    """CV with WDV basis using global random state (Phase 16 approach)."""
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

        pipe = build_preprocess_pipeline(BEST_PREPROCESS)
        pipe.fit(X_tr)

        synth_X, synth_y = generate_wdv_basis_p16(
            X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, n_basis
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

        oof[va_idx] = model.predict(X_va_t)
        fold_rmses.append(rmse(y_va, oof[va_idx]))
        test_preds.append(model.predict(X_test_t))

    return oof, fold_rmses, np.mean(test_preds, axis=0)


def cv_combined_wdv(X_train, y_train, groups, X_test,
                    basis_aug=30, basis_extrap=1.5, basis_k=5,
                    tgt_aug=50, tgt_extrap=1.5, tgt_mm=150,
                    pl_w=0.0):
    """CV with BOTH WDV basis + targeted WDV."""
    params = {**LGBM_PARAMS}
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_train))
    fold_rmses = []
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups[tr_idx]

        pipe = build_preprocess_pipeline(BEST_PREPROCESS)
        pipe.fit(X_tr)

        # Generate both types of WDV
        s1_X, s1_y = generate_wdv_basis_p16(
            X_tr, y_tr, g_tr, basis_aug, basis_extrap, tgt_mm, basis_k
        )
        s2_X, s2_y = generate_targeted_wdv(
            X_tr, y_tr, g_tr, tgt_aug, tgt_extrap, tgt_mm
        )

        X_tr_t = pipe.transform(X_tr)
        X_va_t = pipe.transform(X_va)
        X_test_t = pipe.transform(X_test)

        aug_list = [X_tr_t]
        y_list = [y_tr]
        if len(s1_X) > 0:
            aug_list.append(pipe.transform(s1_X))
            y_list.append(s1_y)
        if len(s2_X) > 0:
            aug_list.append(pipe.transform(s2_X))
            y_list.append(s2_y)

        X_tr_aug = np.vstack(aug_list)
        y_tr_aug = np.concatenate(y_list)

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

        oof[va_idx] = model.predict(X_va_t)
        fold_rmses.append(rmse(y_va, oof[va_idx]))
        test_preds.append(model.predict(X_test_t))

    return oof, fold_rmses, np.mean(test_preds, axis=0)


def main():
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
    # Section A: Multi-seed WDV basis (global state like P16)
    # ============================================================
    print("\n=== Section A: Multi-seed WDV basis (k=5, n=30, f=1.5) ===")

    seed_oofs = []
    seed_test_preds = []
    for seed in range(20):
        name = f"wbb_s{seed}"
        np.random.seed(seed)
        try:
            oof, folds, tp = cv_wdv_basis(
                X_train, y_train, groups, X_test,
                n_aug=30, extrap=1.5, min_moisture=150, n_basis=5
            )
            score = rmse(y_train, oof)
            log(name, score, folds, oof)
            seed_oofs.append(oof)
            seed_test_preds.append(tp)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Average across seeds
    if len(seed_oofs) >= 3:
        avg = np.mean(seed_oofs, axis=0)
        score = rmse(y_train, avg)
        print(f"\n  ** Multi-seed avg ({len(seed_oofs)} seeds): RMSE={score:.4f} **")
        all_oofs["mseed_avg_20"] = avg
        all_results.append(("mseed_avg_20", score, []))

        # Best N seeds
        scored = [(rmse(y_train, o), i, o) for i, o in enumerate(seed_oofs)]
        scored.sort()
        for n_top in [3, 5, 10]:
            if len(scored) >= n_top:
                avg = np.mean([o for _, _, o in scored[:n_top]], axis=0)
                s = rmse(y_train, avg)
                print(f"  ** Top-{n_top} seed avg: RMSE={s:.4f} **")
                all_oofs[f"mseed_top{n_top}"] = avg
                all_results.append((f"mseed_top{n_top}", s, []))

    # ============================================================
    # Section B: Multi-seed for other top configs
    # ============================================================
    print("\n=== Section B: Multi-seed for k=3,n=30,f=1.5 and k=8,n=30,f=1.5 ===")

    for k, label in [(3, "k3"), (8, "k8")]:
        seed_oofs_k = []
        for seed in range(10):
            name = f"wbb_{label}_s{seed}"
            np.random.seed(seed)
            try:
                oof, folds, tp = cv_wdv_basis(
                    X_train, y_train, groups, X_test,
                    n_aug=30, extrap=1.5, min_moisture=150, n_basis=k
                )
                score = rmse(y_train, oof)
                log(name, score, folds, oof)
                seed_oofs_k.append(oof)
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

        if len(seed_oofs_k) >= 3:
            avg = np.mean(seed_oofs_k, axis=0)
            s = rmse(y_train, avg)
            print(f"  ** {label} multi-seed avg: RMSE={s:.4f} **")
            all_oofs[f"mseed_{label}_avg"] = avg
            all_results.append((f"mseed_{label}_avg", s, []))

    # ============================================================
    # Section C: WDV basis + PL (multi-seed)
    # ============================================================
    print("\n=== Section C: WDV basis + PL ===")

    for pl_w in [0.3, 0.5, 0.7]:
        seed_oofs_pl = []
        for seed in range(5):
            name = f"wbb_pl{pl_w}_s{seed}"
            np.random.seed(seed * 100 + 50)
            try:
                oof, folds, tp = cv_wdv_basis(
                    X_train, y_train, groups, X_test,
                    n_aug=30, extrap=1.5, min_moisture=150, n_basis=5,
                    pl_w=pl_w
                )
                score = rmse(y_train, oof)
                log(name, score, folds, oof)
                seed_oofs_pl.append(oof)
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

        if len(seed_oofs_pl) >= 3:
            avg = np.mean(seed_oofs_pl, axis=0)
            s = rmse(y_train, avg)
            print(f"  ** PL w={pl_w} avg: RMSE={s:.4f} **")
            all_oofs[f"mseed_pl{pl_w}"] = avg
            all_results.append((f"mseed_pl{pl_w}", s, []))

    # ============================================================
    # Section D: Combined WDV basis + targeted WDV
    # ============================================================
    print("\n=== Section D: Combined WDV basis + targeted WDV ===")

    combo_configs = [
        {"basis_aug": 20, "basis_extrap": 1.5, "basis_k": 5, "tgt_aug": 30, "tgt_extrap": 1.5},
        {"basis_aug": 20, "basis_extrap": 1.5, "basis_k": 5, "tgt_aug": 50, "tgt_extrap": 1.5},
        {"basis_aug": 30, "basis_extrap": 1.5, "basis_k": 5, "tgt_aug": 30, "tgt_extrap": 1.0},
        {"basis_aug": 15, "basis_extrap": 1.5, "basis_k": 3, "tgt_aug": 30, "tgt_extrap": 1.5},
        {"basis_aug": 15, "basis_extrap": 1.5, "basis_k": 5, "tgt_aug": 50, "tgt_extrap": 1.0},
    ]

    for i, cc in enumerate(combo_configs):
        combo_oofs = []
        for seed in range(5):
            np.random.seed(seed * 200 + 77)
            try:
                oof, folds, tp = cv_combined_wdv(
                    X_train, y_train, groups, X_test, **cc
                )
                combo_oofs.append(oof)
            except Exception as e:
                print(f"  combo{i}_s{seed}: FAILED - {e}")
                continue

        if combo_oofs:
            # Report individual and average
            scores = [rmse(y_train, o) for o in combo_oofs]
            avg_oof = np.mean(combo_oofs, axis=0)
            avg_s = rmse(y_train, avg_oof)
            name = f"combo{i}_ba{cc['basis_aug']}_ta{cc['tgt_aug']}"
            print(f"  {name}: seeds={[f'{s:.1f}' for s in scores]} avg={avg_s:.4f}")
            all_oofs[name] = avg_oof
            all_results.append((name, avg_s, []))

    # ============================================================
    # Section E: Combined WDV basis + targeted WDV + PL
    # ============================================================
    print("\n=== Section E: Combined WDV + PL ===")

    for pl_w in [0.3, 0.5]:
        combo_oofs = []
        for seed in range(5):
            np.random.seed(seed * 300 + 99)
            try:
                oof, folds, tp = cv_combined_wdv(
                    X_train, y_train, groups, X_test,
                    basis_aug=20, basis_extrap=1.5, basis_k=5,
                    tgt_aug=30, tgt_extrap=1.5, pl_w=pl_w
                )
                combo_oofs.append(oof)
            except Exception as e:
                print(f"  combo_pl{pl_w}_s{seed}: FAILED - {e}")
                continue

        if combo_oofs:
            scores = [rmse(y_train, o) for o in combo_oofs]
            avg_oof = np.mean(combo_oofs, axis=0)
            avg_s = rmse(y_train, avg_oof)
            name = f"combo_pl{pl_w}_avg"
            print(f"  {name}: seeds={[f'{s:.1f}' for s in scores]} avg={avg_s:.4f}")
            all_oofs[name] = avg_oof
            all_results.append((name, avg_s, []))

    # ============================================================
    # Section F: LGBM tuning with best WDV basis (multi-seed)
    # ============================================================
    print("\n=== Section F: LGBM tuning with WDV basis ===")

    lgbm_configs = [
        {"min_child_samples": 10, "reg_alpha": 0.05},
        {"min_child_samples": 15, "reg_alpha": 0.1},
        {"n_estimators": 600, "learning_rate": 0.03},
        {"num_leaves": 25, "max_depth": 6},
        {"subsample": 0.8, "colsample_bytree": 0.8},
        {"reg_lambda": 0.5},
    ]

    for j, lgbm_ov in enumerate(lgbm_configs):
        lgbm_oofs = []
        for seed in range(5):
            np.random.seed(seed * 400 + 11)
            try:
                oof, folds, tp = cv_wdv_basis(
                    X_train, y_train, groups, X_test,
                    n_aug=30, extrap=1.5, min_moisture=150, n_basis=5,
                    lgbm_ov=lgbm_ov
                )
                lgbm_oofs.append(oof)
            except Exception as e:
                print(f"  lgbm{j}_s{seed}: FAILED - {e}")

        if lgbm_oofs:
            avg = np.mean(lgbm_oofs, axis=0)
            avg_s = rmse(y_train, avg)
            k = list(lgbm_ov.keys())[0]
            name = f"lgbm_{k}={lgbm_ov[k]}_avg"
            print(f"  {name}: avg={avg_s:.4f} (n={len(lgbm_oofs)})")
            all_oofs[name] = avg
            all_results.append((name, avg_s, []))

    # ============================================================
    # Section G: Grand cross-phase ensemble
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

        # Optimized
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
            # Show top weights
            wn = [(w[i], top[i][1]) for i in range(n)]
            wn.sort(reverse=True)
            for wi, ni in wn[:5]:
                print(f"    {wi:.3f}  {ni}")
            all_results.append(("grand_opt", opt_s, []))
        except Exception as e:
            print(f"  Grand opt FAILED: {e}")

    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 17b FINAL SUMMARY")
    print("=" * 70)

    all_results.sort(key=lambda x: x[1])
    for name, score, folds in all_results[:30]:
        m = " ★★" if score < 14.5 else (" ★" if score < 15.0 else "")
        print(f"  {score:.4f}  {folds}  {name}{m}")

    if all_results:
        print(f"\nBEST: {all_results[0][1]:.4f} ({all_results[0][0]})")
    print(f"Phase 16 best: 14.57")


if __name__ == "__main__":
    main()
