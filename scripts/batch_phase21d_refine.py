#!/usr/bin/env python
"""Phase 21d: Refine the winning combination.

Phase 21c discovery:
  bin4_mm170 (14.47) + bin4_n1000lr01 (14.56) → ensemble 14.26 → stretch 14.12

Goals:
  A. mm170 × LGBM HP sweep (combine best HP with best mm)
  B. More mm values around 170 (165, 170, 175, 180, 185)
  C. Multi-seed averaging for stability
  D. 4+ round iterative PL
  E. bin3 and bin5 variants for ensemble diversity
  F. Mega ensemble of everything + aggressive stretch search
"""

from __future__ import annotations

import sys
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


def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 21d: Refine winning combination")
    print("=" * 70)

    X_train, y_train, groups, X_test, test_ids = load_data()
    all_models = {}

    def log(name, oof, folds, tp):
        score = rmse(y_train, oof)
        fr = [round(f, 1) for f in folds]
        star = " ★★" if score < 14.0 else (" ★" if score < 14.5 else "")
        print(f"  {score:.4f} {fr} {name}{star}")
        all_models[name] = {"oof": oof, "test": tp, "rmse": score}

    # ==================================================================
    # A: mm170 × LGBM HP combinations
    # ==================================================================
    print("\n=== A: mm170 × LGBM HP ===")

    hp_combos = [
        ("base", {}),
        ("n600lr03", {"n_estimators": 600, "learning_rate": 0.03}),
        ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
        ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01}),
        ("n1200lr008", {"n_estimators": 1200, "learning_rate": 0.008}),
        ("n1500lr005", {"n_estimators": 1500, "learning_rate": 0.005}),
        ("d6l25", {"max_depth": 6, "num_leaves": 25}),
        ("d6l25_n600", {"max_depth": 6, "num_leaves": 25, "n_estimators": 600, "learning_rate": 0.03}),
        ("d6l25_n1000", {"max_depth": 6, "num_leaves": 25, "n_estimators": 1000, "learning_rate": 0.01}),
        ("d4l15_n1000", {"max_depth": 4, "num_leaves": 15, "n_estimators": 1000, "learning_rate": 0.01}),
        ("ss06", {"subsample": 0.6, "colsample_bytree": 0.6}),
        ("ss06_n800", {"subsample": 0.6, "colsample_bytree": 0.6, "n_estimators": 800, "learning_rate": 0.02}),
        ("mcs15_n800", {"min_child_samples": 15, "n_estimators": 800, "learning_rate": 0.02}),
    ]

    for hp_name, hp_ov in hp_combos:
        params = {**LGBM_BASE, **hp_ov}
        name = f"mm170_{hp_name}"
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                lgbm_params=params, min_moisture=170,
                                pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

    # ==================================================================
    # B: Fine-grain mm sweep
    # ==================================================================
    print("\n=== B: Fine-grain mm sweep ===")

    for mm in [160, 165, 170, 175, 180, 185, 190, 200]:
        for hp_name, hp_ov in [("base", {}), ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01})]:
            params = {**LGBM_BASE, **hp_ov}
            name = f"mm{mm}_{hp_name}"
            oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                    lgbm_params=params, min_moisture=mm,
                                    pl_w=0.5, pl_rounds=2)
            log(name, oof, f, tp)

    # ==================================================================
    # C: PL rounds 3,4,5
    # ==================================================================
    print("\n=== C: More PL rounds ===")

    for pr in [3, 4, 5]:
        # mm170 base
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                min_moisture=170, pl_w=0.5, pl_rounds=pr)
        log(f"mm170_r{pr}", oof, f, tp)

        # mm170 n1000lr01
        params = {**LGBM_BASE, "n_estimators": 1000, "learning_rate": 0.01}
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                lgbm_params=params, min_moisture=170,
                                pl_w=0.5, pl_rounds=pr)
        log(f"mm170_n1000_r{pr}", oof, f, tp)

    # ==================================================================
    # D: bin3 and bin5 variants for diversity
    # ==================================================================
    print("\n=== D: bin3/bin5/bin6 variants ===")

    for bs in [3, 5, 6]:
        pp = [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": bs},
            {"name": "standard_scaler"},
        ]
        # mm170 base
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                preprocess=pp, min_moisture=170,
                                pl_w=0.5, pl_rounds=2)
        log(f"bin{bs}_mm170", oof, f, tp)

        # mm170 n1000
        params = {**LGBM_BASE, "n_estimators": 1000, "learning_rate": 0.01}
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                preprocess=pp, lgbm_params=params,
                                min_moisture=170, pl_w=0.5, pl_rounds=2)
        log(f"bin{bs}_mm170_n1000", oof, f, tp)

    # ==================================================================
    # E: WDV tuning around mm170
    # ==================================================================
    print("\n=== E: WDV tuning with mm170 ===")

    for n_aug in [20, 25, 30, 35, 40]:
        for extrap in [1.0, 1.5, 2.0]:
            if n_aug == 30 and extrap == 1.5:
                continue  # already done
            name = f"mm170_uw{n_aug}_f{extrap}"
            oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                    n_aug=n_aug, extrap=extrap, min_moisture=170,
                                    pl_w=0.5, pl_rounds=2)
            log(name, oof, f, tp)

    # ==================================================================
    # F: Mega ensemble + stretch
    # ==================================================================
    print("\n=== F: Mega ensemble ===")

    names = list(all_models.keys())
    oofs = np.column_stack([all_models[n]["oof"] for n in names])
    tests = np.column_stack([all_models[n]["test"] for n in names])
    rmses = [all_models[n]["rmse"] for n in names]

    print(f"\n  {len(names)} models. Top 10:")
    for n, r in sorted(zip(names, rmses), key=lambda x: x[1])[:10]:
        print(f"    {r:.4f}  {n}")

    # Greedy
    order = sorted(range(len(names)), key=lambda i: rmses[i])
    selected = [order[0]]
    for _ in range(min(30, len(order) - 1)):
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
            print(f"    +{len(selected)}: {names[best_i][:50]:50s} ens={best_s:.4f}")
        else:
            break

    greedy_avg = oofs[:, selected].mean(axis=1)
    greedy_test = tests[:, selected].mean(axis=1)
    greedy_s = rmse(y_train, greedy_avg)
    print(f"  Greedy ({len(selected)} models): {greedy_s:.4f}")

    # NM optimization
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

    all_models["greedy_ens"] = {"oof": greedy_avg, "test": greedy_test, "rmse": greedy_s}
    all_models["nm_opt"] = {"oof": opt_oof, "test": opt_test, "rmse": best_opt}

    # NM on ALL models (not just greedy)
    n_all = len(names)
    def obj_all(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (oofs * wn).sum(axis=1))

    best_all = 999
    best_w_all = np.ones(n_all) / n_all
    for trial in range(300):
        w0 = np.random.dirichlet(np.ones(n_all) * 1)
        res = minimize(obj_all, w0, method="Nelder-Mead",
                       options={"maxiter": 50000, "xatol": 1e-10, "fatol": 1e-10})
        if res.fun < best_all:
            best_all = res.fun
            w = np.abs(res.x)
            best_w_all = w / w.sum()

    all_opt_oof = (oofs * best_w_all).sum(axis=1)
    all_opt_test = (tests * best_w_all).sum(axis=1)
    print(f"  NM all-model: {best_all:.4f}")
    for i in np.argsort(-best_w_all)[:10]:
        if best_w_all[i] > 0.01:
            print(f"    {best_w_all[i]:.3f}  {names[i]}")
    all_models["nm_all"] = {"oof": all_opt_oof, "test": all_opt_test, "rmse": best_all}

    # ==================================================================
    # Stretch on top models and ensembles
    # ==================================================================
    print("\n=== Stretch optimization ===")

    top_candidates = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:15]
    for base_name, base_data in top_candidates:
        if "stretch" in base_name:
            continue
        base_oof = base_data["oof"]
        base_tp = base_data["test"]
        base_score = base_data["rmse"]
        best_stretch = base_score

        for pct in [90, 93, 95, 97]:
            threshold = np.percentile(base_oof, pct)
            for stretch in [1.02, 1.05, 1.08, 1.1, 1.15, 1.2, 1.3, 1.5]:
                oof_s = base_oof.copy()
                mask = oof_s > threshold
                oof_s[mask] = threshold + (oof_s[mask] - threshold) * stretch
                s = rmse(y_train, oof_s)
                if s < best_stretch - 0.005:
                    best_stretch = s
                    tp_s = base_tp.copy()
                    mask_t = tp_s > threshold
                    tp_s[mask_t] = threshold + (tp_s[mask_t] - threshold) * stretch
                    sname = f"stretch_{base_name}_p{pct}_s{stretch}"
                    all_models[sname] = {"oof": oof_s, "test": tp_s, "rmse": s}

        if best_stretch < base_score - 0.01:
            print(f"  {base_name}: {base_score:.4f} → {best_stretch:.4f}")

    # ==================================================================
    # FINAL
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 21d FINAL SUMMARY")
    print("=" * 70)

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:40]:
        star = " ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else "")
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 21c best: 14.12")

    # Save
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)
    import datetime, json
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        sub = pd.DataFrame({
            "sample number": test_ids.values,
            "含水率": data["test"]
        })
        path = submissions_dir / f"submission_phase21d_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    oof_dir = Path("runs") / f"phase21d_{ts}"
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


if __name__ == "__main__":
    main()
