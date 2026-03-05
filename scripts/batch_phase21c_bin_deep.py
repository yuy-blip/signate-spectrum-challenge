#!/usr/bin/env python
"""Phase 21c: Deep-dive on Binning discovery.

Phase 21b key discovery: Binning(4) >> Binning(8) for UW+iterPL.
  bin4: 14.83 → stretch: 14.68
  bin8: 15.63

Goals:
  A. Binning sweep: 1,2,3,4,5,6,8,12 — find the true optimum
  B. Full tuning around best binning: LGBM HP, WDV, PL, stretch
  C. Multi-binning stacking: combine diverse binning models
  D. Best config × multi-seed for stability check
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
              model_type="lgbm", model_params=None):
    """CV with correct cross-fold iterative PL."""
    if preprocess is None:
        preprocess = [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": 4},
            {"name": "standard_scaler"},
        ]
    if model_params is None:
        model_params = {**(lgbm_params or LGBM_BASE)}

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
                temp = create_model(model_type, model_params)
                temp.fit(X_aug, y_aug)
                pl_pred = temp.predict(X_test_t)
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, pl_pred])
                w = np.ones(len(y_final))
                w[-len(pl_pred):] = pl_w
                model = create_model(model_type, model_params)
                model.fit(X_final, y_final, sample_weight=w)
            elif pl_round > 0 and test_preds_prev is not None:
                X_final = np.vstack([X_aug, X_test_t])
                y_final = np.concatenate([y_aug, test_preds_prev])
                w = np.ones(len(y_final))
                w[-len(test_preds_prev):] = pl_w
                model = create_model(model_type, model_params)
                model.fit(X_final, y_final, sample_weight=w)
            else:
                model = create_model(model_type, model_params)
                model.fit(X_aug, y_aug)

            oof[va_idx] = model.predict(X_va_t).ravel()
            fold_rmses.append(rmse(y_va, oof[va_idx]))
            test_preds_folds.append(model.predict(X_test_t).ravel())

        test_preds_prev = np.mean(test_preds_folds, axis=0)

    return oof, fold_rmses, test_preds_prev


def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 21c: Deep-dive on Binning + Full Tuning")
    print("=" * 70)

    X_train, y_train, groups, X_test, test_ids = load_data()
    print(f"Data: train={X_train.shape}, test={X_test.shape}")

    all_models = {}

    def log(name, oof, folds, tp):
        score = rmse(y_train, oof)
        fr = [round(f, 1) for f in folds]
        star = " ★★" if score < 14.0 else (" ★" if score < 14.8 else "")
        print(f"  {score:.4f} {fr} {name}{star}")
        all_models[name] = {"oof": oof, "test": tp, "rmse": score}

    # ==================================================================
    # Section A: Binning sweep
    # ==================================================================
    print("\n=== Section A: Binning sweep (UW30 + iterPL2 pw0.5) ===")

    for bs in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        pp = [
            {"name": "emsc", "poly_order": 2},
            {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
            {"name": "binning", "bin_size": bs},
            {"name": "standard_scaler"},
        ]
        name = f"bin{bs}_base"
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                preprocess=pp, n_aug=30, extrap=1.5,
                                pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

    # No binning at all
    pp_nobin = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "standard_scaler"},
    ]
    oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                            preprocess=pp_nobin, n_aug=30, extrap=1.5,
                            pl_w=0.5, pl_rounds=2)
    log("nobin_base", oof, f, tp)

    # ==================================================================
    # Section B: Best binning × LGBM HP tuning
    # ==================================================================
    print("\n=== Section B: LGBM HP tuning with bin4 ===")

    hp_configs = [
        ("d6l25", {"max_depth": 6, "num_leaves": 25}),
        ("d7l32", {"max_depth": 7, "num_leaves": 32}),
        ("d4l15", {"max_depth": 4, "num_leaves": 15}),
        ("n600lr03", {"n_estimators": 600, "learning_rate": 0.03}),
        ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
        ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01}),
        ("mcs10", {"min_child_samples": 10}),
        ("mcs15", {"min_child_samples": 15}),
        ("mcs30", {"min_child_samples": 30}),
        ("ss08", {"subsample": 0.8, "colsample_bytree": 0.8}),
        ("ss06", {"subsample": 0.6, "colsample_bytree": 0.6}),
        ("ss05", {"subsample": 0.5, "colsample_bytree": 0.5}),
        ("ra05rl3", {"reg_alpha": 0.5, "reg_lambda": 3.0}),
        ("ra002rl05", {"reg_alpha": 0.02, "reg_lambda": 0.5}),
        ("d6l25_n600", {"max_depth": 6, "num_leaves": 25, "n_estimators": 600, "learning_rate": 0.03}),
        ("d5l20_n600", {"n_estimators": 600, "learning_rate": 0.03}),
    ]

    pp_bin4 = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 4},
        {"name": "standard_scaler"},
    ]

    for hp_name, hp_ov in hp_configs:
        params = {**LGBM_BASE, **hp_ov}
        name = f"bin4_{hp_name}"
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                preprocess=pp_bin4, lgbm_params=params,
                                n_aug=30, extrap=1.5, pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

    # ==================================================================
    # Section C: WDV parameter tuning with bin4
    # ==================================================================
    print("\n=== Section C: WDV tuning with bin4 ===")

    for n_aug, extrap in [(20, 1.0), (20, 1.5), (40, 1.5), (40, 2.0), (50, 1.5), (30, 2.0)]:
        name = f"bin4_uw{n_aug}_f{extrap}"
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                preprocess=pp_bin4, n_aug=n_aug, extrap=extrap,
                                pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

    for mm in [100, 120, 130, 140, 160, 170]:
        name = f"bin4_mm{mm}"
        oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                preprocess=pp_bin4, min_moisture=mm,
                                pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

    # ==================================================================
    # Section D: PL parameter tuning with bin4
    # ==================================================================
    print("\n=== Section D: PL tuning with bin4 ===")

    for pw in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        for pr in [1, 2, 3]:
            name = f"bin4_pl{pw}_r{pr}"
            oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                    preprocess=pp_bin4, pl_w=pw, pl_rounds=pr)
            log(name, oof, f, tp)

    # ==================================================================
    # Section E: SG parameter tuning with bin4
    # ==================================================================
    print("\n=== Section E: SG tuning with bin4 ===")

    for sg_w in [5, 7, 9, 11]:
        for sg_d in [0, 1, 2]:
            pp = [
                {"name": "emsc", "poly_order": 2},
                {"name": "sg", "window_length": sg_w, "polyorder": 2, "deriv": sg_d},
                {"name": "binning", "bin_size": 4},
                {"name": "standard_scaler"},
            ]
            name = f"bin4_sg{sg_w}d{sg_d}"
            oof, f, tp = cv_iterpl(X_train, y_train, groups, X_test,
                                    preprocess=pp, pl_w=0.5, pl_rounds=2)
            log(name, oof, f, tp)

    # ==================================================================
    # Section F: Multi-binning stacking
    # ==================================================================
    print("\n=== Section F: Multi-binning stacking ===")

    # Collect OOF from different bin sizes (already computed in Section A)
    bin_names = [n for n in all_models if n.startswith("bin") and n.endswith("_base")]
    if len(bin_names) >= 3:
        bin_oofs = np.column_stack([all_models[n]["oof"] for n in bin_names])
        bin_tests = np.column_stack([all_models[n]["test"] for n in bin_names])
        bin_rmses = [all_models[n]["rmse"] for n in bin_names]

        # Simple average of top bins
        for k in [2, 3, 4, 5]:
            top_k = sorted(range(len(bin_names)), key=lambda i: bin_rmses[i])[:k]
            avg_oof = bin_oofs[:, top_k].mean(axis=1)
            avg_test = bin_tests[:, top_k].mean(axis=1)
            s = rmse(y_train, avg_oof)
            selected_names = [bin_names[i] for i in top_k]
            print(f"  Top-{k} bin avg: {s:.4f} ({', '.join(selected_names)})")
            all_models[f"top{k}_bin_avg"] = {"oof": avg_oof, "test": avg_test, "rmse": s}

        # Ridge stacking
        gkf = GroupKFold(n_splits=5)
        for alpha in [1.0, 10.0, 100.0]:
            oof_st = np.zeros(len(y_train))
            tp_st = []
            fs = []
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(bin_oofs, y_train, groups)):
                m = Ridge(alpha=alpha)
                m.fit(bin_oofs[tr_idx], y_train[tr_idx])
                oof_st[va_idx] = m.predict(bin_oofs[va_idx])
                tp_st.append(m.predict(bin_tests))
                fs.append(rmse(y_train[va_idx], oof_st[va_idx]))
            s = rmse(y_train, oof_st)
            print(f"  Bin Ridge a={alpha}: {s:.4f} folds={[round(x,1) for x in fs]}")
            all_models[f"bin_ridge_a{alpha}"] = {"oof": oof_st, "test": np.mean(tp_st, axis=0), "rmse": s}

    # ==================================================================
    # Section G: Full stacking of ALL models
    # ==================================================================
    print("\n=== Section G: Full ensemble of all models ===")

    names = list(all_models.keys())
    oofs = np.column_stack([all_models[n]["oof"] for n in names])
    tests = np.column_stack([all_models[n]["test"] for n in names])
    rmses = [all_models[n]["rmse"] for n in names]

    # Greedy selection
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
            print(f"  +{len(selected)}: {names[best_i][:50]:50s} ens={best_s:.4f}")
        else:
            break

    greedy_avg = oofs[:, selected].mean(axis=1)
    greedy_test = tests[:, selected].mean(axis=1)
    greedy_s = rmse(y_train, greedy_avg)
    print(f"  Greedy ({len(selected)} models): {greedy_s:.4f}")
    all_models["greedy_ens"] = {"oof": greedy_avg, "test": greedy_test, "rmse": greedy_s}

    # Nelder-Mead on greedy subset
    sub_oofs = oofs[:, selected]
    sub_tests = tests[:, selected]
    ns = len(selected)
    sub_names = [names[i] for i in selected]

    def obj(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub_oofs * wn).sum(axis=1))

    best_opt = 999
    best_w = np.ones(ns) / ns
    for trial in range(300):
        w0 = np.random.dirichlet(np.ones(ns) * 2)
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-9})
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
    # Section H: Post-processing stretch on top models
    # ==================================================================
    print("\n=== Section H: Post-processing stretch ===")

    # Apply stretch to top 5 base models and all ensembles
    top_candidates = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:10]

    for base_name, base_data in top_candidates:
        if "stretch" in base_name:
            continue
        base_oof = base_data["oof"]
        base_tp = base_data["test"]
        base_score = base_data["rmse"]
        best_stretch = base_score

        for pct in [90, 95, 97]:
            threshold = np.percentile(base_oof, pct)
            for stretch in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
                oof_s = base_oof.copy()
                mask = oof_s > threshold
                oof_s[mask] = threshold + (oof_s[mask] - threshold) * stretch
                s = rmse(y_train, oof_s)
                if s < best_stretch - 0.01:
                    best_stretch = s
                    tp_s = base_tp.copy()
                    mask_t = tp_s > threshold
                    tp_s[mask_t] = threshold + (tp_s[mask_t] - threshold) * stretch
                    sname = f"stretch_{base_name}_p{pct}_s{stretch}"
                    all_models[sname] = {"oof": oof_s, "test": tp_s, "rmse": s}

        if best_stretch < base_score:
            print(f"  {base_name}: {base_score:.4f} → {best_stretch:.4f}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 21c FINAL SUMMARY")
    print("=" * 70)

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:40]:
        star = " ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.8 else "")
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 21b best: 14.67")
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
        path = submissions_dir / f"submission_phase21c_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    # Save artifacts
    oof_dir = Path("runs") / f"phase21c_{ts}"
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
