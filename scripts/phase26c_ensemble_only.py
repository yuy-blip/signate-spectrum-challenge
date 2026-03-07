#!/usr/bin/env python
"""Phase 26c: Ensemble + Calibration only.

Uses saved OOF/test predictions from Phase 23 + Phase 26b individual models.
Focuses on:
  1. Greedy forward selection
  2. NM weight optimization
  3. Conditional ensemble (very lightweight - top 5 only, 50 trials)
  4. Piecewise calibration
  5. Re-ensemble with calibrated models
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
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse

DATA_DIR = Path("data/raw")


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


def analyze_residuals(y_true, y_pred, label=""):
    bins = [0, 30, 60, 100, 150, 200, 300]
    residuals = y_pred - y_true
    overall = rmse(y_true, y_pred)
    mask200 = y_true >= 200
    bias200 = residuals[mask200].mean() if mask200.sum() > 0 else 0
    mask150 = y_true >= 150
    rmse150 = np.sqrt((residuals[mask150] ** 2).mean()) if mask150.sum() > 0 else 0
    mask30 = y_true < 30
    bias30 = residuals[mask30].mean() if mask30.sum() > 0 else 0

    print(f"  {label}")
    print(f"    Overall RMSE: {overall:.4f} | 0-30 bias: {bias30:+.2f} | 200+ bias: {bias200:+.2f} | 150+ RMSE: {rmse150:.2f}")
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() == 0:
            continue
        r = residuals[mask]
        print(f"    {bins[i]:>5d}-{bins[i+1]:<5d} n={mask.sum():>4d} RMSE={np.sqrt((r**2).mean()):>7.2f} bias={r.mean():>+7.2f}")


def piecewise_calibrate(oof, test_pred, y_true):
    """3-zone piecewise calibration (ChatGPT + Claude)."""
    best_oof = oof.copy()
    best_test = test_pred.copy()
    best_score = rmse(y_true, oof)

    for t_low in [20, 25, 30, 35, 40]:
        for t_high in [100, 120, 140, 150, 160, 180]:
            for a_low in np.arange(0.85, 1.01, 0.005):
                for s_high in np.arange(1.01, 1.60, 0.01):
                    oof_cal = oof.copy()
                    mask_low = oof_cal < t_low
                    oof_cal[mask_low] *= a_low
                    mask_high = oof_cal > t_high
                    oof_cal[mask_high] = t_high + (oof_cal[mask_high] - t_high) * s_high
                    s = rmse(y_true, oof_cal)
                    if s < best_score - 0.001:
                        best_score = s
                        best_oof = oof_cal.copy()
                        test_cal = test_pred.copy()
                        test_cal[test_cal < t_low] *= a_low
                        m_ht = test_cal > t_high
                        test_cal[m_ht] = t_high + (test_cal[m_ht] - t_high) * s_high
                        best_test = test_cal.copy()

    return best_oof, best_test, best_score


def conditional_ensemble_micro(oofs_dict, tests_dict, y_true, n_trials=100):
    """Ultra-lightweight conditional ensemble — top 5 models, 4 configs."""
    names = list(oofs_dict.keys())
    n = len(names)
    oofs = np.column_stack([oofs_dict[n_] for n_ in names])
    tests = np.column_stack([tests_dict[n_] for n_ in names])

    best_score = 999
    best_oof = None
    best_test = None

    for threshold in [140, 150]:
        for width in [15, 20]:
            proxy = oofs.mean(axis=1)
            gate = 1 / (1 + np.exp(-(proxy - threshold) / width))

            def obj(params):
                w_lo = np.abs(params[:n])
                w_hi = np.abs(params[n:])
                w_lo = w_lo / (w_lo.sum() + 1e-8)
                w_hi = w_hi / (w_hi.sum() + 1e-8)
                pred = (1 - gate) * (oofs * w_lo).sum(axis=1) + gate * (oofs * w_hi).sum(axis=1)
                return rmse(y_true, pred)

            for _ in range(n_trials):
                w0 = np.random.dirichlet(np.ones(n) * 2, size=2).ravel()
                res = minimize(obj, w0, method="Nelder-Mead",
                               options={"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-9})
                if res.fun < best_score:
                    best_score = res.fun
                    w_lo = np.abs(res.x[:n])
                    w_hi = np.abs(res.x[n:])
                    w_lo = w_lo / (w_lo.sum() + 1e-8)
                    w_hi = w_hi / (w_hi.sum() + 1e-8)
                    best_oof = ((1 - gate) * (oofs * w_lo).sum(axis=1) +
                                gate * (oofs * w_hi).sum(axis=1))
                    proxy_t = tests.mean(axis=1)
                    g_t = 1 / (1 + np.exp(-(proxy_t - threshold) / width))
                    best_test = ((1 - g_t) * (tests * w_lo).sum(axis=1) +
                                 g_t * (tests * w_hi).sum(axis=1))

    return best_oof, best_test, best_score


def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 26c: Ensemble + Calibration")
    print("=" * 70)

    _, y_train, groups, _, test_ids = load_data()
    all_models = {}

    # ==================================================================
    # LOAD ALL SAVED MODELS
    # ==================================================================
    print("\n=== Loading saved models ===")

    for phase_pattern in ["phase23_*", "phase24_*", "phase25_*", "phase26*"]:
        phase_dirs = sorted(Path("runs").glob(phase_pattern))
        for p_dir in phase_dirs:
            summary_path = p_dir / "summary.json"
            if not summary_path.exists():
                continue
            p_summary = json.loads(summary_path.read_text())
            loaded = 0
            for name, rmse_val in p_summary.get("all_results", {}).items():
                oof_path = p_dir / f"oof_{name}.npy"
                test_path = p_dir / f"test_{name}.npy"
                if oof_path.exists() and test_path.exists():
                    oof_data = np.load(oof_path)
                    test_data = np.load(test_path)
                    if (np.isfinite(oof_data).all() and np.isfinite(test_data).all()
                            and len(oof_data) == len(y_train)):
                        key = f"{p_dir.name}_{name}"
                        all_models[key] = {"oof": oof_data, "test": test_data, "rmse": rmse_val}
                        loaded += 1
            print(f"  Loaded {loaded} from {p_dir}")

    print(f"  Total loaded: {len(all_models)}")

    # ==================================================================
    # STEP 1: Greedy + NM
    # ==================================================================
    print("\n=== STEP 1: Greedy + NM Ensemble ===")

    valid_names = list(all_models.keys())
    oofs = np.column_stack([all_models[n]["oof"] for n in valid_names])
    tests = np.column_stack([all_models[n]["test"] for n in valid_names])
    rmses_list = [all_models[n]["rmse"] for n in valid_names]

    ranked_idx = sorted(range(len(valid_names)), key=lambda i: rmses_list[i])
    print(f"  Top 10:")
    for i in ranked_idx[:10]:
        print(f"    {rmses_list[i]:.4f}  {valid_names[i]}")

    # Greedy
    selected = [ranked_idx[0]]
    for _ in range(min(60, len(ranked_idx) - 1)):
        cur_avg = oofs[:, selected].mean(axis=1)
        cur_s = rmse(y_train, cur_avg)
        best_s, best_i = cur_s, -1
        for i in ranked_idx[:200]:  # only consider top 200
            if i in selected:
                continue
            new_avg = (cur_avg * len(selected) + oofs[:, i]) / (len(selected) + 1)
            s = rmse(y_train, new_avg)
            if s < best_s - 0.001:
                best_s = s
                best_i = i
        if best_i >= 0:
            selected.append(best_i)
            if len(selected) <= 15 or len(selected) % 5 == 0:
                print(f"    +{len(selected)}: {valid_names[best_i][:60]:60s} {best_s:.4f}")
        else:
            break

    greedy_s = rmse(y_train, oofs[:, selected].mean(axis=1))
    print(f"  Greedy ({len(selected)} models): {greedy_s:.4f}")
    all_models["greedy_ens"] = {
        "oof": oofs[:, selected].mean(axis=1),
        "test": tests[:, selected].mean(axis=1),
        "rmse": greedy_s
    }

    # NM (1500 trials for better convergence)
    sub_oofs = oofs[:, selected]
    sub_tests = tests[:, selected]
    ns = len(selected)

    def obj(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub_oofs * wn).sum(axis=1))

    best_opt = 999
    best_w = np.ones(ns) / ns
    for trial in range(1500):
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
            print(f"    {best_w[i]:.3f}  {valid_names[idx]}")
    all_models["nm_opt"] = {"oof": opt_oof, "test": opt_test, "rmse": best_opt}
    analyze_residuals(y_train, opt_oof, "NM optimized")

    # ==================================================================
    # STEP 2: Conditional Ensemble (micro — top 5 only)
    # ==================================================================
    print("\n=== STEP 2: Conditional Ensemble (micro) ===")

    # Select top 5 DIVERSE models (not just stretch variants)
    # Filter for unique base models
    seen_base = set()
    diverse_top = []
    for i in ranked_idx:
        name = valid_names[i]
        # Skip stretch/calibrated variants to get diverse bases
        base = name.split("stretch_")[0] if "stretch_" in name else name
        if base not in seen_base:
            seen_base.add(base)
            diverse_top.append((name, all_models[name]))
            if len(diverse_top) >= 5:
                break

    print(f"  Diverse top 5:")
    for n, d in diverse_top:
        print(f"    {d['rmse']:.4f}  {n}")

    ce_oofs = {n: d["oof"] for n, d in diverse_top}
    ce_tests = {n: d["test"] for n, d in diverse_top}
    ce_oof, ce_test, ce_score = conditional_ensemble_micro(
        ce_oofs, ce_tests, y_train, n_trials=100)

    if ce_oof is not None:
        print(f"  Conditional ensemble: {ce_score:.4f}")
        all_models["cond_ens"] = {"oof": ce_oof, "test": ce_test, "rmse": ce_score}
        analyze_residuals(y_train, ce_oof, "Conditional Ensemble")

    # Also try with nm_opt + top diverse
    ce_oofs2 = {"nm_opt": opt_oof}
    ce_tests2 = {"nm_opt": opt_test}
    for n, d in diverse_top[:4]:
        ce_oofs2[n] = d["oof"]
        ce_tests2[n] = d["test"]
    ce_oof2, ce_test2, ce_score2 = conditional_ensemble_micro(
        ce_oofs2, ce_tests2, y_train, n_trials=100)
    if ce_oof2 is not None:
        print(f"  Conditional ensemble (with NM): {ce_score2:.4f}")
        all_models["cond_ens_nm"] = {"oof": ce_oof2, "test": ce_test2, "rmse": ce_score2}

    # ==================================================================
    # STEP 3: Piecewise Calibration
    # ==================================================================
    print("\n=== STEP 3: Piecewise Calibration ===")

    top_cands = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:10]
    for base_name, base_data in top_cands:
        if "pwcal" in base_name:
            continue
        oof_pw, test_pw, score_pw = piecewise_calibrate(
            base_data["oof"], base_data["test"], y_train)
        if score_pw < base_data["rmse"] - 0.005:
            all_models[f"pwcal_{base_name}"] = {"oof": oof_pw, "test": test_pw, "rmse": score_pw}
            print(f"  PW: {base_data['rmse']:.4f} → {score_pw:.4f} ({base_name})")

    # ==================================================================
    # STEP 4: Final re-ensemble
    # ==================================================================
    print("\n=== STEP 4: Final Re-Ensemble ===")

    valid_names2 = [n for n in all_models
                    if np.isfinite(all_models[n]["oof"]).all()
                    and np.isfinite(all_models[n]["test"]).all()
                    and len(all_models[n]["oof"]) == len(y_train)]
    oofs2 = np.column_stack([all_models[n]["oof"] for n in valid_names2])
    tests2 = np.column_stack([all_models[n]["test"] for n in valid_names2])
    rmses2 = [all_models[n]["rmse"] for n in valid_names2]

    order2 = sorted(range(len(valid_names2)), key=lambda i: rmses2[i])
    selected2 = [order2[0]]
    for _ in range(min(60, len(order2) - 1)):
        cur_avg = oofs2[:, selected2].mean(axis=1)
        cur_s = rmse(y_train, cur_avg)
        best_s, best_i = cur_s, -1
        for i in order2[:200]:
            if i in selected2:
                continue
            new_avg = (cur_avg * len(selected2) + oofs2[:, i]) / (len(selected2) + 1)
            s = rmse(y_train, new_avg)
            if s < best_s - 0.001:
                best_s = s
                best_i = i
        if best_i >= 0:
            selected2.append(best_i)
            if len(selected2) <= 10 or len(selected2) % 5 == 0:
                print(f"    +{len(selected2)}: {valid_names2[best_i][:60]:60s} {best_s:.4f}")
        else:
            break

    greedy_s2 = rmse(y_train, oofs2[:, selected2].mean(axis=1))
    print(f"  Final greedy ({len(selected2)} models): {greedy_s2:.4f}")

    sub2 = oofs2[:, selected2]
    sub_t2 = tests2[:, selected2]
    ns2 = len(selected2)

    def obj2(w):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (sub2 * wn).sum(axis=1))

    best_opt2 = 999
    best_w2 = np.ones(ns2) / ns2
    for trial in range(1500):
        w0 = np.random.dirichlet(np.ones(ns2) * 2)
        res = minimize(obj2, w0, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-10})
        if res.fun < best_opt2:
            best_opt2 = res.fun
            w = np.abs(res.x)
            best_w2 = w / w.sum()

    final_oof = (sub2 * best_w2).sum(axis=1)
    final_test = (sub_t2 * best_w2).sum(axis=1)
    print(f"  Final NM: {best_opt2:.4f}")
    for i, idx in enumerate(selected2):
        if best_w2[i] > 0.01:
            print(f"    {best_w2[i]:.3f}  {valid_names2[idx]}")
    all_models["final_nm"] = {"oof": final_oof, "test": final_test, "rmse": best_opt2}
    all_models["final_greedy"] = {
        "oof": oofs2[:, selected2].mean(axis=1),
        "test": tests2[:, selected2].mean(axis=1),
        "rmse": greedy_s2
    }

    # Final piecewise on best
    for cname in ["final_nm", "final_greedy", "cond_ens", "cond_ens_nm"]:
        if cname not in all_models:
            continue
        d = all_models[cname]
        oof_pw, test_pw, score_pw = piecewise_calibrate(d["oof"], d["test"], y_train)
        if score_pw < d["rmse"] - 0.005:
            all_models[f"pwcal_{cname}"] = {"oof": oof_pw, "test": test_pw, "rmse": score_pw}
            print(f"  Final PW: {d['rmse']:.4f} → {score_pw:.4f} ({cname})")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 26c FINAL SUMMARY")
    print("=" * 70)

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:30]:
        star = " ★★★" if data["rmse"] < 13.5 else (" ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else ""))
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 25 best: 13.61")
    improvement = 13.61 - best_data["rmse"]
    print(f"  Improvement: {improvement:+.4f}")
    analyze_residuals(y_train, best_data["oof"], f"BEST ({best_name})")

    # Save
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        sub = pd.DataFrame({
            "sample number": test_ids.values,
            "含水率": data["test"]
        })
        path = submissions_dir / f"submission_phase26c_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    oof_dir = Path("runs") / f"phase26c_{ts}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked},
        "phase": "26c",
        "description": "Ensemble + Piecewise Calibration",
    }
    with open(oof_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save only new models (not previous phase ones)
    new_models = {n: d for n, d in all_models.items()
                  if not n.startswith("phase2")}
    for name, data in new_models.items():
        np.save(oof_dir / f"oof_{name}.npy", data["oof"])
        np.save(oof_dir / f"test_{name}.npy", data["test"])

    print(f"\n  Artifacts: {oof_dir}")
    print(f"  Total models: {len(all_models)} (new: {len(new_models)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
