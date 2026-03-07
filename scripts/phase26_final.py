#!/usr/bin/env python
"""Phase 26 Final: Targeted attack with wavelet features + piecewise calibration.

Key findings from Phase 26a:
  - wavelet_detail_db4_L4 = 14.00 ★ (new diversity source!)
  - MCR-ALS = 14.42 (marginal)
  - Test-ref EMSC = 16+ (failed)

Strategy: Build 10-15 new individual models with wavelet features,
combine with Phase 23's 325 models, then:
  1. Greedy + NM ensemble
  2. Piecewise calibration (fix both low + high bias)
  3. Save for submission
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
from sklearn.preprocessing import StandardScaler
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


# ======================================================================
# Wavelet features
# ======================================================================

def wavelet_features(X, wavelet='db4', level=4, keep_approx=False, keep_details=None):
    import pywt
    if keep_details is None:
        keep_details = [1, 2]
    out = []
    for i in range(X.shape[0]):
        coeffs = pywt.wavedec(X[i], wavelet, level=level)
        parts = []
        if keep_approx:
            parts.append(coeffs[0])
        for d in keep_details:
            if d <= len(coeffs) - 1:
                parts.append(coeffs[d])
        out.append(np.concatenate(parts))
    return np.array(out)


# ======================================================================
# WDV projection + augmentation
# ======================================================================

def compute_wdv(X_tr, y_tr, groups_tr):
    deltas, dys = [], []
    for sp in np.unique(groups_tr):
        m = groups_tr == sp
        Xs, ys = X_tr[m], y_tr[m]
        if len(ys) < 5: continue
        med = np.median(ys)
        hi, lo = Xs[ys >= med], Xs[ys < med]
        if len(hi) < 2 or len(lo) < 2: continue
        d = hi.mean(0) - lo.mean(0)
        dy = ys[ys >= med].mean() - ys[ys < med].mean()
        if dy > 5:
            deltas.append(d / dy)
            dys.append(dy)
    if len(deltas) < 3: return None
    deltas = np.array(deltas)
    pca = PCA(n_components=1)
    pca.fit(deltas)
    wv = pca.components_[0]
    if np.corrcoef(deltas @ wv, dys)[0, 1] < 0:
        wv = -wv
    return wv


def wdv_augment(X_tr, y_tr, g_tr, n_aug=30, extrap=1.5, min_m=170):
    wv = compute_wdv(X_tr, y_tr, g_tr)
    if wv is None: return np.empty((0, X_tr.shape[1])), np.empty(0)
    from numpy.polynomial.polynomial import polyfit
    proj = X_tr @ wv
    c = polyfit(proj, y_tr, 1)
    scale = c[1]
    sX, sy = [], []
    for sp in np.unique(g_tr):
        m = g_tr == sp
        Xs, ys = X_tr[m], y_tr[m]
        for idx in np.where(ys >= min_m)[0]:
            dy = extrap * (ys[idx] * 0.3 + 30)
            step = dy / (scale + 1e-8)
            sX.append(Xs[idx] + step * wv)
            sy.append(ys[idx] + dy)
    if not sX: return np.empty((0, X_tr.shape[1])), np.empty(0)
    sX, sy = np.array(sX), np.array(sy)
    if len(sX) > n_aug:
        idx = np.linspace(0, len(sX)-1, n_aug, dtype=int)
        sX, sy = sX[idx], sy[idx]
    return sX, sy


# ======================================================================
# CV
# ======================================================================

def cv_run(X_train, y_train, groups, X_test, preprocess=PP_BIN4,
           lgbm_params=None, n_aug=30, min_moisture=170,
           pl_w=0.5, pl_rounds=2,
           extra_tr=None, extra_te=None, use_wdv_proj=False):
    params = {**(lgbm_params or LGBM_BASE)}
    gkf = GroupKFold(n_splits=5)
    tp_prev = None

    for pl_r in range(pl_rounds):
        oof = np.zeros(len(y_train))
        folds_r = []
        tp_folds = []

        for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups)):
            Xtr, Xva = X_train[tr], X_train[va]
            ytr, yva = y_train[tr], y_train[va]
            gtr = groups[tr]

            pipe = build_preprocess_pipeline(preprocess)
            pipe.fit(Xtr)
            Xtr_t = pipe.transform(Xtr)
            Xva_t = pipe.transform(Xva)
            Xte_t = pipe.transform(X_test)

            if use_wdv_proj:
                wv = compute_wdv(Xtr_t, ytr, gtr)
                if wv is not None:
                    Xtr_t = np.hstack([Xtr_t, (Xtr_t @ wv).reshape(-1,1)])
                    Xva_t = np.hstack([Xva_t, (Xva_t @ wv).reshape(-1,1)])
                    Xte_t = np.hstack([Xte_t, (Xte_t @ wv).reshape(-1,1)])

            if extra_tr is not None:
                Xtr_t = np.hstack([Xtr_t, extra_tr[tr]])
                Xva_t = np.hstack([Xva_t, extra_tr[va]])
                Xte_t = np.hstack([Xte_t, extra_te])

            # Augmentation
            aX, ay = [], []
            if n_aug > 0:
                sX, sy = wdv_augment(Xtr, ytr, gtr, n_aug, 1.5, min_moisture)
                if len(sX) > 0:
                    sXt = pipe.transform(sX)
                    if use_wdv_proj and wv is not None:
                        sXt = np.hstack([sXt, (sXt @ wv).reshape(-1,1)])
                    if extra_tr is not None:
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
                        _, ni = nn.kneighbors(sX)
                        sXt = np.hstack([sXt, extra_tr[tr][ni.ravel()]])
                    aX.append(sXt); ay.append(sy)

            Xall = np.vstack([Xtr_t]+aX) if aX else Xtr_t
            yall = np.concatenate([ytr]+ay) if ay else ytr

            if pl_r == 0 and pl_w > 0:
                tmp = create_model("lgbm", params); tmp.fit(Xall, yall)
                plp = tmp.predict(Xte_t)
                Xf = np.vstack([Xall, Xte_t])
                yf = np.concatenate([yall, plp])
                w = np.ones(len(yf)); w[-len(plp):] *= pl_w
                m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
            elif pl_r > 0 and tp_prev is not None:
                Xf = np.vstack([Xall, Xte_t])
                yf = np.concatenate([yall, tp_prev])
                w = np.ones(len(yf)); w[-len(tp_prev):] *= pl_w
                m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
            else:
                m = create_model("lgbm", params); m.fit(Xall, yall)

            oof[va] = m.predict(Xva_t).ravel()
            folds_r.append(rmse(yva, oof[va]))
            tp_folds.append(m.predict(Xte_t).ravel())

        tp_prev = np.mean(tp_folds, axis=0)

    return oof, folds_r, tp_prev


# ======================================================================
# Piecewise calibration
# ======================================================================

def piecewise_cal(oof, tp, y_true):
    best_o, best_t, best_s = oof.copy(), tp.copy(), rmse(y_true, oof)
    for tl in [20, 25, 30, 35]:
        for th in [100, 120, 140, 150, 160]:
            for al in np.arange(0.86, 1.01, 0.005):
                for sh in np.arange(1.01, 1.50, 0.01):
                    o = oof.copy()
                    o[o < tl] *= al
                    mh = o > th
                    o[mh] = th + (o[mh] - th) * sh
                    s = rmse(y_true, o)
                    if s < best_s - 0.001:
                        best_s = s
                        best_o = o.copy()
                        t = tp.copy()
                        t[t < tl] *= al
                        mt = t > th
                        t[mt] = th + (t[mt] - th) * sh
                        best_t = t.copy()
    return best_o, best_t, best_s


def analyze_res(y_true, y_pred, label=""):
    bins = [0, 30, 60, 100, 150, 200, 300]
    r = y_pred - y_true
    overall = rmse(y_true, y_pred)
    b200 = r[y_true >= 200].mean() if (y_true >= 200).sum() > 0 else 0
    b30 = r[y_true < 30].mean() if (y_true < 30).sum() > 0 else 0
    r150 = np.sqrt((r[y_true >= 150]**2).mean()) if (y_true >= 150).sum() > 0 else 0
    print(f"  {label}: RMSE={overall:.4f} | 0-30 bias={b30:+.2f} | 200+ bias={b200:+.2f} | 150+ RMSE={r150:.2f}")
    for i in range(len(bins)-1):
        m = (y_true >= bins[i]) & (y_true < bins[i+1])
        if m.sum() == 0: continue
        ri = r[m]
        print(f"    {bins[i]:>5d}-{bins[i+1]:<5d} n={m.sum():>4d} RMSE={np.sqrt((ri**2).mean()):>7.2f} bias={ri.mean():>+7.2f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 26 FINAL: Wavelet + Piecewise + Mega Ensemble")
    print("=" * 70)

    X_train, y_train, groups, X_test, test_ids = load_data()
    all_models = {}

    def log(name, oof, folds, tp):
        score = rmse(y_train, oof)
        fr = [round(f, 1) for f in folds]
        star = " ★★★" if score < 13.5 else (" ★★" if score < 14.0 else (" ★" if score < 14.5 else ""))
        print(f"  {score:.4f} {fr} {name}{star}")
        all_models[name] = {"oof": oof.copy(), "test": tp.copy(), "rmse": score}

    # ==================================================================
    # INDIVIDUAL MODELS with wavelet features
    # ==================================================================
    print("\n=== Individual Models (with wavelet innovation) ===")

    # Prepare wavelet features
    PP_EMSC = [{"name": "emsc", "poly_order": 2}]
    pipe_e = build_preprocess_pipeline(PP_EMSC)
    pipe_e.fit(X_train)
    Xtr_e = pipe_e.transform(X_train)
    Xte_e = pipe_e.transform(X_test)

    # Wavelet detail features (the Phase 26a winner)
    wl_configs = [
        ("db4_L4_d12", 'db4', 4, False, [1, 2]),
        ("db4_L4_d123", 'db4', 4, False, [1, 2, 3]),
        ("db6_L4_d123", 'db6', 4, False, [1, 2, 3]),
        ("db4_L4_ad12", 'db4', 4, True, [1, 2]),
        ("db4_L5_d12", 'db4', 5, False, [1, 2]),
    ]

    wl_feats = {}
    for wl_name, wav, lvl, approx, dets in wl_configs:
        wtr = wavelet_features(Xtr_e, wav, lvl, approx, dets)
        wte = wavelet_features(Xte_e, wav, lvl, approx, dets)
        sc = StandardScaler()
        wtr = sc.fit_transform(wtr)
        wte = sc.transform(wte)
        wl_feats[wl_name] = (wtr, wte)

    # Run models: base configs + wavelet configs + combined
    hp_configs = [
        ("base", {}),
        ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
        ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01}),
        ("d3l10", {"max_depth": 3, "num_leaves": 10, "n_estimators": 1500, "learning_rate": 0.005}),
        ("d4l15", {"max_depth": 4, "num_leaves": 15, "n_estimators": 500, "learning_rate": 0.04}),
    ]

    # Base models (no wavelet)
    for hp_name, hp in hp_configs:
        params = {**LGBM_BASE, **hp}
        oof, f, tp = cv_run(X_train, y_train, groups, X_test, lgbm_params=params)
        log(f"std_{hp_name}", oof, f, tp)

        # With WDV proj
        if hp_name in ["base", "n800lr02"]:
            oof, f, tp = cv_run(X_train, y_train, groups, X_test,
                                lgbm_params=params, use_wdv_proj=True)
            log(f"std_{hp_name}_wdv", oof, f, tp)

    # Wavelet models
    for wl_name, (wtr, wte) in wl_feats.items():
        for hp_name, hp in hp_configs[:3]:  # base, n800lr02, n1000lr01
            params = {**LGBM_BASE, **hp}
            oof, f, tp = cv_run(X_train, y_train, groups, X_test,
                                lgbm_params=params, extra_tr=wtr, extra_te=wte)
            log(f"wl_{wl_name}_{hp_name}", oof, f, tp)

    # ==================================================================
    # LOAD PREVIOUS MODELS
    # ==================================================================
    print("\n=== Loading previous phase models ===")

    for pat in ["phase23_*", "phase24_*", "phase25_*"]:
        for d in sorted(Path("runs").glob(pat)):
            sp = d / "summary.json"
            if not sp.exists(): continue
            s = json.loads(sp.read_text())
            loaded = 0
            for name, rv in s.get("all_results", {}).items():
                op = d / f"oof_{name}.npy"
                tp_path = d / f"test_{name}.npy"
                if op.exists() and tp_path.exists():
                    od = np.load(op); td = np.load(tp_path)
                    if np.isfinite(od).all() and np.isfinite(td).all() and len(od) == len(y_train):
                        all_models[f"prev_{d.name}_{name}"] = {"oof": od, "test": td, "rmse": rv}
                        loaded += 1
            print(f"  {loaded} from {d}")

    # ==================================================================
    # GREEDY + NM ENSEMBLE
    # ==================================================================
    print("\n=== Greedy + NM Ensemble ===")

    vn = [n for n in all_models
          if np.isfinite(all_models[n]["oof"]).all()
          and np.isfinite(all_models[n]["test"]).all()]
    oofs = np.column_stack([all_models[n]["oof"] for n in vn])
    tests = np.column_stack([all_models[n]["test"] for n in vn])
    rs = [all_models[n]["rmse"] for n in vn]

    print(f"  {len(vn)} valid models")
    ri = sorted(range(len(vn)), key=lambda i: rs[i])
    for i in ri[:15]:
        print(f"    {rs[i]:.4f}  {vn[i]}")

    # Greedy
    sel = [ri[0]]
    for _ in range(min(50, len(ri)-1)):
        ca = oofs[:, sel].mean(axis=1)
        cs = rmse(y_train, ca)
        bs, bi = cs, -1
        for i in ri[:200]:
            if i in sel: continue
            na = (ca * len(sel) + oofs[:, i]) / (len(sel) + 1)
            s = rmse(y_train, na)
            if s < bs - 0.001:
                bs, bi = s, i
        if bi >= 0:
            sel.append(bi)
            if len(sel) <= 15 or len(sel) % 5 == 0:
                print(f"    +{len(sel)}: {vn[bi][:60]:60s} {bs:.4f}")
        else:
            break

    gs = rmse(y_train, oofs[:, sel].mean(axis=1))
    print(f"  Greedy ({len(sel)} models): {gs:.4f}")
    all_models["greedy"] = {"oof": oofs[:, sel].mean(axis=1),
                             "test": tests[:, sel].mean(axis=1), "rmse": gs}

    # NM
    so = oofs[:, sel]; st = tests[:, sel]; ns = len(sel)
    def obj(w):
        wp = np.abs(w); wn = wp / (wp.sum() + 1e-8)
        return rmse(y_train, (so * wn).sum(axis=1))

    bo, bw = 999, np.ones(ns)/ns
    for _ in range(1500):
        w0 = np.random.dirichlet(np.ones(ns)*2)
        r = minimize(obj, w0, method="Nelder-Mead",
                     options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-10})
        if r.fun < bo:
            bo = r.fun
            w = np.abs(r.x); bw = w / w.sum()

    nm_oof = (so * bw).sum(axis=1)
    nm_test = (st * bw).sum(axis=1)
    print(f"  NM ({ns} models): {bo:.4f}")
    for i, idx in enumerate(sel):
        if bw[i] > 0.01:
            print(f"    {bw[i]:.3f}  {vn[idx]}")
    all_models["nm_opt"] = {"oof": nm_oof, "test": nm_test, "rmse": bo}
    analyze_res(y_train, nm_oof, "NM opt")

    # ==================================================================
    # CONDITIONAL ENSEMBLE (ultra-light: top 3 diverse, 50 trials)
    # ==================================================================
    print("\n=== Conditional Ensemble (ultra-light) ===")

    # Pick 3 diverse models
    seen = set()
    div3 = []
    for i in ri:
        n = vn[i]
        base = n.split("stretch_")[0]
        if base not in seen:
            seen.add(base)
            div3.append(i)
            if len(div3) >= 3: break

    print(f"  Diverse 3: {[vn[i] for i in div3]}")
    co = oofs[:, div3]; ct = tests[:, div3]

    best_ce = 999; best_ce_oof = None; best_ce_test = None
    for threshold in [140, 150, 160]:
        for width in [15, 20]:
            proxy = co.mean(axis=1)
            gate = 1 / (1 + np.exp(-(proxy - threshold) / width))

            def obj_ce(p):
                wl = np.abs(p[:3]); wh = np.abs(p[3:])
                wl = wl / (wl.sum() + 1e-8); wh = wh / (wh.sum() + 1e-8)
                pl = (co * wl).sum(axis=1); ph = (co * wh).sum(axis=1)
                return rmse(y_train, (1-gate)*pl + gate*ph)

            for _ in range(50):
                w0 = np.random.dirichlet(np.ones(3)*2, size=2).ravel()
                r = minimize(obj_ce, w0, method="Nelder-Mead",
                             options={"maxiter": 5000})
                if r.fun < best_ce:
                    best_ce = r.fun
                    wl = np.abs(r.x[:3]); wh = np.abs(r.x[3:])
                    wl = wl/(wl.sum()+1e-8); wh = wh/(wh.sum()+1e-8)
                    best_ce_oof = (1-gate)*(co*wl).sum(axis=1) + gate*(co*wh).sum(axis=1)
                    pt = ct.mean(axis=1)
                    gt = 1/(1+np.exp(-(pt-threshold)/width))
                    best_ce_test = (1-gt)*(ct*wl).sum(axis=1) + gt*(ct*wh).sum(axis=1)

    if best_ce_oof is not None:
        print(f"  Conditional ensemble: {best_ce:.4f}")
        all_models["cond_ens"] = {"oof": best_ce_oof, "test": best_ce_test, "rmse": best_ce}

    # ==================================================================
    # PIECEWISE CALIBRATION
    # ==================================================================
    print("\n=== Piecewise Calibration ===")

    top10 = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:10]
    for bn, bd in top10:
        if "pwcal" in bn: continue
        po, pt, ps = piecewise_cal(bd["oof"], bd["test"], y_train)
        if ps < bd["rmse"] - 0.005:
            all_models[f"pwcal_{bn}"] = {"oof": po, "test": pt, "rmse": ps}
            print(f"  {bd['rmse']:.4f} → {ps:.4f} ({bn})")

    # ==================================================================
    # FINAL RE-ENSEMBLE
    # ==================================================================
    print("\n=== Final Re-Ensemble ===")

    vn2 = [n for n in all_models
           if np.isfinite(all_models[n]["oof"]).all()
           and np.isfinite(all_models[n]["test"]).all()
           and len(all_models[n]["oof"]) == len(y_train)]
    o2 = np.column_stack([all_models[n]["oof"] for n in vn2])
    t2 = np.column_stack([all_models[n]["test"] for n in vn2])
    r2 = [all_models[n]["rmse"] for n in vn2]

    ord2 = sorted(range(len(vn2)), key=lambda i: r2[i])
    sel2 = [ord2[0]]
    for _ in range(min(50, len(ord2)-1)):
        ca = o2[:, sel2].mean(axis=1)
        cs = rmse(y_train, ca)
        bs, bi = cs, -1
        for i in ord2[:200]:
            if i in sel2: continue
            na = (ca*len(sel2)+o2[:,i])/(len(sel2)+1)
            s = rmse(y_train, na)
            if s < bs - 0.001: bs, bi = s, i
        if bi >= 0:
            sel2.append(bi)
            if len(sel2) <= 10 or len(sel2) % 5 == 0:
                print(f"    +{len(sel2)}: {vn2[bi][:60]:60s} {bs:.4f}")
        else:
            break

    gs2 = rmse(y_train, o2[:,sel2].mean(axis=1))
    print(f"  Final greedy ({len(sel2)}): {gs2:.4f}")

    so2 = o2[:, sel2]; st2 = t2[:, sel2]; ns2 = len(sel2)
    def obj2(w):
        wp = np.abs(w); wn = wp/(wp.sum()+1e-8)
        return rmse(y_train, (so2*wn).sum(axis=1))

    bo2, bw2 = 999, np.ones(ns2)/ns2
    for _ in range(1500):
        w0 = np.random.dirichlet(np.ones(ns2)*2)
        r = minimize(obj2, w0, method="Nelder-Mead",
                     options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-10})
        if r.fun < bo2:
            bo2 = r.fun
            w = np.abs(r.x); bw2 = w/w.sum()

    final_oof = (so2*bw2).sum(axis=1)
    final_test = (st2*bw2).sum(axis=1)
    print(f"  Final NM: {bo2:.4f}")
    for i, idx in enumerate(sel2):
        if bw2[i] > 0.01:
            print(f"    {bw2[i]:.3f}  {vn2[idx]}")

    all_models["final_nm"] = {"oof": final_oof, "test": final_test, "rmse": bo2}
    all_models["final_greedy"] = {"oof": o2[:,sel2].mean(axis=1),
                                   "test": t2[:,sel2].mean(axis=1), "rmse": gs2}

    # Final piecewise on final
    for cn in ["final_nm", "final_greedy"]:
        d = all_models[cn]
        po, pt, ps = piecewise_cal(d["oof"], d["test"], y_train)
        if ps < d["rmse"] - 0.005:
            all_models[f"pwcal_{cn}"] = {"oof": po, "test": pt, "rmse": ps}
            print(f"  Final PW: {d['rmse']:.4f} → {ps:.4f} ({cn})")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 26 FINAL SUMMARY")
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
    analyze_res(y_train, best_data["oof"], f"BEST ({best_name})")

    # Save
    sd = Path("submissions"); sd.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        sub = pd.DataFrame({"sample number": test_ids.values, "含水率": data["test"]})
        p = sd / f"submission_phase26_{i+1}_{ts}.csv"
        sub.to_csv(p, index=False)
        print(f"  Saved: {p} ({data['rmse']:.4f} {name})")

    od = Path("runs") / f"phase26_{ts}"
    od.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked},
        "phase": "26",
    }
    with open(od / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, data in all_models.items():
        if not name.startswith("prev_"):
            np.save(od / f"oof_{name}.npy", data["oof"])
            np.save(od / f"test_{name}.npy", data["test"])

    print(f"\n  Artifacts: {od}")
    print(f"  Total: {len(all_models)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
