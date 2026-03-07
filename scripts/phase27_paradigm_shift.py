#!/usr/bin/env python
"""Phase 27: Paradigm Shift — 予測対象の再定義による壁の突破

4つの新アプローチを実験:
  A. Quantile/Expectile LGBM — 分位点回帰で「上に出たがるモデル」を作る
  B. Pairwise Ranking Water Axis — 順序学習で水分座標を学び、calibrationで戻す
  C. 2D-COS 特徴量 — 2次元相関分光で波長間の共変動パターンを抽出
  D. DANN — Species-Adversarial Latent で樹種不変な表現を学習

全て既存アンサンブルへの合流を前提とし、OOF予測+テスト予測を保存。
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
from sklearn.isotonic import IsotonicRegression
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
# WDV (reused from Phase 26)
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
# Standard CV with PL (baseline for comparison)
# ======================================================================

def cv_standard(X_train, y_train, groups, X_test, lgbm_params=None,
                n_aug=30, min_moisture=170, pl_w=0.5, pl_rounds=2,
                sample_weight_fn=None):
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

            pipe = build_preprocess_pipeline(PP_BIN4)
            pipe.fit(Xtr)
            Xtr_t = pipe.transform(Xtr)
            Xva_t = pipe.transform(Xva)
            Xte_t = pipe.transform(X_test)

            # Augmentation
            aX, ay = [], []
            if n_aug > 0:
                sX, sy = wdv_augment(Xtr, ytr, gtr, n_aug, 1.5, min_moisture)
                if len(sX) > 0:
                    sXt = pipe.transform(sX)
                    aX.append(sXt); ay.append(sy)

            Xall = np.vstack([Xtr_t]+aX) if aX else Xtr_t
            yall = np.concatenate([ytr]+ay) if ay else ytr

            # Sample weights
            sw = None
            if sample_weight_fn is not None:
                sw = sample_weight_fn(yall)

            if pl_r == 0 and pl_w > 0:
                tmp = create_model("lgbm", params)
                tmp.fit(Xall, yall, sample_weight=sw)
                plp = tmp.predict(Xte_t)
                Xf = np.vstack([Xall, Xte_t])
                yf = np.concatenate([yall, plp])
                w = np.ones(len(yf))
                w[-len(plp):] *= pl_w
                if sw is not None:
                    w[:len(sw)] *= sw
                m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
            elif pl_r > 0 and tp_prev is not None:
                Xf = np.vstack([Xall, Xte_t])
                yf = np.concatenate([yall, tp_prev])
                w = np.ones(len(yf)); w[-len(tp_prev):] *= pl_w
                if sw is not None:
                    w[:len(sw)] *= sw
                m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
            else:
                m = create_model("lgbm", params); m.fit(Xall, yall, sample_weight=sw)

            oof[va] = m.predict(Xva_t).ravel()
            folds_r.append(rmse(yva, oof[va]))
            tp_folds.append(m.predict(Xte_t).ravel())

        tp_prev = np.mean(tp_folds, axis=0)

    return oof, folds_r, tp_prev


# ======================================================================
# PART A: Quantile / Expectile LGBM
# ======================================================================

def cv_quantile(X_train, y_train, groups, X_test, alpha=0.5,
                n_aug=30, min_moisture=170, pl_w=0.5, pl_rounds=2):
    """LGBM with quantile objective."""
    params = {**LGBM_BASE}
    params["objective"] = "quantile"
    params["alpha"] = alpha
    # Quantile regression needs more trees for stable convergence
    params["n_estimators"] = 600
    params["learning_rate"] = 0.03

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

            pipe = build_preprocess_pipeline(PP_BIN4)
            pipe.fit(Xtr)
            Xtr_t = pipe.transform(Xtr)
            Xva_t = pipe.transform(Xva)
            Xte_t = pipe.transform(X_test)

            aX, ay = [], []
            if n_aug > 0:
                sX, sy = wdv_augment(Xtr, ytr, gtr, n_aug, 1.5, min_moisture)
                if len(sX) > 0:
                    aX.append(pipe.transform(sX)); ay.append(sy)

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
# PART B: Pairwise Ranking Water Axis (PyTorch)
# ======================================================================

def pairwise_ranking_axis(X_train, y_train, groups, X_test):
    """Learn a 1D water score via pairwise ranking, then calibrate."""
    import torch
    import torch.nn as nn

    gkf = GroupKFold(n_splits=5)
    oof_score = np.zeros(len(y_train))
    test_scores = []

    for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups)):
        Xtr, Xva = X_train[tr], X_train[va]
        ytr, yva = y_train[tr], y_train[va]
        gtr = groups[tr]

        # Preprocess
        pipe = build_preprocess_pipeline(PP_BIN4)
        pipe.fit(Xtr)
        Xtr_t = pipe.transform(Xtr)
        Xva_t = pipe.transform(Xva)
        Xte_t = pipe.transform(X_test)

        # WDV augment
        sX, sy = wdv_augment(Xtr, ytr, gtr, 30, 1.5, 170)
        if len(sX) > 0:
            sXt = pipe.transform(sX)
            Xtr_t = np.vstack([Xtr_t, sXt])
            # Need groups for pair generation
            # Assign augmented samples to their original species
            aug_groups = np.full(len(sy), -1)  # special group for augmented
            ytr = np.concatenate([ytr, sy])
            gtr = np.concatenate([gtr, aug_groups])

        d_in = Xtr_t.shape[1]

        # Generate intra-species pairs
        pairs_i, pairs_j, labels = [], [], []
        for sp in np.unique(gtr):
            idx = np.where(gtr == sp)[0]
            if len(idx) < 3: continue
            n_pairs = min(len(idx) * 5, 2000)
            for _ in range(n_pairs):
                i, j = np.random.choice(idx, 2, replace=False)
                pairs_i.append(i); pairs_j.append(j)
                labels.append(1.0 if ytr[i] > ytr[j] else 0.0)

        pairs_i = np.array(pairs_i)
        pairs_j = np.array(pairs_j)
        labels = np.array(labels)

        # Small MLP encoder → 1D score
        class RankEncoder(nn.Module):
            def __init__(self, d_in):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, 128), nn.ReLU(), nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                )
            def forward(self, x):
                return self.net(x).squeeze(-1)

        device = "cpu"
        model = RankEncoder(d_in).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()

        X_tensor = torch.FloatTensor(Xtr_t).to(device)

        # Train
        model.train()
        bs = 2048
        for epoch in range(100):
            perm = np.random.permutation(len(labels))
            epoch_loss = 0
            n_batch = 0
            for start in range(0, len(labels), bs):
                end = min(start + bs, len(labels))
                idx_b = perm[start:end]
                si = model(X_tensor[pairs_i[idx_b]])
                sj = model(X_tensor[pairs_j[idx_b]])
                target = torch.FloatTensor(labels[idx_b]).to(device)
                loss = bce(si - sj, target)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
                n_batch += 1

        # Extract scores
        model.eval()
        with torch.no_grad():
            s_tr = model(torch.FloatTensor(Xtr_t).to(device)).cpu().numpy()
            s_va = model(torch.FloatTensor(Xva_t).to(device)).cpu().numpy()
            s_te = model(torch.FloatTensor(Xte_t).to(device)).cpu().numpy()

        oof_score[va] = s_va
        test_scores.append(s_te)

    test_score = np.mean(test_scores, axis=0)

    # Calibrate: water score → absolute moisture
    # Method 1: Isotonic regression on OOF (careful — fit on OOF, apply to test)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_score, y_train)
    oof_pred_iso = iso.predict(oof_score)
    test_pred_iso = iso.predict(test_score)

    # Method 2: Polynomial fit
    from numpy.polynomial.polynomial import polyfit, polyval
    coeffs = polyfit(oof_score, y_train, 3)
    oof_pred_poly = polyval(oof_score, coeffs)
    test_pred_poly = polyval(test_score, coeffs)

    # Method 3: Use score as LGBM feature (combine with standard features)
    # We'll return both predictions and scores for downstream use
    return {
        "oof_score": oof_score,
        "test_score": test_score,
        "oof_iso": oof_pred_iso,
        "test_iso": test_pred_iso,
        "oof_poly": oof_pred_poly,
        "test_poly": test_pred_poly,
    }


def cv_with_ranking_feature(X_train, y_train, groups, X_test, rank_results):
    """Standard LGBM CV but with ranking score as additional feature."""
    params = {**LGBM_BASE}
    gkf = GroupKFold(n_splits=5)
    tp_prev = None
    oof_score = rank_results["oof_score"]
    test_score = rank_results["test_score"]

    for pl_r in range(2):  # 2 rounds PL
        oof = np.zeros(len(y_train))
        folds_r = []
        tp_folds = []

        for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups)):
            Xtr, Xva = X_train[tr], X_train[va]
            ytr, yva = y_train[tr], y_train[va]
            gtr = groups[tr]

            pipe = build_preprocess_pipeline(PP_BIN4)
            pipe.fit(Xtr)
            Xtr_t = pipe.transform(Xtr)
            Xva_t = pipe.transform(Xva)
            Xte_t = pipe.transform(X_test)

            # Add ranking score as feature
            Xtr_t = np.hstack([Xtr_t, oof_score[tr].reshape(-1, 1)])
            Xva_t = np.hstack([Xva_t, oof_score[va].reshape(-1, 1)])
            Xte_t = np.hstack([Xte_t, test_score.reshape(-1, 1)])

            # WDV augment
            sX, sy = wdv_augment(Xtr, ytr, gtr, 30, 1.5, 170)
            aX, ay = [], []
            if len(sX) > 0:
                sXt = pipe.transform(sX)
                # For augmented, use nearest neighbor's score
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
                _, ni = nn.kneighbors(sX)
                sXt = np.hstack([sXt, oof_score[tr][ni.ravel()].reshape(-1, 1)])
                aX.append(sXt); ay.append(sy)

            Xall = np.vstack([Xtr_t]+aX) if aX else Xtr_t
            yall = np.concatenate([ytr]+ay) if ay else ytr

            if pl_r == 0:
                tmp = create_model("lgbm", params); tmp.fit(Xall, yall)
                plp = tmp.predict(Xte_t)
                Xf = np.vstack([Xall, Xte_t])
                yf = np.concatenate([yall, plp])
                w = np.ones(len(yf)); w[-len(plp):] *= 0.5
                m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
            elif tp_prev is not None:
                Xf = np.vstack([Xall, Xte_t])
                yf = np.concatenate([yall, tp_prev])
                w = np.ones(len(yf)); w[-len(tp_prev):] *= 0.5
                m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
            else:
                m = create_model("lgbm", params); m.fit(Xall, yall)

            oof[va] = m.predict(Xva_t).ravel()
            folds_r.append(rmse(yva, oof[va]))
            tp_folds.append(m.predict(Xte_t).ravel())

        tp_prev = np.mean(tp_folds, axis=0)

    return oof, folds_r, tp_prev


# ======================================================================
# PART C: 2D-COS Features
# ======================================================================

def compute_2dcos_features(X_train, y_train, groups, X_test, n_components=20):
    """2D Correlation Spectroscopy features.

    For each species, sort samples by moisture, compute synchronous map,
    then compress via PCA.
    """
    # Use EMSC-preprocessed spectra
    PP_EMSC = [{"name": "emsc", "poly_order": 2}]
    pipe = build_preprocess_pipeline(PP_EMSC)
    pipe.fit(X_train)
    Xtr_e = pipe.transform(X_train)
    Xte_e = pipe.transform(X_test)

    # Compute mean-centered dynamic spectra per species
    # Then average the synchronous maps across species
    n_wl = Xtr_e.shape[1]

    # We'll compute a global synchronous map from all species
    sync_maps = []
    for sp in np.unique(groups):
        mask = groups == sp
        Xs = Xtr_e[mask]
        ys = y_train[mask]
        if len(ys) < 5: continue

        # Sort by moisture
        order = np.argsort(ys)
        Xs_sorted = Xs[order]

        # Mean-center
        Xs_mc = Xs_sorted - Xs_sorted.mean(axis=0)

        # Synchronous map: Phi = X^T @ X / (n-1)
        sync = Xs_mc.T @ Xs_mc / (len(Xs_mc) - 1)
        sync_maps.append(sync)

    if not sync_maps:
        return None, None

    # Average synchronous map
    avg_sync = np.mean(sync_maps, axis=0)

    # Extract features from sync map:
    # 1. Diagonal (autopower spectrum)
    diag = np.diag(avg_sync)

    # 2. PCA of sync map rows
    pca = PCA(n_components=min(n_components, n_wl))
    pca.fit(avg_sync)

    # Project original spectra onto sync-derived components
    # This captures "how each sample relates to the moisture-correlated structure"
    sc = StandardScaler()
    sync_proj_tr = sc.fit_transform(Xtr_e @ pca.components_.T)
    sync_proj_te = sc.transform(Xte_e @ pca.components_.T)

    # Also add diagonal-weighted features
    diag_norm = diag / (diag.max() + 1e-8)
    weighted_tr = Xtr_e * diag_norm[np.newaxis, :]
    weighted_te = Xte_e * diag_norm[np.newaxis, :]

    # Compress weighted spectra
    pca2 = PCA(n_components=min(n_components, weighted_tr.shape[1]))
    wtr = pca2.fit_transform(weighted_tr)
    wte = pca2.transform(weighted_te)

    # Combine
    sc2 = StandardScaler()
    feats_tr = sc2.fit_transform(np.hstack([sync_proj_tr, wtr]))
    feats_te = sc2.transform(np.hstack([sync_proj_te, wte]))

    return feats_tr, feats_te


# ======================================================================
# PART D: DANN — Domain Adversarial Neural Network
# ======================================================================

def dann_features(X_train, y_train, groups, X_test, latent_dim=32):
    """Train a DANN to extract species-invariant features."""
    import torch
    import torch.nn as nn

    gkf = GroupKFold(n_splits=5)
    oof_latent = np.zeros((len(y_train), latent_dim))
    test_latents = []
    oof_pred = np.zeros(len(y_train))
    test_preds = []

    for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups)):
        Xtr, Xva = X_train[tr], X_train[va]
        ytr, yva = y_train[tr], y_train[va]
        gtr = groups[tr]

        pipe = build_preprocess_pipeline(PP_BIN4)
        pipe.fit(Xtr)
        Xtr_t = pipe.transform(Xtr)
        Xva_t = pipe.transform(Xva)
        Xte_t = pipe.transform(X_test)

        # WDV augment
        sX, sy = wdv_augment(Xtr, ytr, gtr, 30, 1.5, 170)
        if len(sX) > 0:
            sXt = pipe.transform(sX)
            Xtr_t = np.vstack([Xtr_t, sXt])
            ytr = np.concatenate([ytr, sy])
            # Assign augmented to random existing species for adversary
            aug_species = np.random.choice(gtr, len(sy))
            gtr = np.concatenate([gtr, aug_species])

        d_in = Xtr_t.shape[1]
        n_species = len(np.unique(gtr))
        sp_map = {s: i for i, s in enumerate(sorted(np.unique(gtr)))}
        sp_labels = np.array([sp_map[s] for s in gtr])

        # Gradient Reversal Layer
        class GRL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, alpha):
                ctx.alpha = alpha
                return x.view_as(x)
            @staticmethod
            def backward(ctx, grad_output):
                return -ctx.alpha * grad_output, None

        class DANN(nn.Module):
            def __init__(self, d_in, latent_dim, n_species):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(d_in, 128), nn.ReLU(), nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, latent_dim), nn.ReLU(),
                )
                self.regressor = nn.Linear(latent_dim, 1)
                self.domain_clf = nn.Sequential(
                    nn.Linear(latent_dim, 32), nn.ReLU(),
                    nn.Linear(32, n_species),
                )

            def forward(self, x, alpha=1.0):
                z = self.encoder(x)
                y_pred = self.regressor(z).squeeze(-1)
                z_rev = GRL.apply(z, alpha)
                sp_pred = self.domain_clf(z_rev)
                return y_pred, sp_pred, z

        device = "cpu"
        model = DANN(d_in, latent_dim, n_species).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(Xtr_t).to(device)
        y_t = torch.FloatTensor(ytr).to(device)
        sp_t = torch.LongTensor(sp_labels).to(device)

        # Train
        n_epochs = 200
        bs = 256
        model.train()
        for epoch in range(n_epochs):
            # Alpha schedule: 0 → 1.0 over training
            p = epoch / n_epochs
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            lam = 0.5  # domain loss weight

            perm = np.random.permutation(len(ytr))
            for start in range(0, len(ytr), bs):
                end = min(start + bs, len(ytr))
                idx = perm[start:end]
                yp, sp, z = model(X_t[idx], alpha)
                loss_reg = mse_loss(yp, y_t[idx])
                loss_dom = ce_loss(sp, sp_t[idx])
                loss = loss_reg + lam * loss_dom
                opt.zero_grad(); loss.backward(); opt.step()

        # Extract latent features
        model.eval()
        with torch.no_grad():
            _, _, z_tr = model(torch.FloatTensor(Xtr_t[:len(X_train[tr])]).to(device))
            _, _, z_va = model(torch.FloatTensor(Xva_t).to(device))
            _, _, z_te = model(torch.FloatTensor(Xte_t).to(device))
            y_va_pred, _, _ = model(torch.FloatTensor(Xva_t).to(device))
            y_te_pred, _, _ = model(torch.FloatTensor(Xte_t).to(device))

        oof_latent[va] = z_va.cpu().numpy()
        test_latents.append(z_te.cpu().numpy())
        oof_pred[va] = y_va_pred.cpu().numpy()
        test_preds.append(y_te_pred.cpu().numpy())

    test_latent = np.mean(test_latents, axis=0)
    test_pred_dann = np.mean(test_preds, axis=0)

    return {
        "oof_latent": oof_latent,
        "test_latent": test_latent,
        "oof_pred": oof_pred,
        "test_pred": test_pred_dann,
    }


# ======================================================================
# Piecewise calibration (reused)
# ======================================================================

def piecewise_cal(oof, tp, y_true):
    best_o, best_t, best_s = oof.copy(), tp.copy(), rmse(y_true, oof)
    for tl in [20, 25, 30]:
        for th in [120, 140, 160]:
            for al in np.arange(0.88, 1.01, 0.01):
                for sh in np.arange(1.02, 1.40, 0.02):
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
    print(f"  {label}: RMSE={overall:.4f}")
    for i in range(len(bins)-1):
        m = (y_true >= bins[i]) & (y_true < bins[i+1])
        if m.sum() == 0: continue
        ri = r[m]
        print(f"    {bins[i]:>5d}-{bins[i+1]:<5d} n={m.sum():>4d} "
              f"RMSE={np.sqrt((ri**2).mean()):>7.2f} bias={ri.mean():>+7.2f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 27: PARADIGM SHIFT — 予測対象の再定義")
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
    # Baseline for comparison
    # ==================================================================
    print("\n=== Baseline (std bin4 mm170 UW iterPL) ===")
    oof, f, tp = cv_standard(X_train, y_train, groups, X_test)
    log("baseline", oof, f, tp)

    # With moisture weighting
    def mw_fn(y):
        return np.maximum(0.5, 1.0 + 1.5 * ((y - y.mean()) / (y.max() - y.min() + 1)))
    oof, f, tp = cv_standard(X_train, y_train, groups, X_test, sample_weight_fn=mw_fn)
    log("baseline_mw", oof, f, tp)

    # ==================================================================
    # PART A: Quantile / Expectile LGBM
    # ==================================================================
    print("\n=== PART A: Quantile LGBM ===")

    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        oof, f, tp = cv_quantile(X_train, y_train, groups, X_test, alpha=alpha)
        log(f"quantile_a{alpha}", oof, f, tp)

    # Conditional quantile blend: use median for low, q0.8 for high
    q50 = all_models.get("quantile_a0.5", {})
    q80 = all_models.get("quantile_a0.8", {})
    q90 = all_models.get("quantile_a0.9", {})

    if q50 and q80 and q90:
        # Blend: below threshold use q50, above use q80/q90
        for thresh in [80, 100, 120, 140]:
            for width in [20, 30]:
                gate_oof = 1 / (1 + np.exp(-(q50["oof"] - thresh) / width))
                gate_test = 1 / (1 + np.exp(-(q50["test"] - thresh) / width))
                blend_oof = (1-gate_oof)*q50["oof"] + gate_oof*q80["oof"]
                blend_test = (1-gate_test)*q50["test"] + gate_test*q80["test"]
                s = rmse(y_train, blend_oof)
                if s < 18.0:
                    log(f"qblend_50_80_t{thresh}_w{width}", blend_oof,
                        [0]*5, blend_test)

                # q50 + q90 blend
                blend_oof90 = (1-gate_oof)*q50["oof"] + gate_oof*q90["oof"]
                blend_test90 = (1-gate_test)*q50["test"] + gate_test*q90["test"]
                s90 = rmse(y_train, blend_oof90)
                if s90 < 18.0:
                    log(f"qblend_50_90_t{thresh}_w{width}", blend_oof90,
                        [0]*5, blend_test90)

    # Optimal weight blend of all quantiles
    valid_q = [(n, d) for n, d in all_models.items() if n.startswith("quantile_")]
    if len(valid_q) >= 3:
        qo = np.column_stack([d["oof"] for _, d in valid_q])
        qt = np.column_stack([d["test"] for _, d in valid_q])
        def obj_q(w):
            wp = np.abs(w); wn = wp/(wp.sum()+1e-8)
            return rmse(y_train, (qo*wn).sum(axis=1))
        best_qs, best_qw = 999, None
        for _ in range(500):
            w0 = np.random.dirichlet(np.ones(len(valid_q))*2)
            r = minimize(obj_q, w0, method="Nelder-Mead", options={"maxiter": 5000})
            if r.fun < best_qs:
                best_qs = r.fun
                w = np.abs(r.x); best_qw = w/w.sum()
        nm_oof = (qo*best_qw).sum(axis=1)
        nm_test = (qt*best_qw).sum(axis=1)
        log("quantile_nm_blend", nm_oof, [0]*5, nm_test)
        print(f"    Quantile NM weights: {dict(zip([n for n,_ in valid_q], best_qw.round(3)))}")

    # ==================================================================
    # PART B: Pairwise Ranking Water Axis
    # ==================================================================
    print("\n=== PART B: Pairwise Ranking Water Axis ===")

    rank_results = pairwise_ranking_axis(X_train, y_train, groups, X_test)

    # Evaluate ranking score correlation with true y
    from scipy.stats import spearmanr
    rho, _ = spearmanr(rank_results["oof_score"], y_train)
    print(f"  Ranking score Spearman ρ = {rho:.4f}")

    # Compare with WDV projection
    pipe_full = build_preprocess_pipeline(PP_BIN4)
    pipe_full.fit(X_train)
    Xtr_full = pipe_full.transform(X_train)
    wdv = compute_wdv(Xtr_full, y_train, groups)
    if wdv is not None:
        wdv_score = Xtr_full @ wdv
        rho_wdv, _ = spearmanr(wdv_score, y_train)
        print(f"  WDV projection Spearman ρ = {rho_wdv:.4f}")

    # Isotonic calibration
    s_iso = rmse(y_train, rank_results["oof_iso"])
    log("rank_isotonic", rank_results["oof_iso"], [0]*5, rank_results["test_iso"])

    # Poly calibration
    s_poly = rmse(y_train, rank_results["oof_poly"])
    log("rank_poly3", rank_results["oof_poly"], [0]*5, rank_results["test_poly"])

    # Ranking score as LGBM feature
    print("  Training LGBM with ranking score as extra feature...")
    oof_rf, f_rf, tp_rf = cv_with_ranking_feature(X_train, y_train, groups, X_test, rank_results)
    log("lgbm_rank_feat", oof_rf, f_rf, tp_rf)

    # ==================================================================
    # PART C: 2D-COS Features
    # ==================================================================
    print("\n=== PART C: 2D-COS Features ===")

    for n_comp in [10, 20, 30]:
        cos_tr, cos_te = compute_2dcos_features(X_train, y_train, groups, X_test, n_components=n_comp)
        if cos_tr is not None:
            print(f"  2D-COS features shape: {cos_tr.shape}")
            # Use as extra features in LGBM
            oof_c, f_c, tp_c = cv_standard(X_train, y_train, groups, X_test)
            # Re-run with 2D-COS features appended
            # Need a custom CV for this
            params = {**LGBM_BASE}
            gkf = GroupKFold(n_splits=5)
            oof_cos = np.zeros(len(y_train))
            folds_cos = []
            tp_cos_folds = []
            tp_prev_cos = None

            for pl_r in range(2):
                oof_cos = np.zeros(len(y_train))
                folds_cos = []
                tp_cos_folds = []
                for fold_i, (tr, va) in enumerate(gkf.split(X_train, y_train, groups)):
                    Xtr, Xva = X_train[tr], X_train[va]
                    ytr, yva = y_train[tr], y_train[va]
                    gtr = groups[tr]

                    pipe2 = build_preprocess_pipeline(PP_BIN4)
                    pipe2.fit(Xtr)
                    Xtr_t = pipe2.transform(Xtr)
                    Xva_t = pipe2.transform(Xva)
                    Xte_t = pipe2.transform(X_test)

                    # Append 2D-COS features
                    Xtr_t = np.hstack([Xtr_t, cos_tr[tr]])
                    Xva_t = np.hstack([Xva_t, cos_tr[va]])
                    Xte_t = np.hstack([Xte_t, cos_te])

                    sX, sy = wdv_augment(Xtr, ytr, gtr, 30, 1.5, 170)
                    aX, ay = [], []
                    if len(sX) > 0:
                        sXt = pipe2.transform(sX)
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
                        _, ni = nn.kneighbors(sX)
                        sXt = np.hstack([sXt, cos_tr[tr][ni.ravel()]])
                        aX.append(sXt); ay.append(sy)

                    Xall = np.vstack([Xtr_t]+aX) if aX else Xtr_t
                    yall = np.concatenate([ytr]+ay) if ay else ytr

                    if pl_r == 0:
                        tmp = create_model("lgbm", params); tmp.fit(Xall, yall)
                        plp = tmp.predict(Xte_t)
                        Xf = np.vstack([Xall, Xte_t])
                        yf = np.concatenate([yall, plp])
                        w = np.ones(len(yf)); w[-len(plp):] *= 0.5
                        m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
                    elif tp_prev_cos is not None:
                        Xf = np.vstack([Xall, Xte_t])
                        yf = np.concatenate([yall, tp_prev_cos])
                        w = np.ones(len(yf)); w[-len(tp_prev_cos):] *= 0.5
                        m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
                    else:
                        m = create_model("lgbm", params); m.fit(Xall, yall)

                    oof_cos[va] = m.predict(Xva_t).ravel()
                    folds_cos.append(rmse(yva, oof_cos[va]))
                    tp_cos_folds.append(m.predict(Xte_t).ravel())

                tp_prev_cos = np.mean(tp_cos_folds, axis=0)

            log(f"cos2d_n{n_comp}", oof_cos, folds_cos, tp_prev_cos)

    # ==================================================================
    # PART D: DANN — Species-Adversarial Latent
    # ==================================================================
    print("\n=== PART D: DANN (Species-Adversarial Latent) ===")

    for ld in [16, 32]:
        dann_res = dann_features(X_train, y_train, groups, X_test, latent_dim=ld)

        # DANN direct prediction
        s_dann = rmse(y_train, dann_res["oof_pred"])
        log(f"dann_direct_d{ld}", dann_res["oof_pred"], [0]*5, dann_res["test_pred"])

        # DANN latent as LGBM features
        dann_tr = dann_res["oof_latent"]
        dann_te = dann_res["test_latent"]

        params = {**LGBM_BASE}
        gkf2 = GroupKFold(n_splits=5)
        oof_dl = np.zeros(len(y_train))
        folds_dl = []
        tp_dl_folds = []
        tp_prev_dl = None

        for pl_r in range(2):
            oof_dl = np.zeros(len(y_train))
            folds_dl = []
            tp_dl_folds = []
            for fold_i, (tr, va) in enumerate(gkf2.split(X_train, y_train, groups)):
                Xtr, Xva = X_train[tr], X_train[va]
                ytr, yva = y_train[tr], y_train[va]
                gtr = groups[tr]

                pipe3 = build_preprocess_pipeline(PP_BIN4)
                pipe3.fit(Xtr)
                Xtr_t = pipe3.transform(Xtr)
                Xva_t = pipe3.transform(Xva)
                Xte_t = pipe3.transform(X_test)

                Xtr_t = np.hstack([Xtr_t, dann_tr[tr]])
                Xva_t = np.hstack([Xva_t, dann_tr[va]])
                Xte_t = np.hstack([Xte_t, dann_te])

                sX, sy = wdv_augment(Xtr, ytr, gtr, 30, 1.5, 170)
                aX, ay = [], []
                if len(sX) > 0:
                    sXt = pipe3.transform(sX)
                    from sklearn.neighbors import NearestNeighbors
                    nn2 = NearestNeighbors(n_neighbors=1).fit(Xtr)
                    _, ni = nn2.kneighbors(sX)
                    sXt = np.hstack([sXt, dann_tr[tr][ni.ravel()]])
                    aX.append(sXt); ay.append(sy)

                Xall = np.vstack([Xtr_t]+aX) if aX else Xtr_t
                yall = np.concatenate([ytr]+ay) if ay else ytr

                if pl_r == 0:
                    tmp = create_model("lgbm", params); tmp.fit(Xall, yall)
                    plp = tmp.predict(Xte_t)
                    Xf = np.vstack([Xall, Xte_t])
                    yf = np.concatenate([yall, plp])
                    w = np.ones(len(yf)); w[-len(plp):] *= 0.5
                    m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
                elif tp_prev_dl is not None:
                    Xf = np.vstack([Xall, Xte_t])
                    yf = np.concatenate([yall, tp_prev_dl])
                    w = np.ones(len(yf)); w[-len(tp_prev_dl):] *= 0.5
                    m = create_model("lgbm", params); m.fit(Xf, yf, sample_weight=w)
                else:
                    m = create_model("lgbm", params); m.fit(Xall, yall)

                oof_dl[va] = m.predict(Xva_t).ravel()
                folds_dl.append(rmse(yva, oof_dl[va]))
                tp_dl_folds.append(m.predict(Xte_t).ravel())

            tp_prev_dl = np.mean(tp_dl_folds, axis=0)

        log(f"lgbm_dann_d{ld}", oof_dl, folds_dl, tp_prev_dl)

    # ==================================================================
    # MEGA ENSEMBLE: all new + previous best
    # ==================================================================
    print("\n=== MEGA ENSEMBLE ===")

    # Load previous phase models
    for pat in ["phase23_*", "phase25_*", "phase26_*"]:
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
                    if (np.isfinite(od).all() and np.isfinite(td).all()
                            and len(od) == len(y_train) and len(td) == len(X_test)):
                        all_models[f"prev_{d.name}_{name}"] = {"oof": od, "test": td, "rmse": rv}
                        loaded += 1
            if loaded > 0:
                print(f"  {loaded} from {d}")

    # Greedy + NM
    vn = [n for n in all_models
          if np.isfinite(all_models[n]["oof"]).all()
          and np.isfinite(all_models[n]["test"]).all()
          and len(all_models[n]["oof"]) == len(y_train)
          and len(all_models[n]["test"]) == len(X_test)]
    oofs = np.column_stack([all_models[n]["oof"] for n in vn])
    tests = np.column_stack([all_models[n]["test"] for n in vn])
    rs = [all_models[n]["rmse"] for n in vn]

    print(f"\n  {len(vn)} valid models for ensemble")
    ri = sorted(range(len(vn)), key=lambda i: rs[i])
    for i in ri[:20]:
        print(f"    {rs[i]:.4f}  {vn[i]}")

    # Greedy forward selection
    sel = [ri[0]]
    for _ in range(min(50, len(ri)-1)):
        ca = oofs[:, sel].mean(axis=1)
        cs = rmse(y_train, ca)
        bs, bi = cs, -1
        for i in ri[:300]:
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

    # NM optimization
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
    all_models["mega_nm"] = {"oof": nm_oof, "test": nm_test, "rmse": bo}
    analyze_res(y_train, nm_oof, "NM ensemble")

    # Piecewise cal on top
    po, pt, ps = piecewise_cal(nm_oof, nm_test, y_train)
    if ps < bo - 0.005:
        all_models["mega_nm_pwcal"] = {"oof": po, "test": pt, "rmse": ps}
        print(f"  + Piecewise: {bo:.4f} → {ps:.4f}")
        analyze_res(y_train, po, "NM + pwcal")

    # Also try greedy + pwcal
    g_oof = oofs[:, sel].mean(axis=1)
    g_test = tests[:, sel].mean(axis=1)
    po2, pt2, ps2 = piecewise_cal(g_oof, g_test, y_train)
    all_models["mega_greedy_pwcal"] = {"oof": po2, "test": pt2, "rmse": ps2}
    print(f"  Greedy + pwcal: {ps2:.4f}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 27 SUMMARY — PARADIGM SHIFT RESULTS")
    print("=" * 70)

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    # Only show new models (not prev_*)
    new_ranked = [(n, d) for n, d in ranked if not n.startswith("prev_")]
    print("\n  New models (Phase 27):")
    for name, data in new_ranked[:30]:
        star = " ★★★" if data["rmse"] < 13.5 else (" ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else ""))
        print(f"    {data['rmse']:.4f}  {name}{star}")

    print("\n  Overall top 10 (including previous):")
    for name, data in ranked[:10]:
        star = " ★★★" if data["rmse"] < 13.5 else (" ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else ""))
        print(f"    {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 26 best: 13.5954")
    improvement = 13.5954 - best_data["rmse"]
    print(f"  Improvement: {improvement:+.4f}")
    analyze_res(y_train, best_data["oof"], f"BEST ({best_name})")

    # Highlight new approach contributions
    print("\n  === New Approach Contributions ===")
    bl = all_models.get("baseline", {}).get("rmse", 99)
    for name in ["quantile_nm_blend", "rank_isotonic", "rank_poly3",
                 "lgbm_rank_feat", "cos2d_n20", "dann_direct_d32",
                 "lgbm_dann_d32", "mega_nm"]:
        if name in all_models:
            d = all_models[name]
            delta = d["rmse"] - bl
            print(f"    {d['rmse']:.4f} ({delta:+.2f} vs baseline) {name}")

    # Save
    sd = Path("submissions"); sd.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        if not name.startswith("prev_"):
            sub = pd.DataFrame({"sample number": test_ids.values, "含水率": data["test"]})
            p = sd / f"submission_phase27_{i+1}_{ts}.csv"
            sub.to_csv(p, index=False)
            print(f"  Saved: {p} ({data['rmse']:.4f} {name})")

    od = Path("runs") / f"phase27_{ts}"
    od.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked if not n.startswith("prev_")},
        "phase": "27",
    }
    with open(od / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, data in all_models.items():
        if not name.startswith("prev_"):
            np.save(od / f"oof_{name}.npy", data["oof"])
            np.save(od / f"test_{name}.npy", data["test"])

    print(f"\n  Artifacts: {od}")
    print(f"  Total new models: {len([n for n in all_models if not n.startswith('prev_')])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
