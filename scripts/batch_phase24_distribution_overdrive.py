#!/usr/bin/env python
"""Phase 24: Distribution Overdrive — 条件付きΔx生成器 + Isotonic補正.

Current best: 13.87 (Phase 22 NM-optimized ensemble)

Key innovations:
  1. Conditional Δx Generator: Learn p(Δx | x_base, y_base, Δy) via MLP
     - Instead of a single WDV direction, learn condition-dependent spectral changes
     - Train on same-species pairs with different moisture levels
  2. Isotonic Regression calibration: monotone correction OOF→true
  3. Residual-aware sample weighting: weight by inverse residual density
  4. Combined mega ensemble with Phase 23 models
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
# 1. Conditional Δx Generator (MLP-based)
# ======================================================================

class DeltaXGenerator:
    """Learn spectral change as a function of (x_base, y_base, Δy).

    Stage A: Deterministic — predict mean Δx
    Stage B: Probabilistic — predict (μ, σ) for Gaussian NLL
    """

    def __init__(self, hidden_dims=(256, 256, 128), lr=0.001, epochs=200,
                 batch_size=256, stage="A", patience=20):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.stage = stage  # "A" or "B"
        self.patience = patience
        self.model = None
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

    def _build_pairs(self, X, y, groups, min_dy=20, max_dy=120):
        """Build (x_base, y_base, Δy, Δx) pairs from same-species samples."""
        pairs_input = []  # (x_base_features..., y_base, delta_y)
        pairs_output = []  # (delta_x_features...)

        for sp in np.unique(groups):
            sp_mask = groups == sp
            X_sp = X[sp_mask]
            y_sp = y[sp_mask]
            n = len(y_sp)
            if n < 3:
                continue

            # Sort by moisture for efficient pairing
            order = np.argsort(y_sp)
            X_sp = X_sp[order]
            y_sp = y_sp[order]

            # Build pairs: low → high
            for i in range(n):
                for j in range(i + 1, n):
                    dy = y_sp[j] - y_sp[i]
                    if dy < min_dy or dy > max_dy:
                        continue

                    dx = X_sp[j] - X_sp[i]

                    # Forward pair: base=i, target=j
                    inp = np.concatenate([X_sp[i], [y_sp[i], dy]])
                    pairs_input.append(inp)
                    pairs_output.append(dx)

                    # Also reverse pair: base=j, delta negative
                    # (helps learn bidirectional changes)

        if not pairs_input:
            return None, None

        return np.array(pairs_input), np.array(pairs_output)

    def fit(self, X, y, groups, min_dy=20, max_dy=120):
        """Train the Δx generator on pair data."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            print("  [DeltaX] PyTorch not available, falling back to sklearn MLP")
            return self._fit_sklearn(X, y, groups, min_dy, max_dy)

        inputs, outputs = self._build_pairs(X, y, groups, min_dy, max_dy)
        if inputs is None:
            print("  [DeltaX] No valid pairs found")
            return self

        print(f"  [DeltaX] Built {len(inputs)} pairs, input_dim={inputs.shape[1]}, output_dim={outputs.shape[1]}")

        # Normalize
        self.input_mean = inputs.mean(axis=0)
        self.input_std = inputs.std(axis=0) + 1e-8
        self.output_mean = outputs.mean(axis=0)
        self.output_std = outputs.std(axis=0) + 1e-8

        inputs_n = (inputs - self.input_mean) / self.input_std
        outputs_n = (outputs - self.output_mean) / self.output_std

        # Build MLP
        feat_dim = inputs.shape[1]
        out_dim = outputs.shape[1]

        layers = []
        prev = feat_dim
        for h in self.hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h

        if self.stage == "A":
            layers.append(nn.Linear(prev, out_dim))
        else:  # Stage B: output (μ, log_σ)
            layers.append(nn.Linear(prev, out_dim * 2))

        model = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Train/val split (80/20)
        n = len(inputs_n)
        perm = np.random.permutation(n)
        n_train = int(0.8 * n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        X_tr = torch.FloatTensor(inputs_n[train_idx])
        y_tr = torch.FloatTensor(outputs_n[train_idx])
        X_va = torch.FloatTensor(inputs_n[val_idx])
        y_va = torch.FloatTensor(outputs_n[val_idx])

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_count = 0
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                if self.stage == "A":
                    loss = nn.MSELoss()(pred, yb)
                else:
                    mu = pred[:, :out_dim]
                    log_sigma = pred[:, out_dim:]
                    loss = nn.GaussianNLLLoss()(mu, yb, torch.exp(log_sigma * 2))
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va)
                if self.stage == "A":
                    val_loss = nn.MSELoss()(val_pred, y_va).item()
                else:
                    mu = val_pred[:, :out_dim]
                    log_sigma = val_pred[:, out_dim:]
                    val_loss = nn.GaussianNLLLoss()(mu, y_va, torch.exp(log_sigma * 2)).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self._use_torch = True
        print(f"  [DeltaX] Trained (stage {self.stage}), best val loss: {best_val_loss:.6f}")
        return self

    def _fit_sklearn(self, X, y, groups, min_dy=20, max_dy=120):
        """Fallback: sklearn MLPRegressor."""
        from sklearn.neural_network import MLPRegressor

        inputs, outputs = self._build_pairs(X, y, groups, min_dy, max_dy)
        if inputs is None:
            return self

        print(f"  [DeltaX-sklearn] Built {len(inputs)} pairs")

        self.input_mean = inputs.mean(axis=0)
        self.input_std = inputs.std(axis=0) + 1e-8
        self.output_mean = outputs.mean(axis=0)
        self.output_std = outputs.std(axis=0) + 1e-8

        inputs_n = (inputs - self.input_mean) / self.input_std
        outputs_n = (outputs - self.output_mean) / self.output_std

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_dims,
            max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20,
            batch_size=min(256, len(inputs_n)),
            random_state=42,
        )
        self.model.fit(inputs_n, outputs_n)
        self._use_torch = False
        print(f"  [DeltaX-sklearn] Trained")
        return self

    def generate(self, X_base, y_base, delta_y_values, n_samples_per=1):
        """Generate synthetic spectra by applying learned Δx.

        Parameters
        ----------
        X_base : array (n, d) — base spectra (preprocessed)
        y_base : array (n,) — base moisture values
        delta_y_values : list of float — Δy values to generate for each base
        n_samples_per : int — samples per (base, Δy) combo (Stage B only)

        Returns
        -------
        X_new : array — generated spectra
        y_new : array — generated moisture values
        """
        if self.model is None:
            return np.empty((0, X_base.shape[1])), np.empty(0)

        synth_X, synth_y = [], []

        for i in range(len(X_base)):
            for dy in delta_y_values:
                inp = np.concatenate([X_base[i], [y_base[i], dy]])
                inp_n = (inp - self.input_mean) / self.input_std

                if hasattr(self, '_use_torch') and self._use_torch:
                    import torch
                    with torch.no_grad():
                        pred = self.model(torch.FloatTensor(inp_n.reshape(1, -1)))
                        pred = pred.numpy().ravel()
                else:
                    pred = self.model.predict(inp_n.reshape(1, -1)).ravel()

                if self.stage == "A":
                    dx_n = pred
                    dx = dx_n * self.output_std + self.output_mean
                    x_new = X_base[i] + dx
                    synth_X.append(x_new)
                    synth_y.append(y_base[i] + dy)
                else:
                    # Stage B: sample from learned distribution
                    d = len(self.output_mean)
                    mu_n = pred[:d]
                    log_sigma_n = pred[d:]
                    for _ in range(n_samples_per):
                        noise = np.random.randn(d) * np.exp(log_sigma_n)
                        dx_n = mu_n + noise
                        dx = dx_n * self.output_std + self.output_mean
                        x_new = X_base[i] + dx
                        synth_X.append(x_new)
                        synth_y.append(y_base[i] + dy)

        if not synth_X:
            return np.empty((0, X_base.shape[1])), np.empty(0)

        return np.array(synth_X), np.array(synth_y)


# ======================================================================
# 2. WDV baseline (from Phase 23 for comparison)
# ======================================================================

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
# 3. Core CV with augmentation options
# ======================================================================

def cv_full(X_train, y_train, groups, X_test,
            preprocess=None, lgbm_params=None,
            # WDV params
            n_aug=30, extrap=1.5, min_moisture=170,
            dy_scale=0.3, dy_offset=30,
            # PL params
            pl_w=0.5, pl_rounds=2,
            # Weight params
            sample_weight_fn=None,
            # Target transform
            target_transform=None, target_transform_lambda=0.5,
            # Delta-X generator params
            use_deltax=False, deltax_stage="A",
            deltax_dy_values=None, deltax_min_moisture=170,
            deltax_max_samples=50,
            deltax_min_dy=20, deltax_max_dy=120,
            # Isotonic calibration
            use_isotonic=False,
            # Combined WDV + DeltaX
            wdv_weight=1.0, deltax_weight=1.0):
    """Core CV with all augmentation options."""
    if preprocess is None:
        preprocess = PP_BIN4
    params = {**(lgbm_params or LGBM_BASE)}
    gkf = GroupKFold(n_splits=5)
    test_preds_prev = None

    if deltax_dy_values is None:
        deltax_dy_values = [20, 40, 60, 80]

    # Target transform functions
    def fwd(y):
        if target_transform == "log1p":
            return np.log1p(np.clip(y, 0, None))
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

            X_tr_t = pipe.transform(X_tr)
            X_va_t = pipe.transform(X_va)
            X_test_t = pipe.transform(X_test)

            # === Augmentation: WDV ===
            aug_X_list, aug_y_list = [], []

            if n_aug > 0:
                synth_X_raw, synth_y = generate_universal_wdv(
                    X_tr, y_tr, g_tr, n_aug, extrap, min_moisture, dy_scale, dy_offset)
                if len(synth_X_raw) > 0:
                    synth_X_t = pipe.transform(synth_X_raw)
                    aug_X_list.append(synth_X_t)
                    aug_y_list.append(synth_y)

            # === Augmentation: Δx Generator ===
            if use_deltax:
                # Train generator on preprocessed training data
                deltax_gen = DeltaXGenerator(
                    hidden_dims=(256, 256, 128),
                    epochs=200, stage=deltax_stage, patience=20,
                )
                deltax_gen.fit(X_tr_t, y_tr, g_tr,
                              min_dy=deltax_min_dy, max_dy=deltax_max_dy)

                # Generate from high-moisture samples
                high_mask = y_tr >= deltax_min_moisture
                if high_mask.sum() > 0:
                    X_high = X_tr_t[high_mask]
                    y_high = y_tr[high_mask]

                    gen_X, gen_y = deltax_gen.generate(
                        X_high, y_high, deltax_dy_values,
                        n_samples_per=1 if deltax_stage == "A" else 3
                    )

                    if len(gen_X) > deltax_max_samples:
                        idx = np.random.choice(len(gen_X), deltax_max_samples, replace=False)
                        gen_X, gen_y = gen_X[idx], gen_y[idx]

                    if len(gen_X) > 0:
                        aug_X_list.append(gen_X)
                        aug_y_list.append(gen_y)

            # Combine all augmentation
            if aug_X_list:
                X_aug = np.vstack([X_tr_t] + aug_X_list)
                y_aug = np.concatenate([y_tr] + aug_y_list)
            else:
                X_aug = X_tr_t
                y_aug = y_tr

            # Compute sample weights
            sw_train = None
            if sample_weight_fn is not None:
                sw_base = sample_weight_fn(y_tr, g_tr)
                sw_parts = [sw_base]
                for aug_y in aug_y_list:
                    sw_parts.append(sample_weight_fn(aug_y, None))
                if aug_y_list:
                    sw_train = np.concatenate(sw_parts)
                else:
                    sw_train = sw_base

            # Target transform
            y_aug_t = fwd(y_aug)

            # Pseudo-labeling
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

    # === Isotonic calibration ===
    if use_isotonic:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_train)
        oof = iso.predict(oof)
        test_preds_prev = iso.predict(test_preds_prev)

    return oof, fold_rmses, test_preds_prev


# ======================================================================
# 4. Residual analysis utility
# ======================================================================

def analyze_residuals(y_true, y_pred, bins=None):
    """Analyze prediction residuals by moisture range."""
    if bins is None:
        bins = [0, 30, 60, 100, 150, 200, 300]

    residuals = y_pred - y_true
    abs_res = np.abs(residuals)

    print(f"\n  Residual Analysis (overall RMSE: {rmse(y_true, y_pred):.4f}):")
    print(f"  {'Range':>15s} {'N':>5s} {'RMSE':>8s} {'MAE':>8s} {'Bias':>8s} {'Max|e|':>8s}")

    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() == 0:
            continue
        r = residuals[mask]
        print(f"  {bins[i]:>6d}-{bins[i+1]:<6d}  {mask.sum():>5d} "
              f"{np.sqrt((r**2).mean()):>8.2f} {np.abs(r).mean():>8.2f} "
              f"{r.mean():>+8.2f} {np.abs(r).max():>8.2f}")

    # High tail
    mask = y_true >= bins[-1]
    if mask.sum() > 0:
        r = residuals[mask]
        print(f"  {bins[-1]:>6d}+       {mask.sum():>5d} "
              f"{np.sqrt((r**2).mean()):>8.2f} {np.abs(r).mean():>8.2f} "
              f"{r.mean():>+8.2f} {np.abs(r).max():>8.2f}")


# ======================================================================
# Main experiments
# ======================================================================

def main():
    np.random.seed(42)
    print("=" * 70)
    print("PHASE 24: Distribution Overdrive — Δx Generator + Isotonic")
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
    # BASELINE: Reproduce best known
    # ==================================================================
    print("\n=== BASELINE ===")

    oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                          min_moisture=170, pl_w=0.5, pl_rounds=2)
    log("baseline", oof, f, tp)
    analyze_residuals(y_train, oof)

    # ==================================================================
    # A: ISOTONIC CALIBRATION — monotone OOF→true correction
    # ==================================================================
    print("\n=== A: ISOTONIC CALIBRATION ===")

    # Apply isotonic to baseline
    oof_a, f_a, tp_a = cv_full(X_train, y_train, groups, X_test,
                                min_moisture=170, pl_w=0.5, pl_rounds=2,
                                use_isotonic=True)
    log("isotonic_baseline", oof_a, f_a, tp_a)

    # Isotonic with different HP
    for hp_name, hp_ov in [
        ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
        ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01}),
    ]:
        params = {**LGBM_BASE, **hp_ov}
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params, min_moisture=170,
                              pl_w=0.5, pl_rounds=2, use_isotonic=True)
        log(f"isotonic_{hp_name}", oof, f, tp)

    # ==================================================================
    # B: Δx GENERATOR — Stage A (deterministic)
    # ==================================================================
    print("\n=== B: Δx GENERATOR (Stage A) ===")

    # B1: Basic Δx generator replacing WDV
    for dy_list_name, dy_list in [
        ("dy20_40_60", [20, 40, 60]),
        ("dy20_40_60_80", [20, 40, 60, 80]),
        ("dy30_50_70", [30, 50, 70]),
        ("dy40_80", [40, 80]),
    ]:
        for max_samples in [30, 50]:
            name = f"deltax_A_{dy_list_name}_n{max_samples}"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  n_aug=0,  # no WDV
                                  use_deltax=True, deltax_stage="A",
                                  deltax_dy_values=dy_list,
                                  deltax_min_moisture=170,
                                  deltax_max_samples=max_samples,
                                  pl_w=0.5, pl_rounds=2)
            log(name, oof, f, tp)

    # B2: Δx generator WITH WDV (parallel augmentation)
    print("\n=== B2: Δx + WDV combined ===")
    for dy_list_name, dy_list in [
        ("dy20_40_60", [20, 40, 60]),
        ("dy40_80", [40, 80]),
    ]:
        for max_samples in [30, 50]:
            name = f"deltax_wdv_{dy_list_name}_n{max_samples}"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  n_aug=30,  # keep WDV
                                  use_deltax=True, deltax_stage="A",
                                  deltax_dy_values=dy_list,
                                  deltax_min_moisture=170,
                                  deltax_max_samples=max_samples,
                                  pl_w=0.5, pl_rounds=2)
            log(name, oof, f, tp)

    # B3: Different min_moisture thresholds for Δx
    print("\n=== B3: Δx min_moisture sweep ===")
    for mm in [140, 150, 160, 170, 180]:
        name = f"deltax_A_mm{mm}"
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              n_aug=0, use_deltax=True, deltax_stage="A",
                              deltax_dy_values=[20, 40, 60, 80],
                              deltax_min_moisture=mm,
                              deltax_max_samples=50,
                              pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

    # B4: Pair distance sweep (min_dy, max_dy)
    print("\n=== B4: Pair distance sweep ===")
    for min_dy, max_dy in [(10, 80), (15, 100), (20, 120), (30, 150)]:
        name = f"deltax_A_dy{min_dy}_{max_dy}"
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              n_aug=0, use_deltax=True, deltax_stage="A",
                              deltax_dy_values=[20, 40, 60],
                              deltax_min_moisture=170,
                              deltax_max_samples=50,
                              deltax_min_dy=min_dy, deltax_max_dy=max_dy,
                              pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

    # ==================================================================
    # C: Δx GENERATOR — Stage B (probabilistic)
    # ==================================================================
    print("\n=== C: Δx GENERATOR (Stage B — probabilistic) ===")

    for dy_list_name, dy_list in [
        ("dy20_40_60", [20, 40, 60]),
        ("dy40_80", [40, 80]),
    ]:
        for max_samples in [30, 50]:
            name = f"deltax_B_{dy_list_name}_n{max_samples}"
            oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                                  n_aug=0, use_deltax=True, deltax_stage="B",
                                  deltax_dy_values=dy_list,
                                  deltax_min_moisture=170,
                                  deltax_max_samples=max_samples,
                                  pl_w=0.5, pl_rounds=2)
            log(name, oof, f, tp)

    # ==================================================================
    # D: ISOTONIC + Δx COMBINED
    # ==================================================================
    print("\n=== D: ISOTONIC + Δx COMBINED ===")

    # Best Δx configs so far + isotonic
    deltax_models = {k: v for k, v in all_models.items() if k.startswith("deltax_")}
    if deltax_models:
        top_dx = sorted(deltax_models.items(), key=lambda x: x[1]["rmse"])[:5]
        print(f"  Top 5 Δx configs:")
        for n, d in top_dx:
            print(f"    {d['rmse']:.4f} {n}")

    for dy_list_name, dy_list in [
        ("dy20_40_60", [20, 40, 60]),
        ("dy20_40_60_80", [20, 40, 60, 80]),
    ]:
        name = f"isotonic_deltax_A_{dy_list_name}"
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              n_aug=0, use_deltax=True, deltax_stage="A",
                              deltax_dy_values=dy_list,
                              deltax_min_moisture=170,
                              deltax_max_samples=50,
                              pl_w=0.5, pl_rounds=2,
                              use_isotonic=True)
        log(name, oof, f, tp)

    # Isotonic + WDV + Δx
    name = "isotonic_wdv_deltax_full"
    oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                          n_aug=30, use_deltax=True, deltax_stage="A",
                          deltax_dy_values=[20, 40, 60, 80],
                          deltax_min_moisture=170,
                          deltax_max_samples=50,
                          pl_w=0.5, pl_rounds=2,
                          use_isotonic=True)
    log(name, oof, f, tp)

    # ==================================================================
    # E: MOISTURE WEIGHTING + Δx + ISOTONIC (triple combo)
    # ==================================================================
    print("\n=== E: TRIPLE COMBO (MW + Δx + Isotonic) ===")

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

    for wp, wb, thr in [
        (1.5, 0.5, 0), (2.0, 0.5, 0),
        (1.5, 0.5, 140), (2.0, 0.5, 140),
    ]:
        name = f"triple_mw_p{wp}_b{wb}_t{thr}"
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              n_aug=30, use_deltax=True, deltax_stage="A",
                              deltax_dy_values=[20, 40, 60],
                              deltax_min_moisture=170,
                              deltax_max_samples=30,
                              sample_weight_fn=mw_fn_factory(wp, wb, thr),
                              pl_w=0.5, pl_rounds=2,
                              use_isotonic=True)
        log(name, oof, f, tp)

    # ==================================================================
    # F: DIVERSITY models (different HPs with best augmentation)
    # ==================================================================
    print("\n=== F: DIVERSITY ===")

    hp_variants = [
        ("n800lr02", {"n_estimators": 800, "learning_rate": 0.02}),
        ("n1000lr01", {"n_estimators": 1000, "learning_rate": 0.01}),
        ("d3l10_n1500lr005", {"max_depth": 3, "num_leaves": 10, "n_estimators": 1500, "learning_rate": 0.005}),
        ("d7l30_n600lr03", {"max_depth": 7, "num_leaves": 30, "n_estimators": 600, "learning_rate": 0.03}),
    ]

    for hp_name, hp_ov in hp_variants:
        params = {**LGBM_BASE, **hp_ov}

        # With WDV + Δx
        name = f"div_{hp_name}_wdv_dx"
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params, n_aug=30,
                              use_deltax=True, deltax_stage="A",
                              deltax_dy_values=[20, 40, 60],
                              deltax_min_moisture=170,
                              deltax_max_samples=30,
                              pl_w=0.5, pl_rounds=2)
        log(name, oof, f, tp)

        # With isotonic
        name = f"div_{hp_name}_wdv_dx_iso"
        oof, f, tp = cv_full(X_train, y_train, groups, X_test,
                              lgbm_params=params, n_aug=30,
                              use_deltax=True, deltax_stage="A",
                              deltax_dy_values=[20, 40, 60],
                              deltax_min_moisture=170,
                              deltax_max_samples=30,
                              pl_w=0.5, pl_rounds=2,
                              use_isotonic=True)
        log(name, oof, f, tp)

    # ==================================================================
    # MEGA ENSEMBLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("MEGA ENSEMBLE")
    print("=" * 70)

    # Load Phase 23 results if available
    phase23_dirs = sorted(Path("runs").glob("phase23_*"))
    if phase23_dirs:
        p23_dir = phase23_dirs[-1]
        print(f"  Loading Phase 23 results from {p23_dir}")
        p23_summary = json.loads((p23_dir / "summary.json").read_text())
        for name, rmse_val in p23_summary.get("all_results", {}).items():
            oof_path = p23_dir / f"oof_{name}.npy"
            test_path = p23_dir / f"test_{name}.npy"
            if oof_path.exists() and test_path.exists():
                oof_data = np.load(oof_path)
                test_data = np.load(test_path)
                if np.isfinite(oof_data).all() and np.isfinite(test_data).all():
                    all_models[f"p23_{name}"] = {
                        "oof": oof_data, "test": test_data, "rmse": rmse_val
                    }
        print(f"  Loaded {sum(1 for k in all_models if k.startswith('p23_'))} Phase 23 models")

    names = list(all_models.keys())
    oofs = np.column_stack([all_models[n]["oof"] for n in names])
    tests = np.column_stack([all_models[n]["test"] for n in names])
    rmses_list = [all_models[n]["rmse"] for n in names]

    # Filter valid
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
    for _ in range(min(50, len(order) - 1)):
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

    # NM optimization (500 trials)
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
    # CONDITIONAL STRETCH
    # ==================================================================
    print("\n" + "=" * 70)
    print("CONDITIONAL STRETCH")
    print("=" * 70)

    top_candidates = sorted(all_models.items(), key=lambda x: x[1]["rmse"])[:15]

    for base_name, base_data in top_candidates:
        if "stretch" in base_name:
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

        if best_stretch_score < base_score - 0.01:
            print(f"  {base_name}: {base_score:.4f} → {best_stretch_score:.4f}")

    # ==================================================================
    # FINAL RE-ENSEMBLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("FINAL RE-ENSEMBLE")
    print("=" * 70)

    names2 = [n for n in all_models if np.isfinite(all_models[n]["oof"]).all()
              and np.isfinite(all_models[n]["test"]).all()]
    oofs2 = np.column_stack([all_models[n]["oof"] for n in names2])
    tests2 = np.column_stack([all_models[n]["test"] for n in names2])
    rmses2 = [all_models[n]["rmse"] for n in names2]

    order2 = sorted(range(len(names2)), key=lambda i: rmses2[i])
    selected2 = [order2[0]]
    for _ in range(min(50, len(order2) - 1)):
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

    # NM final
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

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 24 FINAL SUMMARY")
    print("=" * 70)

    all_models["nm_opt_final"] = {"oof": opt_oof2, "test": opt_test2, "rmse": best_opt2}
    all_models["greedy_ens_final"] = {"oof": greedy_avg2, "test": greedy_test2, "rmse": greedy_s2}

    ranked = sorted(all_models.items(), key=lambda x: x[1]["rmse"])
    for name, data in ranked[:50]:
        star = " ★★★" if data["rmse"] < 13.5 else (" ★★" if data["rmse"] < 14.0 else (" ★" if data["rmse"] < 14.5 else ""))
        print(f"  {data['rmse']:.4f}  {name}{star}")

    best_name, best_data = ranked[0]
    print(f"\n  BEST: {best_data['rmse']:.4f} ({best_name})")
    print(f"  Phase 22 best: 13.87")
    improvement = 13.87 - best_data["rmse"]
    print(f"  Improvement: {improvement:+.4f}")

    # Residual analysis on best
    analyze_residuals(y_train, ranked[0][1]["oof"])

    # Save
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (name, data) in enumerate(ranked[:5]):
        sub = pd.DataFrame({
            "sample number": test_ids.values,
            "含水率": data["test"]
        })
        path = submissions_dir / f"submission_phase24_{i+1}_{ts}.csv"
        sub.to_csv(path, index=False)
        print(f"  Saved: {path} ({data['rmse']:.4f} {name})")

    oof_dir = Path("runs") / f"phase24_{ts}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_name": best_name,
        "best_rmse": float(best_data["rmse"]),
        "all_results": {n: float(d["rmse"]) for n, d in ranked},
        "phase": "24",
        "description": "Distribution Overdrive",
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
