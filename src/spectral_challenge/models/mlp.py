"""Simple MLP regressor for spectral data.

sklearn-compatible wrapper around a PyTorch MLP.
MLPs can extrapolate better than tree models if the learned function is smooth.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class MLPRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible MLP regressor.

    Parameters
    ----------
    hidden_sizes : tuple
        Hidden layer sizes.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early stopping patience.
    seed : int
        Random seed.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        hidden_sizes=(256, 128, 64),
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=500,
        batch_size=64,
        patience=30,
        seed=42,
        verbose=0,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

    def _build_model(self, in_features):
        layers = []
        prev_size = in_features
        for size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model_ = self._build_model(X.shape[1])
        device = torch.device("cpu")
        self.model_.to(device)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        self.y_mean_ = y_t.mean().item()
        self.y_std_ = y_t.std().item() + 1e-8
        y_norm = (y_t - self.y_mean_) / self.y_std_

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model_.train()
        n = len(X_t)

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                X_batch = X_t[idx]
                y_batch = y_norm[idx]

                optimizer.zero_grad()
                pred = self.model_(X_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1

            if self.verbose >= 1 and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

            if patience_counter >= self.patience:
                if self.verbose >= 1:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X):
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            preds = []
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i: i + self.batch_size]
                pred = self.model_(batch).squeeze(-1)
                preds.append(pred.cpu().numpy())

        y_pred = np.concatenate(preds)
        return y_pred * self.y_std_ + self.y_mean_
