"""1D Convolutional Neural Network for spectral regression.

sklearn-compatible wrapper around a PyTorch 1D CNN.
Designed for NIR spectral data where features are ordered wavelengths.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class _SpectralCNN(nn.Module):
    """1D CNN for spectral regression."""

    def __init__(
        self,
        in_features: int,
        n_filters: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        fc_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = []
        in_ch = 1
        current_len = in_features

        for filters, ks in zip(n_filters, kernel_sizes):
            pad = ks // 2
            layers.append(nn.Conv1d(in_ch, filters, ks, padding=pad))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            current_len = current_len // 2
            in_ch = filters

        self.conv = nn.Sequential(*layers)

        # Global average pooling + FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        fc_layers = []
        fc_in = n_filters[-1]
        for fc_size in fc_sizes:
            fc_layers.append(nn.Linear(fc_in, fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            fc_in = fc_size
        fc_layers.append(nn.Linear(fc_in, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)  # (batch, n_filters[-1])
        return self.fc(x).squeeze(-1)  # (batch,)


class CNN1DRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible 1D CNN regressor.

    Parameters
    ----------
    n_filters : tuple
        Number of filters per conv layer.
    kernel_sizes : tuple
        Kernel sizes per conv layer.
    fc_sizes : tuple
        Hidden layer sizes for the FC head.
    dropout : float
        Dropout rate.
    use_batchnorm : bool
        Whether to use batch normalization.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early stopping patience (based on training loss plateau).
    seed : int
        Random seed.
    verbose : int
        Verbosity level (0=silent, 1=per-epoch, 2=detailed).
    """

    def __init__(
        self,
        n_filters=(32, 64, 128),
        kernel_sizes=(7, 5, 3),
        fc_sizes=(128, 64),
        dropout=0.3,
        use_batchnorm=True,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=300,
        batch_size=64,
        patience=30,
        seed=42,
        verbose=0,
    ):
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.fc_sizes = fc_sizes
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n_features = X.shape[1]
        self.model_ = _SpectralCNN(
            in_features=n_features,
            n_filters=tuple(self.n_filters) if not isinstance(self.n_filters, tuple) else self.n_filters,
            kernel_sizes=tuple(self.kernel_sizes) if not isinstance(self.kernel_sizes, tuple) else self.kernel_sizes,
            fc_sizes=tuple(self.fc_sizes) if not isinstance(self.fc_sizes, tuple) else self.fc_sizes,
            dropout=self.dropout,
            use_batchnorm=self.use_batchnorm,
        )

        device = torch.device("cpu")
        self.model_.to(device)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        # Normalize target for stable training
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
            # Shuffle
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                X_batch = X_t[idx]
                y_batch = y_norm[idx]

                optimizer.zero_grad()
                pred = self.model_(X_batch)
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
                print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}, best={best_loss:.6f}")

            if patience_counter >= self.patience:
                if self.verbose >= 1:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X):
        self.model_.eval()
        device = torch.device("cpu")
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            preds = []
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i : i + self.batch_size]
                pred = self.model_(batch)
                preds.append(pred.cpu().numpy())

        y_pred = np.concatenate(preds)
        # Denormalize
        return y_pred * self.y_std_ + self.y_mean_
