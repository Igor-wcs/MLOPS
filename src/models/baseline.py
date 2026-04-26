"""
Baseline models: Scikit-Learn (Ridge Regression) e MLP PyTorch.
Servem como referência de comparação para o modelo LSTM principal.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def train_ridge_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Treina um baseline Ridge Regression (flatten das sequências).

    Args:
        X_train: Shape (n_samples, seq_len, n_features).
        y_train: Shape (n_samples,).
        X_test: Shape (n_samples, seq_len, n_features).
        y_test: Shape (n_samples,).

    Returns:
        Dicionário de métricas {"mae": float, "rmse": float}.
    """
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = Ridge(alpha=1.0)
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    logger.info("Ridge baseline: MAE=%.4f, RMSE=%.4f", metrics["mae"], metrics["rmse"])
    return metrics


class MLPBaseline(nn.Module):
    """MLP simples como baseline PyTorch.

    Args:
        input_dim: Dimensão de entrada (seq_len * n_features).
        hidden_dim: Dimensão da camada oculta.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))
