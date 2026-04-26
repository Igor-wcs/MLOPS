"""
Pipeline de dados para modelos LSTM de séries temporais.
Converte DataFrames em DataLoaders do PyTorch.

Design Pattern: Template Method — define os passos do pipeline de dados
(fetch → feature → normalize → sequence → split → dataloader) com
pontos de extensão para estratégias diferentes.
"""
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.features.feature_engineering import (
    compute_features,
    create_sequences,
    fetch_stock_data,
    normalize_features,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "returns",
    "volatility_20d",
    "sma_20",
    "sma_50",
    "price_range",
    "rsi_14",
]


def prepare_data(
    tickers: list[str],
    period: str = "2y",
    seq_len: int = 30,
    batch_size: int = 32,
    test_size: float = 0.2,
) -> tuple[DataLoader, DataLoader, dict[str, tuple[float, float]]]:
    """Pipeline completo: download → features → normalização → sequências → DataLoader.

    Args:
        tickers: Lista de tickers para download.
        period: Período de dados.
        seq_len: Comprimento da sequência.
        batch_size: Tamanho do batch.
        test_size: Proporção de teste.

    Returns:
        Tupla (train_loader, test_loader, norm_params).
    """
    raw_df = fetch_stock_data(tickers, period)
    featured_df = compute_features(raw_df)

    cols_to_normalize = [c for c in FEATURE_COLS if c in featured_df.columns]
    norm_df, norm_params = normalize_features(featured_df, cols_to_normalize)

    data_array = norm_df[cols_to_normalize].values.astype(np.float32)
    X, y = create_sequences(data_array, seq_len)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        "Data pipeline: train=%d, test=%d, features=%d, seq_len=%d",
        len(X_train),
        len(X_test),
        X_train.shape[2],
        seq_len,
    )

    return train_loader, test_loader, norm_params
