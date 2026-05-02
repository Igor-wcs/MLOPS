"""Fixtures globais para a suíte de testes (FastAPI, PyTorch, Features e LLM)."""

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

# Importa a nossa API para habilitar testes de integração
from src.serving.app import app


# ==========================================
# FIXTURES DE API E SERVING
# ==========================================
@pytest.fixture
def client() -> TestClient:
    """Cliente de teste do FastAPI para simular requisições HTTP."""
    return TestClient(app)


# ==========================================
# FIXTURES DE DADOS E FEATURES (MOCKS)
# ==========================================
@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Dados multivariados sintéticos para testes (6 features: OHLCV + EMA20)."""
    np.random.seed(42)
    n = 100
    base_price = 150.0
    prices = base_price + np.cumsum(np.random.randn(n) * 2)

    df = pd.DataFrame(
        {
            "Open": prices + np.random.randn(n) * 0.5,
            "High": prices + np.abs(np.random.randn(n)) * 2,
            "Low": prices - np.abs(np.random.randn(n)) * 2,
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 50_000_000, size=n).astype(float),
        }
    )
    # Adiciona a feature EMA20 exigida pelo pipeline
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df


@pytest.fixture
def sample_sequences() -> tuple[np.ndarray, np.ndarray]:
    """Sequências sintéticas para testes do modelo StockLSTM (6 features)."""
    np.random.seed(42)
    n_samples, seq_len, n_features = 50, 30, 6
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    return X, y


@pytest.fixture
def lstm_layer_config() -> dict[str, str]:
    """Configuração de camadas padrão para os testes do Factory."""
    return {"lstm1": "LSTM", "linear1": "Linear"}


# ==========================================
# FIXTURES DE AVALIAÇÃO (LLM / RAG)
# ==========================================
@pytest.fixture
def golden_set_data() -> list[dict]:
    """Golden set mínimo sintético para testes de avaliação RAGAS/Juiz."""
    return [
        {
            "question": "Qual o preço atual da AAPL?",
            "ground_truth": "O preço atual da AAPL é $185.50.",
            "contexts": ["AAPL fechou a $185.50 no último pregão."],
        },
        {
            "question": "A MSFT está em tendência de alta?",
            "ground_truth": "Sim, a MSFT está acima da SMA20 e SMA50.",
            "contexts": ["MSFT: preço $420, SMA20=$415, SMA50=$408."],
        },
    ]
