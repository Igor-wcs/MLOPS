"""Testes de feature engineering para séries temporais LSTM."""

import numpy as np
import pytest
from src.features.feature_engineering import preparar_janelas_temporais


@pytest.fixture
def serie_temporal_mock():
    """Mock de uma série multivariada (100 dias, 6 features)."""
    return np.random.randn(100, 6)


def test_dimensoes_janela_temporal(serie_temporal_mock):
    """Garante que as dimensões do tensor X e array y estão corretas."""
    window_size = 10
    X, y, scaler = preparar_janelas_temporais(
        serie_temporal_mock, window_size=window_size
    )

    expected_samples = len(serie_temporal_mock) - window_size
    # Agora o modelo espera 6 features
    assert X.shape == (expected_samples, window_size, 6), "Shape de X incorreto"
    assert y.shape == (expected_samples,), "Shape de y incorreto"


def test_escalonamento_limites(serie_temporal_mock):
    """Garante que os dados passaram pelo MinMaxScaler e estão entre 0 e 1."""
    X, y, scaler = preparar_janelas_temporais(serie_temporal_mock)

    assert np.min(X) >= 0.0, "Existem valores menores que 0 após o scaler"
    assert np.max(X) <= 1.0, "Existem valores maiores que 1 após o scaler"
