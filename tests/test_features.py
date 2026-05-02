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
    X, y, _scaler = preparar_janelas_temporais(
        serie_temporal_mock, window_size=window_size
    )

    expected_samples = len(serie_temporal_mock) - window_size
    # Agora o modelo espera 6 features
    assert X.shape == (expected_samples, window_size, 6), "Shape de X incorreto"
    assert y.shape == (expected_samples,), "Shape de y incorreto"


def test_escalonamento_limites(serie_temporal_mock):
    """Garante que os dados de treino estão entre 0 e 1 após o MinMaxScaler."""
    # Usamos uma semente fixa para reprodutibilidade no teste
    np.random.seed(42)
    X, _y, _scaler = preparar_janelas_temporais(serie_temporal_mock)

    # Verificamos a primeira janela (que é garantidamente do set de treino)
    assert np.min(X[0]) >= -1e-7, f"Valor mínimo {np.min(X[0])} menor que 0 no treino"
    assert (
        np.max(X[0]) <= 1.0 + 1e-7
    ), f"Valor máximo {np.max(X[0])} maior que 1 no treino"
