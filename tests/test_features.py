"""Testes de feature engineering para séries temporais LSTM."""

import numpy as np
import pytest
from src.features.feature_engineering import preparar_janelas_temporais

@pytest.fixture
def serie_temporal_mock():
    """Mock de uma série de preços de fechamento (100 dias)."""
    return np.linspace(10, 110, 100)

def test_dimensoes_janela_temporal(serie_temporal_mock):
    """Garante que as dimensões do tensor X e array y estão corretas."""
    window_size = 10
    X, y, scaler = preparar_janelas_temporais(
        serie_temporal_mock, window_size=window_size
    )
    
    expected_samples = len(serie_temporal_mock) - window_size
    assert X.shape == (expected_samples, window_size, 1), "Shape de X incorreto"
    assert y.shape == (expected_samples,), "Shape de y incorreto"

def test_escalonamento_limites(serie_temporal_mock):
    """Garante que os dados passaram pelo MinMaxScaler e estão entre 0 e 1."""
    X, y, scaler = preparar_janelas_temporais(serie_temporal_mock)
    
    assert np.min(X) >= 0.0, "Existem valores menores que 0 após o scaler"
    assert np.max(X) <= 1.0, "Existem valores maiores que 1 após o scaler"