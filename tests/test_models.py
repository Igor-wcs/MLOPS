"""Testes de modelos LSTM — factory, forward pass, métricas de negócio."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from src.models.lstm_factory import get_model
from src.models.lstm_model import StockLSTM
from src.models.lstm_params import LSTMParams
from src.models.train import compute_sigma_metric


class TestLSTMFactory:
    """Testes da Factory e da Arquitetura StockLSTM."""

    def test_create_basic_model(self) -> None:
        """Garante que a factory instancia a classe correta com os parâmetros Pydantic."""
        params = LSTMParams(input_size=6, hidden_size=32, num_layers=2, output_size=1)
        model = get_model(params)

        assert isinstance(model, StockLSTM)
        assert model.hidden_size == 32
        assert model.num_layers == 2

    def test_forward_pass_shape(self) -> None:
        """Garante que as dimensões do tensor no forward pass estão corretas (Multivariado)."""
        params = LSTMParams(input_size=6, hidden_size=64, num_layers=2, output_size=1)
        model = get_model(params)

        # Batch=4, Sequence Length=30 (janela), Features=6
        x = torch.randn(4, 30, 6)
        output = model(x)

        assert output.shape == (
            4,
            1,
        ), "A saída do modelo deve prever 1 valor por amostra do batch."

    def test_device_consistency(self) -> None:
        """Verifica se o modelo inicializa estados no mesmo device que os dados."""
        params = LSTMParams(input_size=6)
        model = get_model(params)

        x = torch.randn(2, 10, 6)
        # Se o forward pass completar sem erro de RuntimeError
        # significa que h0/c0 foram criados corretamente via x.device
        try:
            _ = model(x)
        except RuntimeError as e:
            pytest.fail(f"Erro de consistência de hardware: {e}")


class TestLSTMParams:
    """Testes de validação de parâmetros com Pydantic."""

    def test_valid_params(self) -> None:
        params = LSTMParams(input_size=6)
        # Verifica os defaults
        assert params.hidden_size == 64
        assert params.output_size == 1

    def test_invalid_input_size(self) -> None:
        """Pydantic deve falhar para valores fora do range ou tipos errados."""
        with pytest.raises(ValidationError):
            LSTMParams(hidden_size=5)  # ge=16

    def test_dropout_range(self) -> None:
        with pytest.raises(ValidationError):
            LSTMParams(input_size=6, dropout=1.5)


class TestSigmaMetric:
    """Testes da métrica de negócio (Tolerância de erro baseada em Volatilidade sigma)."""

    def test_perfect_prediction(self) -> None:
        y_true = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        y_pred = y_true.copy()

        metrics = compute_sigma_metric(y_true, y_pred)
        assert metrics["sigma_error_mean"] == 0.0
        assert metrics["pct_within_tolerance"] == 100.0

    def test_large_error(self) -> None:
        y_true = np.array([100.0] * 30)
        y_pred = np.array([200.0] * 30)

        metrics = compute_sigma_metric(y_true, y_pred)
        assert metrics["pct_within_tolerance"] == 0.0
