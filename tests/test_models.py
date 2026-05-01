"""Testes de modelos LSTM — factory, forward pass, métricas de negócio."""

import numpy as np
import pytest
import torch

from src.models.lstm_factory import LSTM, LSTMFactory
from src.models.lstm_params import LSTMParams
from src.models.train import compute_sigma_metric


class TestLSTMFactory:
    """Testes do padrão Factory para criação de modelos (Injeção de Dependência)."""

    def test_create_basic_model(self, lstm_layer_config: dict) -> None:
        params = LSTMParams(input_size=5, hidden_size=32, num_layers=1, output_size=1)
        factory = LSTMFactory(lstm_layer_config, params)
        model = factory.create()

        assert isinstance(model, LSTM)
        assert len(model.layers) == 2  # LSTM + Linear

    def test_forward_pass_shape(self, lstm_layer_config: dict) -> None:
        """Garante que as dimensões do tensor no forward pass estão corretas."""
        params = LSTMParams(input_size=5, hidden_size=32, num_layers=1, output_size=1)
        factory = LSTMFactory(lstm_layer_config, params)
        model = factory.create()

        # Batch=4, Sequence Length=30 (janela), Features=5
        x = torch.randn(4, 30, 5)
        output = model(x)

        assert output.shape == (
            4,
            1,
        ), "A saída do modelo deve prever 1 valor por amostra do batch."

    def test_invalid_layer_raises(self) -> None:
        """Garante que a Factory falha graciosamente ao receber configurações inválidas."""
        params = LSTMParams(input_size=5)
        factory = LSTMFactory({"bad": "NonExistent"}, params)

        with pytest.raises(ValueError, match="não suportada"):
            factory.create()


class TestLSTMParams:
    """Testes de validação de parâmetros com Pydantic."""

    def test_valid_params(self) -> None:
        params = LSTMParams(input_size=5)
        # Verifica os defaults
        assert params.hidden_size == 64
        assert params.output_size == 1

    def test_invalid_input_size(self) -> None:
        with pytest.raises(Exception):
            LSTMParams(input_size=0)

    def test_dropout_range(self) -> None:
        with pytest.raises(Exception):
            LSTMParams(input_size=5, dropout=1.5)


class TestSigmaMetric:
    """Testes da métrica de negócio (Tolerância de erro baseada em Volatilidade σ)."""

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

    def test_partial_tolerance(self) -> None:
        """Testa se a métrica captura corretamente quem está dentro e fora da banda de tolerância."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        sigma = np.std(y_true)  # std = 11.18

        # Previsões:
        # Índice 0 e 1: Exatas
        # Índice 2: Erro de 0.3 * sigma (Dentro da tolerância padrão de 0.5)
        # Índice 3: Erro de 0.8 * sigma (Fora da tolerância)
        y_pred = np.array([10.0, 20.0, 30.0 + (0.3 * sigma), 40.0 + (0.8 * sigma)])

        metrics = compute_sigma_metric(y_true, y_pred)

        # Exatamente 3 de 4 (75%) devem estar dentro da tolerância
        assert metrics["pct_within_tolerance"] == 75.0
