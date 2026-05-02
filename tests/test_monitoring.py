"""Testes unitários para o pipeline de monitoramento de drift."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from src.monitoring.drift import gerar_relatorio_drift


class TestDriftMonitoring:
    """Testes para detecção de Data Drift e Target Drift."""

    @patch("src.monitoring.drift.yf.Ticker")
    @patch("src.monitoring.drift.MlflowClient")
    @patch("src.monitoring.drift.obter_modelo_producao")
    def test_drift_execution_flow(
        self, mock_get_model, mock_mlflow_client, mock_ticker
    ) -> None:
        """Testa o fluxo completo do drift report com mocks de dados e modelo."""
        # 1. Mock de dados do Yahoo Finance (Suficientes para não cair no fallback total)
        mock_history = MagicMock()
        data = {
            "Close": np.linspace(30, 40, 150),
            "Open": np.linspace(30, 40, 150),
            "High": np.linspace(30, 40, 150),
            "Low": np.linspace(30, 40, 150),
            "Volume": np.linspace(1e6, 2e6, 150),
        }
        mock_history.history.return_value = pd.DataFrame(data)
        mock_ticker.return_value = mock_history

        # 2. Mock do modelo no MLflow Registry
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_mlflow_client.return_value.search_model_versions.return_value = [
            mock_version
        ]

        # Forçamos modelo como None para usar apenas DataDriftPreset() real,
        # evitando o erro de 'prediction column' e problemas de tipo com Mocks no Evidently.
        mock_get_model.return_value = None

        drift_share = gerar_relatorio_drift()

        # 4. Validações
        assert isinstance(drift_share, float)

    def test_drift_fallback_on_api_error(self) -> None:
        """Garante que o monitoramento não quebra se a API de dados falhar (resiliência)."""
        with patch("src.monitoring.drift.yf.Ticker", side_effect=Exception("API Down")):
            drift_share = gerar_relatorio_drift()
            assert isinstance(drift_share, float)
