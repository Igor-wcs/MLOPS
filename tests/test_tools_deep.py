"""Testes específicos para as ferramentas customizadas do Agente."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.agent.tools import consultar_base_conhecimento, obter_previsao_lstm


class TestAgentToolsDeep:
    """Testes detalhados para cobertura de lógica interna das ferramentas."""

    @patch("src.agent.tools.get_rag_pipeline")
    def test_consultar_base_conhecimento_flow(self, mock_get_rag) -> None:
        """Testa o fluxo da tool de RAG quando encontra documentos."""
        mock_rag = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Conteúdo de Teste"
        mock_rag.retrieve.return_value = [mock_doc]
        mock_get_rag.return_value = mock_rag

        res = consultar_base_conhecimento.run("pergunta teste")
        assert "Informações encontradas" in res
        assert "Conteúdo de Teste" in res

    @patch("src.agent.tools.get_rag_pipeline")
    def test_consultar_base_conhecimento_empty(self, mock_get_rag) -> None:
        """Testa o fluxo da tool de RAG quando o banco está vazio."""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = []
        mock_get_rag.return_value = mock_rag

        res = consultar_base_conhecimento.run("pergunta teste")
        assert "Não encontrei informações" in res

    @patch("src.agent.tools.joblib.load")
    @patch("src.agent.tools.get_model")
    @patch("src.agent.tools.yf.Ticker")
    def test_obter_previsao_lstm_error_handling(
        self, mock_yf, mock_get_model, mock_joblib
    ) -> None:
        """Testa o tratamento de erros na ferramenta LSTM (ex: falta de dados)."""
        mock_history = MagicMock()
        mock_history.history.return_value = pd.DataFrame()  # DataFrame vazio
        mock_yf.return_value = mock_history

        # Deve retornar uma mensagem de erro amigável em vez de quebrar
        res = obter_previsao_lstm.run("PETR4.SA")
        assert "Erro" in res or "Dados insuficientes" in res
