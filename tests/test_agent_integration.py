"""Testes de integração para o Agente ReAct e Roteamento."""

import pytest
from unittest.mock import patch, MagicMock
from src.agent.react_agent import RouterAgent
from langchain_core.tools import Tool


@pytest.fixture
def mock_tools():
    """Cria ferramentas mockadas para testar o roteamento."""
    tool1 = Tool(
        name="obter_previsao_lstm", func=lambda x: "Previsão Mock", description="lstm"
    )
    tool2 = Tool(
        name="obter_cotacao_atual", func=lambda x: "Cotação Mock", description="cotação"
    )
    tool3 = Tool(
        name="consultar_base_conhecimento",
        func=lambda x: "RAG Mock",
        description="base",
    )
    return [tool1, tool2, tool3]


class TestRouterAgent:
    """Validação da lógica de roteamento do Agente Router."""

    @patch("src.agent.react_agent.HuggingFacePipeline.from_model_id")
    def test_agent_routing_logic(self, mock_llm_factory, mock_tools) -> None:
        """Testa se o agente escolhe a ferramenta correta baseada em palavras-chave."""
        # Configura o LLM mockado
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Resposta Final Mock"
        mock_llm_factory.return_value = mock_llm

        agent = RouterAgent(mock_tools)

        # Teste 1: Keyword 'previsão' deve disparar LSTM
        res1 = agent.run("Qual a previsão para amanhã?")
        assert res1["intermediate_steps"][0]["tool"] == "obter_previsao_lstm"

        # Teste 2: Keyword 'cotação' deve disparar Cotação Atual
        res2 = agent.run("Qual o preço hoje?")
        assert res2["intermediate_steps"][0]["tool"] == "obter_cotacao_atual"

        # Teste 3: Keyword 'política' deve disparar RAG
        res3 = agent.run("Qual a política de dividendos?")
        assert res3["intermediate_steps"][0]["tool"] == "consultar_base_conhecimento"

    @patch("src.agent.react_agent.HuggingFacePipeline.from_model_id")
    def test_agent_fallback_routing(self, mock_llm_factory, mock_tools) -> None:
        """Testa o fallback para o LLM quando não há keywords claras."""
        mock_llm = MagicMock()
        # O LLM decide qual ferramenta usar
        mock_llm.invoke.side_effect = ["obter_previsao_lstm", "Resposta Final"]
        mock_llm_factory.return_value = mock_llm

        agent = RouterAgent(mock_tools)
        # Uma pergunta vaga sem keywords mapeadas
        res = agent.run("Me conte algo aleatório")

        assert "answer" in res
        assert len(res["intermediate_steps"]) > 0
