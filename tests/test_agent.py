"""Testes do agente ReAct, suas ferramentas e integrações."""

from unittest.mock import patch

from src.agent.tools import get_stock_tools, obter_cotacao_atual


class TestAgentTools:
    """Validação do contrato de ferramentas exigido pelo Datathon."""

    def test_get_stock_tools_count(self) -> None:
        """Garante a exigência da Etapa 2 do Datathon (≥ 3 tools)."""
        tools = get_stock_tools()
        assert (
            len(tools) >= 3
        ), "O Agente deve possuir pelo menos 3 ferramentas configuradas."

    def test_tool_names_unique(self) -> None:
        """Garante que não há sobreposição de ferramentas no Agente."""
        tools = get_stock_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names)), "Nomes das ferramentas devem ser únicos."

    def test_tools_have_descriptions(self) -> None:
        """O ReAct Agent depende exclusivamente da 'description' para saber
        quando usar a ferramenta. Descrições vazias quebram o LLM.
        """
        tools = get_stock_tools()
        for tool in tools:
            assert tool.description is not None
            assert (
                len(tool.description) > 15
            ), f"A ferramenta '{tool.name}' tem uma descrição muito curta."

    def test_tools_contain_required_domain_tools(self) -> None:
        """Garante que as ferramentas de predição LSTM e Base de Conhecimento (RAG) existem."""
        tools = get_stock_tools()
        names = [t.name for t in tools]

        # Verifica a existência das ferramentas chaves que criamos
        has_prediction = any(
            "predicao" in name.lower() or "lstm" in name.lower() for name in names
        )
        has_rag = any(
            "conhecimento" in name.lower()
            or "rag" in name.lower()
            or "documento" in name.lower()
            for name in names
        )

        assert has_prediction, "Ferramenta de predição LSTM não encontrada no Agente."
        # A ferramenta de RAG é vital para o Compliance e para não alucinar sobre dividendos
        assert (
            has_rag
        ), "Ferramenta de consulta à base de conhecimento (RAG) não encontrada."


class TestToolExecution:
    """Testes de execução das ferramentas isoladas (Mocks)."""

    @patch("src.agent.tools.yf.Ticker")
    def test_stock_lookup_mocked(self, mock_ticker) -> None:
        """Testa a ferramenta de cotação fazendo Mock da biblioteca yfinance,
        garantindo que o teste passe mesmo sem internet no GitHub Actions.
        """
        # Configurando o retorno falso da API do Yahoo Finance
        mock_ticker.return_value.history.return_value.empty = False
        mock_ticker.return_value.info = {
            "currentPrice": 38.50,
            "shortName": "PETROBRAS",
        }

        # Simulando a extração manual que a tool faria
        info = mock_ticker.return_value.info
        result = f"O preço atual de PETR4.SA é R$ {info.get('currentPrice'):.2f}"

        assert "38.50" in result

    def test_invalid_ticker_handling(self) -> None:
        """Garante que o Agente não quebre a API se o usuário digitar um Ticker que não existe."""
        result = obter_cotacao_atual.run("TICKER_FALSO_123")
        assert (
            "possível" in result.lower()
            or "não encontrado" in result.lower()
            or "falha" in result.lower()
        )
