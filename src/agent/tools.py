import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
import yfinance as yf
from langchain.tools import tool

from src.agent.rag_pipeline import RAGPipeline

# Componentes Internos
from src.models.lstm_factory import get_model
from src.models.lstm_params import LSTMParams

# Configuração de Logs
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Carrega o arquivo de configuração YAML centralizado.

    Returns:
        Dicionário com as configurações do modelo e do pipeline.
    """
    with open("configs/model_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Singleton para o RAG para evitar recarregar embeddings em cada chamada
_rag_pipeline_instance: RAGPipeline | None = None


def get_project_root() -> Path:
    """Retorna o caminho raiz do projeto de forma robusta."""
    return Path(__file__).parent.parent.parent


def get_rag_pipeline() -> RAGPipeline:
    """Retorna a instância singleton do pipeline RAG.

    Returns:
        Objeto RAGPipeline inicializado.
    """
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance


@tool
def obter_previsao_lstm(ticker: str) -> str:
    """ÚTIL PARA: Obter a predição futura do preço de uma ação usando o modelo de Inteligência Artificial LSTM interno.
    ENTRADA: O código da ação (exemplo: 'PETR4.SA').
    SAÍDA: O preço previsto em Reais (R$) para o próximo dia útil.
    """
    logger.info(f"Tool 'obter_previsao_lstm' acionada para o ticker: {ticker}")
    cfg = load_config()
    root = get_project_root()

    try:
        # 1. Carregamento de Artefatos (Caminhos relativos à raiz)
        scaler_path = root / cfg["paths"]["scaler_path"]
        if not scaler_path.exists():
            return f"Erro: Scaler não encontrado em {scaler_path}. Rode o treinamento primeiro."

        scaler = joblib.load(scaler_path)

        # Instanciação via Factory e Params para consistência total
        params = LSTMParams(
            input_size=cfg["model"]["input_size"],
            hidden_size=cfg["model"]["hidden_size"],
            output_size=cfg["model"]["output_size"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout_rate"],
        )
        modelo = get_model(params)

        # Tentativa de carregar pesos (caso o usuário já tenha rodado o treino)
        model_weights_path = root / "model_weights.pt"
        if model_weights_path.exists():
            modelo.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
        else:
            logger.warning(
                f"Pesos do modelo não encontrados em {model_weights_path}. Usando modelo não treinado."
            )

        modelo.eval()

        # 2. Coleta de Dados para Inferência (Window Size)
        tkt = yf.Ticker(ticker)
        df = tkt.history(period="60d")  # Pega um pouco mais para calcular EMA20
        if len(df) < cfg["data"]["window_size"] + 20:
            return f"Dados insuficientes para {ticker}. Necessário pelo menos 50 dias de histórico."

        # Feature Engineering (Mesma lógica do treino)
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df = df[["Close", "Open", "High", "Low", "Volume", "EMA20"]].tail(
            cfg["data"]["window_size"]
        )

        # 3. Pré-processamento
        input_data = scaler.transform(df.values)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(
            0
        )  # Adiciona batch dim

        # 4. Predição
        with torch.no_grad():
            prediction_scaled = modelo(input_tensor).numpy()

        # 5. Desnormalização (Dummy array trick)
        dummy = np.zeros((1, cfg["model"]["input_size"]))
        dummy[0, 0] = prediction_scaled[0, 0]
        prediction_real = scaler.inverse_transform(dummy)[0, 0]

        return f"A previsão do modelo LSTM para o fechamento de {ticker} no próximo dia útil é de R$ {prediction_real:.2f}."

    except Exception as e:
        logger.error(f"Erro na tool LSTM: {e}")
        return f"Erro ao calcular previsão para {ticker}. Verifique se o modelo foi treinado e o scaler gerado."


@tool
def obter_cotacao_atual(ticker: str) -> str:
    """ÚTIL PARA: Obter o preço de fechamento mais recente (tempo real/hoje) de uma ação no mercado.
    ENTRADA: O código da ação (exemplo: 'PETR4.SA').
    SAÍDA: O preço atual da ação no mercado.
    """
    logger.info(f"Tool 'obter_cotacao_atual' acionada para o ticker: {ticker}")
    try:
        tkt = yf.Ticker(ticker)
        dados = tkt.history(period="1d")
        if dados.empty:
            return f"Não foi possível encontrar dados recentes para o ticker {ticker}."

        preco_atual = dados["Close"].iloc[-1]
        return (
            f"A cotação atual (último fechamento) de {ticker} é R$ {preco_atual:.2f}."
        )

    except Exception as e:
        logger.error(f"Erro na tool de cotação: {e}")
        return f"Falha ao buscar a cotação de {ticker} no Yahoo Finance."


@tool
def consultar_base_conhecimento(query: str) -> str:
    """ÚTIL PARA: Responder perguntas teóricas sobre o negócio, regras de compliance,
    políticas da empresa ou detalhes técnicos dos modelos.
    ENTRADA: A pergunta do usuário.
    SAÍDA: A resposta extraída dos documentos oficiais da empresa.
    """
    logger.info(f"Tool 'consultar_base_conhecimento' acionada com a query: {query}")
    try:
        rag = get_rag_pipeline()
        contextos = rag.retrieve(query)

        if not contextos:
            return "Não encontrei informações específicas sobre isso na base de conhecimento oficial."

        # Formata os contextos para o LLM
        resposta_base = "\n\n".join([doc.page_content for doc in contextos])
        return f"Informações encontradas na base de conhecimento:\n{resposta_base}"

    except Exception as e:
        logger.error(f"Erro na tool de RAG: {e}")
        return "Erro ao consultar o banco de vetores de conhecimento."


def get_stock_tools():
    """Retorna a lista de ferramentas que o Agente ReAct pode utilizar."""
    return [obter_previsao_lstm, obter_cotacao_atual, consultar_base_conhecimento]
