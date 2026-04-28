import logging
import yfinance as yf
from typing import Optional
from langchain.tools import tool

# Configuração de Logs
logger = logging.getLogger(__name__)

@tool
def obter_previsao_lstm(ticker: str) -> str:
    """
    ÚTIL PARA: Obter a predição futura do preço de uma ação usando o modelo de Inteligência Artificial LSTM interno.
    ENTRADA: O código da ação (exemplo: 'PETR4.SA').
    SAÍDA: O preço previsto em Reais (R$) para o próximo dia útil.
    """
    logger.info(f"Tool 'obter_previsao_lstm' acionada para o ticker: {ticker}")
    try:
        # Aqui simulamos a chamada interna para a rota /predict ou lógica do modelo
        # Em um ambiente real, você faria um httpx.post("http://localhost:8000/predict", json={"ticker": ticker})
        # Para simplificar e evitar dependência circular de rede, vamos retornar um valor formatado.
        
        # Simulando o retorno de sucesso da nossa API
        preco_simulado = 36.50 
        return f"A previsão do modelo LSTM para {ticker} é de R$ {preco_simulado:.2f}."
    
    except Exception as e:
        logger.error(f"Erro na tool LSTM: {e}")
        return f"Erro ao calcular previsão para {ticker}. O serviço preditivo pode estar indisponível."

@tool
def obter_cotacao_atual(ticker: str) -> str:
    """
    ÚTIL PARA: Obter o preço de fechamento mais recente (tempo real/hoje) de uma ação no mercado.
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
        return f"A cotação atual (último fechamento) de {ticker} é R$ {preco_atual:.2f}."
    
    except Exception as e:
        logger.error(f"Erro na tool de cotação: {e}")
        return f"Falha ao buscar a cotação de {ticker} no Yahoo Finance."

@tool
def consultar_base_conhecimento(query: str) -> str:
    """
    ÚTIL PARA: Responder perguntas teóricas sobre o negócio, como 'O que é a tolerância sigma?', 
    'Como funciona o modelo LSTM?', ou regras de negócio da empresa.
    ENTRADA: A pergunta do usuário.
    SAÍDA: A resposta extraída dos documentos oficiais da empresa.
    """
    logger.info(f"Tool 'consultar_base_conhecimento' acionada com a query: {query}")
    try:
        # Aqui entraria a conexão real com o ChromaDB / Vector Store do seu RAG.
        # Para estruturação, simulamos o retorno do RAG.
        if "sigma" in query.lower():
            return "A tolerância sigma é uma métrica de negócio que avalia se o erro da predição está dentro de 0.5 desvios-padrão da volatilidade histórica do ativo."
        elif "lstm" in query.lower():
            return "Utilizamos uma rede neural LSTM em PyTorch com janela de 30 dias para prever o fechamento do mercado."
        else:
            return "Não encontrei informações específicas sobre isso na base de conhecimento oficial."
            
    except Exception as e:
        logger.error(f"Erro na tool de RAG: {e}")
        return "Erro ao consultar o banco de vetores de conhecimento."

def get_stock_tools():
    """Retorna a lista de ferramentas que o Agente ReAct pode utilizar."""
    return [obter_previsao_lstm, obter_cotacao_atual, consultar_base_conhecimento]