"""
Implementação do Feature Store (Redis) compatível com o modelo de 11 features.
Garante a persistência incremental e a disponibilidade de dados 
para a inferência em tempo real.
"""
import logging
import json
import redis
import yfinance as yf
import requests
import pandas as pd
from datetime import timedelta

# Importamos a engenharia de features para garantir que o Store guarde os dados processados
from src.features.feature_engineering import compute_features

logger = logging.getLogger(__name__)

class RedisFeatureStore:
    def __init__(self, host="redis", port=6379, db=0):
        try:
            self.client = redis.Redis(
                host=host, port=port, db=db, decode_responses=True
            )
            self.client.ping()
            logger.info("Conectado ao Redis Feature Store.")
        except Exception as e:
            logger.error(f"Erro ao conectar ao Redis: {e}")
            raise

    def upsert_incremental(self, ticker: str, df_processado: pd.DataFrame):
        """
        Implementa o GAP 03: Upsert de todas as 11 features sem destruir o histórico.
        Guarda cada linha como um objeto JSON dentro do Hash do Redis.
        """
        chave_hash = f"features:{ticker}"
        
        # Transformamos o DataFrame em um dicionário de JSONs: {data: "JSON_da_linha"}
        # Orient='index' cria um dict onde a chave é o timestamp e o valor são as colunas
        dados_dict = df_processado.to_dict(orient='index')
        
        updates = {}
        for data_ts, colunas in dados_dict.items():
            data_str = str(data_ts.date())
            # Serializamos todas as 11 colunas em uma string JSON
            updates[data_str] = json.dumps(colunas)

        if updates:
            # HSET realiza o Upsert (Update + Insert) atómico
            self.client.hset(chave_hash, mapping=updates)
            
            # TTL de segurança (7 dias) para evitar dados obsoletos
            self.client.expire(chave_hash, timedelta(days=7))
            logger.info(f"✅ {ticker}: Upsert de {len(updates)} registros (11 features) concluído.")

    def obter_janela_predicao(self, ticker: str, window_size: int = 30):
        """
        Busca os últimos N dias processados para alimentar o modelo LSTM.
        Reconstrói a matriz [window_size, 11] a partir do JSON.
        """
        chave_hash = f"features:{ticker}"
        todos_dados = self.client.hgetall(chave_hash)

        if not todos_dados:
            raise ValueError(f"Ticker {ticker} não encontrado no Store.")

        # Ordenar datas e pegar os últimos N registros
        datas_ordenadas = sorted(todos_dados.keys())
        ultimas_datas = datas_ordenadas[-window_size:]

        # Desserializar os JSONs para listas de valores numéricos
        matriz_features = []
        for d in ultimas_datas:
            linha_dict = json.loads(todos_dados[d])
            # Garante que a ordem das colunas é a mesma do FEATURE_COLS
            valores = list(linha_dict.values())
            matriz_features.append(valores)

        if len(matriz_features) < window_size:
            raise ValueError(f"Dados insuficientes no Redis: {len(matriz_features)}/{window_size}")

        return matriz_features


def executar_atualizacao_diaria():
    """Script disparado pelo cron ou CI/CD para alimentar o Store."""
    store = RedisFeatureStore()
    ticker = "PETR4.SA"

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    logger.info("Coletando e processando dados para o Store...")
    tkt = yf.Ticker(ticker, session=session)
    
    # Pegamos 60 dias para garantir que o compute_features tenha dados para as Médias Móveis (SMA50)
    dados_brutos = tkt.history(period="3mo")

    if not dados_brutos.empty:
        # Aplicamos a engenharia de features antes de salvar no Redis
        dados_processados = compute_features(dados_brutos)
        store.upsert_incremental(ticker, dados_processados)
    else:
        logger.error("Falha na coleta: API retornou vazio.")


if __name__ == "__main__":
    executar_atualizacao_diaria()