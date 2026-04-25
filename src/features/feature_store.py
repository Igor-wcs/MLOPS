import logging
import redis
import json
import yfinance as yf
import requests
import pandas as pd
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RedisFeatureStore:
    def __init__(self, host='redis', port=6379, db=0):
        # Conexão com o serviço Redis do Docker
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            logger.info("Conectado ao Redis Feature Store.")
        except Exception as e:
            logger.error(f"Erro ao conectar ao Redis: {e}")
            raise

    def upsert_incremental(self, ticker: str, novos_dados: pd.DataFrame):
        """
        Implementa o GAP 03: Upsert Incremental sem destruir o store.
        Usa um Hash do Redis para armazenar as datas como chaves.
        """
        chave_hash = f"features:{ticker}"
        
        # Transformamos o DataFrame em um dicionário {data: preço}
        # O Redis HSET aceita múltiplos pares chave-valor de uma vez
        updates = {str(d.date()): float(v) for d, v in novos_dados['Close'].items()}
        
        if updates:
            # HSET realiza o Upsert: se a data existe, atualiza; se não, cria.
            # Nunca usamos FLUSHALL ou DEL aqui.
            self.client.hset(chave_hash, mapping=updates)
            
            # Definimos um TTL (Time-To-Live) para o conjunto inteiro. 
            # Se o pipeline parar por mais de 7 dias, os dados expiram por segurança.
            self.client.expire(chave_hash, timedelta(days=7))
            
            logger.info(f"✅ {ticker}: Upsert de {len(updates)} registros concluído.")

    def obter_janela_predicao(self, ticker: str, window_size: int = 30):
        """Busca os últimos N dias para alimentar o modelo LSTM."""
        chave_hash = f"features:{ticker}"
        
        # Buscamos todos os dados do Hash
        todos_dados = self.client.hgetall(chave_hash)
        
        if not todos_dados:
            raise ValueError(f"Ticker {ticker} não encontrado no Store.")

        # Ordenamos pelas datas e pegamos os últimos 'window_size' registros
        datas_ordenadas = sorted(todos_dados.keys())
        ultimas_datas = datas_ordenadas[-window_size:]
        
        precos = [float(todos_dados[d]) for d in ultimas_datas]
        
        if len(precos) < window_size:
            raise ValueError(f"Dados insuficientes: {len(precos)}/{window_size}")
            
        return precos

def executar_atualizacao_diaria():
    store = RedisFeatureStore()
    ticker = "PETR4.SA"
    
    # Coleta de dados (com o disfarce de navegador para o Docker)
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    logger.info(f"Coletando dados recentes para o Store...")
    tkt = yf.Ticker(ticker, session=session)
    dados = tkt.history(period="1mo") # Pega o último mês para garantir o incremental
    
    if not dados.empty:
        store.upsert_incremental(ticker, dados)
    else:
        logger.error("Falha na coleta: Ingestão retornou vazio.")

if __name__ == "__main__":
    executar_atualizacao_diaria()