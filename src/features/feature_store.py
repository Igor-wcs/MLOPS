import logging
from datetime import timedelta
from typing import Any

import pandas as pd
import redis
import requests
import yaml
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
    """Carrega as configurações do modelo."""
    with open("configs/model_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RedisFeatureStore:
    """Implementa o Feature Store usando Redis para armazenamento de séries temporais."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
    ) -> None:
        """Inicializa a conexão com o Redis."""
        cfg = load_config()
        redis_cfg = cfg.get("redis", {})

        # Prioriza argumentos passados, senão usa config
        host = host or redis_cfg.get("host", "redis")
        port = port or redis_cfg.get("port", 6379)
        db = db or redis_cfg.get("db", 0)

        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            logger.info(f"Conectado ao Redis Feature Store ({host}:{port}).")
        except Exception as e:
            logger.error(f"Erro ao conectar ao Redis: {e}")
            raise

    def upsert_incremental(self, ticker: str, df: pd.DataFrame) -> None:
        """Implementa o GAP 03: Upsert Incremental sem destruir o store.

        Armazena o vetor multivariado (OHLCV + EMA20) como string JSON.
        """
        cfg = load_config()
        ttl_days = cfg.get("redis", {}).get("ttl_days", 90)
        chave_hash = f"features:{ticker}"

        # Engenharia de Features Multivariada
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df_features = df[["Close", "Open", "High", "Low", "Volume", "EMA20"]].dropna()

        # Transformamos o DataFrame em um dicionário {data: json_vector}
        updates = {}
        for idx, row in df_features.iterrows():
            updates[str(idx.date())] = row.to_json()

        if updates:
            self.client.hset(chave_hash, mapping=updates)
            # Define o TTL para garantir que os dados não expirem antes da janela necessária
            self.client.expire(chave_hash, timedelta(days=ttl_days))
            logger.info(
                f"✅ {ticker}: Upsert de {len(updates)} vetores concluído. TTL: {ttl_days} dias."
            )

    def obter_janela_predicao(self, ticker: str, window_size: int = 30) -> pd.DataFrame:
        """Busca os últimos N dias retornando um DataFrame multivariado."""
        chave_hash = f"features:{ticker}"
        todos_dados = self.client.hgetall(chave_hash)

        if not todos_dados:
            raise ValueError(f"Ticker {ticker} não encontrado no Store.")

        # Ordenamos pelas datas e pegamos os últimos 'window_size' registros
        datas_ordenadas = sorted(todos_dados.keys())
        ultimas_datas = datas_ordenadas[-window_size:]

        # Reconstrói o DataFrame a partir do JSON
        rows = [pd.read_json(todos_dados[d], typ="series") for d in ultimas_datas]
        df_result = pd.DataFrame(rows)

        if len(df_result) < window_size:
            raise ValueError(f"Dados insuficientes: {len(df_result)}/{window_size}")

        return df_result


def executar_atualizacao_diaria():
    store = RedisFeatureStore()
    ticker = "PETR4.SA"

    # Coleta de dados (com o disfarce de navegador para o Docker)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    logger.info("Coletando dados recentes para o Store...")
    tkt = yf.Ticker(ticker, session=session)
    dados = tkt.history(period="1mo")  # Pega o último mês para garantir o incremental

    if not dados.empty:
        store.upsert_incremental(ticker, dados)
    else:
        logger.error("Falha na coleta: Ingestão retornou vazio.")


if __name__ == "__main__":
    executar_atualizacao_diaria()
