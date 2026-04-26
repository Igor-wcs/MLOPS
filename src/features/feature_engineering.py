"""src/features/feature_engineering.py

Módulo de extração e transformação de features.
Alimenta o pipeline de dados (data.py) com 11 features e resiliência de API.
"""
import logging
from datetime import date
import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

def fetch_stock_data(tickers: list[str], period: str) -> pd.DataFrame:
    """Baixa os dados do Yahoo Finance com Fallback para dados sintéticos."""
    ticker = tickers[0]
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    try:
        logger.info("A baixar dados de %s (%s)...", ticker, period)
        tkt = yf.Ticker(ticker, session=session)
        df = tkt.history(period=period)
        
        if df.empty:
            raise ValueError("Dataset vazio retornado pela API.")
        return df

    except Exception as e:
        logger.warning("Falha na API YFinance: %s", e)
        logger.warning("FALLBACK: Gerando dados sintéticos para garantir execução (Resiliência).")
        datas = pd.date_range(end=date.today(), periods=500)
        
        # Simula um DataFrame OHLCV
        base_price = 35.0
        close = np.linspace(base_price, 42.0, 500) + np.random.randn(500)
        df = pd.DataFrame({
            "Open": close - np.random.rand(500),
            "High": close + np.random.rand(500),
            "Low": close - np.random.rand(500) - 0.5,
            "Close": close,
            "Volume": np.random.randint(1000000, 50000000, 500)
        }, index=datas)
        return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula os indicadores técnicos (11 features do YAML)."""
    df = df.copy()
    
    # 1. Retornos diários
    df["returns"] = df["Close"].pct_change()
    
    # 2. Volatilidade (20 dias)
    df["volatility_20d"] = df["returns"].rolling(window=20).std()
    
    # 3. Médias Móveis
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    df["sma_50"] = df["Close"].rolling(window=50).mean()
    
    # 4. Range de Preço
    df["price_range"] = df["High"] - df["Low"]
    
    # 5. RSI 14 dias
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Remove os NaNs gerados pelo rolling()
    df.dropna(inplace=True)
    return df

def normalize_features(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Aplica normalização (Z-Score) nas colunas especificadas.
    Retorna o DataFrame normalizado e um dicionário com os parâmetros (média, desvio)
    para permitir a inversão de escala no futuro.
    """
    df_norm = df.copy()
    norm_params = {}
    
    for col in cols:
        mean = df_norm[col].mean()
        std = df_norm[col].std()
        
        # Prevenção de divisão por zero
        std = std if std > 1e-8 else 1.0 
        
        df_norm[col] = (df_norm[col] - mean) / std
        norm_params[col] = (mean, std)
        
    return df_norm, norm_params

def create_sequences(data_array: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Gera janelas deslizantes (X) e o target (y) t+1 para as 11 features."""
    X, y = [], []
    
    # O 'Close' é a 4ª coluna no FEATURE_COLS do data.py (índice 3)
    close_idx = 3 
    
    for i in range(len(data_array) - seq_len):
        X.append(data_array[i : i + seq_len])
        # O alvo (y) continua a ser apenas o preço de fechamento do dia seguinte
        y.append(data_array[i + seq_len, close_idx])
        
    return np.array(X), np.array(y)