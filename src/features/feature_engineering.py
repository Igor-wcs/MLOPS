import logging
import os
import yaml
import yfinance as yf
import requests
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Any
from pathlib import Path

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Carrega as configurações centralizadas."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preparar_janelas_temporais(
    dados: np.ndarray, window_size: int = 30, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Escalona e prepara janelas temporais para treinamento do modelo LSTM multivariado.
    O target é assumido como a primeira coluna do array 'dados'.

    Args:
        dados: Array NumPy (n_samples, n_features). A primeira coluna deve ser o Target.
        window_size: Tamanho da janela de observação.
        train_ratio: Proporção para treino.

    Returns:
        Tupla (X, y, scaler).
    """
    n_features = dados.shape[1]
    tamanho_x_esperado = len(dados) - window_size
    split_idx = int(tamanho_x_esperado * train_ratio) + window_size

    # Fit apenas no conjunto de treino para evitar data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dados[:split_idx])

    dados_escalonados = scaler.transform(dados)

    X, y = [], []
    for i in range(window_size, len(dados_escalonados)):
        X.append(dados_escalonados[i - window_size : i, :])
        y.append(dados_escalonados[i, 0])  # Primeira coluna é o target (Close)

    X_arr, y_arr = np.array(X), np.array(y)
    # X_arr já terá o shape (samples, window_size, n_features)

    return X_arr, y_arr, scaler


def run_feature_engineering():
    """Pipeline de processamento de features para DVC."""
    cfg = load_config()

    # --- INGESTÃO DE DADOS ---
    ticker = cfg["data"]["ticker"]
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    logger.info(f"Baixando dados para {ticker}...")
    try:
        tkt = yf.Ticker(ticker, session=session)
        df = tkt.history(period=cfg["data"]["period"])
        if df.empty:
            raise ValueError("Dataset vazio.")

        # Feature Engineering Multivariada
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df = df[["Close", "Open", "High", "Low", "Volume", "EMA20"]].dropna()
        dados_input = df.values
    except Exception as e:
        logger.warning(f"Falha na API: {e}. Gerando Mock para continuidade.")
        dados_input = np.random.randn(1000, 6)

    # --- PROCESSAMENTO ---
    window = cfg["data"]["window_size"]
    X, y, scaler = preparar_janelas_temporais(dados_input, window_size=window)

    # --- SALVAMENTO (REQUISITO DVC) ---
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    joblib.dump(scaler, output_dir / "scaler_temp.pkl")

    logger.info(f"Features e Scaler salvos com sucesso em {output_dir}")


if __name__ == "__main__":
    run_feature_engineering()
