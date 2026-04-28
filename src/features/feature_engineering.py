import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Any

def preparar_janelas_temporais(
    dados_close: np.ndarray, window_size: int = 30, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Escalona e prepara janelas temporais para treinamento do modelo LSTM,
    garantindo que o fit do scaler ocorra apenas nos dados de treino.

    Args:
        dados_close: Array NumPy contendo a série temporal (ex: preços de fechamento).
        window_size: Tamanho da janela de observação para predição.
        train_ratio: Proporção do dataset a ser utilizada para o treino.

    Returns:
        Tupla contendo as features em janelas (X), os targets (y) e o scaler treinado.
    """
    tamanho_x_esperado = len(dados_close) - window_size
    split_idx = int(tamanho_x_esperado * train_ratio) + window_size

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dados_close[:split_idx].reshape(-1, 1))

    dados_escalonados = scaler.transform(dados_close.reshape(-1, 1))

    X = []
    y = []
    for i in range(window_size, len(dados_escalonados)):
        X.append(dados_escalonados[i - window_size : i, 0])
        y.append(dados_escalonados[i, 0])

    X_arr, y_arr = np.array(X), np.array(y)
    X_arr = np.reshape(X_arr, (X_arr.shape[0], X_arr.shape[1], 1))

    return X_arr, y_arr, scaler