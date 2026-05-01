import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Any

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
        y.append(dados_escalonados[i, 0]) # Primeira coluna é o target (Close)

    X_arr, y_arr = np.array(X), np.array(y)
    # X_arr já terá o shape (samples, window_size, n_features)
    
    return X_arr, y_arr, scaler