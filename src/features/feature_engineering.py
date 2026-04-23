import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preparar_janelas_temporais(dados_close, window_size=30):
    # 1. Escalonando os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_escalonados = scaler.fit_transform(dados_close)

    # 2. Construindo as janelas temporais
    X = []
    y = []
    for i in range(window_size, len(dados_escalonados)):
        X.append(dados_escalonados[i-window_size:i, 0])
        y.append(dados_escalonados[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Devolvemos os dados prontos e o "scaler" (útil para o futuro)
    return X, y, scaler