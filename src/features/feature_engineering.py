import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preparar_janelas_temporais(dados_close, window_size=30, train_ratio=0.8):
    # Calcula onde será o ponto de corte do treino para evitar Data Leakage
    tamanho_x_esperado = len(dados_close) - window_size
    split_idx = int(tamanho_x_esperado * train_ratio) + window_size
    
    # Fit APENAS nos dados de treino!
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dados_close[:split_idx])
    
    # Transform em todo o dataset
    dados_escalonados = scaler.transform(dados_close)

    # Construindo as janelas temporais
    X = []
    y = []
    for i in range(window_size, len(dados_escalonados)):
        X.append(dados_escalonados[i-window_size:i, 0])
        y.append(dados_escalonados[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler