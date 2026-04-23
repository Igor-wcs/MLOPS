import logging
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import numpy as np
import joblib
from datetime import date

from src.features.feature_engineering import preparar_janelas_temporais
from src.models.lstm_model import ModeloLSTM

# Configuração de Logs profissionais
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- FUNÇÃO DE TREINAMENTO PADRONIZADA ---
def train_and_log(ticker="PETR4.SA"):
    # Parâmetros do Modelo e Treino
    params = {
        "window_size": 30,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "num_epochs": 50,
        "batch_size": 32
    }

    # Ingestão e Preparação (Features)
    logger.info(f"Baixando dados para {ticker}...")
    dados = yf.download(ticker, start='2019-01-01', end=date.today().strftime("%Y-%m-%d"))
    dados_close = dados[['Close']].values
    
    # Usando a função importada
    X, y, scaler = preparar_janelas_temporais(dados_close, params["window_size"])

    # Divisão Treino/Teste (Corte Cronológico)
    split = int(len(X) * 0.8)
    X_train = torch.tensor(X[:split], dtype=torch.float32)
    y_train = torch.tensor(y[:split], dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X[split:], dtype=torch.float32)
    y_test = torch.tensor(y[split:], dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    # Inicialização usando a classe importada
    modelo = ModeloLSTM(1, params["hidden_size"], params["num_layers"], 1, params["dropout_rate"])
    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=params["learning_rate"])

    mlflow.set_experiment("Datathon_Previsao_Acoes")

    with mlflow.start_run(run_name=f"Treino_{ticker}"):
        mlflow.log_params(params)
        
        # Tags padronizadas
        mlflow.set_tag("model_type", "regression_lstm")
        mlflow.set_tag("owner", "grupo-XX")
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("framework", "pytorch")

        history = {"train_loss": [], "val_loss": []}

        logger.info("Iniciando treinamento...")
        for epoch in range(params["num_epochs"]):
            modelo.train()
            train_losses = []
            for batch_X, batch_y in loader:
                otimizador.zero_grad()
                loss = criterio(modelo(batch_X), batch_y)
                loss.backward()
                otimizador.step()
                train_losses.append(loss.item())

            # Avaliação Final
            modelo.eval()
            with torch.no_grad():
                val_loss = criterio(modelo(X_test), y_test).item()

            epoch_train_loss = np.mean(train_losses)
            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(val_loss)

            # ✅ Log por época — gera curva no MLflow UI
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss",   val_loss,         step=epoch)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Época [{epoch+1}/{params['num_epochs']}] "
                    f"Train Loss: {epoch_train_loss:.5f} | Val Loss: {val_loss:.5f}"
                )

        # --- Métricas finais (MSE e MAE no espaço original) ---
        modelo.eval()
        with torch.no_grad():
            previsoes = modelo(X_test)
            mse = criterio(previsoes, y_test).item()
            mae = torch.mean(torch.abs(previsoes - y_test)).item()

        mlflow.log_metrics({"final_mse": mse, "final_mae": mae})

        mlflow.pytorch.log_model(
            modelo,
            artifact_path="model",
            registered_model_name="LSTM_Petrobras"
        )

        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        logger.info(f"Treino Finalizado. MSE: {mse:.6f} | MAE: {mae:.6f}")
        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    train_and_log()