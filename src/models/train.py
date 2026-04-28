import logging
import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
import yfinance as yf
from datetime import date
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# Importando componentes internos
from src.features.feature_engineering import preparar_janelas_temporais
from src.models.lstm_model import ModeloLSTM

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
#       FUNÇÕES AUXILIARES E DE MLOPS
# ==========================================

def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Carrega as configurações centralizadas."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_sigma_metric(y_true: np.ndarray, y_pred: np.ndarray, window: int = 30, tolerance: float = 0.5) -> dict:
    """
    Calcula a métrica de negócio: erro em desvios-padrão.
    Erros acima do threshold (ex: 0.5σ) são inaceitáveis para trading.
    """
    errors = np.abs(y_true - y_pred)
    sigma = float(np.std(y_true[-window:])) if len(y_true) >= window else float(np.std(y_true))
    sigma = max(sigma, 1e-8) # Evita divisão por zero
    
    sigma_errors = errors / sigma

    return {
        "sigma_error_mean": float(np.mean(sigma_errors)),
        "sigma_error_max": float(np.max(sigma_errors)),
        "pct_within_tolerance": float(np.mean(sigma_errors <= tolerance) * 100),
        "sigma_value": sigma,
    }

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Executa uma época de treinamento de forma isolada e limpa."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)

@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Avalia o modelo calculando a perda de validação."""
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        total_loss += loss.item()
        
    return total_loss / max(len(loader), 1)

# ==========================================
#       PIPELINE PRINCIPAL DE TREINAMENTO
# ==========================================

def train_and_log():
    """Orquestra a ingestão, treino, avaliação e tracking no MLflow."""
    cfg = load_config()
    
    # Detecção de Hardware (Intel Arc XPU / NVIDIA CUDA / CPU)
    device = torch.device(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )
    logger.info(f"Iniciando treinamento da LSTM utilizando device: {device}")

    # --- INGESTÃO DE DADOS ---
    ticker = cfg["data"]["ticker"]
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    try:
        tkt = yf.Ticker(ticker, session=session)
        dados = tkt.history(period=cfg["data"]["period"])
        if dados.empty: raise ValueError("Dataset vazio.")
        dados_close = dados[["Close"]].values
    except Exception as e:
        logger.warning(f"Falha na API: {e}. Ativando Fallback Sintético.")
        datas = pd.date_range(end=date.today(), periods=1000)
        dados_close = (np.linspace(25, 42, 1000) + np.random.randn(1000)).reshape(-1, 1)

    # --- PREPARAÇÃO DE DADOS ---
    window = cfg["data"]["window_size"]
    X, y, scaler = preparar_janelas_temporais(dados_close, window_size=window)

    split_idx = int(len(X) * (1 - cfg["data"]["test_size"]))
    
    X_train_t = torch.tensor(X[:split_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[:split_idx], dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X[split_idx:], dtype=torch.float32)
    y_test_t = torch.tensor(y[split_idx:], dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=cfg["training"]["batch_size"], shuffle=False)

    # --- INICIALIZAÇÃO DO MODELO ---
    modelo = ModeloLSTM(
        input_size=cfg["model"]["input_size"],
        hidden_size=cfg["model"]["hidden_size"],
        output_size=cfg["model"]["output_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout_rate=cfg["model"]["dropout_rate"],
    ).to(device)
    
    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=cfg["training"]["learning_rate"])

    # --- MLFLOW TRACKING ---
    mlflow.set_experiment(cfg["paths"]["experiment_name"])
    
    with mlflow.start_run(run_name=f"Treino_{ticker}") as run:
        # Logs de Governança
        mlflow.log_params(cfg["model"])
        mlflow.log_params(cfg["training"])
        mlflow.log_param("window_size", window)
        
        mlflow.set_tag("model_type", "lstm_time_series")
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("business_metric", f"sigma_tolerance_{cfg['business_metric']['tolerance']}")

        # Loop de Treinamento
        logger.info("Iniciando treinamento das épocas...")
        num_epochs = cfg["training"]["num_epochs"]
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(modelo, train_loader, otimizador, criterio, device)
            val_loss = evaluate_model(modelo, test_loader, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Época [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # --- AVALIAÇÃO FINAL (ESCALA REAL E MÉTRICA DE NEGÓCIO) ---
        modelo.eval()
        with torch.no_grad():
            previsoes_scaled = modelo(X_test_t.to(device)).cpu().numpy()
            y_test_scaled = y_test_t.numpy()

            # Desnormalização para Reais (R$)
            previsoes_real = scaler.inverse_transform(previsoes_scaled)
            y_test_real = scaler.inverse_transform(y_test_scaled)

            # Métricas Tradicionais
            mse_real = float(mean_squared_error(y_test_real, previsoes_real))
            mae_real = float(mean_absolute_error(y_test_real, previsoes_real))
            rmse_real = float(np.sqrt(mse_real))

            # Métricas de Negócio (O grande diferencial)
            sigma_metrics = compute_sigma_metric(
                y_true=y_test_real, 
                y_pred=previsoes_real, 
                window=cfg["business_metric"]["window_size"],
                tolerance=cfg["business_metric"]["tolerance"]
            )

        mlflow.log_metrics({
            "rmse_real": rmse_real,
            "mae_real": mae_real,
            **sigma_metrics
        })

        # --- SALVAMENTO DE ARTEFATOS ---
        mlflow.pytorch.log_model(
            modelo, 
            artifact_path="model", 
            registered_model_name=cfg["paths"]["registered_model_name"]
        )
        
        scaler_path = cfg["paths"]["scaler_path"]
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        logger.info(
            f"Treino Finalizado! RMSE: {rmse_real:.2f} | "
            f"Taxa de Acerto (< {cfg['business_metric']['tolerance']}σ): {sigma_metrics['pct_within_tolerance']:.1f}%"
        )
        return run.info.run_id

if __name__ == "__main__":
    train_and_log()