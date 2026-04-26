"""src/models/train.py

Pipeline de treinamento com MLflow tracking padronizado.
Refatorado para Nível 2 de MLOps:
- Padrão Factory (LSTMFactory) para construção flexível do modelo.
- Validação de hiperparâmetros via Pydantic (LSTMParams).
- Leitura centralizada de parâmetros via YAML (configs/model_config.yaml).
- Métrica de negócio (Sigma Tolerance).
"""
import logging
import yaml
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Importando a nova arquitetura (Factory + Pydantic + Pipeline de Dados)
from src.models.data import prepare_data
from src.models.lstm_factory import LSTMFactory
from src.models.lstm_params import LSTMParams

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / ".models"
MODEL_DIR.mkdir(exist_ok=True)


def carregar_configuracao(config_path: str = "configs/model_config.yaml") -> dict:
    """Carrega os hiperparâmetros centralizados do ficheiro YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sigma_metric(y_true: np.ndarray, y_pred: np.ndarray, window: int = 30) -> dict[str, float]:
    """Calcula a métrica de negócio exigida: erro em desvios-padrão (σ).
    A tolerância de negócio definida no YAML é 0.5σ.
    """
    errors = np.abs(y_true - y_pred)
    sigma = float(np.std(y_true[-window:])) if len(y_true) >= window else float(np.std(y_true))
    sigma = max(sigma, 1e-8)
    
    sigma_errors = errors / sigma

    return {
        "sigma_error_mean": float(np.mean(sigma_errors)),
        "sigma_error_max": float(np.max(sigma_errors)),
        "pct_within_0_5_sigma": float(np.mean(sigma_errors <= 0.5) * 100),
        "sigma_value": sigma,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Executa uma época de treinamento de forma isolada."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Avalia o modelo e retorna predições e ground truth."""
    model.eval()
    all_preds, all_targets = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        all_preds.append(output.cpu().numpy().flatten())
        all_targets.append(y_batch.numpy().flatten())

    return np.concatenate(all_targets), np.concatenate(all_preds)


def train_and_log() -> str:
    """Orquestra o treino, instancia a Factory e loga no MLflow."""
    # 1. Carrega as configurações do YAML (Single Source of Truth)
    config = carregar_configuracao()
    lstm_cfg = config["lstm"]
    train_cfg = config["training"]
    layer_cfg = config["layer_config"]
    metric_cfg = config["business_metric"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilizando o device: {device}")

    # 2. Pipeline de Dados (Centralizado em data.py)
    train_loader, test_loader, norm_params = prepare_data(
        tickers=train_cfg["tickers"],
        period=train_cfg["period"],
        seq_len=train_cfg["seq_len"],
        batch_size=train_cfg["batch_size"],
        test_size=train_cfg.get("test_size", 0.2)
    )
    
    # Extrai o número de features dinamicamente a partir do DataLoader
    n_features = next(iter(train_loader))[0].shape[2]

    # 3. Governança via Pydantic e Criação via Factory
    params = LSTMParams(
        input_size=n_features,
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        output_size=lstm_cfg["output_size"],
        batch_first=lstm_cfg["batch_first"],
        dropout=lstm_cfg["dropout"]
    )
    
    factory = LSTMFactory(layer_config=layer_cfg, params=params)
    model = factory.create().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # 4. Rastreamento MLflow
    mlflow.set_experiment("datathon-fase05")
    
    ticker_name = "-".join(train_cfg["tickers"])
    with mlflow.start_run(run_name=f"LSTM_Factory_{ticker_name}") as run:
        
        # Loga hiperparâmetros (YAML + Pydantic)
        mlflow.log_params({
            "tickers": ticker_name,
            "period": train_cfg["period"],
            "seq_len": train_cfg["seq_len"],
            "num_epochs": train_cfg["num_epochs"],
            "learning_rate": train_cfg["learning_rate"],
            "batch_size": train_cfg["batch_size"],
            "n_features": n_features,
            "hidden_size": params.hidden_size,
            "num_layers": params.num_layers,
            "dropout": params.dropout,
            "device": str(device),
        })

        # Tags de Governança (Edital GAP 05)
        mlflow.set_tags({
            "model_type": "lstm_time_series",
            "framework": "pytorch",
            "owner": "grupo-XX",
            "phase": "datathon-fase05",
            "business_metric": f"sigma_tolerance_{metric_cfg['tolerance']}"
        })

        logger.info("Iniciando o loop de treino...")
        for epoch in range(train_cfg["num_epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                logger.info("Epoch [%d/%d] — loss: %.6f", epoch + 1, train_cfg["num_epochs"], train_loss)

        # 5. Avaliação e Métricas
        y_true_scaled, y_pred_scaled = evaluate_model(model, test_loader, device)
        
        # Inversão de Escala: Extrai a média e desvio da variável Close a partir do norm_params
        if "Close" in norm_params:
            close_mean, close_std = norm_params["Close"]
            y_true_real = (y_true_scaled * close_std) + close_mean
            y_pred_real = (y_pred_scaled * close_std) + close_mean
            
            mae_real = float(np.mean(np.abs(y_true_real - y_pred_real)))
            rmse_real = float(np.sqrt(np.mean((y_true_real - y_pred_real) ** 2)))
        else:
            logger.warning("Feature 'Close' não encontrada em norm_params. Avaliando dados em escala.")
            y_true_real, y_pred_real = y_true_scaled, y_pred_scaled
            mae_real = float(np.mean(np.abs(y_true_real - y_pred_real)))
            rmse_real = float(np.sqrt(np.mean((y_true_real - y_pred_real) ** 2)))

        # Métrica de Negócio
        sigma_metrics = compute_sigma_metric(y_true_real, y_pred_real, metric_cfg["window_size"])

        # Log de Resultados
        mlflow.log_metrics({
            "mae": mae_real,
            "rmse": rmse_real,
            **sigma_metrics
        })

        # 6. Salvar Artefatos
        artifact_path = MODEL_DIR / "lstm_model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "layer_config": layer_cfg,
                "lstm_params": params.model_dump(),
                "training_params": train_cfg,
                "norm_params": {k: list(v) for k, v in norm_params.items()},
            },
            artifact_path,
        )

        mlflow.log_artifact(str(artifact_path))
        mlflow.pytorch.log_model(model, "model")

        logger.info(
            "Treinamento concluído! MAE=%.4f | Erro Sigma=%.4fσ | Dentro do Alvo (%.1fσ)=%.1f%%",
            mae_real,
            sigma_metrics["sigma_error_mean"],
            metric_cfg["tolerance"],
            sigma_metrics["pct_within_0_5_sigma"],
        )

        return run.info.run_id


if __name__ == "__main__":
    run_id = train_and_log()
    logger.info(f"MLflow run_id: {run_id}")