import logging

import mlflow
import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

# Importando componentes internos

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Carrega o arquivo de configuração YAML centralizado."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================
# 1. FUNÇÕES DO CÓDIGO QUE VOCÊ ACHOU (Adaptadas)
# ==========================================


def train_ridge_baseline(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> tuple[Ridge, dict]:
    """Treina o baseline Ridge e retorna o modelo e métricas."""
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    model = Ridge(alpha=1.0)
    model.fit(x_train_flat, y_train)
    y_pred = model.predict(x_test_flat)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mse": float(mean_squared_error(y_test, y_pred)),
    }
    return model, metrics


class MLPBaseline(nn.Module):
    """MLP simples como baseline PyTorch."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa o forward pass do modelo MLP."""
        # O .view aqui faz o flatten direto no PyTorch
        return self.net(x.view(x.size(0), -1))


# ==========================================
# 2. INTEGRAÇÃO COM MLOPS E DADOS
# ==========================================


def run_baselines() -> None:
    """Pipeline de execução dos modelos baseline para comparação."""
    cfg = load_config()
    ticker = cfg["data"]["ticker"]

    # --- CARREGAMENTO DE DADOS (DVC OUTS - Mesmo que o modelo campeão) ---
    try:
        x_all = np.load("data/processed/X.npy")
        y_all = np.load("data/processed/y.npy")
        logger.info("Baselines carregando dados processados do DVC.")
    except FileNotFoundError:
        logger.warning("Dados processados não encontrados. Usando Mock para baseline.")
        x_all = np.random.randn(
            500, cfg["data"]["window_size"], cfg["model"]["input_size"]
        )
        y_all = np.random.randn(500)

    # Split cronológico consistente
    split_idx = int(len(x_all) * (1 - cfg["data"]["test_size"]))

    x_train, x_test = x_all[:split_idx], x_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    mlflow.set_experiment(cfg["paths"]["experiment_name"])

    # --- RODO 1: O RIDGE ---
    with mlflow.start_run(run_name=f"Baseline_Ridge_{ticker}"):
        logger.info("Executando Ridge...")
        ridge_model, ridge_metrics = train_ridge_baseline(
            x_train, y_train, x_test, y_test
        )

        mlflow.log_params(
            {"model_type": "baseline_ridge", "window_size": cfg["data"]["window_size"]}
        )
        mlflow.log_metrics(ridge_metrics)
        mlflow.sklearn.log_model(ridge_model, "model")
        logger.info(f"Ridge Finalizado: RMSE={ridge_metrics['rmse']:.4f}")

    # --- RODO 2: O MLP ---
    with mlflow.start_run(run_name=f"Baseline_MLP_{ticker}"):
        logger.info("Executando MLP...")
        # Lógica resumida de treino do MLP para o baseline
        input_dim = cfg["data"]["window_size"] * cfg["model"]["input_size"]
        mlp_model = MLPBaseline(input_dim=input_dim, hidden_dim=64)
        criterio = nn.MSELoss()
        otimizador = torch.optim.Adam(mlp_model.parameters(), lr=0.005)

        x_t_train = torch.tensor(x_train, dtype=torch.float32)
        y_t_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        mlp_model.train()
        for _ in range(15):  # Apenas 15 épocas para o baseline ser rápido
            otimizador.zero_grad()
            perda = criterio(mlp_model(x_t_train), y_t_train)
            perda.backward()
            otimizador.step()

        mlp_model.eval()
        with torch.no_grad():
            x_t_test = torch.tensor(x_test, dtype=torch.float32)
            y_pred_mlp = mlp_model(x_t_test).numpy()

        mlp_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_mlp)))

        mlflow.log_params({"model_type": "baseline_mlp", "epochs": 15})
        mlflow.log_metric("rmse", mlp_rmse)
        mlflow.pytorch.log_model(mlp_model, "model")
        logger.info(f"MLP Finalizado: RMSE={mlp_rmse:.4f}")


if __name__ == "__main__":
    run_baselines()
