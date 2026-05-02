import logging
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import mlflow.pytorch
import requests
from datetime import date
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from mlflow.tracking import MlflowClient

# Importando a nossa preparação de dados
from src.features.feature_engineering import preparar_janelas_temporais

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_configs() -> tuple[dict, dict]:
    """Carrega as configurações centrais."""
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/monitoring_config.yaml", "r", encoding="utf-8") as f:
        mon_cfg = yaml.safe_load(f)
    return model_cfg, mon_cfg


def obter_modelo_producao(model_name: str, device: torch.device):
    """Busca a versão mais recente do modelo no MLflow Registry e aloca no device correto."""
    logger.info(f"Buscando o modelo '{model_name}' no Registry...")
    client = MlflowClient()

    try:
        versoes = client.search_model_versions(f"name='{model_name}'")
        if not versoes:
            logger.warning("Nenhum modelo encontrado no Registry.")
            return None

        ultima_versao = max(versoes, key=lambda v: int(v.version))
        model_uri = f"models:/{model_name}/{ultima_versao.version}"

        logger.info(f"Carregando Modelo Versão {ultima_versao.version}...")
        modelo = mlflow.pytorch.load_model(model_uri).to(device)
        modelo.eval()
        return modelo

    except Exception as e:
        logger.warning(f"Não foi possível carregar o modelo do MLflow: {e}")
        return None


def gerar_relatorio_drift() -> float:
    # 1. Carregamento de Configurações e Hardware
    model_cfg, mon_cfg = load_configs()
    ticker = model_cfg["data"]["ticker"]
    window_size = model_cfg["data"]["window_size"]

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Iniciando análise de Drift para {ticker} usando {device}...")

    # 2. Obtenção Robusta de Dados
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    try:
        tkt = yf.Ticker(ticker, session=session)
        # Usa os períodos configurados no YAML
        ref_period = mon_cfg["drift"].get("reference_period", "1y")
        df_yf = tkt.history(period=ref_period)
        
        if len(df_yf) < mon_cfg["drift"].get("min_samples", 50):
             raise ValueError(f"Dados insuficientes para análise: {len(df_yf)} amostras.")

        # Feature Engineering Multivariada
        df_yf["EMA20"] = df_yf["Close"].ewm(span=20, adjust=False).mean()
        df_yf = df_yf[["Close", "Open", "High", "Low", "Volume", "EMA20"]].dropna()
        dados_input = df_yf.values
    except Exception as e:
        logger.warning(f"Falha na coleta de dados: {e}. Usando fallback de dados sintéticos.")
        # Fallback de dados sintéticos (Mock)
        dados_input = np.random.randn(200, 6)

    # 3. Preparação das janelas temporais
    X, y, _ = preparar_janelas_temporais(dados_input, window_size)

    # 4. Split de Referência (Passado) vs Atual (Recente)
    split = len(X) - 30
    X_ref_np, X_curr_np = X[:split], X[split:]

    # Criando nomes de colunas para as features multivariadas
    features_base = ["Close", "Open", "High", "Low", "Volume", "EMA20"]
    colunas_features = []
    for t in range(window_size, 0, -1):
        for feat in features_base:
            colunas_features.append(f"{feat}_t-{t}")

    df_ref = pd.DataFrame(X_ref_np.reshape(len(X_ref_np), -1), columns=colunas_features)
    df_curr = pd.DataFrame(
        X_curr_np.reshape(len(X_curr_np), -1), columns=colunas_features
    )

    # 5. Geração de Predições para Target Drift
    nome_modelo = model_cfg["paths"]["registered_model_name"]
    modelo = obter_modelo_producao(nome_modelo, device)

    if modelo is not None:
        try:
            with torch.no_grad():
                preds_ref = (
                    modelo(torch.tensor(X_ref_np, dtype=torch.float32).to(device))
                    .cpu()
                    .numpy()
                )
                preds_curr = (
                    modelo(torch.tensor(X_curr_np, dtype=torch.float32).to(device))
                    .cpu()
                    .numpy()
                )

            df_ref["prediction"] = preds_ref.flatten()
            df_curr["prediction"] = preds_curr.flatten()
            logger.info("Predições geradas. Analisando Data Drift e Prediction Drift.")
            metrics_preset = [DataDriftPreset(), TargetDriftPreset()]
        except Exception as e:
            logger.error(f"Erro nas predições: {e}. Analisando apenas Features.")
            metrics_preset = [DataDriftPreset()]
    else:
        logger.info("Sem modelo. Analisando apenas Feature Drift.")
        metrics_preset = [DataDriftPreset()]

    # 6. Rastreamento MLflow e Execução Evidently
    mlflow.set_experiment(model_cfg["paths"]["experiment_name"])

    with mlflow.start_run(run_name="Monitoring_Drift_Evidently"):
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("pipeline_step", "monitoring")

        report = Report(metrics=metrics_preset)
        report.run(reference_data=df_ref, current_data=df_curr)

        caminho_html = mon_cfg["paths"]["report_html"]
        report.save_html(caminho_html)

        # Clipar relatório no MLflow
        mlflow.log_artifact(caminho_html)

        # Extração e Log de Métricas
        drift_result = report.as_dict()
        drift_share = drift_result["metrics"][0]["result"]["share_of_drifted_columns"]

        mlflow.log_metric("drift_share", float(drift_share))

        retrain_th = mon_cfg["drift"]["retrain_threshold"]
        if drift_share > retrain_th:
            logger.warning(
                f"🚨 ALERTA CRÍTICO: Degradação detectada ({drift_share * 100:.1f}%). Necessário Retreino."
            )
            mlflow.set_tag("status", "CRITICAL_DRIFT")
            return float(drift_share)
        else:
            logger.info(
                f"✅ Estabilidade confirmada. Drift atual: {drift_share * 100:.1f}%."
            )
            mlflow.set_tag("status", "HEALTHY")
            return float(drift_share)


if __name__ == "__main__":
    gerar_relatorio_drift()
