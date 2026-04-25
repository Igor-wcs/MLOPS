import logging
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import mlflow.pytorch
import requests
from datetime import date
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from mlflow.tracking import MlflowClient

# Importando a nossa preparação de dados
from src.features.feature_engineering import preparar_janelas_temporais

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def obter_modelo_producao():
    """Busca a versão mais recente do modelo no MLflow Registry."""
    logger.info("Buscando o modelo mais recente no Model Registry...")
    client = MlflowClient()

    try:
        # Busca todas as versões registradas para este nome
        versoes = client.search_model_versions("name='LSTM_Petrobras'")
        if not versoes:
            logger.warning(
                "Nenhum modelo encontrado no Registry. O primeiro treino ainda não ocorreu."
            )
            return None

        # Identifica a versão mais alta
        ultima_versao = max(versoes, key=lambda v: int(v.version))
        model_uri = f"models:/LSTM_Petrobras/{ultima_versao.version}"

        logger.info(f"Carregando Modelo Versão {ultima_versao.version}...")
        modelo = mlflow.pytorch.load_model(model_uri)
        modelo.eval()
        return modelo

    except Exception as e:
        logger.warning(f"Não foi possível carregar o modelo do MLflow: {e}")
        return None


def gerar_relatorio_drift(ticker="PETR4.SA", window_size=30):
    logger.info(f"Iniciando análise de Drift para {ticker}...")

    # 1. Configurar Sessão Robusta (Disfarce de Navegador para evitar bloqueio no Docker)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        }
    )

    # 2. Tentar baixar dados reais do mercado
    try:
        logger.info(f"Tentando baixar histórico de {ticker} via Yahoo Finance...")
        tkt = yf.Ticker(ticker, session=session)
        dados = tkt.history(period="2y")

        if dados.empty:
            raise ValueError("O Yahoo Finance retornou um dataset vazio.")

        logger.info(f"Sucesso! {len(dados)} linhas baixadas da API.")
        dados_close = dados[["Close"]].values

    except Exception as e:
        logger.warning(f"🚨 Falha na conexão com a API externa: {e}")
        logger.warning(
            "Ativando Fallback: Gerando dados sintéticos para garantir a continuidade da DAG."
        )

        datas_mock = pd.date_range(end=date.today(), periods=500)
        dados_mock = pd.DataFrame(
            {"Close": np.linspace(32, 38, 500) + np.random.randn(500)}, index=datas_mock
        )
        dados_close = dados_mock[["Close"]].values

    # 3. Preparar as janelas (Processamento isolado)
    X, y, _ = preparar_janelas_temporais(dados_close, window_size)

    # 4. Dividir Referência (80%) vs Atual (20%)
    split = int(len(X) * 0.8)
    X_ref_np, X_curr_np = X[:split], X[split:]

    colunas_features = [f"preco_dia_menos_{i}" for i in range(window_size, 0, -1)]
    df_ref = pd.DataFrame(X_ref_np.reshape(-1, window_size), columns=colunas_features)
    df_curr = pd.DataFrame(X_curr_np.reshape(-1, window_size), columns=colunas_features)

    # 5. Tentar gerar predições (Se o modelo existir)
    modelo = obter_modelo_producao()

    if modelo is not None:
        try:
            with torch.no_grad():
                logger.info(
                    "Gerando predições com o modelo Champion para análise de desvio..."
                )
                preds_ref = modelo(torch.tensor(X_ref_np, dtype=torch.float32)).numpy()
                preds_curr = modelo(
                    torch.tensor(X_curr_np, dtype=torch.float32)
                ).numpy()

            df_ref["predicao_modelo"] = preds_ref
            df_curr["predicao_modelo"] = preds_curr
        except Exception as e:
            logger.error(
                f"Erro ao gerar predições: {e}. Prosseguindo apenas com análise de features."
            )
    else:
        logger.info(
            "Pulando análise de predições. O relatório focarão apenas no Drift das Features (preços)."
        )

    # 6. Rodar o Evidently Report
    logger.info("Executando Data Drift Report (Evidently)...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_ref, current_data=df_curr)

    # 7. Salvar e Avaliar
    caminho_relatorio = "drift_report_petr4.html"
    report.save_html(caminho_relatorio)

    drift_result = report.as_dict()
    drift_share = drift_result["metrics"][0]["result"]["share_of_drifted_columns"]

    logger.info(f"Relatório gerado em: {caminho_relatorio}")
    logger.info(f"Proporção de dados com drift: {drift_share * 100:.2f}%")

    if drift_share > 0.2:
        logger.warning("ALERTA: Degradação de dados detectada (> 20%).")
    else:
        logger.info("✅ Estabilidade de dados confirmada.")


if __name__ == "__main__":
    gerar_relatorio_drift()
