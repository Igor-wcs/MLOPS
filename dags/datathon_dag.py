import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Adiciona o diretório raiz ao path para que os módulos src sejam encontrados
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from mlflow.tracking import MlflowClient

from src.monitoring.drift import gerar_relatorio_drift


def check_drift_and_decide() -> bool:
    """Executa o monitoramento de drift e decide se o retreino é necessário.

    Retorna True se o drift_share exceder o threshold definido (Event-Driven Retraining).
    """
    # 1. Executa o monitoramento e obtém o drift_share
    drift_share = gerar_relatorio_drift()

    # 2. Carrega as configurações para ler o threshold
    config_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "configs/monitoring_config.yaml",
    )
    with open(config_path, encoding="utf-8") as f:
        mon_cfg = yaml.safe_load(f)

    threshold = mon_cfg["drift"]["retrain_threshold"]

    logger.info("--- [DRIFT CHECK] ---")
    logger.info(f"Drift Detectado: {drift_share:.4f} | Limite para Retreino: {threshold:.4f}")

    # 3. Retorna True se o drift for maior que o threshold
    return bool(drift_share > threshold)


def validate_champion_challenger() -> bool:
    """Compara o novo modelo (Challenger) com o modelo atual (Champion) no MLflow.

    Implementa a Governança de Promoção (GAP 07).
    """
    config_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "configs/model_config.yaml",
    )
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["paths"]["registered_model_name"]
    client = MlflowClient()

    # 1. Busca as versões do modelo no Registry
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if len(versions) < 2:
            logger.info("Apenas uma versão disponível. Promoção automática para Champion.")
            return True

        # Ordena por versão para pegar as duas mais recentes
        sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        challenger = sorted_versions[0]  # Versão recém treinada
        champion = sorted_versions[1]  # Versão anterior

        # 2. Busca métricas de performance (RMSE Real)
        run_challenger = client.get_run(challenger.run_id)
        run_champion = client.get_run(champion.run_id)

        rmse_challenger = run_challenger.data.metrics.get("rmse_real", 999999)
        rmse_champion = run_champion.data.metrics.get("rmse_real", 999999)

        logger.info("--- [CHAMPION vs CHALLENGER] ---")
        logger.info(f"Challenger (v{challenger.version}) RMSE: {rmse_challenger:.4f}")
        logger.info(f"Champion (v{champion.version}) RMSE: {rmse_champion:.4f}")

        # 3. Critério de Aceite: O erro deve ser menor ou igual ao anterior
        if rmse_challenger <= rmse_champion:
            logger.info("✅ APROVADO: O novo modelo é superior ou igual. Promoção permitida.")
            # Aqui
            return True

        logger.info("❌ REPROVADO: O novo modelo degradou a performance. Mantendo Champion atual.")
        return False

    except Exception as e:
        logger.error(f"Erro na validação: {e}")
        return False


# Configurações padrão
default_args = {
    "owner": "grupo-XX",
    "depends_on_past": False,
    "start_date": datetime(2026, 4, 23),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="datathon_pipeline_mlops_v2",
    default_args=default_args,
    description="Pipeline MLOps Nível 2: Ingestão -> Drift -> DVC -> Champion-Challenger",
    schedule_interval="@daily",
    catchup=False,
    tags=["datathon", "mlops", "nivel-2", "automated-governance"],
) as dag:
    # Task 1: Ingestão Incremental (Resolve GAP 03)
    task_ingestao = BashOperator(
        task_id="ingestao_incremental_dados",
        bash_command="python -m src.features.feature_store",
    )

    # Task 2: Detecção de Drift Condicional (Event-Driven Retraining)
    task_monitorar_drift = ShortCircuitOperator(
        task_id="validacao_drift_condicional",
        python_callable=check_drift_and_decide,
    )

    # Task 3: Retreino Reprodutível via DVC
    task_retreino_dvc = BashOperator(
        task_id="executar_pipeline_dvc",
        bash_command="dvc repro",
    )

    # Task 4: Validação Champion-Challenger (Governança de Modelo)
    task_validacao_modelo = PythonOperator(
        task_id="validacao_champion_challenger",
        python_callable=validate_champion_challenger,
    )

    # Fluxo de Execução
    task_ingestao >> task_monitorar_drift >> task_retreino_dvc >> task_validacao_modelo
