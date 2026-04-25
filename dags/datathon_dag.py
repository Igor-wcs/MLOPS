from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# 1. Configurações padrão do pipeline
default_args = {
    'owner': 'grupo-XX',
    'depends_on_past': False,
    'start_date': datetime(2026, 4, 23),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# 2. Definição da DAG (O Fluxo Declarativo)
with DAG(
    dag_id='datathon_pipeline_mlops',
    default_args=default_args,
    description='Pipeline de ML Contínuo: Monitoramento de Drift e Retreino',
    schedule_interval='@daily', # Configurado para rodar todos os dias à meia-noite
    catchup=False,
    tags=['datathon', 'mlops', 'nivel-2'],
) as dag:

    # 3. Tasks (Isolamento de Compute garantido pelo BashOperator)
    # Cada task roda como um comando independente no sistema operacional do container
    
    task_monitorar_drift = BashOperator(
        task_id='monitoramento_drift_evidently',
        # O Airflow chamaria o seu script exatamente como você faz no terminal
        bash_command='cd /opt/airflow && python -m src.monitoring.drift'
    )

    task_treinar_modelo = BashOperator(
        task_id='treinamento_modelo_champion',
        bash_command='cd /opt/airflow && python -m src.models.train'
    )

    # 4. A Mágica do Airflow: Definindo a Ordem (Topologia)
    # A setinha (>>) significa "roda o drift primeiro. Se der certo, roda o treino."
    task_monitorar_drift >> task_treinar_modelo