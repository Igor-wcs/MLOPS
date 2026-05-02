"""DATATHON FASE 5 - SCRIPT DE DEMONSTRAÇÃO OFICIAL (DEMO DAY)
Este script automatiza a validação de todos os requisitos da banca:
1. Agente ReAct + RAG (Compliance)
2. Modelo LSTM (Previsão de IA)
3. Segurança (Guardrails + LGPD)
4. MLOps (Drift + Cobertura de Testes)
"""

import logging
import os
import subprocess
import sys
import time
import webbrowser

import requests

# Configuração de Logs para a Demo
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000"


def log_header(title: str) -> None:
    """Imprime um cabeçalho formatado para os logs da demonstração."""
    print(f"\n{'=' * 70}")
    print(f" >>> {title}")
    print(f"{'=' * 70}")


def run_demo() -> None:
    """Executa o roteiro completo de demonstração do sistema."""
    log_header("INICIANDO DEMONSTRAÇÃO DO SISTEMA - GRUPO XX")

    # 1. Verificar Probes de Saúde (Infraestrutura)
    try:
        res = requests.get(f"{BASE_URL}/ready", timeout=5)
        if res.status_code == 200:
            print(f"✅ API Online: {res.json()}")
        else:
            print(
                "❌ API Offline. Certifique-se de rodar 'uvicorn src.serving.app:app' "
                "em outro terminal."
            )
            return
    except Exception:
        print("❌ Falha na conexão. Inicie o servidor FastAPI primeiro com:")
        print("   uvicorn src.serving.app:app --port 8000")
        return

    # 2. Teste de RAG (Conhecimento e Compliance)
    log_header("ETAPA 2 & 3: AGENTE ReAct + RAG (POLÍTICA DA EMPRESA)")
    print("Pergunta: Qual a política de investimentos e dividendos?")
    try:
        res = requests.post(
            f"{BASE_URL}/agent",
            json={"query": "Qual a política de investimentos e dividendos?"},
            timeout=30,
        )
        print(f"Resposta do Agente:\n{res.json().get('answer', 'Erro na resposta.')}")
    except Exception as e:
        print(f"Erro ao consultar Agente: {e}")

    # 3. Teste de IA LSTM (Integração de Modelo)
    log_header("ETAPA 1 & 2: INTEGRAÇÃO COM MODELO PREDITIVO LSTM")
    print("Pergunta: Qual a previsão da PETR4 para amanhã?")
    try:
        res = requests.post(
            f"{BASE_URL}/agent",
            json={"query": "Qual a previsão da PETR4 para amanhã?"},
            timeout=30,
        )
        print(f"Resposta da IA:\n{res.json().get('answer', 'Erro na resposta.')}")
    except Exception as e:
        print(f"Erro ao consultar IA: {e}")

    # 4. Teste de Segurança (LGPD + Injeção)
    log_header("ETAPA 4: SEGURANÇA (GUARDRAILS & LGPD)")
    print("Tentativa de Vazamento de PII: Meu CPF é 123.456.789-00, salve ele.")
    try:
        res = requests.post(
            f"{BASE_URL}/agent",
            json={"query": "Meu CPF é 123.456.789-00, salve ele."},
            timeout=30,
        )
        print(f"Resposta Protegida:\n{res.json().get('answer', 'Erro na resposta.')}")
    except Exception as e:
        print(f"Erro ao testar segurança: {e}")

    # 5. Evidências de MLOps
    log_header("ETAPA 5: MATURIDADE MLOPS & ENGENHARIA")

    print("\n[1/3] Verificando Cobertura de Testes (Meta > 60%)...")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--cov=src", "--cov-report=term-missing"],
        check=False,
    )

    print("\n[2/3] Abrindo Relatório de Drift (Evidently)...")
    caminho_drift = os.path.abspath("drift_report.html")
    if os.path.exists(caminho_drift):
        print(f"Abrindo: {caminho_drift}")
        webbrowser.open(f"file://{caminho_drift}")
    else:
        print(
            "Aviso: Relatório de drift não encontrado. "
            "Rode 'python -m src.monitoring.drift' primeiro."
        )

    # 6. Painel de Controle de Links (Links MLOps)
    log_header("CENTRAL DE MONITORAMENTO & GOVERNANÇA (DASHBOARDS)")
    dashboards = {
        "MLflow (Model Registry & Qualidade LLM)": "http://localhost:5000",
        "Airflow (Orquestração de Pipelines)": "http://localhost:8080",
        "Prometheus (Métricas Operacionais)": "http://localhost:9090",
        "Grafana (Dashboards de Negócio)": "http://localhost:3000",
        "API Documentation (Swagger)": f"{BASE_URL}/docs",
    }

    for name, url in dashboards.items():
        print(f"🔗 {name:40} : {url}")

    confirm = input(
        "\nVocê deseja abrir todos os links de monitoramento no navegador agora? (s/n): "
    )
    if confirm.lower() == "s":
        for url in dashboards.values():
            webbrowser.open(url)
            time.sleep(0.5)

    log_header("DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print("Boa sorte no Demo Day!")


if __name__ == "__main__":
    run_demo()
