import logging
import time

from prometheus_client import Counter, Gauge, Summary, start_http_server

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# DEFINIÇÃO DE MÉTRICAS PROMETHEUS
# ==========================================

# 1. Latência do Modelo (Tempo de Inferência)
LATENCY = Summary(
    "model_inference_latency_seconds", "Tempo gasto na predição do modelo LSTM"
)

# 2. Contador de Requisições por Status
REQUEST_COUNT = Counter(
    "agent_requests_total", "Total de requisições ao Agente ReAct", ["status"]
)

# 3. Erro Sigma (Métrica de Negócio em Tempo Real)
SIGMA_ERROR = Gauge("model_sigma_error", "Erro da predição em desvios-padrão (Sigma)")

# 4. Drift Share (Detectado pelo Evidently)
DRIFT_SHARE = Gauge("model_drift_share", "Proporção de colunas com drift detectado")

# 5. Qualidade do LLM (RAGAS / Feedback)
LLM_FAITHFULNESS = Gauge(
    "agent_llm_faithfulness", "Métrica de fidelidade da resposta ao contexto"
)
LLM_RELEVANCY = Gauge(
    "agent_llm_relevancy", "Métrica de relevância da resposta à pergunta"
)


def start_metrics_server(port: int = 9090):
    """Inicia o servidor de métricas do Prometheus."""
    logger.info(f"Iniciando exportador Prometheus na porta {port}...")
    start_http_server(port)


# Exemplo de uso para integração posterior no app.py
def track_prediction(sigma_val: float):
    SIGMA_ERROR.set(sigma_val)


def track_drift(share: float):
    DRIFT_SHARE.set(share)


@LATENCY.time()
def process_request(success: bool = True):
    status = "success" if success else "error"
    REQUEST_COUNT.labels(status=status).inc()


if __name__ == "__main__":
    # Mantém o processo vivo se rodado isoladamente para teste
    start_metrics_server(9090)
    while True:
        time.sleep(1)
