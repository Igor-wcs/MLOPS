import logging
from typing import Any

import joblib
import mlflow.artifacts
import mlflow.pytorch
import numpy as np
import torch
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from torch import nn

# Importações Internas
from src.agent.react_agent import create_datathon_agent, query_agent
from src.agent.tools import get_stock_tools
from src.features.feature_store import RedisFeatureStore
from src.models.train import train_and_log
from src.security.guardrails import input_guard, output_guard

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
#   ESTADO GLOBAL E CONFIGURAÇÃO
# ==========================================

tags_metadata = [
    {
        "name": "Predição",
        "description": "Inferência de preços com Feature Store (Redis).",
    },
    {"name": "Agente", "description": "Consulta ao agente ReAct LLM."},
    {"name": "Treinamento", "description": "Disparo assíncrono do pipeline MLflow."},
    {"name": "Configuração", "description": "Probes de Health e Readiness."},
]

app = FastAPI(
    title="Datathon Fase 05 — Stock Price Prediction",
    description="LSTM PyTorch + Agente LLM (Qwen) com governança total.",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrumentação Automática (Prometheus)
Instrumentator().instrument(app).expose(app)


class AppState:
    """Contêiner para o estado global da aplicação para evitar o uso de 'global'."""

    model: nn.Module | None = None
    scaler: Any = None
    config: dict[str, Any] | None = None
    device: torch.device | None = None
    feature_store: RedisFeatureStore | None = None
    agent_executor: Any = None


state = AppState()

# ==========================================
#    SCHEMAS
# ==========================================


class PredictRequest(BaseModel):
    """Corpo da requisição para predição de preços."""

    ticker: str


class AgentRequest(BaseModel):
    """Corpo da requisição para consulta ao Agente."""

    query: str


class AgentResponse(BaseModel):
    """Resposta formatada do Agente."""

    answer: str


# ==========================================
#    STARTUP E PROBES (Estilo Kubernetes)
# ==========================================


@app.on_event("startup")
def startup_event() -> None:
    """Inicializa configurações, hardware, Redis e baixa artefatos do MLflow."""
    try:
        with open("configs/model_config.yaml", encoding="utf-8") as f:
            cfg_raw = yaml.safe_load(f)
            if not cfg_raw:
                raise RuntimeError("Arquivo de configuração vazio ou inválido.")
            state.config = dict(cfg_raw)

        state.device = torch.device(
            "xpu"
            if hasattr(torch, "xpu") and torch.xpu.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Servidor inicializado com aceleração em: {state.device}")

        # 1. Redis Parametrizado
        redis_cfg = state.config.get("redis", {})
        try:
            state.feature_store = RedisFeatureStore(
                host=redis_cfg.get("host", "redis"),
                port=redis_cfg.get("port", 6379),
                db=redis_cfg.get("db", 0),
            )
        except Exception as e:
            logger.warning(f"Aviso: Redis não acessível. Detalhe: {e}")

        # 2. Modelo via MLflow Registry
        nome_modelo = state.config["paths"]["registered_model_name"]
        logger.info(f"Buscando modelo '{nome_modelo}' no MLflow Registry...")

        state.model = mlflow.pytorch.load_model(f"models:/{nome_modelo}/latest").to(
            state.device
        )
        state.model.eval()

        client = MlflowClient()
        versoes = client.search_model_versions(f"name='{nome_modelo}'")
        if versoes:
            ultima_versao = max(versoes, key=lambda v: int(v.version))

            local_scaler_path = mlflow.artifacts.download_artifacts(
                run_id=ultima_versao.run_id,
                artifact_path=state.config["paths"]["scaler_path"],
            )
            state.scaler = joblib.load(local_scaler_path)

        # 3. Inicialização do Agente Singleton
        logger.info("Inicializando Agente ReAct (Singleton)...")
        tools = get_stock_tools()
        state.agent_executor = create_datathon_agent(tools)

        logger.info("Artefatos de inferência e Agente carregados com sucesso.")

    except Exception as e:
        logger.error(f"Erro no startup: {e}")


@app.get("/ready", tags=["Configuração"])
async def readiness() -> dict[str, str]:
    """Verifica se os componentes vitais (Modelo/Redis) estão carregados."""
    is_ready = (
        state.model is not None
        and state.scaler is not None
        and state.feature_store is not None
        and state.config is not None
    )
    return {"status": "ready" if is_ready else "not_ready", "device": str(state.device)}


@app.exception_handler(RequestValidationError)
async def validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Captura erros de validação do Pydantic e retorna 400 em vez de 422."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(
            {"detail": exc.errors(), "message": "Parâmetros inválidos."}
        ),
    )


# ==========================================
#    ENDPOINTS CORE
# ==========================================


@app.post("/train", tags=["Treinamento"])
async def trigger_training(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Dispara o pipeline de treinamento em background (sem travar a API)."""
    background_tasks.add_task(train_and_log)
    return {
        "message": "Treinamento assíncrono disparado com sucesso.",
        "status": "running",
    }


@app.post("/predict", tags=["Predição"])
def predict(req: PredictRequest) -> dict[str, Any]:
    """Executa a predição LSTM multivariada para um ticker."""
    if not state.model or not state.scaler or not state.feature_store or not state.config:
        raise HTTPException(
            status_code=503, detail="Serviço indisponível (Model/Redis/Config não carregados)."
        )

    try:
        window_size = int(state.config["data"]["window_size"])
        input_size = int(state.config["model"]["input_size"])

        # 1. Puxa do Feature Store (Dataframe Multivariado)
        df_features = state.feature_store.obter_janela_predicao(
            req.ticker, window_size=window_size
        )

        # 2. Pré-processamento Multivariado
        dados_escalonados = state.scaler.transform(df_features.values)

        tensor_entrada = torch.tensor(
            dados_escalonados.reshape(1, window_size, input_size), dtype=torch.float32
        ).to(state.device)

        # 3. Inferência
        with torch.no_grad():
            predicao_tensor = state.model(tensor_entrada)
            resultado_escalonado = predicao_tensor.cpu().numpy()

        # 4. Desnormalização do Target (Close está na coluna 0)
        dummy = np.zeros((1, input_size))
        dummy[0, 0] = resultado_escalonado[0, 0]
        resultado_reais = state.scaler.inverse_transform(dummy)[0, 0]

        return {
            "ticker": req.ticker,
            "predicted_price_brl": round(float(resultado_reais), 2),
            "features_used": list(df_features.columns),
            "source": "Redis Feature Store (Multivariate)",
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.error(f"Erro na inferência: {e}")
        raise HTTPException(
            status_code=500, detail="Erro interno no pipeline de predição."
        ) from e


@app.post("/agent", tags=["Agente"], response_model=AgentResponse)
async def agent_query(data: AgentRequest) -> AgentResponse:
    """Consulta o agente ReAct protegido por Guardrails (Baixa Latência)."""
    # 1. Barreira de Entrada (Input Guardrail - OWASP LLM01)
    is_valid, reason = input_guard.validate(data.query)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)

    if state.agent_executor is None:
        raise HTTPException(
            status_code=503, detail="Agente LLM não inicializado no startup."
        )

    try:
        # 2. Processamento do LLM (Usa Singleton Singleton carregado no startup)
        result = query_agent(state.agent_executor, data.query)
        resposta_bruta = str(result.get("answer", "Desculpe, não consegui processar a resposta."))

        # 3. Barreira de Saída (Output Guardrail - OWASP LLM06 / LGPD)
        resposta_segura = output_guard.sanitize(resposta_bruta)

        return AgentResponse(answer=resposta_segura)

    except Exception as e:
        logger.error(f"Erro no Agente ReAct: {e}")
        raise HTTPException(
            status_code=500, detail="Falha na geração da resposta do LLM."
        ) from e