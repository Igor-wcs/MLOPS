import logging
import joblib
import yaml
import mlflow.pytorch
import mlflow.artifacts
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator

# Importações Internas
from src.features.feature_store import RedisFeatureStore
from src.security.guardrails import input_guard, output_guard
from src.models.train import train_and_log

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

model = None
scaler = None
config = None
device = None
feature_store = None

# ==========================================
#    SCHEMAS
# ==========================================


class PredictRequest(BaseModel):
    ticker: str = "PETR4.SA"


class AgentRequest(BaseModel):
    query: str


class AgentResponse(BaseModel):
    answer: str


# ==========================================
#    STARTUP E PROBES (Estilo Kubernetes)
# ==========================================


@app.on_event("startup")
def startup_event():
    """Inicializa configurações, hardware, Redis e baixa artefatos do MLflow."""
    global model, scaler, config, device, feature_store

    try:
        with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        device = torch.device(
            "xpu"
            if hasattr(torch, "xpu") and torch.xpu.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Servidor inicializado com aceleração em: {device}")

        try:
            feature_store = RedisFeatureStore(host="redis", port=6379)
        except Exception as e:
            logger.warning(f"Aviso: Redis não acessível. Detalhe: {e}")

        # Carregamento via MLflow Registry
        nome_modelo = config["paths"]["registered_model_name"]
        logger.info(f"Buscando modelo '{nome_modelo}' no MLflow Registry...")

        model = mlflow.pytorch.load_model(f"models:/{nome_modelo}/latest").to(device)
        model.eval()

        client = MlflowClient()
        versoes = client.search_model_versions(f"name='{nome_modelo}'")
        ultima_versao = max(versoes, key=lambda v: int(v.version))

        local_scaler_path = mlflow.artifacts.download_artifacts(
            run_id=ultima_versao.run_id, artifact_path=config["paths"]["scaler_path"]
        )
        scaler = joblib.load(local_scaler_path)
        logger.info("Artefatos de inferência carregados com sucesso.")

    except Exception as e:
        logger.error(f"Erro no startup: {e}")


@app.get("/ready", tags=["Configuração"])
async def readiness():
    is_ready = model is not None and scaler is not None and feature_store is not None
    return {"status": "ready" if is_ready else "not_ready", "device": str(device)}


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
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
async def trigger_training(background_tasks: BackgroundTasks):
    """Dispara o pipeline de treinamento em background (sem travar a API)."""
    background_tasks.add_task(train_and_log)
    return {
        "message": "Treinamento assíncrono disparado com sucesso.",
        "status": "running",
    }


@app.post("/predict", tags=["Predição"])
def predict(req: PredictRequest):
    if not model or not scaler or not feature_store:
        raise HTTPException(
            status_code=503, detail="Serviço indisponível (Model/Redis não carregados)."
        )

    try:
        window_size = config["data"]["window_size"]
        input_size = config["model"]["input_size"]

        # 1. Puxa do Feature Store (Dataframe Multivariado)
        df_features = feature_store.obter_janela_predicao(
            req.ticker, window_size=window_size
        )

        # 2. Pré-processamento Multivariado
        dados_escalonados = scaler.transform(df_features.values)

        tensor_entrada = torch.tensor(
            dados_escalonados.reshape(1, window_size, input_size), dtype=torch.float32
        ).to(device)

        # 3. Inferência
        with torch.no_grad():
            predicao_tensor = model(tensor_entrada)
            resultado_escalonado = predicao_tensor.cpu().numpy()

        # 4. Desnormalização do Target (Close está na coluna 0)
        dummy = np.zeros((1, input_size))
        dummy[0, 0] = resultado_escalonado[0, 0]
        resultado_reais = scaler.inverse_transform(dummy)[0, 0]

        return {
            "ticker": req.ticker,
            "predicted_price_brl": round(float(resultado_reais), 2),
            "features_used": list(df_features.columns),
            "source": "Redis Feature Store (Multivariate)",
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erro na inferência: {e}")
        raise HTTPException(
            status_code=500, detail="Erro interno no pipeline de predição."
        )


@app.post("/agent", tags=["Agente"], response_model=AgentResponse)
async def agent_query(data: AgentRequest):
    """Consulta o agente ReAct protegido por Guardrails."""

    # 1. Barreira de Entrada (Input Guardrail - OWASP LLM01)
    is_valid, reason = input_guard.validate(data.query)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)

    # Lazy Load do LLM
    from src.agent.react_agent import create_datathon_agent
    from src.agent.tools import get_stock_tools

    try:
        tools = get_stock_tools()
        agent = create_datathon_agent(tools)

        # 2. Processamento do LLM
        result = agent.invoke({"input": data.query})
        resposta_bruta = result.get(
            "output", "Desculpe, não consegui processar a resposta."
        )

        # 3. Barreira de Saída (Output Guardrail - OWASP LLM06 / LGPD)
        resposta_segura = output_guard.sanitize(resposta_bruta)

        return AgentResponse(answer=resposta_segura)

    except Exception as e:
        logger.error(f"Erro no Agente ReAct: {e}")
        raise HTTPException(
            status_code=500, detail="Falha na geração da resposta do LLM."
        )
