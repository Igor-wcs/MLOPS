from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pytorch
import torch
import numpy as np
import joblib
import mlflow.artifacts
from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator

# Importando o nosso novo Feature Store (Redis)
from src.features.feature_store import RedisFeatureStore

# Configuração do App
app = FastAPI(title="API de Previsão PETR4 - Datathon")

# Instrumentação do Prometheus (GAP 01)
Instrumentator().instrument(app).expose(app)

# --- Inicialização do Feature Store (Redis) ---
try:
    # No Docker, o host se chama 'redis'. Se rodar local, mude para 'localhost'
    feature_store = RedisFeatureStore(host='redis', port=6379)
    print("Feature Store (Redis) conectado com sucesso!")
except Exception as e:
    print(f"Aviso: Não foi possível conectar ao Redis na inicialização: {e}")
    feature_store = None

# --- Carregamento do Modelo e do Scaler (MLflow) ---
MODEL_URI = "models:/LSTM_Petrobras/latest"

try:
    print(f"Carregando modelo de {MODEL_URI}...")
    modelo = mlflow.pytorch.load_model(MODEL_URI)
    modelo.eval() # Modo de inferência
    
    client = MlflowClient()
    versoes = client.search_model_versions("name='LSTM_Petrobras'")
    ultima_versao = max(versoes, key=lambda v: int(v.version))
    run_id = ultima_versao.run_id
    
    # Baixa o scaler salvo durante o treinamento
    local_scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
    scaler = joblib.load(local_scaler_path)
    print(f"✅ Modelo (Versão {ultima_versao.version}) e Scaler carregados com sucesso!")

except Exception as e:
    print(f"Erro ao carregar MLflow: {e}")
    modelo = None
    scaler = None


@app.get("/")
def home():
    return {"status": "API Online", "modelo": "LSTM_Petrobras", "feature_store": "Redis"}


# --- ROTA DE PREVISÃO (GAP 03 Compliant) ---
class RequisicaoPrevisao(BaseModel):
    ticker: str = "PETR4.SA"

@app.post("/predict")
def predict(req: RequisicaoPrevisao):
    if modelo is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo ou Scaler não carregados no servidor.")
    
    if feature_store is None:
        raise HTTPException(status_code=500, detail="Conexão com o Redis (Feature Store) indisponível.")
    
    try:
        # Busca os últimos 30 dias direto do Redis (Sem Flush, via Upsert Incremental)
        ultimos_30_precos = feature_store.obter_janela_predicao(req.ticker, window_size=30)
        
        # Normalização (De Reais para 0-1) usando o Scaler do MLflow
        precos_np = np.array(ultimos_30_precos).reshape(-1, 1)
        precos_escalonados = scaler.transform(precos_np)
        
        # Formatação para o PyTorch (batch_size, seq_len, input_size) -> (1, 30, 1)
        tensor_entrada = torch.tensor(precos_escalonados.reshape(1, 30, 1), dtype=torch.float32)

        # Predição
        with torch.no_grad():
            predicao_tensor = modelo(tensor_entrada)
            resultado_escalonado = predicao_tensor.item()

        # Transformação Inversa (De 0-1 de volta para Reais)
        resultado_reais = scaler.inverse_transform([[resultado_escalonado]])[0][0]

        return {
            "ticker": req.ticker,
            "previsao_reais": f"R$ {resultado_reais:.2f}",
            "fonte_dados": "Redis Feature Store (Upsert Incremental)",
            "observacao": "GAP 03 Resolvido - Risco de Janela Vazia Mitigado"
        }

    except ValueError as ve:
        # Tratamento de erro se a API for chamada antes do Airflow popular o Redis
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)