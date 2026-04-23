from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pytorch
import torch
import numpy as np
from typing import List
from src.models.lstm_model import ModeloLSTM
import joblib
import mlflow.artifacts
from mlflow.tracking import MlflowClient

# 1. Configuração do App
app = FastAPI(title="API de Previsão PETR4 - Datathon")

# 2. Definição do formato de entrada (o que a API espera receber)
class DadosEntrada(BaseModel):
    precos: List[float] # Uma lista com os últimos 30 preços de fechamento

# 3. Carregamento do Modelo (Direto do MLflow Model Registry)
# Usamos 'models:/NomeDoModelo/Versao' ou 'latest'
MODEL_URI = "models:/LSTM_Petrobras/latest"

try:
    print(f"Carregando modelo de {MODEL_URI}...")
    modelo = mlflow.pytorch.load_model(MODEL_URI)
    modelo.eval() # Modo de inferência
    client = MlflowClient()
    versoes = client.search_model_versions("name='LSTM_Petrobras'")
    
    # Encontra a versão com o número mais alto (a última que você treinou)
    ultima_versao = max(versoes, key=lambda v: int(v.version))
    run_id = ultima_versao.run_id
    local_scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
    scaler = joblib.load(local_scaler_path)
    print(f"Modelo (Versão {ultima_versao.version}) e Scaler carregados com sucesso!")

except Exception as e:
    print(f"Erro ao carregar: {e}")
    modelo = None
    scaler = None

@app.get("/")
def home():
    return {"status": "API Online", "modelo": "LSTM_Petrobras"}

# 4. Rota de Previsão (POST)
@app.post("/predict")
def predict(entrada: DadosEntrada):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
    
    if len(entrada.precos) != 30:
        raise HTTPException(status_code=400, detail="A API exige exatamente 30 preços de fechamento anteriores.")

    try:
        # Preparação dos dados para o PyTorch
        # 1. Converter para array e normalizar (ou usar o scaler salvo se necessário)
        # Nota: Idealmente o Scaler também deveria ser um artefato salvo no MLflow
        dados_np = np.array(entrada.precos).reshape(1, 30, 1)
        tensor_entrada = torch.tensor(dados_np, dtype=torch.float32)

        # 2. Predição
        with torch.no_grad():
            predicao_tensor = modelo(tensor_entrada)
            resultado_escalonado = predicao_tensor.item()

        # 3. TRANSFORMAÇÃO INVERSA (De 0-1 para Reais)
        # O scaler espera um formato 2D, por isso usamos [[ ]]
        resultado_reais = scaler.inverse_transform([[resultado_escalonado]])[0][0]

        return {
            "ticker": "PETR4.SA",
            "previsao_escalonada": round(resultado_escalonado, 6),
            "previsao_reais": f"R$ {resultado_reais:.2f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)