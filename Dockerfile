# ==============================================================================
# DATATHON - MULTI-STAGE DOCKERFILE
# Centraliza a build para Airflow e API (FastAPI) garantindo paridade de versões.
# ==============================================================================

# --- ESTÁGIO 1: BASE (Dependências de Sistema e Python) ---
FROM python:3.11-slim as base-image

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Copia APENAS o requirements para a raiz do container (Otimização de Cache)
COPY requirements.txt ./

# 3. Instala as dependências Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# 4. Agora copia o restante dos arquivos de configuração e metadados
COPY pyproject.toml README.md ./

# 5. Copia as pastas de código e recursos
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

# --- ESTÁGIO 2: API (Imagem Otimizada para Inferência) ---
FROM base-image as api
EXPOSE 8000
# Comando de inicialização da API
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

# --- ESTÁGIO 3: AIRFLOW (Imagem para Orquestração) ---
FROM apache/airflow:2.8.4-python3.11 as airflow

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/airflow

# Copia arquivos necessários para instalação do pacote local no Airflow
COPY --chown=airflow:root pyproject.toml README.md ./
# Nota: O Airflow aqui instala as dependências extras necessárias para orquestração
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir mlflow dvc[s3] redis yfinance && \
    pip install --no-cache-dir "SQLAlchemy<2.0"

# Copia o código e DAGs para o contexto do Airflow
COPY --chown=airflow:root src/ ./src/
COPY --chown=airflow:root dags/ ./dags/
COPY --chown=airflow:root configs/ ./configs/