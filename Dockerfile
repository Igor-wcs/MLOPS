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

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia definições de pacotes
COPY pyproject.toml README.md ./
COPY src/serving/requirements.txt ./serving-requirements.txt

# Instala dependências (Usa o requirements.txt do serving para ser mais leve na API)
RUN pip install --upgrade pip && \
    pip install -r serving-requirements.txt

# Copia o código fonte
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

# --- ESTÁGIO 2: API (Imagem Otimizada para Inferência) ---
FROM base-image as api
EXPOSE 8000
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

# Copia o código e dependências para o contexto do Airflow
COPY --chown=airflow:root pyproject.toml README.md ./
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir mlflow dvc[s3] redis yfinance

COPY --chown=airflow:root src/ ./src/
COPY --chown=airflow:root dags/ ./dags/
COPY --chown=airflow:root configs/ ./configs/
