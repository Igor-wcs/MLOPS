# 📂 Estrutura do Projeto (Filtrada)

Esta listagem contém as pastas e arquivos do código-fonte e configurações, excluindo arquivos binários, dados pesados e diretórios de cache/ambiente.

## 📁 Diretório Raiz
- `.env.example` - Template de variáveis de ambiente.
- `.gitignore` - Arquivos ignorados pelo Git.
- `contexto_projeto.txt` - Contexto do projeto.
- `docker-compose.yaml` - Configuração do Docker Compose.
- `Dockerfile` - Receita da imagem Docker.
- `dvc.yaml` - Pipeline do DVC.
- `dvc.lock` - Lockfile do DVC.
- `GEMINI.md` - Instruções do projeto.
- `Makefile` - Atalhos de comando.
- `mlflow.db` - Banco de dados do MLflow (Metadados).
- `prompt_auditoria.txt` - Prompt para auditoria.
- `pyproject.toml` - Gerenciamento de dependências e ferramentas.
- `README.md` - Documentação principal.

## 📁 configs/
- `model_config.yaml` - Configurações do modelo e caminhos.
- `monitoring_config.yaml` - Configurações de monitoramento e drift.
- `prometheus.yaml` - Configuração do Prometheus.

## 📁 dags/
- `datathon_dag.py` - DAG do Airflow para o pipeline.

## 📁 data/
- `documents/politica_investimento.txt` - Documento para a base RAG.
- `golden_set/golden_set.json` - Conjunto de testes para avaliação.

## 📁 docs/
- `ARCHITECTURE.md` - Documentação da arquitetura.
- `LGPD_PLAN.md` - Plano de conformidade LGPD.
- `MODEL_CARD.md` - Ficha técnica do modelo.
- `OWASP_MAPPING.md` - Mapeamento de segurança OWASP.
- `RED_TEAM_REPORT.md` - Relatório de testes adversariais.
- `REQUIREMENTS.md` - Requisitos extraídos do PDF.
- `SYSTEM_CARD.md` - Ficha técnica do sistema de agentes.

## 📁 evaluation/
- `llm_benchmark.py` - Script de benchmark de LLMs.
- `llm_judge.py` - Avaliação qualitativa via LLM.
- `ragas_eval.py` - Avaliação quantitativa de RAG.

## 📁 notebooks/
- `01_eda.ipynb` - Notebook de análise exploratória.
- `02_eda.ipynb` - Notebook complementar.

## 📁 src/
- `__init__.py`

### 📁 src/agent/
- `__init__.py`
- `rag_pipeline.py` - Lógica do pipeline RAG.
- `react_agent.py` - Implementação do Agente ReAct/Router.
- `tools.py` - Ferramentas disponíveis para o agente.

### 📁 src/features/
- `__init__.py`
- `feature_engineering.py` - Lógica de transformação de dados.
- `feature_store.py` - Interface com o Redis Feature Store.

### 📁 src/models/
- `__init__.py`
- `baseline.py` - Modelo de baseline simples.
- `lstm_factory.py` - Factory para criação do modelo LSTM.
- `lstm_model.py` - Arquitetura da rede neural LSTM.
- `lstm_params.py` - Definição de hiperparâmetros.
- `train.py` - Script principal de treinamento e tracking.

### 📁 src/monitoring/
- `__init__.py`
- `drift.py` - Detecção de Data e Target Drift.
- `metrics.py` - Métricas customizadas para Prometheus.

### 📁 src/security/
- `__init__.py`
- `guardrails.py` - Implementação de Input/Output Guardrails.
- `pii_detection.py` - Motor de detecção de PII (Presidio).

### 📁 src/serving/
- `__init__.py`
- `app.py` - API FastAPI para servir predições e o agente.
- `requirements.txt` - Dependências específicas para o container de serving.

## 📁 tests/
- `conftest.py` - Configurações e fixtures do pytest.
- `test_agent.py` - Testes do agente financeiro.
- `test_api.py` - Testes dos endpoints da API.
- `test_features.py` - Testes de engenharia de features.
- `test_guardrails.py` - Testes de segurança e guardrails.
- `test_models.py` - Testes da lógica do modelo e métricas.
