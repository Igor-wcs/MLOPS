# 📂 Estrutura do Projeto (Completa)

Esta listagem contém a estrutura organizacional do projeto Datathon Fase 5, detalhando a função de cada diretório e arquivos principais.

## 📁 Diretório Raiz
- `.env` - Variáveis de ambiente locais (não versionado).
- `.env.example` - Template para configuração de chaves de API.
- `.gitignore` - Regras de exclusão para o Git.
- `coverage.xml` - Relatório de cobertura de testes gerado pelo Pytest.
- `Datathon - Fase 5 (1).pdf` - Documento de requisitos oficiais.
- `demo_day.py` - Script para demonstração final do projeto.
- `docker-compose.yaml` - Orquestração de serviços (Redis, Prometheus, Grafana, MLflow).
- `Dockerfile` - Definição da imagem para o container da API.
- `dvc.yaml` / `dvc.lock` - Gerenciamento de pipelines de dados e reprodutibilidade.
- `Makefile` - Comandos utilitários para automação de tarefas.
- `mlflow.db` - Banco de dados local para rastreamento de experimentos.
- `model_weights.pt` - Pesos do modelo LSTM treinado (Champion).
- `pyproject.toml` - Configuração central de dependências e ferramentas (Ruff, Mypy, Pytest).
- `ragas_detailed_report.csv` - Relatório detalhado de avaliação do RAG.
- `README.md` - Guia principal de inicialização e uso.
- `requirements.txt` - Lista de dependências para instalação rápida.
- `scaler.pkl` - Objeto scaler utilizado para normalização de dados.

## 📁 configs/
- `model_config.yaml` - Hiperparâmetros do modelo e configurações do pipeline.
- `monitoring_config.yaml` - Regras de drift, alertas e guardrails de segurança.
- `prometheus.yaml` - Configuração do sistema de métricas.

## 📁 dags/
- `datathon_dag.py` - Orquestração do pipeline de retreino via Airflow.

## 📁 data/
- `chroma_db/` - Banco de dados vetorial para o RAG.
- `documents/` - Documentos de referência (ex: politica_investimento.txt).
- `golden_set/` - Conjunto de teste para avaliação quantitativa do LLM.
- `processed/` - Dados limpos e preparados para treino.

## 📁 docs/
- `ARCHITECTURE.md` - Visão técnica e diagramas do sistema.
- `LGPD_PLAN.md` - Estratégia de privacidade e anonimização.
- `MODEL_CARD.md` - Detalhes do modelo LSTM (PETR4).
- `OWASP_MAPPING.md` - Mitigação de vulnerabilidades de IA e API.
- `PROJECT_STRUCTURE.md` - Este documento.
- `RED_TEAM_REPORT.md` - Resultados de testes adversariais.
- `REQUIREMENTS.md` - Checklist de requisitos cumpridos.
- `SYSTEM_CARD.md` - Detalhes do agente inteligente e governança.

## 📁 evaluation/
- `llm_benchmark.py` - Comparação de performance entre modelos.
- `llm_judge.py` - Avaliação qualitativa automatizada.
- `ragas_eval.py` - Cálculo de métricas de fidelidade e relevância do RAG.

## 📁 notebooks/
- `01_eda.ipynb` - Análise exploratória de dados inicial.
- `02_eda.ipynb` - Experimentos e visualizações adicionais.

## 📁 src/ (Código Fonte)
- `agent/` - Implementação do Agente ReAct, RAG e ferramentas.
- `features/` - Engenharia de features e integração com Redis.
- `models/` - Arquitetura, treinamento e factory do modelo LSTM.
- `monitoring/` - Detecção de drift e métricas operacionais.
- `security/` - Guardrails e detecção de PII (LGPD).
- `serving/` - API FastAPI para disponibilização do sistema.

## 📁 tests/
- Testes unitários e de integração cobrindo todos os módulos do `src/`.
