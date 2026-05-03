# 📋 Requisitos do Projeto - Datathon Fase 05

Status de cumprimento dos requisitos baseados no documento oficial e nos padrões de maturidade MLOps Nível 2.

## 1. Maturidade MLOps (Objetivo: Nível 2)
- [x] **Experiment Management:** MLflow integrado para tracking de todos os runs de treino.
- [x] **Model Management:** Uso de MLflow Model Registry para governança do modelo `LSTM_Petrobras`.
- [x] **CI/CD:** Pipeline automatizado via GitHub Actions (`ci.yaml`) com lint e testes.
- [x] **Monitoring:** Stack Prometheus + Grafana e detecção de drift via Evidently AI.
- [x] **Data Management:** Versionamento de dados e artefatos via DVC.
- [x] **Feature Management:** Implementação de Feature Store com Redis para materialização incremental.

## 2. Etapa 1: Dados + Baseline
- [x] **EDA:** Notebooks `01_eda.ipynb` e `02_eda.ipynb` com análise profunda.
- [x] **Baseline:** Script `src/models/baseline.py` funcional e reportado.
- [x] **Pipeline:** DVC configurado para garantir reprodutibilidade total.
- [x] **Métricas:** Mapeamento de métricas financeiras (RMSE, MAE) e de negócio (Sigma Tolerance).

## 3. Etapa 2: LLM + Agente
- [x] **Serviço LLM:** API FastAPI com Qwen 2.5 local e roteamento eficiente.
- [x] **Agente ReAct:** Router Agent com 3 ferramentas: LSTM, yFinance e RAG.
- [x] **RAG:** Pipeline ChromaDB com documentos de compliance e política de investimento.
- [x] **Benchmark:** Documentado em `evaluation/llm_benchmark.py`.

## 4. Etapa 3: Avaliação + Observabilidade
- [x] **Golden Set:** Conjunto de dados para teste localizado em `data/golden_set/`.
- [x] **RAGAS:** Avaliação quantitativa implementada em `evaluation/ragas_eval.py`.
- [x] **LLM-as-judge:** Avaliação qualitativa em `evaluation/llm_judge.py`.
- [x] **Telemetria:** Endpoints `/metrics` funcionais e monitorados.
- [x] **Drift:** Script `src/monitoring/drift.py` gerando relatórios HTML automáticos.

## 5. Etapa 4: Segurança + Governança
- [x] **OWASP LLM:** Mapeamento completo e mitigações em `docs/OWASP_MAPPING.md`.
- [x] **Guardrails:** Proteção bidirecional funcional (Input/Output).
- [x] **Red Teaming:** Relatório de testes adversariais em `docs/RED_TEAM_REPORT.md`.
- [x] **LGPD:** Plano de anonimização com Microsoft Presidio em `docs/LGPD_PLAN.md`.
- [x] **Cards:** Model Card e System Card detalhados.

## 6. Qualidade Técnica
- [x] **Tipagem:** Uso extensivo de *Type Hints* e validação Mypy.
- [x] **Docstrings:** Segue o padrão Google/Pydocstyle.
- [x] **Testes:** Cobertura superior a 60% (verificado via `coverage.xml`).
- [x] **Segurança de Secrets:** Uso de `.env` e `.gitignore` para proteção de credenciais.
