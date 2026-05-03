# 🎯 Objetivos e Requisitos do Datathon - Fase 5

Este documento consolida os requisitos oficiais extraídos do arquivo `Datathon - Fase 5 (1).pdf`, servindo como guia mestre para a avaliação da banca.

## 1. Meta de Maturidade MLOps (Nível 2)
O projeto deve atingir o **Nível 2** do Modelo de Maturidade MLOps da Microsoft nas seguintes dimensões:

- **Experiment Management**: MLflow padronizado com registro de métricas, parâmetros e artefatos.
- **Model Management**: Model Registry com versionamento, linhagem e metadados obrigatórios.
- **CI/CD**: Pipeline automatizado (ex: GitHub Actions) com lint -> test -> build -> deploy (staging).
- **Monitoring**: Observabilidade completa (métricas, detecção de drift, dashboards e alertas).
- **Data Management**: Versionamento de dados via DVC/Delta Lake e uso de dados sintéticos em ambiente de dev.
- **Feature Management**: Uso de Feature Store com materialização incremental (não destrutiva).

---

## 2. Requisitos por Etapa de Desenvolvimento

### Etapa 1: Dados + Baseline (Fases 01-02)
- [ ] **EDA**: Documentada com insights relevantes.
- [ ] **Feature Engineering**: Com validação de schema.
- [ ] **Baseline**: Implementação (ex: Scikit-Learn ou MLP PyTorch) integrada ao MLflow.
- [ ] **Tracking**: Registro padronizado de métricas e parâmetros.

### Etapa 2: LLM + Agente (Fases 03-05)
- [ ] **Serving**: LLM servido via API (ex: vLLM, BentoML ou FastAPI).
- [ ] **Agente ReAct**: Implementação funcional com **no mínimo 3 ferramentas (tools)**.
- [ ] **RAG**: Pipeline completo com Embedding + Vector Store (ChromaDB).
- [ ] **FastAPI**: Endpoint documentado e integrado ao CI/CD.

### Etapa 3: Avaliação + Observabilidade (Fases 03-05)
- [ ] **Golden Set**: Pelo menos 20 pares de query/expected answer.
- [ ] **RAGAS**: Avaliação quantitativa com as 4 métricas obrigatórias.
- [ ] **LLM-as-judge**: Avaliação qualitativa com pelo menos 3 critérios de negócio.
- [ ] **Telemetria**: Dashboard (Prometheus + Grafana) para métricas operacionais.
- [ ] **Drift Detection**: Implementação via Evidently para dados e predições.

### Etapa 4: Segurança + Governança (Fases 04-05)
- [ ] **Guardrails**: Implementação funcional de barreiras de Input e Output.
- [ ] **OWASP Top 10**: Mapeamento de pelo menos 5 ameaças e suas mitigações.
- [ ] **Red Teaming**: Pelo menos 5 cenários adversariais testados e documentados.
- [ ] **LGPD + Fairness**: Plano de conformidade e documentação de explicabilidade.
- [ ] **Documentação**: Model Card e System Card completos.

---

## 3. Padrões de Engenharia e Qualidade (GAPs a evitar)
A banca avaliará o rigor técnico através dos seguintes pontos:
- **Testes**: Cobertura de código (pytest) com no mínimo **60%**.
- **Clean Code**: Uso de *Type Hints*, *Docstrings* (padrão Google/NumPy) e *Logging Estruturado*.
- **Arquitetura**: Evitar SPOF (Single Point of Failure), especialmente o uso de notebooks em produção.
- **Governança**: Registro de modelos no MLflow com tags obrigatórias (`model_name`, `version`, `owner`, `risk_level`, `git_sha`, etc.).
- **Retraining**: Demonstrar estratégia de retreino (agendado ou baseado em eventos/drift).

---

## 4. Critérios de Avaliação (Pesos)
- **Pipeline de dados + baseline**: 10%
- **LLM serving + agente**: 15%
- **Avaliação de qualidade**: 10%
- **Observabilidade + monitoramento**: 10%
- **Segurança + guardrails**: 10%
- **Governança + conformidade**: 5%
- **Documentação + arquitetura**: 5%
- **PyTorch + MLflow**: 5%
- **Critérios de negócio (Empresa)**: 30%
