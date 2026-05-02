# 📋 Requisitos do Projeto - Datathon Fase 05

Com base no documento oficial `Datathon - Fase 5 (1).pdf`, os requisitos para a entrega final estão estruturados em quatro etapas principais, focando na maturidade MLOps (Nível 2) e segurança.

## 1. Maturidade MLOps (Objetivo: Nível 2)
O sistema deve atingir o nível esperado nas seguintes dimensões:
- **Experiment Management:** MLflow padronizado com registro de métricas, parâmetros e artefatos.
- **Model Management:** Uso de Model Registry com versionamento e metadados obrigatórios.
- **CI/CD:** Pipeline automatizado (ex: GitHub Actions) com etapas de lint, teste, build e deploy.
- **Monitoring:** Observabilidade completa com métricas, detecção de drift, dashboards e alertas.
- **Data Management:** Versionamento via DVC/Delta Lake e uso de dados sintéticos em ambiente de dev.
- **Feature Management:** Features compartilhadas e estratégia de materialização incremental.

## 2. Etapa 1: Dados + Baseline
- [ ] **EDA:** Análise exploratória documentada com insights relevantes.
- [ ] **Baseline:** Modelo treinado e métricas reportadas no MLflow.
- [ ] **Pipeline:** Versionamento via DVC + Docker, garantindo reprodutibilidade.
- [ ] **Métricas:** Mapeamento claro de métricas de negócio para métricas técnicas.
- [ ] **Dependências:** `pyproject.toml` com todas as dependências gerenciadas.

## 3. Etapa 2: LLM + Agente
- [ ] **Serviço LLM:** API com quantização aplicada para baixa latência.
- [ ] **Agente ReAct:** Funcional com pelo menos 3 ferramentas (tools) relevantes.
- [ ] **RAG:** Recuperação de contexto relevante a partir dos documentos fornecidos.
- [ ] **CI/CD:** Pipeline funcional (GitHub Actions).
- [ ] **Benchmark:** Documentado com pelo menos 3 configurações distintas.

## 4. Etapa 3: Avaliação + Observabilidade
- [ ] **Golden Set:** Pelo menos 20 pares de query/resposta relevantes ao domínio.
- [ ] **RAGAS:** Avaliação do pipeline RAG usando as 4 métricas obrigatórias.
- [ ] **LLM-as-judge:** Avaliação qualitativa com pelo menos 3 critérios de negócio.
- [ ] **Telemetria:** Dashboard funcional (Prometheus/Grafana/Langfuse) end-to-end.
- [ ] **Drift:** Detecção de drift (dados e predições) implementada e documentada.

## 5. Etapa 4: Segurança + Governança
- [ ] **OWASP LLM:** Mapeamento de pelo menos 5 ameaças e suas mitigações.
- [ ] **Guardrails:** Implementação funcional de barreiras de entrada (input) e saída (output).
- [ ] **Red Teaming:** Testes com pelo menos 5 cenários adversariais documentados.
- [ ] **LGPD:** Plano de conformidade aplicado ao caso de uso real.
- [ ] **Explicabilidade:** Documentação de explicabilidade e fairness do modelo.
- [ ] **Cards:** Model Card e System Card completos.

## 6. Padrões Técnicos e Engenharia (GAPs a evitar)
- **Qualidade de Código:** Uso de Type hints, Docstrings (Google/NumPy), Logging estruturado.
- **Testes:** Cobertura de testes funcional (pytest) com no mínimo 60%.
- **Infraestrutura:** Evitar SPOF em notebooks; os componentes do pipeline devem ser isolados (DAGs).
- **Feature Store:** Estratégia de atualização incremental (não destrutiva).
- **Secrets:** Nunca usar hardcoded; utilizar `.env` ou gerenciadores de segredos.

## 7. Demo Day
- [ ] Pitch de no máximo 10 minutos (Problema -> Abordagem -> Demo -> Resultados -> Impacto).
- [ ] Preparação para Q&A técnico e de negócio.
