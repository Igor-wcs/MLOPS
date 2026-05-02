# 📈 ML + LLM Stock Analysis System (Datathon Fase 05)

  !MLOps Maturity (https://img.shields.io/badge/MLOps-N%C3%ADvel%202-blue)<br>
  !Python (https://img.shields.io/badge/Python-3.11-green)<br>
  !PyTorch (https://img.shields.io/badge/PyTorch-2.2-red)<br>
  !FastAPI (https://img.shields.io/badge/FastAPI-0.110-teal)<br>
  !License (https://img.shields.io/badge/License-MIT-yellow)<br>

  Este projeto representa a entrega final do Datathon - Fase 05, um sistema end-to-end que integra modelagem preditiva de séries temporais
  (LSTM) com Inteligência Artificial Generativa (Agentes ReAct). O sistema foi projetado sob os mais rigorosos padrões de Engenharia de
  Machine Learning (MLOps), Segurança (OWASP) e Governança (LGPD).

  ---

  ## 📖 Descrição

  O sistema resolve o desafio de análise de ativos financeiros (focado em PETR4.SA) através de duas frentes principais:
   1. Predição Quantitativa: Modelo LSTM multivariado que utiliza dados históricos (OHLCV) e indicadores técnicos (EMA20) para prever o
      fechamento do próximo dia.
   2. Agente Inteligente: Um assistente financeiro baseado no modelo Qwen 2.5 (Local) que utiliza o padrão ReAct para decidir entre consultar
      previsões da IA, cotações em tempo real via Yahoo Finance ou buscar informações em uma base de conhecimento oficial (RAG com ChromaDB).

  ## 🚀 Diferenciais Técnicos
   - Maturidade MLOps Nível 2: Automação total via Airflow, versionamento de dados com DVC e registro de modelos com MLflow.
   - Feature Store Incremental: Uso de Redis para baixa latência, resolvendo o GAP de "Full-Flush" destrutivo.
   - Segurança em Profundidade: Guardrails contra injeção de prompt (Multi-idioma) e motor de anonimização de PII (LGPD) via Microsoft
     Presidio.
   - Observabilidade: Monitoramento de Data Drift e Prediction Drift com Evidently e telemetria completa via Prometheus/Grafana.

  ---

 ## 🏗️ Arquitetura do Sistema

  O projeto está dividido em 4 etapas lógicas fundamentais:

   1. Etapa 1 (Dados + Baseline): Ingestão via DVC, Feature Engineering e treinamento de modelos de base trackeados no MLflow.
   2. Etapa 2 (LLM + Agente): Implementação do Agente ReAct com 3 ferramentas customizadas e pipeline RAG para documentos financeiros.
   3. Etapa 3 (Avaliação + Observabilidade): Avaliação quantitativa (RAGAS) e qualitativa (LLM-as-a-judge), além de dashboards de saúde do
      sistema.
   4. Etapa 4 (Segurança + Governança): Implementação de Guardrails, Red Teaming e documentação técnica (Model/System Cards).

  ---

## 🛠️ Instalação e Requisitos

  Pré-requisitos
   - Docker & Docker Compose (Recomendado)
   - Python 3.11+
   - Poetry ou Pip

  Passo a Passo

   1. Clone o repositório:

   1     git clone https://github.com/seu-usuario/datathon-grupo-xx.git
   2     cd datathon-grupo-63

   2. Configuração de Ambiente:
      Crie um arquivo .env baseado no template (necessário para avaliação de LLMs):

   1     cp .env.example .env
   2     # Edite o .env e insira sua OPENAI_API_KEY

   3. Instalação Local (Opcional - para Dev):

   1     make setup
   2     python -m spacy download en_core_web_sm

  ---

## 🐳 Execução via Docker

  A infraestrutura completa pode ser iniciada com um único comando:

   1 make docker-up

  Painéis de Controle e Endereços:

  ┌─────────────────┬─────────────────────────────────────────────────────────┬───────────────────────────────────────────────┐
  │ Serviço         │ Endereço                                                │ Função                                        │
  ├─────────────────┼─────────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ Swagger UI      │ http://localhost:8000/docs (http://localhost:8000/docs) │ Testar endpoints de predição e agente.        │
  │ MLflow Registry │ http://localhost:5000 (http://localhost:5000)           │ Ver experimentos, métricas e pesos do modelo. │
  │ Airflow         │ http://localhost:8080 (http://localhost:8080)           │ Orquestração (Login: admin / admin).          │
  │ Prometheus      │ http://localhost:9090 (http://localhost:9090)           │ Métricas de sistema e negócio.                │
  │ Grafana         │ http://localhost:3000 (http://localhost:3000)           │ Dashboards de Observabilidade.                │
  └─────────────────┴─────────────────────────────────────────────────────────┴───────────────────────────────────────────────┘
  ---

##  📑 Uso e Comandos Principais

  O projeto utiliza um Makefile para padronizar as tarefas de MLOps:

   - Testes Automáticos (Cobertura > 80%):
   1     make test
   - Executar Pipeline DVC:
   1     make pipeline
   - Treinar Modelo LSTM:
   1     datathon-train
   - Avaliação RAGAS do Agente:
   1     datathon-eval

  ---

 ## 🛡️ Segurança e LGPD

  O sistema implementa o princípio de Privacy by Design:
   - Input Guardrail: Bloqueia tentativas de Jailbreak em Português, Inglês e Espanhol.
   - Output Guardrail: Utiliza o Microsoft Presidio para detectar e mascarar CPFs, nomes e e-mails em tempo real antes de entregar a resposta
     ao usuário.
   - Auditabilidade: Todas as decisões do Agente e detecções de PII são logadas no MLflow para auditoria técnica.

  ---

##  📈 Roteiro de Evolução (Roadmap)

   - [ ] Implementação de Fine-Tuning do modelo Qwen para terminologia financeira brasileira.
   - [ ] Integração de conectores de dados via Streaming (Kafka) para predição intra-day.
   - [ ] Expansão dos Guardrails para detecção de viés (Bias Detection) em recomendações.
   - [ ] Dashboards customizados no Grafana para visualização de lucro/prejuízo hipotético.

  ---

##  👥 Autores e Agradecimentos

   - Igor Wasiljew - MLE Specialist & Arquiteto de Sistemas
   - Agradecimentos à equipe da FIAP e à empresa convidada pelo desafio técnico.

  ---

##  📄 Licença

  Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE (LICENSE) para detalhes.

  ---
