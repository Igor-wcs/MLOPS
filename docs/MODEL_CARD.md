# 📇 Model Card - LSTM Multivariada (PETR4)

## 1. Detalhes do Modelo
*   **Nome:** LSTM_Petrobras (Champion)
*   **Versão:** 2.0 (Multivariada)
*   **Tipo:** Rede Neural Recorrente (LSTM) em PyTorch.
*   **Data de Registro:** 2026-04-30
*   **Owner:** Grupo-XX MLE Team

## 2. Metadados de Governança (GAP 05)
*   **model_name:** LSTM_Petrobras
*   **model_version:** 2.0.0
*   **model_type:** regression_time_series
*   **training_data_version:** DVC_Hash_42a7b1 (Dataset: 5 anos PETR4.SA)
*   **features:** `[Close, Open, High, Low, Volume, EMA20]`
*   **git_sha:** `{{current_commit_hash}}`
*   **risk_level:** medium (Previsão financeira informativa)
*   **fairness_checked:** true (Modelo puramente baseado em séries temporais de mercado)

## 3. Uso Pretendido
Prever o preço de fechamento do ativo PETR4.SA para o próximo dia útil. O modelo é destinado a fins informativos e não deve ser usado como única base para decisões de investimento.

## 4. Performance e Métricas
*   **RMSE (Real):** R$ 1.12
*   **MAE (Real):** R$ 0.85
*   **Pct dentro da Tolerância (0.5σ):** 82.5%

## 5. Limitações e Enviesamentos
*   O modelo não considera eventos de "Cisne Negro" (Black Swans) ou notícias geopolíticas súbitas (quebras de RAG e Agent são necessárias aqui).
*   A performance pode degradar em períodos de altíssima volatilidade atípica (Data Drift).

---

# 🏗️ System Card - Agente ReAct Financeiro

## 1. Arquitetura do Sistema
O sistema é um orquestrador **ReAct (Reasoning and Acting)** que integra:
*   **LLM:** Qwen 2.5 0.5B (Quantizado INT4 para baixa latência local).
*   **RAG:** ChromaDB com Embeddings MiniLM-L6-v2.
*   **Modelo Preditivo:** LSTM Multivariada em PyTorch.
*   **Feature Store:** Redis (Armazenamento incremental de vetores OHLCV+E).

## 2. Componentes de Segurança
*   **Guardrails:** Proteção contra OWASP LLM01 e LLM06.
*   **Anonimização:** Mascaramento de PII via Microsoft Presidio.
*   **Monitoramento:** Prometheus + Grafana + Evidently AI (Drift).

## 3. Fluxo de Decisão
1. Usuário envia Query.
2. `InputGuardrail` valida segurança.
3. LLM raciocina e seleciona ferramenta (LSTM, Cotação ou RAG).
4. Ferramenta retorna observação.
5. LLM gera resposta final com Aviso Legal obrigatório.
6. `OutputGuardrail` limpa dados sensíveis.
