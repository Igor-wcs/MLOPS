# 📇 Model Card - LSTM Multivariada (PETR4)

## 1. Detalhes do Modelo
*   **Nome:** `LSTM_Petrobras`
*   **Versão:** 2.0.0 (Multivariada)
*   **Tipo:** Rede Neural Recorrente (LSTM) desenvolvida em PyTorch.
*   **Data de Registro:** 2026-05-03
*   **Owner:** Grupo-XX MLE Team

## 2. Metadados de Governança
*   **model_name:** `LSTM_Petrobras`
*   **model_type:** `regression_time_series`
*   **training_data_version:** `DVC_Managed_PETR4_5Y`
*   **features:** `[Close, Open, High, Low, Volume, EMA20]`
*   **risk_level:** medium (Previsão financeira de caráter informativo)
*   **fairness_checked:** true (Modelo treinado exclusivamente com séries temporais de mercado público)

## 3. Uso Pretendido
*   **Objetivo:** Prever o preço de fechamento do ativo PETR4.SA para o próximo dia útil com base em uma janela de 30 dias.
*   **Público-alvo:** Investidores e analistas que buscam suporte tecnológico para análise de tendências.
*   **Aviso:** Este modelo NÃO deve ser utilizado como única ferramenta de decisão. Não constitui recomendação de investimento.

## 4. Performance e Métricas (Treinamento Final)
*   **RMSE (Root Mean Squared Error):** ~ R$ 1.10
*   **MAE (Mean Absolute Error):** ~ R$ 0.82
*   **Sigma Tolerance (0.5σ):** > 80% de acertos dentro da margem de tolerância.
*   **Experiment Tracking:** Todos os runs estão registrados no MLflow no experimento `Datathon_Previsao_Acoes`.

## 5. Limitações e Enviesamentos
*   **Fatores Exógenos:** O modelo não reage instantaneamente a notícias políticas ou macroeconômicas extremas (Cisnes Negros).
*   **Variação de Janela:** A performance é otimizada para a janela de 30 dias; variações bruscas no volume podem causar degradação temporária.
*   **Data Drift:** A performance depende da estabilidade da distribuição estatística dos preços. Monitorado via `src/monitoring/drift.py`.

---

# 🏗️ System Card - Agente ReAct Financeiro

## 1. Arquitetura do Sistema
O sistema é um assistente inteligente baseado no padrão **ReAct (Reasoning and Acting)**:
*   **LLM Principal:** Qwen 2.5 0.5B (Quantizado para eficiência).
*   **RAG Engine:** ChromaDB + Embeddings MiniLM-L6-v2 para consulta de documentos de compliance.
*   **Modelo de Inferência:** Pipeline LSTM acoplado via ferramentas customizadas.
*   **Feature Store:** Redis para recuperação ultra-rápida de janelas de tempo.

## 2. Camadas de Segurança e Confiança
*   **Guardrails:** Implementação de `InputGuardrail` (anti-injection) e `OutputGuardrail` (anti-PII).
*   **Anonimização:** Integração nativa com Microsoft Presidio para conformidade LGPD.
*   **Monitoramento:** Telemetria via Prometheus e Dashboards Grafana.

## 3. Fluxo de Execução
1. Entrada do usuário passa pelo **InputGuardrail**.
2. O **Router Agent** decide a melhor ferramenta (LSTM, Yahoo Finance ou RAG).
3. A ferramenta retorna a observação técnica.
4. O LLM sintetiza a resposta final.
5. O **OutputGuardrail** aplica máscaras de PII e insere o Disclaimer Legal.
