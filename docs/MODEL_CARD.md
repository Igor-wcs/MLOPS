# 📄 Model Card: Previsor de Ações PETR4 (LSTM)

## 1. Detalhes do Modelo
* **Nome do Modelo:** `LSTM_Petrobras`
* **Versão:** 1.0 (Rastreado via MLflow)
* **Desenvolvedores:** Grupo XX - Datathon Pós-Tech
* **Tipo de Modelo:** Rede Neural Recorrente (Long Short-Term Memory - LSTM)
* **Framework:** PyTorch
* **Data de Treinamento:** Abril/2026

## 2. Uso Pretendido
* **Propósito Primário:** Prever o preço de fechamento das ações da Petrobras (PETR4.SA) para o próximo dia útil.
* **Casos de Uso Adequados:** Auxílio em dashboards analíticos de investimentos e sinalização de tendências de curto prazo.
* **Casos de Uso Inadequados (Out of Scope):** Operações de trading automatizado de alta frequência (HFT) e tomada de decisão financeira sem supervisão humana (Human-in-the-loop é obrigatório).

## 3. Dados de Treinamento e Avaliação
* **Fonte de Dados:** Yahoo Finance API (`yfinance`).
* **Período Histórico:** 01/01/2019 até a data atual.
* **Feature Engineering:** Janelas temporais de 30 dias de preços de fechamento (Close).
* **Pré-processamento:** Escalonamento Min-Max (0 a 1) usando `scikit-learn` (Artefato `scaler.pkl` versionado).
* **Split de Dados:** Divisão cronológica estrita (80% Treino / 20% Teste) para evitar vazamento de dados do futuro (Data Leakage).

## 4. Métricas de Desempenho
As métricas oficiais da última versão do modelo podem ser consultadas no servidor do MLflow da equipe:
* **Métrica de Otimização:** MSE (Mean Squared Error)
* **Métrica de Negócio (Interpretabilidade):** MAE (Mean Absolute Error) em Reais (R$).

## 5. Governança e Observabilidade
* **Monitoramento Operacional:** API FastAPI instrumentada com Prometheus para coleta de latência e throughput.
* **Data Drift:** Relatório automatizado com `Evidently AI` para comparação de distribuições de preços da janela de treinamento vs. produção. Threshold de alerta de degradação configurado para > 20% de features com drift.
* **Gatilho de Retreino:** Alertas do Evidently acionam a necessidade de validação Champion-Challenger para uma nova versão do modelo.

## 6. Limitações e Riscos
* **Risco de Concept Drift:** O modelo é puramente autoregressivo (baseado em preços passados). Ele não tem contexto sobre notícias, crises políticas ou relatórios trimestrais, podendo falhar abruptamente em eventos de "Cisne Negro".
* **Impacto Econômico:** Baixo risco sistêmico desde que respeitado o uso pretendido (Apenas suporte à decisão).