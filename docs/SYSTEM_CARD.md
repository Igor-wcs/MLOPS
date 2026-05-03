# 🏗️ System Card - Agente ReAct Financeiro

## 1. Descrição do Sistema
*   **Nome:** Datathon Financial Agent
*   **Versão:** 1.0.0
*   **Modelo Base:** `Qwen/Qwen2.5-0.5B-Instruct` (Execução local).
*   **Framework:** LangChain v0.3.
*   **Objetivo:** Prover informações financeiras em tempo real, previsões de preço via IA e consulta a documentos de compliance.

## 2. Componentes e Ferramentas
*   **Motor RAG:** ChromaDB com Embeddings MiniLM-L6-v2 para busca semântica em manuais da empresa.
*   **Integração LSTM:** Chamada direta ao modelo PyTorch Champion para previsões de PETR4.
*   **Market Tools:** Integração com a API do Yahoo Finance para cotações e indicadores atuais.

## 3. Matriz de Segurança e Governança
| Categoria | Medida de Segurança | Descrição |
| :--- | :--- | :--- |
| **Integridade** | `InputGuardrail` | Filtro contra injeção de prompt e jailbreaks (OWASP LLM01). |
| **Privacidade** | `OutputGuardrail` | Mascaramento de dados pessoais (CPF, e-mail) via Presidio (LGPD). |
| **Robustez** | `max_input_tokens` | Limitação de tamanho de entrada para evitar estouro de memória. |
| **Confiança** | Disclaimer Automático | Todas as respostas incluem aviso legal de não recomendação. |

## 4. Avaliação de Performance
*   **Qualidade das Respostas:** Auditada via RAGAS (Fidelidade > 0.8).
*   **Latência de Resposta:** Otimizada via quantização e hardware-aware dispatch (GPU/XPU/CPU).
*   **Precisão do Roteamento:** Avaliado via testes de integração no Agente Router.

## 5. Monitoramento Operacional
*   **Observabilidade:** Métricas customizadas de latência e taxa de erro via Prometheus.
*   **Drift de Conceito:** Verificação periódica se as queries do usuário mudaram significativamente de perfil.
*   **Logs:** Registro de pensamentos intermediários do agente (Thought/Action/Observation) para auditoria.
