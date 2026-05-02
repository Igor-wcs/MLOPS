# 🏗️ System Card - Agente ReAct Financeiro

## 1. Descrição do Sistema
*   **Nome:** Datathon Financial Agent
*   **Versão:** 1.0.0
*   **Desenvolvedores:** Grupo-XX
*   **Arquitetura:** Agente ReAct com Roteamento Híbrido (Keyword + LLM).

## 2. Componentes Técnicos
*   **LLM Base:** Qwen/Qwen2.5-0.5B-Instruct (Quantizado).
*   **Orquestrador:** LangChain v0.3.
*   **Banco Vetorial (RAG):** ChromaDB.
*   **Embeddings:** sentence-transformers/all-MiniLM-L6-v2.
*   **Ferramentas Integradas:**
    *   `obter_previsao_lstm`: Inferência em tempo real via PyTorch.
    *   `obter_cotacao_atual`: Integração com Yahoo Finance API.
    *   `consultar_base_conhecimento`: Busca semântica em documentos de compliance.

## 3. Matriz de Segurança (OWASP LLM Top 10)
| Risco | Mitigação Implementada |
| :--- | :--- |
| **LLM01: Prompt Injection** | `InputGuardrail` com regex multicamadas. |
| **LLM04: Model Denial of Service** | Limite de 4096 tokens por input. |
| **LLM06: Sensitive Info Disclosure** | `OutputGuardrail` com Microsoft Presidio (LGPD). |
| **LLM07: Insecure Output Handling** | Sanitização de strings e validação de schema. |
| **LLM08: Insecure Plugin Design** | Ferramentas com escopo restrito e sem execução de código dinâmico. |

## 4. Governança e Ética
*   **Conformidade LGPD:** Mascaramento automático de CPF, Nomes e Contatos.
*   **Aviso Legal:** Todas as respostas contêm o disclaimer "Não constitui recomendação de investimento".
*   **Explicabilidade:** O Agente reporta os passos intermediários de raciocínio (Thought/Action).

## 5. Monitoramento e Observabilidade
*   **Data/Target Drift:** Monitorado via Evidently AI.
*   **Métricas Técnicas:** Latência e Status de Requisição via Prometheus.
*   **Qualidade do RAG:** Avaliação automática via RAGAS (Faithfulness, Relevancy).
