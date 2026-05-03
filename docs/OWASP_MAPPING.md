# 🛡️ Mapeamento de Segurança (OWASP) - Defesa em Profundidade

**Projeto:** Datathon Fase 05 - Predição de Séries Temporais e Agentes LLM
**Módulo:** Governança e Segurança

---

## 1. Visão Executiva
Implementamos uma estratégia de **Defesa em Profundidade** que protege desde a entrada do usuário até a saída do modelo, cobrindo vulnerabilidades de IA e de APIs modernas.

---

## 2. Segurança do Agente de IA (OWASP Top 10 for LLMs)

### 🛡️ LLM01: Prompt Injection
* **Ameaça:** Comandos maliciosos para burlar instruções do sistema (Jailbreak).
* **Mitigação:** `InputGuardrail` (`src/security/guardrails.py`) utiliza Regex multi-idioma para detectar padrões como `"ignore all previous instructions"`, `"act as a"` ou `"você agora é"`.
* **Ação:** Requisições suspeitas são bloqueadas antes de atingir o LLM.

### 🛡️ LLM06: Sensitive Information Disclosure
* **Ameaça:** Revelação acidental de dados pessoais (PII) nas respostas.
* **Mitigação:** `OutputGuardrail` integrado com **Microsoft Presidio**. Detecção de `BR_CPF`, `BR_PHONE`, `EMAIL` e `PERSON` com anonimização automática.
* **Ação:** O texto é mascarado (ex: `<BR_CPF>`) antes de sair da API.

### 🛡️ LLM04: Model Denial of Service
* **Ameaça:** Prompts excessivamente longos para exaurir memória (Context Stuffing).
* **Mitigação:** Validação de `max_input_tokens` (default 4096 chars) no `InputGuardrail`.
* **Ação:** Payloads gigantes são rejeitados imediatamente.

### 🛡️ LLM09: Overreliance
* **Ameaça:** Dependência cega de saídas do LLM que podem conter alucinações.
* **Mitigação:** Uso de ferramentas determinísticas (`yfinance`, LSTM PyTorch) e auditoria via **RAGAS**.
* **Ação:** O Agente cita fontes e usa dados reais, não apenas geração probabilística.

---

## 3. Segurança da Infraestrutura (OWASP API Security)

### 🛡️ API3:2023 - Broken Object Property Level Authorization
* **Mitigação:** Validação estrita de schema via **Pydantic** no FastAPI. Tipagem forte impede injeção de tipos inesperados nos tensores do modelo.

### 🛡️ API4:2023 - Unrestricted Resource Consumption
* **Mitigação:** Rate limiting (via Redis) e timeouts configurados para chamadas externas à API do Yahoo Finance e processamento de LLM.

### 🛡️ API7:2023 - Security Misconfiguration
* **Mitigação:** Logs estruturados que ocultam informações de sistema. Erros são encapsulados em mensagens amigáveis sem expor o *stack trace* do servidor.

---

## 4. Matriz de Guardrails Implementados

| Camada | Ferramenta | Objetivo |
| :--- | :--- | :--- |
| **Input** | `InputGuardrail` | Anti-Injection, Tamanho de Payload. |
| **Logic** | Pydantic Models | Integridade de Dados, Schema Validation. |
| **Output** | `OutputGuardrail` | Conformidade LGPD, Mascaramento de PII. |
| **Ops** | Prometheus/Drift | Saúde do Modelo, Detecção de Anomalias. |
