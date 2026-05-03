# 🛡️ Mapeamento de Segurança (OWASP) - Defesa em Profundidade

**Projeto:** Datathon Fase 05 - Predição de Séries Temporais e Agentes LLM
**Módulo:** Governança e Segurança

---

## 1. Visão Executiva
Para garantir a resiliência do nosso ecossistema financeiro, implementamos uma arquitetura de **Defesa em Profundidade**. Este documento mapeia as mitigações adotadas contra duas frentes de ataque distintas: as vulnerabilidades de Inteligência Artificial Generativa (**OWASP Top 10 for LLM Applications**) e as vulnerabilidades de Infraestrutura (**OWASP Top 10 API Security**).

---

## 2. Camada 1: Segurança do Agente de IA (OWASP for LLMs)

### 🛡️ OWASP LLM01: Prompt Injection (Injeção de Comandos)
* **A Ameaça:** Atacantes utilizam comandos maliciosos embutidos no prompt para ignorar as instruções do sistema (*Jailbreak*) e forçar o LLM a executar ações não intencionais.
* **A Nossa Mitigação:** A classe `InputGuardrail` (`src/security/guardrails.py`) utiliza inspeção baseada em Regex para interceptar padrões clássicos como `"ignore previous instructions"` ou `"act as a"`. Requisições infectadas são rejeitadas (Erro HTTP 400) antes de alcançarem o LLM Qwen.

### 🛡️ OWASP LLM06: Sensitive Information Disclosure (Vazamento de PII)
* **A Ameaça:** O LLM revela, de forma acidental ou induzida, Informações Pessoalmente Identificáveis (PII) nas suas respostas.
* **A Nossa Mitigação:** A classe `OutputGuardrail` varre a resposta usando o motor de NLP **Microsoft Presidio**. Expressões regulares bloqueiam a exposição de CPFs (`BR_CPF`) e telefones, substituindo-os por máscaras de segurança (Ex: `<BR_CPF>`) antes do envio da resposta.

### 🛡️ OWASP LLM09: Overreliance (Excesso de Confiança / Alucinação)
* **A Ameaça:** O sistema depende excessivamente do LLM para a tomada de decisões financeiras exatas, levando a alucinações matemáticas.
* **A Nossa Mitigação:** O Agente ReAct não calcula preços; ele consulta obrigatoriamente o modelo determinístico (LSTM) para séries temporais e a base vetorial (RAG) para contexto de negócios. A qualidade é auditada pelo framework RAGAS (mantendo *Faithfulness* elevado).

---

## 3. Camada 2: Segurança da Infraestrutura (OWASP API Security)

### 🛡️ API3:2023 - Broken Object Property Level Authorization (Injeção de Dados)
* **Ameaça:** Atacantes tentam enviar códigos maliciosos ou formatos incorretos no corpo do JSON para quebrar o treinamento ou a inferência da rede neural (Data Poisoning).
* **Mitigação:** Utilizamos o framework **Pydantic** no FastAPI. Os contratos (`PredictRequest` e `AgentRequest`) exigem tipagem estrita. Qualquer envio de texto ou formato não esperado é barrado automaticamente na porta de entrada (Erro 422 - Unprocessable Entity), blindando os tensores do PyTorch.

### 🛡️ API4:2023 - Unrestricted Resource Consumption (Negação de Serviço / DoS)
* **Ameaça:** Envio intencional de *payloads* gigantescos ou *prompts* imensos para estourar a memória RAM/VRAM do servidor (OOM) e causar instabilidade (Context Stuffing).
* **Mitigação:** Implementamos uma validação estrita de Length (Tamanho). No Agente, o *input* é validado contra o limite máximo de *tokens* configurado no YAML. Na inferência temporal, a API checa a janela exata de dias no Feature Store. Requisições maiores são rejeitadas.

### 🛡️ API7:2023 - Security Misconfiguration (Exposição via Traceback)
* **Ameaça:** Erros internos não tratados retornam mensagens de sistema (Tracebacks) que revelam a estrutura de pastas ou versões de bibliotecas ao atacante.
* **Mitigação:** O FastAPI intercepta falhas (como timeouts do Redis ou do MLflow) e encapsula os erros em respostas seguras usando `HTTPException`. O *Traceback* é ocultado do usuário final, retornando apenas mensagens controladas (ex: "Erro interno no pipeline de predição").
