# Plano de Conformidade LGPD

> Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais

## 1. Aplicabilidade ao Sistema

### 1.1. Dados Processados

| Tipo de Dado | Classificação | Origem | Tratamento |
|---|---|---|---|
| Preços de ações | Dado público | Yahoo Finance | Sem restrição LGPD |
| Queries do usuário | Dado pessoal (potencial) | Input da API | Sanitização via InputGuardrail |
| Respostas do LLM | Dado derivado | Gerado pelo modelo | PII scan (OutputGuardrail) antes da devolução |

### 1.2. Princípios LGPD Aplicados

1. **Finalidade (Art. 6°, I)**: Processamento focado exclusivamente em análise financeira e suporte ao usuário.
2. **Adequação (Art. 6°, II)**: Coleta mínima de dados necessária para a predição e resposta do agente.
3. **Necessidade (Art. 6°, III)**: Minimização de dados; o sistema não exige cadastro de usuários.
4. **Transparência (Art. 6°, VI)**: Documentação completa via System Card e Model Card.
5. **Segurança (Art. 6°, VII)**: Uso de Guardrails, criptografia em trânsito e isolamento de processos.
6. **Prevenção (Art. 6°, VIII)**: Monitoramento contínuo de drift e integridade do modelo.

---

## 2. Medidas Técnicas e Arquitetura (Privacy by Design)

O sistema utiliza a biblioteca **Microsoft Presidio** integrada ao `OutputGuardrail` (`src/security/guardrails.py` e `src/security/pii_detection.py`).

### 2.1. Localização para o Cenário Brasileiro
O motor do Presidio foi estendido com padrões customizados para o mercado brasileiro:
- **`BR_CPF`**: Regex para detecção de Cadastro de Pessoa Física.
- **`BR_PHONE`**: Padrões para telefones fixos e celulares brasileiros.
- **`PERSON` / `EMAIL_ADDRESS`**: Reconhecedores baseados em NLP (Spacy `pt_core_news_lg`).

### 2.2. Otimização de Performance
- **Lazy Loading**: O motor Presidio e o modelo SpaCy são carregados apenas quando necessários e mantidos em cache (`@lru_cache`).
- **Singleton**: Garantia de uma única instância pesada em memória durante o ciclo de vida da API.

### 2.3. Gestão de Configurações
O mascaramento é controlado centralizadamente em `configs/monitoring_config.yaml` sob a chave `pii_detection_enabled`, permitindo ajustes rápidos de governança.

---

## 3. Política de Retenção e Direitos do Titular

### 3.1. Retenção
| Dado | Retenção | Justificativa |
|---|---|---|
| Dados de mercado | Cache Redis (90 dias) | Eficiência de inferência (TTL configurável). |
| Queries de usuário | Não persistidas | Privacidade por padrão (Privacy by Default). |
| Logs de operação | 30 dias | Segurança e auditoria, sem PII. |
| Artefatos MLflow | Vitalício (Ciclo de Vida) | Rastreabilidade e conformidade MLOps. |

### 3.2. Direitos do Titular
O sistema foi projetado para não armazenar dados pessoais identificáveis. Em caso de logs residuais, o processo de anonimização garante que o titular não seja identificado, atendendo ao direito de anonimização e eliminação.

---

## 4. Matriz de Riscos e Mitigação

1. **Risco**: Injeção de dados pessoais em queries.
   - **Mitigação**: O `InputGuardrail` bloqueia padrões suspeitos e o `OutputGuardrail` anonimiza qualquer eco de dado sensível.
2. **Risco**: Alucinação de PII pelo LLM.
   - **Mitigação**: Varredura de 100% das saídas do LLM pelo motor Presidio.
3. **Risco**: Acesso indevido aos artefatos de treino.
   - **Mitigação**: Controle de acesso ao servidor MLflow e armazenamento seguro de pesos e scalers.
