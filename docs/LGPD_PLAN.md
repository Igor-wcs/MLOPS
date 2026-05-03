# Plano de Conformidade LGPD

> Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais

## 1. Aplicabilidade ao Sistema

### 1.1. Dados Processados

| Tipo de Dado | Classificação | Origem | Tratamento |
|---|---|---|---|
| Preços de ações | Dado público | Yahoo Finance | Sem restrição LGPD |
| Queries do usuário | Dado pessoal (potencial) | Input da API | Sanitização + sem armazenamento |
| Respostas do LLM | Dado derivado | Gerado pelo modelo | PII scan (OutputGuardrail) antes da devolução |

### 1.2. Princípios LGPD Aplicados

1. **Finalidade (Art. 6°, I)**: Dados de mercado usados exclusivamente para análise financeira.
2. **Adequação (Art. 6°, II)**: Features derivadas são proporcionais ao objetivo preditivo.
3. **Necessidade (Art. 6°, III)**: Apenas OHLCV + indicadores técnicos, sem persistência de dados pessoais.
4. **Transparência (Art. 6°, VI)**: System Card documenta todos os dados e processamentos de IA.
5. **Segurança (Art. 6°, VII)**: Guardrails de input/output, deteção de PII via NLP e arquitetura isolada.
6. **Prevenção (Art. 6°, VIII)**: Módulo de *Drift Detection* previne a degradação silenciosa do modelo.

---

## 2. Medidas Técnicas e Arquitetura (Privacy by Design)

Para garantir a conformidade sem comprometer a latência da API, o sistema utiliza a biblioteca **Microsoft Presidio** acoplada a um padrão de **Output Guardrail** (`src/security/pii_detection.py`).

### 2.1. Localização para o Cenário Brasileiro
O motor do Presidio foi estendido com *Pattern Recognizers* customizados utilizando Expressões Regulares (Regex) rigorosas para evitar falsos negativos no mercado local:
- `BR_CPF`: Deteta e mascara o Cadastro de Pessoa Física.
- `BR_PHONE`: Deteta e mascara telefones com DDD brasileiro.
- `PERSON` / `EMAIL_ADDRESS`: Deteta e mascara nomes e e-mails via modelos de NLP do *Spacy*.

### 2.2. Otimização de Performance
O motor de análise é instanciado em modo *Singleton* utilizando a diretiva `@lru_cache(maxsize=1)`. O modelo de NLP é carregado em memória apenas uma vez no arranque do *FastAPI*, garantindo que a anonimização ocorra em milissegundos.

### 2.3. Gestão de Configurações
O mascaramento pode ser ativado ou desativado de forma centralizada pelo ficheiro `configs/monitoring_config.yaml` (chave `pii_detection_enabled`), garantindo a governança operacional.

---

## 3. Política de Retenção e Direitos do Titular

### 3.1. Retenção
| Dado | Retenção | Justificativa |
|---|---|---|
| Dados de mercado | Cache local (Redis) | Dados públicos, expiração via TTL |
| Queries de usuário | Não armazenadas | Minimização (Privacy by Default) |
| Logs de operação | 30 dias, sem PII | Observabilidade e Auditoria |
| Artefatos MLflow | Indefinido | Rastreabilidade do Modelo Campeão |

### 3.2. Direitos do Titular
- **Acesso**: Dados de mercado são públicos. Não há dados pessoais armazenados.
- **Eliminação**: Sem dados pessoais persistidos no banco de dados da aplicação.
- **Portabilidade**: Não aplicável (sistema isento de cadastros de utilizadores).

---

## 4. Matriz de Riscos

1. **Risco**: Utilizador insere PII de terceiros na *query* do Agente.
   - **Mitigação**: O `OutputGuardrail` sanitiza qualquer eco dessa informação na resposta do LLM.
   - **Risco Residual**: Baixo.

2. **Risco**: LLM gera PII acidental no output (Alucinação de dados similares a CPF).
   - **Mitigação**: Interceção de 100% das respostas pelo Microsoft Presidio antes de devolver o JSON à API.
   - **Risco Residual**: Médio (falsos positivos podem mascarar números financeiros legítimos, o que é preferível a vazar dados reais).

3. **Risco**: Logs da aplicação conterem dados pessoais.
   - **Mitigação**: *Logging* estruturado focado em latência e *thresholds* (`app.py`), sem imprimir o corpo (payload) das *queries* não tratadas.
   - **Risco Residual**: Baixo.
