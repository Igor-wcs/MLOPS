# Plano de Conformidade LGPD

> Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais

## Aplicabilidade ao Sistema

### Dados Processados

| Tipo de Dado | Classificação | Origem | Tratamento |
|---|---|---|---|
| Preços de ações | Dado público | Yahoo Finance | Sem restrição LGPD |
| Queries do usuário | Dado pessoal (potencial) | Input da API | Sanitização + sem armazenamento |
| Respostas do LLM | Dado derivado | Gerado pelo modelo | PII scan antes do output |

### Princípios LGPD Aplicados

1. **Finalidade (Art. 6°, I)**: Dados de mercado usados exclusivamente para análise financeira.
2. **Adequação (Art. 6°, II)**: Features derivadas são proporcionais ao objetivo.
3. **Necessidade (Art. 6°, III)**: Apenas OHLCV + indicadores técnicos, sem dados pessoais.
4. **Transparência (Art. 6°, VI)**: System Card documenta todos os dados e processamentos.
5. **Segurança (Art. 6°, VII)**: Guardrails de input/output, PII detection, encryption at rest.
6. **Prevenção (Art. 6°, VIII)**: Drift detection previne degradação silenciosa.

## Medidas Técnicas

### Proteção de Dados Pessoais em Queries

```python
# Presidio detecta e anonimiza PII antes de processar
entities = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "BR_CPF"]
results = analyzer.analyze(text=user_input, entities=entities)
if results:
    anonymized = anonymizer.anonymize(text=user_input, analyzer_results=results)
```

### Política de Retenção

| Dado | Retenção | Justificativa |
|---|---|---|
| Dados de mercado | Cache local, sem PII | Dados públicos |
| Queries de usuário | Não armazenadas | Minimização |
| Logs de operação | 30 dias, sem PII | Observabilidade |
| Artefatos MLflow | Indefinido | Rastreabilidade |

### Direitos do Titular

- **Acesso**: Dados de mercado são públicos, sem dados pessoais armazenados.
- **Eliminação**: Sem dados pessoais persistidos.
- **Portabilidade**: Não aplicável (sem dados pessoais).

## Riscos Identificados

1. **Risco**: Usuário insere PII na query do agente
   - **Mitigação**: InputGuardrail + Presidio scan
   - **Residual**: Baixo (PII detectado e anonimizado)

2. **Risco**: LLM gera PII no output (hallucination)
   - **Mitigação**: OutputGuardrail com Presidio
   - **Residual**: Médio (modelos podem gerar padrões semelhantes a CPF)

3. **Risco**: Logs contêm dados pessoais
   - **Mitigação**: Logging estruturado sem conteúdo de queries
   - **Residual**: Baixo