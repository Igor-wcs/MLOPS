# 📈 Datathon Fase 5 - Sistema de ML + LLM Agent

Bem-vindo ao projeto do **Grupo XX** para a Fase 5 do Datathon. Este sistema é uma plataforma completa de Engenharia de Machine Learning (MLE) que combina previsões de séries temporais (LSTM) com um Agente Inteligente (ReAct + RAG) protegido por camadas de segurança e monitoramento de última geração.

---

## 📖 Guia de Inicialização (Passo a Passo)

Este guia foi feito para que você consiga rodar o projeto do zero, mesmo sem conhecer todos os comandos.

### 🏗️ 1. Preparando o Terreno
Abra o seu terminal (PowerShell no Windows ou Terminal no Mac/Linux).

1.  **Baixe o Código:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd datathon-grupo-XX
    ```

2.  **Configure suas Chaves (Secrets):**
    Renomeie o arquivo `.env.example` para apenas `.env` e abra-o em um editor de texto. Insira sua chave da OpenAI na linha:
    `OPENAI_API_KEY=sua_chave_aqui`

3.  **Crie o Ambiente de Trabalho (Python):**
    ```bash
    python -m venv .venv
    # Para Windows:
    .\.venv\Scripts\activate
    # Para Mac/Linux:
    source .venv/bin/activate
    
    pip install -e .
    ```

---

### 🐳 2. Ligando as Ferramentas (Docker)
O projeto usa o Docker para gerenciar bancos de dados e painéis de controle.
```bash
docker-compose up -d
```
*Isso ativa o Redis, Prometheus, Grafana, Airflow e o MLflow.*

---

### 🧠 3. Treinando a Inteligência Artificial
Agora, vamos ensinar o sistema a prever ações e a ler os manuais da empresa.

1.  **Treinar a Previsão de Ações (Petrobras):**
    ```bash
    datathon-train
    ```
    *Ele baixará os dados mais recentes, treinará a IA e salvará a "versão oficial" no sistema.*

2.  **Ensinar as Regras da Empresa (RAG):**
    ```bash
    python src/agent/rag_pipeline.py
    ```
    *Isso alimenta o banco de conhecimento com os documentos de compliance e política.*

---

### 🚀 4. Iniciando o Servidor (O Cérebro)
Com tudo pronto, ligue o servidor que responde às perguntas.
```bash
uvicorn src.serving.app:app --port 8000
```
> **Atenção:** Mantenha este terminal aberto! Se fechar, o sistema para de responder.

---

### 🧪 5. Prova de Qualidade (Testes e Drift)
Para mostrar que o código é profissional e seguro:

1.  **Rodar Testes Automatizados:**
    ```bash
    pytest --cov=src
    ```
    *Verifique se a cobertura está acima de 60%.*

2.  **Gerar Relatório de Mudança de Mercado (Drift):**
    ```bash
    python -m src.monitoring.drift
    ```
    *Isso cria um arquivo chamado `drift_report.html` na pasta do projeto. Abra-o no seu navegador para ver os gráficos.*

---

### 📊 6. Dashboards e Apresentação (Demo Day)
Acesse estes links no seu navegador para mostrar as evidências à banca:

*   **Agente Inteligente (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs) (Use o botão "Try it out" no endpoint `/agent`).
*   **Gestão de Modelos (MLflow):** [http://localhost:5000](http://localhost:5000) (Mostre a versão do modelo e as tags de governança).
*   **Orquestração (Airflow):** [http://localhost:8080](http://localhost:8080) (Mostre o fluxo de treinamento automático).
*   **Monitoramento (Grafana):** [http://localhost:3000](http://localhost:3000).

---

## 🛠️ Resumo de Comandos

| Objetivo | Comando |
| :--- | :--- |
| **Iniciar Infraestrutura** | `docker-compose up -d` |
| **Treinar IA** | `datathon-train` |
| **Subir Servidor** | `uvicorn src.serving.app:app` |
| **Limpar Tudo** | `make clean` |

---
**Grupo XX - Datathon 2026**
