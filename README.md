# 📈 Datathon Fase 5 - Sistema de MLOps & Agente Inteligente

Bem-vindo ao projeto do **Grupo 63** para a Fase 5 do Datathon. Este repositório contém uma plataforma completa de Engenharia de Machine Learning (MLE), unindo previsões de séries temporais (LSTM) com um Agente Inteligente de IA (LLM) protegido por camadas robustas de segurança e monitoramento.

---

## 📂 Estrutura do Projeto

Para facilitar sua navegação, aqui está o que você encontrará em cada pasta:

*   **`configs/`**: Arquivos de configuração do modelo, monitoramento e métricas.
*   **`dags/`**: Fluxos de automação para o Airflow (Retreino automático).
*   **`data/`**: Base de conhecimento (RAG), documentos de compliance e conjuntos de teste.
*   **`docs/`**: Documentação técnica detalhada (Arquitetura, LGPD, OWASP, etc).
*   **`evaluation/`**: Scripts para avaliar a qualidade das respostas da IA.
*   **`notebooks/`**: Análises exploratórias e experimentos iniciais.
*   **`src/`**: O "coração" do sistema (Agente, Modelo LSTM, Segurança e API).
*   **`tests/`**: Testes automatizados para garantir que nada quebre.

---

## 🛠️ Guia de Instalação Passo a Passo (PowerShell)

Este guia foi feito para que qualquer pessoa consiga rodar o projeto do zero. Siga os comandos na ordem exata.

### 1. Clonar e Preparar o Ambiente
Abra o seu **PowerShell** e cole os comandos abaixo:

```powershell
# 1. Baixe o código do projeto
git clone <URL_DO_REPOSITORIO>
cd datathon-grupo-XX

# 2. Configure suas chaves de acesso
# (Renomeia o exemplo e você deve abrir o arquivo .env e colocar sua OPENAI_API_KEY)
Copy-Item .env.example .env

# 3. Crie o ambiente isolado do Python
python -m venv .venv

# 4. Ative o ambiente
.\.venv\Scripts\Activate.ps1

# 5. Instale as dependências e o projeto
pip install -e .
```

### 2. Iniciar a Infraestrutura (Docker)
O sistema precisa de bancos de dados e painéis que rodam dentro do Docker. Certifique-se de que o **Docker Desktop** esteja aberto.

```powershell
docker-compose up -d
```
*Aguarde alguns minutos até que todos os containers fiquem verdes no Docker Desktop.*

### 3. Treinar e Preparar a IA
Agora vamos ensinar o sistema a prever preços e ler os manuais.

```powershell
# 1. Treinar o modelo de previsão (Petrobras)
datathon-train

# 2. Indexar os documentos na base de conhecimento (RAG)
python src/agent/rag_pipeline.py
```

### 4. Ligar o Servidor Principal
Este é o comando que ativa a API para você conversar com o agente.

```powershell
docker-compose up --build
ou
uvicorn src.serving.app:app --port 8000 --reload
```
> **Dica:** Mantenha esta janela do terminal aberta. Se fechar, o sistema para de responder.

---

## 📊 Central de Dashboards (Acesso Rápido)

Com o sistema rodando, você pode acessar todas as ferramentas de governança através do seu navegador:

| Ferramenta | O que faz? | Endereço (URL) |
| :--- | :--- | :--- |
| **🤖 Chat da IA** | Interface para testar o Agente (Swagger) | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **📈 MLflow** | Gestão de Modelos e Experimentos | [http://127.0.0.1:5000](http://localhost:5000) |
| **📉 Grafana** | Painéis de monitoramento técnico | [http://localhost:3000](http://localhost:3000) |
| **⏱️ Airflow** | Agendamento de tarefas e retreino | [http://localhost:8080](http://localhost:8080) |
| **📡 Prometheus** | Coleta de métricas em tempo real | [http://localhost:9090](http://localhost:9090) |

---

## 🧪 Comandos Úteis para Auditoria

Se precisar rodar testes ou verificar a saúde do sistema:

*   **Rodar Testes de Qualidade:** `pytest`
*   **Gerar Relatório de Drift (Mudança de Dados):** `python -m src.monitoring.drift`
*   **Limpar arquivos temporários:** `make clean` (requer `make` instalado)

---
**Grupo 63 - Datathon 2026**
*Este projeto segue rigorosamente os padrões de MLOps Nível 2 e conformidade LGPD.*
