# ==============================================================================
# DATATHON - FASE 05: AUTO-ML + RAG + GOVERNANÇA
# Makefile para automação de tarefas de desenvolvimento e MLOps.
# ==============================================================================

.PHONY: help setup lint format test train serve docker-up docker-down clean security

# Variáveis
PYTHON = python
PIP = pip
DVC = dvc
DOCKER_COMPOSE = docker-compose

help: ## Mostra esta ajuda
	@echo "--------------------------------------------------------------------------------"
	@echo "Datathon CLI - Comandos disponíveis:"
	@echo "--------------------------------------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Instala as dependências e configura o ambiente (incluindo pre-commit)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install

format: ## Formata o código usando Black e Ruff (isort)
	black src/ tests/ evaluation/ dags/
	ruff check src/ tests/ evaluation/ dags/ --fix

lint: ## Executa verificações de linting (Ruff, Mypy)
	ruff check src/ tests/ evaluation/ dags/
	mypy src/ dags/ evaluation/

security: ## Executa scans de segurança (Bandit)
	bandit -r src/ evaluation/ -ll

test: ## Executa a suíte de testes com cobertura
	pytest tests/ -x --cov=src --cov-report=term-missing

pipeline: ## Executa o pipeline completo via DVC
	$(DVC) repro

train: ## Executa apenas o treinamento do modelo LSTM
	datathon-train

serve: ## Inicia a API FastAPI localmente
	datathon-serve

eval: ## Executa a avaliação do RAG via RAGAS
	datathon-eval

docker-up: ## Sobe a infraestrutura completa (Airflow, MLflow, API, Redis)
	$(DOCKER_COMPOSE) up -d --build

docker-down: ## Para todos os serviços do Docker
	$(DOCKER_COMPOSE) down

docker-logs: ## Mostra os logs dos serviços Docker
	$(DOCKER_COMPOSE) logs -f

clean: ## Limpa arquivos temporários, cache e artefatos de teste
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "reports/test-results.xml" -delete
	@echo "Limpeza concluída."
