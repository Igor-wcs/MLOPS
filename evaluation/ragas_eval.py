import json
import logging
import os

import mlflow
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, reresponse_relevancy

# Importações do nosso ecossistema
from src.agent.rag_pipeline import RAGPipeline
from src.agent.react_agent import create_datathon_agent, query_agent
from src.agent.tools import get_stock_tools

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def load_config() -> dict:
    """Carrega as configurações do projeto."""
    with open("configs/model_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_agent_response(question: str) -> dict:
    """Gera respostas dinâmicas passando pelo Agente Qwen e RAG local."""
    try:
        rag = RAGPipeline()
        tools = get_stock_tools()
        agent = create_datathon_agent(tools)

        result = query_agent(agent, question)
        answer = result.get("answer", "")

        # Recupera contextos usados para o RAGAS validar
        contexts = [doc.page_content for doc in rag.retrieve(question)]

        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        logger.error(f"Erro ao gerar resposta para RAGAS: {e}")
        return {"answer": "Erro", "contexts": []}


def run_ragas_evaluation() -> None:
    """Executa o framework RAGAS para medir Fidelidade e Relevância do Agente."""
    cfg = load_config()
    logger.info("\n--- INICIANDO AVALIAÇÃO RAGAS (MÉTRICAS LLM) ---")

    # 1. Carregamento do Golden Set (Gabarito)
    try:
        with open("data/golden_set/golden_set.json", encoding="utf-8") as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        logger.error("Golden Set não encontrado.")
        return

    # 2. Coleta de Respostas do Agente em Tempo Real
    logger.info(f"Processando {len(golden_data)} perguntas pelo Agente local...")
    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in golden_data:
        question = item["question"]
        logger.info(f"Agente respondendo: {question[:60]}...")

        res = get_agent_response(question)

        dataset_dict["question"].append(question)
        dataset_dict["answer"].append(res["answer"])
        dataset_dict["contexts"].append(res["contexts"])
        dataset_dict["ground_truth"].append(item["ground_truth"])

    # 3. Conversão para Dataset HuggingFace (Exigido pelo RAGAS)
    dataset = Dataset.from_dict(dataset_dict)

    # 4. Execução da Avaliação
    # Inicializa o LLM-Juiz (Requer OPENAI_API_KEY no .env)
    logger.info("Inicializando LLM Juiz (OpenAI) para avaliação sintética...")

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("Erro: OPENAI_API_KEY necessária para rodar RAGAS.")
        return

    # RAGAS Metrics
    # Faithfulness: A resposta é baseada apenas no contexto? (Evita Alucinação)
    # Relevancy: A resposta foca no que foi perguntado?
    metrics = [faithfulness, reresponse_relevancy]

    logger.info("Calculando métricas RAGAS... (Isso pode levar alguns minutos)")
    result = evaluate(
        dataset,
        metrics=metrics,
    )

    # 5. Exportação de Resultados
    logger.info("\n" + "=" * 40)
    logger.info("RESULTADOS RAGAS (0.0 a 1.0):")
    for metric_name, value in result.items():
        logger.info(f"- {metric_name:20}: {value:.4f}")
    logger.info("=" * 40)

    # Log no MLflow
    try:
        mlflow.set_experiment(cfg["paths"]["experiment_name"])
        with mlflow.start_run(run_name="RAGAS_Evaluation"):
            mlflow.log_metrics(result)
            # Salva o dataframe detalhado como artefato
            df_detailed = result.to_pandas()
            df_detailed.to_csv("ragas_detailed_report.csv", index=False)
            mlflow.log_artifact("ragas_detailed_report.csv")
            logger.info("Métricas RAGAS enviadas ao MLflow com sucesso.")
    except Exception as e:
        logger.warning(f"Erro ao logar no MLflow: {e}")


if __name__ == "__main__":
    run_ragas_evaluation()
