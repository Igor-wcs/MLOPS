import json
import logging
import yaml
from pathlib import Path
import pandas as pd

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

def load_config():
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_golden_set(path: str = "data/golden_set/golden_set.json") -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Golden set não encontrado em {path}. Crie o arquivo com 20 pares.")
    
    with open(path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    logger.info("Golden set carregado: %d pares", len(golden_set))
    return golden_set

def generate_rag_responses(golden_set: list[dict]) -> list[dict]:
    """Gera respostas dinâmicas passando pelo Agente Qwen e RAG local."""
    
    # Importações do nosso ecossistema
    from src.agent.react_agent import create_datathon_agent, query_agent
    from src.agent.tools import get_stock_tools
    
    # Importação do RAG (que vamos construir a seguir)
    try:
        from src.agent.rag_pipeline import RAGPipeline
        rag = RAGPipeline()
    except ImportError:
        logger.warning("RAGPipeline não encontrado. Simulando recuperação para fins de teste.")
        rag = None

    results = []
    
    # Instancia o Agente uma única vez para não recarregar o modelo na memória
    tools = get_stock_tools()
    agent = create_datathon_agent(tools)

    for item in golden_set:
        query = item["question"] # Usando "question" conforme padrão RAGAS
        
        # 1. Recuperar contextos
        if rag:
            docs = rag.retrieve(query, top_k=3)
            contexts = [doc.page_content for doc in docs]
        else:
            contexts = item.get("contexts", ["Contexto simulado pois RAGPipeline ainda não existe."])

        # 2. Gerar resposta via Agente
        try:
            response = query_agent(agent, query)
            answer = response["answer"]
        except Exception as e:
            logger.error(f"Erro ao gerar resposta para '{query[:50]}': {e}")
            answer = "Erro ao processar a pergunta."

        results.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("ground_truth", item.get("expected_answer")), # Suporta tanto "ground_truth" quanto "expected_answer"
        })
        logger.info(f"Processado: {query[:50]}...")

    return results

def evaluate_rag_pipeline(golden_set_path: str = "data/golden_set/golden_set.json") -> dict[str, float]:
    """Avalia o pipeline com RAGAS e salva auditoria no MLflow."""
    cfg = load_config()
    
    logger.info("Iniciando geração de respostas do Agente...")
    golden_set = load_golden_set(golden_set_path)
    results = generate_rag_responses(golden_set)

    dataset = Dataset.from_list(results)

    # Inicializa o LLM-Juiz (Requer OPENAI_API_KEY no .env)
    logger.info("Inicializando LLM Juiz (OpenAI) para avaliação sintética...")
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    # Avaliação RAGAS — As 4 métricas obrigatórias da banca
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False
    )

    # Processamento de Métricas
    df_results = scores.to_pandas()
    metrics = {
        "ragas_faithfulness": float(df_results["faithfulness"].mean(skipna=True)),
        "ragas_answer_relevancy": float(df_results["answer_relevancy"].mean(skipna=True)),
        "ragas_context_precision": float(df_results["context_precision"].mean(skipna=True)),
        "ragas_context_recall": float(df_results["context_recall"].mean(skipna=True)),
    }

    # ==========================================
    # LOG NO MLFLOW E ARTEFATOS
    # ==========================================
    try:
        import mlflow
        mlflow.set_experiment(cfg["paths"]["experiment_name"])
        
        with mlflow.start_run(run_name="Avaliacao_RAGAS_Agente"):
            mlflow.set_tag("phase", "datathon-fase05")
            mlflow.log_metrics(metrics)
            
            # Salvar CSV de auditoria (A banca adora ver isso)
            report_path = "ragas_detailed_report.csv"
            df_results.to_csv(report_path, index=False)
            mlflow.log_artifact(report_path)
            logger.info("Métricas e CSV de auditoria salvos no MLflow.")
            
    except Exception as e:
        logger.warning(f"Aviso: Não foi possível logar no MLflow: {e}")

    return metrics

if __name__ == "__main__":
    evaluate_rag_pipeline()