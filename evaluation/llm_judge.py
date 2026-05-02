import logging
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


# ==========================================
# SCHEMAS DO JUIZ (Com validação nativa)
# ==========================================
class EvaluationResult(BaseModel):
    # O Pydantic (ge=1, le=5) já substitui a necessidade da função _clamp_score
    technical_correctness: int = Field(ge=1, le=5, description="A resposta é factualmente correta?")
    relevance: int = Field(
        ge=1, le=5, description="A resposta aborda diretamente a pergunta feita?"
    )
    clarity: int = Field(
        ge=1,
        le=5,
        description="A resposta é clara, bem organizada e fácil de entender?",
    )
    investor_utility: int = Field(
        ge=1,
        le=5,
        description="A resposta fornece informações úteis para tomada de decisão de negócio?",
    )
    risk_disclaimers: int = Field(
        ge=1,
        le=5,
        description="A resposta inclui avisos de que não é recomendação de investimento?",
    )
    justification: str = Field(description="Justificativa geral e concisa para as notas aplicadas.")


# ==========================================
#  PROMPT DO JUIZ
# ==========================================
JUDGE_SYSTEM_PROMPT = """Você é um avaliador especializado em sistemas de análise financeira.
Avalie a resposta do assistente comparando-a com a Pergunta e o Gabarito Esperado, considerando o contexto esperado.

Atribua notas rigorosas de 1 a 5 para os seguintes critérios:
1. Correção Técnica
2. Relevância
3. Clareza
4. Utilidade para Investidor
5. Disclaimers de Risco (Se a pergunta não exige disclaimer, dê nota 5 por padrão).
"""


def load_config() -> dict:
    with open("configs/model_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_llm_judge(results_file_path: str = "ragas_detailed_report.csv"):
    """Avalia as respostas (já geradas pelo RAGAS) usando LLM-as-a-judge
    com 5 critérios de negócio e registra no MLflow.
    """
    cfg = load_config()
    file_path = Path(results_file_path)

    if not file_path.exists():
        logger.error(
            f"Arquivo '{results_file_path}' não encontrado. Rode o ragas_eval.py primeiro."
        )
        return

    # 1. Lê as respostas que o pipeline já gerou (Evita reprocessamento)
    df = pd.read_csv(file_path)
    required_cols = ["question", "answer", "ground_truth"]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"O CSV precisa conter as colunas: {required_cols}")
        return

    logger.info("Inicializando o LLM Juiz (GPT-4o-mini)...")
    try:
        # LangChain Wrapper com Structured Output
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        structured_llm = llm.with_structured_output(EvaluationResult)
    except Exception as e:
        logger.error(f"Falha ao iniciar o LLM Juiz: {e}. Verifique a sua OPENAI_API_KEY no .env.")
        return

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            (
                "human",
                "PERGUNTA: {question}\nGABARITO ESPERADO: {ground_truth}\nRESPOSTA DO ASSISTENTE: {answer}",
            ),
        ]
    )

    judge_chain = prompt | structured_llm

    # 2. Execução e Rastreamento (MLflow)
    mlflow.set_experiment(cfg["paths"]["experiment_name"])

    with mlflow.start_run(run_name="Avaliacao_LLM_Judge") as run:
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("evaluation_type", "llm-as-a-judge")

        evaluations = []
        logger.info(f"Julgando {len(df)} respostas com 5 critérios rigorosos...")

        for idx, row in df.iterrows():
            try:
                result: EvaluationResult = judge_chain.invoke(
                    {
                        "question": row.get("question", ""),
                        "ground_truth": row.get("ground_truth", ""),
                        "answer": row.get("answer", ""),
                    }
                )

                # O Pydantic já garantiu que os valores estão entre 1 e 5
                evaluations.append(
                    {
                        "question": row.get("question", ""),
                        "technical_correctness": result.technical_correctness,
                        "relevance": result.relevance,
                        "clarity": result.clarity,
                        "investor_utility": result.investor_utility,
                        "risk_disclaimers": result.risk_disclaimers,
                        "overall_score": (
                            result.technical_correctness
                            + result.relevance
                            + result.clarity
                            + result.investor_utility
                            + result.risk_disclaimers
                        )
                        / 5.0,
                        "justification": result.justification,
                    }
                )
            except Exception as e:
                logger.warning(f"Falha ao avaliar a linha {idx}: {e}")

        # 3. Consolidação de Métricas
        df_evals = pd.DataFrame(evaluations)
        if df_evals.empty:
            return

        avg_metrics = {
            "judge_technical_avg": df_evals["technical_correctness"].mean(),
            "judge_relevance_avg": df_evals["relevance"].mean(),
            "judge_clarity_avg": df_evals["clarity"].mean(),
            "judge_utility_avg": df_evals["investor_utility"].mean(),
            "judge_disclaimer_avg": df_evals["risk_disclaimers"].mean(),
            "judge_overall_avg": df_evals["overall_score"].mean(),
        }

        mlflow.log_metrics(avg_metrics)
        logger.info(f"Métricas Médias do Juiz: {avg_metrics}")

        # Exportação de Artefatos
        report_path = "llm_judge_detailed_report.csv"
        df_evals.to_csv(report_path, index=False)
        mlflow.log_artifact(report_path)

        logger.info("Julgamento finalizado. CSV de auditoria salvo no MLflow.")


if __name__ == "__main__":
    run_llm_judge()
