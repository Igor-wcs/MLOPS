import json
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


# ==========================================
# 1. SCHEMA DE AVALIAÇÃO (Pydantic)
# ==========================================
class EvaluationScore(BaseModel):
    """Schema para o LLM Juiz retornar notas estruturadas."""

    fidelidade: int = Field(description="Nota de 1 a 5 para fidelidade ao contexto (hallucination)")
    relevancia: int = Field(description="Nota de 1 a 5 para relevância à pergunta")
    comentario: str = Field(description="Breve explicação da nota")


# ==========================================
# 2. PROMPT DO JUIZ
# ==========================================
JUDGE_SYSTEM_PROMPT = """Você é um avaliador especializado em sistemas de análise financeira.
Avalie a resposta do assistente comparando-a com a Pergunta e o Gabarito Esperado.
Considere o contexto esperado para a análise.

Atribua notas rigorosas de 1 a 5 para os seguintes critérios:
1. Fidelidade: A resposta condiz com os fatos do gabarito? (5=Total, 1=Invenção)
2. Relevância: A resposta atende ao que foi perguntado? (5=Direta, 1=Fugiu do tema)

Responda APENAS em formato JSON."""

# ==========================================
# 3. CLASSE DO JUIZ
# ==========================================


class LLMJudge:
    """Juiz Sintético (LLM-as-a-Judge) para validar qualidade do Agente ReAct."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY não encontrada no ambiente.")

        self.llm = ChatOpenAI(model=model, temperature=0).with_structured_output(EvaluationScore)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", JUDGE_SYSTEM_PROMPT),
                (
                    "human",
                    "PERGUNTA: {question}\nGABARITO ESPERADO: {ground_truth}\n"
                    "RESPOSTA DO ASSISTENTE: {answer}",
                ),
            ]
        )

    def evaluate(self, question: str, ground_truth: str, answer: str) -> EvaluationScore:
        """Executa a avaliação de uma única resposta."""
        chain = self.prompt | self.llm
        return chain.invoke({"question": question, "ground_truth": ground_truth, "answer": answer})


# ==========================================
# 4. EXECUÇÃO DO BENCHMARK DE QUALIDADE
# ==========================================


def run_llm_judge() -> None:
    """Roda a avaliação em lote sobre o Golden Set."""
    logger.info("\n--- INICIANDO LLM-AS-A-JUDGE (QUALIDADE AGENTE) ---")

    # Carrega Golden Set
    try:
        with open("data/golden_set/golden_set.json", encoding="utf-8") as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        logger.error("Golden Set não encontrado em data/golden_set/golden_set.json")
        return

    judge = LLMJudge()
    results = []

    for item in golden_data:
        logger.info(f"Avaliando: {item['question'][:50]}...")

        # Aqui simulamos ou chamamos a API real do nosso Agente
        # Para o benchmark, assumimos que temos as respostas salvas ou chamamos local
        # mock_answer = "A Petrobras planeja investir US$ 102 bilhões entre 2024 e 2028."
        answer = item.get("actual_answer", "Resposta não coletada.")

        score = judge.evaluate(
            question=item["question"], ground_truth=item["ground_truth"], answer=answer
        )

        results.append(
            {
                "question": item["question"],
                "fidelidade": score.fidelidade,
                "relevancia": score.relevancia,
                "comentario": score.comentario,
            }
        )

    # Consolidação
    df = pd.DataFrame(results)
    logger.info("\n" + "=" * 30)
    logger.info(f"Média Fidelidade: {df['fidelidade'].mean():.2f}/5")
    logger.info(f"Média Relevância: {df['relevancia'].mean():.2f}/5")
    logger.info("=" * 30)

    df.to_csv("llm_judge_results.csv", index=False)
    logger.info("Relatório detalhado salvo em llm_judge_results.csv")


if __name__ == "__main__":
    run_llm_judge()
