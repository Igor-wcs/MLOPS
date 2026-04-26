"""ragas_eval.py

Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias.

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of Retrieval
            Augmented Generation. https://arxiv.org/abs/2309.15217
"""

import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "golden_set" / "golden_set.json"
)


def evaluate_rag_pipeline(
    golden_set_path: str | Path | None = None,
    rag_fn=None,
) -> dict[str, float]:
    """Avalia pipeline RAG contra golden set.

    Args:
        golden_set_path: Caminho para JSON com golden set (≥ 20 pares).
        rag_fn: Função que recebe query e retorna (answer, contexts).

    Returns:
        Dicionário com 4 métricas RAGAS.
    """
    golden_set_path = Path(golden_set_path) if golden_set_path else GOLDEN_SET_PATH

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    if len(golden_set) < 20:
        logger.warning("Golden set com %d pares. Datathon exige ≥ 20.", len(golden_set))

    results = []
    for item in golden_set:
        if rag_fn:
            answer, contexts = rag_fn(item["query"])
        else:
            answer = item.get("generated_answer", "")
            contexts = item.get("contexts", [])

        results.append(
            {
                "question": item["query"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
            }
        )

    dataset = Dataset.from_list(results)

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    metrics = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
    }

    logger.info("RAGAS scores: %s", metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scores = evaluate_rag_pipeline()
    print(json.dumps(scores, indent=2))
