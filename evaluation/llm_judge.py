"""llm_judge.py

Avaliação LLM-as-judge com ≥ 3 critérios.
Inclui critério de negócio (adequação financeira).
Blindado com Regex contra erros de formatação (JSONDecodeError).
"""

import json
import logging
import re
from pathlib import Path

from src.serving.llm_serving import generate_response

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """Você é um avaliador especializado em análise financeira.
Avalie a resposta gerada abaixo em 3 critérios, dando nota de 1 a 5 para cada:

1. **Precisão técnica**: A resposta contém informações financeiras corretas?
2. **Relevância**: A resposta aborda diretamente a pergunta?
3. **Adequação ao negócio**: A resposta é útil para decisões de investimento?

Pergunta: {question}
Resposta Gerada: {answer}
Gabarito Esperado: {ground_truth}

Responda APENAS em JSON com o formato exato abaixo, sem adicionar comentários antes ou depois:
{{"precisao_tecnica": <1-5>, "relevancia": <1-5>, "adequacao_negocio": <1-5>, "justificativa": "<texto>"}}
"""


def extract_json_from_llm_response(raw_text: str) -> dict:
    """Extrai e faz o parse seguro do JSON da resposta do LLM usando Regex."""
    # Busca qualquer bloco de texto que comece com { e termine com }
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if json_match:
        clean_json_string = json_match.group(0)
        try:
            return json.loads(clean_json_string)
        except json.JSONDecodeError:
            logger.error("Falha no parse do JSON extraído: %s", clean_json_string[:100])
            return {
                "precisao_tecnica": 0,
                "relevancia": 0,
                "adequacao_negocio": 0,
                "justificativa": f"Erro de parse no JSON limpo: {clean_json_string[:100]}",
            }
    else:
        logger.error("Nenhum padrão JSON encontrado na resposta: %s", raw_text[:100])
        return {
            "precisao_tecnica": 0,
            "relevancia": 0,
            "adequacao_negocio": 0,
            "justificativa": f"LLM não retornou nenhum formato JSON válido: {raw_text[:100]}",
        }


def evaluate_with_judge(
    golden_set_path: str | Path,
    answers: list[str],
) -> list[dict]:
    """Avalia respostas usando LLM-as-judge.

    Args:
        golden_set_path: Caminho para golden set JSON.
        answers: Lista de respostas geradas pelo agente.

    Returns:
        Lista de avaliações com scores e justificativas.
    """
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    evaluations = []
    for item, answer in zip(golden_set, answers):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=item["query"],
            answer=answer,
            ground_truth=item["expected_answer"],
        )

        # Gera a avaliação usando temperatura zero para máxima consistência
        raw = generate_response(prompt, temperature=0.0)

        # Usa a nossa nova função blindada
        result = extract_json_from_llm_response(raw)
        evaluations.append(result)

    # Resumo Matemático
    avg_scores = {
        "precisao_tecnica_avg": sum(e.get("precisao_tecnica", 0) for e in evaluations)
        / max(len(evaluations), 1),
        "relevancia_avg": sum(e.get("relevancia", 0) for e in evaluations)
        / max(len(evaluations), 1),
        "adequacao_negocio_avg": sum(e.get("adequacao_negocio", 0) for e in evaluations)
        / max(len(evaluations), 1),
    }

    logger.info("LLM-as-judge scores médios: %s", avg_scores)
    return evaluations


def evaluate_prompt_variants(
    question: str,
    prompt_variants: list[str],
) -> list[dict[str, float]]:
    """A/B test de variantes de prompt com extração segura."""
    results = []
    for variant in prompt_variants:
        # Gera a resposta do agente com a variante atual
        answer = generate_response(f"{variant}\n\nPergunta: {question}")

        # O Juiz avalia a resposta gerada
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            ground_truth="N/A",  # Teste cego, sem gabarito
        )

        raw = generate_response(judge_prompt, temperature=0.0)

        # Extração segura
        scores = extract_json_from_llm_response(raw)
        results.append(scores)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Script llm_judge.py pronto para uso com blindagem Regex.")
