"""Sistema de detecção e Anonimização de dados sensíveis PII.

Especializado para dados brasileiros (CPF, telefone BR).
Conformidade com a LGPDM e Mitigação OWASP LLM06.
"""

import logging
from functools import lru_cache
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_br_analyzer() -> AnalyzerEngine:
    """Cria e faz o cache do analisador com reconhecedores brasileiros."""
    # Configuração explícita do motor NLP para Português
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "pt", "model_name": "pt_core_news_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)

    # Reconhecedor de CPF Brasileiro
    cpf_pattern = Pattern(
        name="cpf_pattern",
        regex=r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
        score=0.9,
    )
    cpf_recognizer = PatternRecognizer(
        supported_entity="BR_CPF",
        patterns=[cpf_pattern],
        supported_language="pt",
    )
    analyzer.registry.add_recognizer(cpf_recognizer)

    # Reconhecedor de Telefone Brasileiro
    phone_pattern = Pattern(
        name="br_phone_pattern",
        regex=r"\b\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b",
        score=0.7,
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="BR_PHONE",
        patterns=[phone_pattern],
        supported_language="pt",
    )
    analyzer.registry.add_recognizer(phone_recognizer)

    logger.info("Presidio Analyzer com padrões Brasileiros (LGPD) carregado com sucesso.")
    return analyzer


class PIIDetector:
    """Classe responsável por escanear e anonimizar textos contra PII."""

    def __init__(self, language: str = "pt") -> None:
        """Inicializa o detector com suporte a múltiplos idiomas."""
        self.language = language

        try:
            self.analyzer = get_br_analyzer()
            self.anonymizer = AnonymizerEngine()
            self.is_active = True
        except Exception as e:
            logger.error(f"Erro ao inicializar Presidio. O mascaramento falhará aberto. Erro: {e}")
            self.is_active = False

        self.entities_to_find = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "BR_CPF",
            "BR_PHONE",
            "CREDIT_CARD",
        ]

    def scan(self, text: str) -> list[dict[str, Any]]:
        """Escaneia o texto em busca de PII e retorna os detalhes dos achados."""
        if not self.is_active or not text:
            return []

        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities_to_find,
        )

        findings = [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start : r.end],
            }
            for r in results
        ]

        if findings:
            logger.warning(
                f"🛡️ Alerta LGPD: Detectadas {len(findings)} ocorrências de PII no output da IA."
            )

        return findings

    def anonymize(self, text: str) -> str:
        """Escaneia o texto e substitui automaticamente o PII por máscaras."""
        if not self.is_active or not text:
            return text

        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities_to_find,
        )

        if not results:
            return text

        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )

        return str(anonymized_result.text)

    def sanitize_text(self, text: str) -> str:
        """Alias para integração com o OutputGuardrail."""
        return self.anonymize(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector_test = PIIDetector()
    texto_exemplo = (
        "O cliente João Silva, portador do CPF 123.456.789-00, "
        "solicitou contato no email joao@banco.com."
    )
    logger.info(f"Original: {texto_exemplo}")
    logger.info(f"Anonimizado: {detector_test.sanitize_text(texto_exemplo)}")
