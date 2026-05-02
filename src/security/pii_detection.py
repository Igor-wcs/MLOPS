"""
Sistema de detecção e Anonimização de dados sensíveis PII (Personally Identifiable Information) com Presidio.
Especializado para dados brasileiros (CPF, telefone BR). Conformidade com a LGPDM e Mitigação OWASP LLM06.
"""

import logging
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_br_analyzer() -> AnalyzerEngine:
    """
    Cria e faz o cache do analisador com reconhecedores brasileiros.
    O lru_cache garante que o motor do Presidio seja instanciado apenas uma vez.
    """
    analyzer = AnalyzerEngine()

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

    logger.info(
        "Presidio Analyzer com padrões Brasileiros (LGPD) carregado com sucesso."
    )
    return analyzer


class PIIDetector:
    """Classe responsável por escanear e anonimizar textos contra PII."""

    def __init__(self, language: str = "pt"):
        self.language = language

        try:
            self.analyzer = get_br_analyzer()
            self.anonymizer = AnonymizerEngine()
            self.is_active = True
        except Exception as e:
            logger.error(
                f"Erro ao inicializar Presidio. O mascaramento falhará aberto. Erro: {e}"
            )
            self.is_active = False

        self.entities_to_find = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "BR_CPF",
            "BR_PHONE",
            "CREDIT_CARD",
        ]

    def scan(self, text: str) -> list[dict]:
        """Escaneia o texto em busca de PII e retorna os detalhes dos achados para auditoria."""
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

        return anonymized_result.text

    def sanitize_text(self, text: str) -> str:
        """Alias para integração com o OutputGuardrail."""
        return self.anonymize(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = PIIDetector()
    texto_teste = "O cliente João Silva, portador do CPF 123.456.789-00, solicitou contato no email joao@banco.com."
    print(f"Original: {texto_teste}")
    print(f"Anonimizado: {detector.sanitize_text(texto_teste)}")
