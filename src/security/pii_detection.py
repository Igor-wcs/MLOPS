"""src/security/pii_detection.py

Detecção e Anonimização de PII (Personally Identifiable Information) com Presidio.
Especializado para dados brasileiros (CPF, telefone BR).
"""

import logging
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_br_analyzer() -> AnalyzerEngine:
    """Cria e faz o cache do analisador com reconhecedores brasileiros.
    O lru_cache garante que o motor pesado do Presidio seja instanciado apenas uma vez.
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

    logger.info("Presidio Analyzer com padrões Brasileiros carregado com sucesso.")
    return analyzer


class PiiDetector:
    """Classe responsável por escanear e anonimizar textos contra PII."""

    def __init__(self, language: str = "pt"):
        self.language = language
        self.analyzer = get_br_analyzer()
        self.anonymizer = AnonymizerEngine()

        self.entities_to_find = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "BR_CPF",
            "BR_PHONE",
            "CREDIT_CARD",
        ]

    def scan(self, text: str) -> list[dict]:
        """Escaneia o texto em busca de PII e retorna os detalhes dos achados."""
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
                "Alerta LGPD: Detetadas %d ocorrências de PII.", len(findings)
            )

        return findings

    def anonymize(self, text: str) -> str:
        """Escaneia o texto e substitui automaticamente o PII por máscaras."""
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities_to_find,
        )

        if not results:
            return text

        # O Presidio Anonymizer substitui a entidade pelo seu nome (ex: "João" -> "<PERSON>")
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )

        return anonymized_result.text


# Teste local rápido (executa apenas se rodar este ficheiro diretamente)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = PiiDetector()

    texto_teste = "O cliente João Silva, portador do CPF 123.456.789-00, solicitou contato no número (11) 98765-4321."

    print("Texto Original:", texto_teste)
    print("\nAchados:", detector.scan(texto_teste))
    print("\nTexto Anonimizado:", detector.anonymize(texto_teste))
