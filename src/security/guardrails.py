"""src/security/guardrails.py

Guardrails de segurança para input e output do agente.
Implementa proteções contra o OWASP Top 10 for LLM Applications e conformidade LGPD.
"""

import logging
import re

# Importa o detector otimizado e focado no Brasil que criamos em pii_detection.py
from src.security.pii_detection import PiiDetector

logger = logging.getLogger(__name__)


class InputGuardrail:
    """Valida e sanitiza input do usuário antes de enviar ao LLM."""

    # Padrões expandidos para mitigar LLM01: Prompt Injection e Jailbreak
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+a",
        r"system:\s*",
        r"<\|im_start\|>",
        r"\[INST\]",
        r"forget\s+(everything|all|your\s+instructions)",
        r"act\s+as\s+(a\s+)?",
        r"pretend\s+you\s+are",
    ]

    def __init__(self, max_length: int = 4096):
        self.max_length = max_length
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida input do usuário contra injeções e context stuffing.

        Args:
            user_input: Texto do usuário.

        Returns:
            Tupla (is_valid, reason).
        """
        # Check 1: Prompt injection detection
        for pattern in self._compiled_patterns:
            if pattern.search(user_input):
                logger.warning("Prompt injection detectado: %s", user_input[:100])
                return False, "Input bloqueado: padrão de instrução suspeito detectado."

        # Check 2: Tamanho máximo (evitar context stuffing/DoS - OWASP LLM04)
        if len(user_input) > self.max_length:
            return (
                False,
                f"Input bloqueado: excede tamanho máximo permitido ({self.max_length} chars).",
            )

        return True, "OK"


class OutputGuardrail:
    """Valida e sanitiza output do LLM antes de retornar ao usuário (LGPD)."""

    def __init__(self, language: str = "pt"):
        # Instancia o detector que já contém o Lazy Loading e as regras do Brasil
        self.detector = PiiDetector(language=language)

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.

        Args:
            llm_output: Texto gerado pelo LLM.

        Returns:
            Texto sanitizado com placeholders (ex: <BR_CPF>, <PERSON>).
        """
        # A nossa classe PiiDetector já escaneia e anonimiza com segurança
        anonymized_text = self.detector.anonymize(llm_output)

        if anonymized_text != llm_output:
            logger.info(
                "OutputGuardrail: Dados sensíveis foram mascarados antes da resposta."
            )

        return anonymized_text
