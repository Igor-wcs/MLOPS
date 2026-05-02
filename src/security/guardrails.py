"""
Guardrails de segurança para input e output do agente.
Implementa proteções contra o OWASP Top 10 for LLM Applications e conformidade LGPD.
"""

import logging
import re
import yaml

# Importa o detector otimizado e focado no Brasil que criamos em pii_detection.py
from src.security.pii_detection import PIIDetector

logger = logging.getLogger(__name__)


def load_security_config() -> dict:
    """Carrega as regras de segurança do YAML para evitar hardcode (Governança)."""
    try:
        with open("configs/monitoring_config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("security_guardrails", {})
    except Exception as e:
        logger.warning(f"Aviso: Usando defaults de segurança pois o YAML falhou: {e}")
        return {
            "max_input_tokens": 4096,
            "pii_detection_enabled": True,
            "block_prompt_injection": True,
        }


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

    def __init__(self):
        self.config = load_security_config()
        self.max_length = self.config.get("max_input_tokens", 4096)
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
        # Check 1: Prompt injection detection (OWASP LLM01)
        if self.config.get("block_prompt_injection", True):
            for pattern in self._compiled_patterns:
                if pattern.search(user_input):
                    logger.warning("Prompt injection detectado: %s", user_input[:100])
                    return (
                        False,
                        "Input bloqueado: padrão de instrução suspeito detectado.",
                    )

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
        self.config = load_security_config()
        self.is_enabled = self.config.get("pii_detection_enabled", True)

        # Instancia o detector que já contém o Lazy Loading e as regras do Brasil
        if self.is_enabled:
            self.detector = PIIDetector(language=language)

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.

        Args:
            llm_output: Texto gerado pelo LLM.

        Returns:
            Texto sanitizado com placeholders (ex: <BR_CPF>, <PERSON>).
        """
        if not self.is_enabled:
            return llm_output

        # A nossa classe PIIDetector já escaneia e anonimiza com segurança
        anonymized_text = self.detector.sanitize_text(llm_output)

        if anonymized_text != llm_output:
            logger.info(
                "OutputGuardrail: Dados sensíveis (PII) foram mascarados antes da resposta."
            )

        return anonymized_text


# Instâncias prontas para o app.py
input_guard = InputGuardrail()
output_guard = OutputGuardrail()
