import logging
from typing import Any

import yaml
from langchain_core.tools import Tool

# --- Imports para Roteamento Simples ---
from langchain_huggingface import HuggingFacePipeline

logger = logging.getLogger(__name__)

# ==========================================
# PROMPTS ENXUTOS (Slim Prompts)
# ==========================================

# Prompt focado apenas em CLASSIFICAR a intenção
ROUTER_PROMPT = """Classifique a pergunta abaixo em uma das ferramentas:
- obter_previsao_lstm: para previsões futuras.
- obter_cotacao_atual: para preço atual ou hoje.
- consultar_base_conhecimento: para regras, dividendos ou história.

Pergunta: {input}
Ferramenta:"""

# Prompt focado apenas em FORMATAR a resposta final
FINAL_PROMPT = """Você é um assistente financeiro.
Use a Informação abaixo para responder à Pergunta de forma curta.
AVISO: Esta análise não constitui recomendação de investimento.

Pergunta: {input}
Informação: {observation}
Resposta Final:"""


def load_config() -> dict[str, Any]:
    """Carrega as configurações do sistema a partir do arquivo YAML."""
    with open("configs/model_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RouterAgent:
    """Versão enxuta para hardware limitado."""

    def __init__(self, tools: list[Tool]) -> None:
        self.cfg = load_config()
        self.ticker = self.cfg["data"]["ticker"]
        self.tools = {t.name: t for t in tools}
        self.llm = self._init_llm()

    def _init_llm(self) -> HuggingFacePipeline:
        """Inicializa o modelo de linguagem local."""
        llm_cfg = self.cfg["llm"]
        temp = llm_cfg.get("temperature", 0.1)

        # Correção da lógica de amostragem:
        # se temp for 0, desativa do_sample para ser determinístico
        do_sample = True if temp > 0 else False

        return HuggingFacePipeline.from_model_id(
            model_id=llm_cfg["model_name"],
            task="text-generation",
            device_map=llm_cfg.get("device_map", "auto"),
            pipeline_kwargs={
                "max_new_tokens": llm_cfg.get("max_new_tokens", 100),
                "temperature": temp if do_sample else None,
                "do_sample": do_sample,
            },
        )

    def run(self, input_text: str) -> dict[str, Any]:
        """Executa a lógica de roteamento e resposta do agente."""
        # 1. Roteamento (Lógica Híbrida: LLM + Keywords para robustez em SLM)
        query_lower = input_text.lower()

        if any(w in query_lower for w in ["prev", "futuro", "amanhã", "modelo", "ia", "lstm"]):
            selected = "obter_previsao_lstm"
        elif any(w in query_lower for w in ["preço", "cotação", "valor", "hoje", "agora", "atual"]):
            selected = "obter_cotacao_atual"
        elif any(
            w in query_lower
            for w in ["política", "regra", "dividendos", "história", "sobre", "quem"]
        ):
            selected = "consultar_base_conhecimento"
        else:
            # Fallback para o LLM classificar
            route_query = ROUTER_PROMPT.format(input=input_text)
            tool_name = self.llm.invoke(route_query).strip().lower()
            selected = "consultar_base_conhecimento"
            for name in self.tools.keys():
                if name in tool_name:
                    selected = name
                    break

        # 2. Execução
        tool = self.tools[selected]
        # Se for RAG, passa a pergunta toda. Se for Ticker, usa o padrão.
        arg = input_text if selected == "consultar_base_conhecimento" else self.ticker
        obs = tool.run(arg)

        # 3. Resposta Final
        final_query = FINAL_PROMPT.format(input=input_text, observation=obs)
        answer = self.llm.invoke(final_query).strip()

        # Limpeza de rastro de prompt (comum em modelos pequenos)
        if "Resposta Final:" in answer:
            answer = answer.split("Resposta Final:")[-1].strip()

        return {
            "answer": answer,
            "intermediate_steps": [{"tool": selected, "input": arg, "output": obs}],
        }


# Mantendo compatibilidade com scripts existentes
def create_datathon_agent(tools: list[Tool]) -> RouterAgent:
    """Cria e retorna uma instância do RouterAgent."""
    return RouterAgent(tools)


def query_agent(agent: RouterAgent, question: str) -> dict[str, Any]:
    """Consulta o agente e formata a resposta para compatibilidade."""
    res = agent.run(question)
    return {
        "answer": res["answer"],
        "intermediate_steps": [
            type("obj", (object,), {"tool": s["tool"], "tool_input": s["input"]})
            for s in res["intermediate_steps"]
        ],
    }
