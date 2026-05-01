import torch
import logging
import yaml
from typing import List, Dict, Any

# --- Imports para Roteamento Simples ---
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

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


def load_config():
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RouterAgent:
    """Versão enxuta para hardware limitado."""

    def __init__(self, tools: List[Tool]):
        self.cfg = load_config()
        self.ticker = self.cfg["data"]["ticker"]
        self.tools = {t.name: t for t in tools}
        self.llm = self._init_llm()

    def _init_llm(self):
        llm_cfg = self.cfg["llm"]
        temp = llm_cfg.get("temperature", 0.1)
        return HuggingFacePipeline.from_model_id(
            model_id=llm_cfg["model_name"],
            task="text-generation",
            device_map="auto",
            pipeline_kwargs={
                "max_new_tokens": 100,
                "temperature": temp if temp > 0 else None,
                "do_sample": True if temp > 0 else False,
            },
        )

    def run(self, input_text: str) -> Dict[str, Any]:
        # 1. Roteamento
        route_query = ROUTER_PROMPT.format(input=input_text)
        tool_name = self.llm.invoke(route_query).strip().lower()

        # Seleção segura
        selected = "consultar_base_conhecimento"
        for name in self.tools.keys():
            if name in tool_name:
                selected = name
                break

        # 2. Execução
        tool = self.tools[selected]
        obs = tool.run(
            self.ticker if selected != "consultar_base_conhecimento" else input_text
        )

        # 3. Resposta Final
        final_query = FINAL_PROMPT.format(input=input_text, observation=obs)
        answer = self.llm.invoke(final_query)

        return {
            "answer": answer,
            "intermediate_steps": [
                {"tool": selected, "input": input_text, "output": obs}
            ],
        }


# Mantendo compatibilidade com scripts existentes
def create_datathon_agent(tools: List[Tool]):
    return RouterAgent(tools)


def query_agent(agent: RouterAgent, question: str):
    res = agent.run(question)
    return {
        "answer": res["answer"],
        "intermediate_steps": [
            type("obj", (object,), {"tool": s["tool"], "tool_input": s["input"]})
            for s in res["intermediate_steps"]
        ],
    }
