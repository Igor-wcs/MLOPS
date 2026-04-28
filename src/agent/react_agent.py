import logging
import yaml
from typing import List, Dict, Any

# --- Imports Corrigidos para LangChain v0.3+ ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool                       
from langchain_huggingface import HuggingFacePipeline 

logger = logging.getLogger(__name__)

# ==========================================
# PROMPT PADRÃO REACT (Com Compliance e Segurança)
# ==========================================
# Variáveis dinâmicas para não engessar o ativo
SYSTEM_PROMPT = """Você é um assistente financeiro especializado na análise do ativo {ticker}.
Você tem acesso a predições de um modelo LSTM, cotações de mercado e base de conhecimento corporativa.
Sua missão é responder essas perguntas sobre ações e modelos preditivos utilizando AS FERRAMENTAS DISPONÍVEIS.
NUNCA invente preços ou previsões. Se não souber, use uma ferramenta ou diga que não sabe.

REGRAS DE COMPLIANCE E SEGURANÇA (OBRIGATÓRIAS):
1. AVISO LEGAL: Sempre que fizer uma predição ou citar valores futuros, termine a resposta com: "Aviso: Esta análise não constitui recomendação de investimento."
2. NEUTRALIDADE: NUNCA sugira comprar, vender ou manter o ativo.
3. GATILHO DE FERRAMENTA: Se o usuário mencionar "dividendos", "história", "regras" ou "políticas", você DEVE usar a ferramenta 'consultar_base_conhecimento'.
4. IDIOMA: Responda estritamente em português do Brasil.

Ferramentas disponíveis:
{tools}

Use EXATAMENTE este formato de raciocínio:

Question: a pergunta inicial do usuário
Thought: pensar sobre o que fazer para responder a pergunta
Action: nome_da_ferramenta (deve ser uma destas: [{tool_names}])
Action Input: input para a ferramenta
Observation: resultado da ferramenta
... (repita Thought/Action/Observation quantas vezes for necessário)
Thought: Agora sei a resposta final
Final Answer: resposta completa e formatada para o usuário

Question: {input}
{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)

def load_config():
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# CONSTRUÇÃO DO AGENTE (Qwen INT4 Local)
# ==========================================

def create_datathon_agent(tools: List[Tool]) -> AgentExecutor:
    """Cria o agente ReAct integrado ao LLM quantizado local."""
    cfg = load_config()
    llm_cfg = cfg["llm"]
    ticker = cfg["data"]["ticker"]
    
    if len(tools) < 3:
        logger.warning(f"O Datathon exige >= 3 tools. Fornecidas: {len(tools)}")

    logger.info(f"Inicializando pipeline {llm_cfg['model_name']} ({llm_cfg['quantization']})...")
    
    # Inicialização da IA Generativa Local (Garante a ADR-002)
    try:
        llm = HuggingFacePipeline.from_model_id(
            model_id=llm_cfg["model_name"],
            task="text-generation",
            device_map=llm_cfg["device_map"],
            pipeline_kwargs={
                "max_new_tokens": llm_cfg["max_new_tokens"],
                "temperature": llm_cfg["temperature"],
                "do_sample": True if llm_cfg["temperature"] > 0 else False
            },
            model_kwargs={
                "load_in_4bit": True if llm_cfg["quantization"] == "int4" else False,
            }
        )
    except Exception as e:
        logger.error(f"Erro ao alocar o modelo na GPU/CPU: {e}. Usando Fallback.")
        # <-- Atualizado: Importando o FakeListLLM do core ao invés do community
        from langchain_core.language_models.fake import FakeListLLM
        llm = FakeListLLM(responses=["Final Answer: Serviço de LLM temporariamente indisponível."])

    # O partial garante que o prompt já venha com o Ticker configurado no YAML
    prompt_with_ticker = REACT_PROMPT.partial(ticker=ticker)
    
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_with_ticker)
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=7,              # Limite de segurança contra loops infinitos
        handle_parsing_errors=True,
        return_intermediate_steps=True # Vital para a telemetria da Fase 3
    )
    
    return executor

# ==========================================
#    INTERFACE DE TELEMETRIA E EXECUÇÃO
# ==========================================

def query_agent(agent: AgentExecutor, question: str) -> Dict[str, Any]:
    """Executa a query e empacota os passos intermediários para telemetria/API."""
    try:
        result = agent.invoke({"input": question})
        
        # Extração limpa do raciocínio da IA para o Langfuse/Frontend
        intermediate_steps = [
            {
                "tool_used": step[0].tool,
                "tool_input": step[0].tool_input,
                "tool_output": str(step[1])[:500]
            }
            for step in result.get("intermediate_steps", [])
        ]
        
        return {
            "success": True,
            "answer": result.get("output", "Não foi possível formular uma resposta."),
            "intermediate_steps": intermediate_steps
        }
        
    except Exception as e:
        logger.error(f"Erro na orquestração do agente: {e}")
        return {
            "success": False,
            "answer": f"Desculpe, ocorreu uma falha técnica ao processar a requisição.",
            "intermediate_steps": []
        }