import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Importando o TruLens (Observabilidade Local)
from trulens_eval import Tru, TruChain

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Inicializa o TruLens e o banco de dados local (default.sqlite)
tru = Tru()


def inicializar_qwen_local():
    """Baixa e carrega o modelo Qwen open-source diretamente na memória."""
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    logger.info(f"Carregando o modelo {model_id} localmente...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    modelo = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Vai para a GPU se existir, senão CPU
        torch_dtype="auto",
    )

    # Criando o pipeline de geração de texto
    pipe = pipeline(
        "text-generation",
        model=modelo,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.1,  # Baixa temperatura para respostas mais factuais
        repetition_penalty=1.1,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("✅ Modelo Qwen carregado com sucesso!")
    return llm


def gerar_analise_com_telemetria():
    llm = inicializar_qwen_local()

    # --- 1. O Prompt ---
    template = """<|im_start|>system
Você é um analista financeiro sênior especializado em Petrobras (PETR4).
Responda de forma objetiva e baseada apenas no contexto fornecido.<|im_end|>
<|im_start|>user
Contexto Atual do Mercado: {contexto}

Pergunta: {pergunta}<|im_end|>
<|im_start|>assistant
"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    # --- 2. Envelopando com o TruLens ---
    # O TruChain atua como um wrapper que intercepta e grava as requisições
    tru_recorder = TruChain(chain, app_id="Agente_Investimentos_Qwen")

    logger.info("Enviando requisição ao LLM e gravando no TruLens...")

    # Usamos o 'with' para garantir que a gravação aconteça durante a execução
    with tru_recorder as recording:
        resposta = chain.invoke(
            {
                "contexto": "O preço do barril de petróleo Brent caiu 3% hoje, mas a Petrobras anunciou um aumento recorde na distribuição de dividendos.",
                "pergunta": "Qual o impacto esperado nas ações da PETR4 no curto prazo considerando este cenário misto?",
            }
        )

    print("\n" + "=" * 50)
    print("🤖 RESPOSTA DO QWEN LOCAL:")
    resposta_limpa = resposta.split("<|im_start|>assistant\n")[-1].strip()
    print(resposta_limpa)
    print("=" * 50 + "\n")

    logger.info(
        "Telemetria gravada localmente. Para ver o Dashboard, rode: trulens-eval"
    )


if __name__ == "__main__":
    gerar_analise_com_telemetria()
