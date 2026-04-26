"""llm_serving.py

Serving do LLM Qwen 2.5-0.5B-Instruct com quantização INT4 (bitsandbytes).
Segue o padrão de quantização do repositório de referência para o Datathon.

A quantização NF4 reduz o consumo de VRAM de ~1GB (FP16) para ~300MB,
permitindo co-localizar o modelo LSTM e o LLM numa única GPU.
"""

import logging
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_quantized_model(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quantize: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Carrega o modelo Qwen com quantização INT4 via bitsandbytes.

    Args:
        model_name: Nome do modelo no HuggingFace Hub.
        quantize: Se True, aplica a configuração de quantização NF4.

    Returns:
        Tupla contendo o modelo e o tokenizer carregados.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if quantize:
        # Configuração NF4 para máxima eficiência de memória (Etapa 2 do Datathon)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Modelo %s carregado com quantização INT4 (NF4)", model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Modelo %s carregado em FP16", model_name)

    model.eval()
    return model, tokenizer


def generate_response(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    quantize: bool = True,
) -> str:
    """Gera uma resposta do LLM para um dado prompt.

    Args:
        prompt: Texto de entrada para o modelo.
        model_name: Identificador do modelo no HuggingFace.
        max_new_tokens: Limite de tokens na geração.
        temperature: Controla a aleatoriedade (0.0 para respostas determinísticas).
        quantize: Define se utiliza a versão comprimida do modelo.

    Returns:
        Texto gerado pelo modelo, limpo de tokens especiais.
    """
    model, tokenizer = load_quantized_model(model_name, quantize)

    messages = [
        {"role": "system", "content": "Você é um analista financeiro especializado."},
        {"role": "user", "content": prompt},
    ]

    # Aplicação do template oficial do Qwen para garantir o formato correto das mensagens
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Configuração dinâmica para evitar avisos e erros de amostragem (sampling)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        # Quando temperature=0.0 (ex: no Juiz), desativamos o sampling para maior precisão
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    # Decodificação ignorando os tokens de input para retornar apenas a resposta
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)
