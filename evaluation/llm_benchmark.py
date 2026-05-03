import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_benchmark() -> None:
    """Roda um benchmark de velocidade de inferência (Tokens/Sec)."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # SLM de Referência do projeto

    logger.info(f"\n--- BENCHMARK DE INFERÊNCIA LOCAL ({model_name}) ---")

    device = (
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    t = time.time()
    logger.info("Carregando modelo e tokenizer...")

    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
    )

    logger.info(f"Tempo de Load: {time.time() - t:.1f}s | Device: {device}\n")

    prompt = (
        "Contexto: O preço do barril de petróleo Brent caiu 3%. \n"
        "Resuma o impacto para a Petrobras: "
    )
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    # Warm-up
    _ = mod.generate(ids, max_new_tokens=5)

    # Benchmark real
    t_start = time.time()
    out = mod.generate(
        ids,
        max_new_tokens=100,
        do_sample=False,
    )
    t_end = time.time()

    total_tokens = out.shape[1] - ids.shape[1]
    duration = t_end - t_start
    tps = total_tokens / duration

    logger.info(f"Resposta Gerada: {tok.decode(out[0], skip_special_tokens=True)[len(prompt) :]}")
    logger.info("-" * 50)
    logger.info(f"Tokens Gerados: {total_tokens}")
    logger.info(f"Duração: {duration:.2f}s")
    logger.info(f"Velocidade Média: {tps:.1f} tokens/seg")
    logger.info("-" * 50)


if __name__ == "__main__":
    run_benchmark()
