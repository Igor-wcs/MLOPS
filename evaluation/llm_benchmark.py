import logging
import time

import mlflow
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("configs/model_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def executar_benchmark():
    set_seed(20260420)
    cfg = load_config()

    logger.info("Iniciando Benchmark de Latência do LLM...\n")

    t = time.time()
    # Usando o modelo configurado ou o default Qwen
    model_id = cfg.get("llm", {}).get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id)

    # Detecção de Hardware Consistente com o resto do projeto
    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if str(device) != "cpu" else torch.float32
    ).to(device)
    model.eval()

    logger.info(f"Tempo de Load: {time.time() - t:.1f}s | Device: {device}\n")

    prompt = "Contexto: O preço do barril de petróleo Brent caiu 3%. \nResuma o impacto para a Petrobras: "
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    # Aquecimento (Warm-up)
    with torch.inference_mode():
        _ = model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)

    N = 10  # Número de chamadas por teste
    configuracoes_tokens = [16, 64, 128]

    # --- MLFLOW TRACKING ---
    mlflow.set_experiment(cfg["paths"]["experiment_name"])

    with mlflow.start_run(run_name="Benchmark_LLM_Latency"):
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.set_tag("device", str(device))
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("iterations_per_config", N)

        for max_tokens in configuracoes_tokens:
            logger.info(f"--- Configuração: Gerando {max_tokens} tokens ---")
            t_inicio = time.time()

            with torch.inference_mode():
                for i in range(N):
                    ids_teste = tok(prompt + str(i), return_tensors="pt").input_ids.to(device)
                    _ = model.generate(
                        ids_teste,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )

            dt = time.time() - t_inicio
            avg_ms = (dt / N) * 1000

            logger.info(f"Resultado: {N} chamadas em {dt:.2f}s => {avg_ms:.1f} ms/chamada\n")

            # Log de métricas por configuração
            mlflow.log_metric(f"latency_ms_avg_{max_tokens}tokens", avg_ms)
            mlflow.log_metric(f"total_time_s_{max_tokens}tokens", dt)


if __name__ == "__main__":
    executar_benchmark()
