import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def executar_benchmark():
    set_seed(20260420)
    print("Iniciando Benchmark de Latência do LLM...\n")

    t = time.time()
    # Usando o modelo menor para o benchmark ser rápido
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)

    # Carregando na GPU se disponível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    print(f"Tempo de Load: {time.time()-t:.1f}s | Device: {device}\n")

    prompt = "Contexto: O preço do barril de petróleo Brent caiu 3%. \nResuma o impacto para a Petrobras: "
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    # Aquecimento (Warm-up) da GPU
    with torch.inference_mode():
        _ = model.generate(
            ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id
        )

    N = 10  # Número de chamadas por teste

    # As 3 configurações exigida
    configuracoes_tokens = [16, 64, 128]

    for max_tokens in configuracoes_tokens:
        print(f"--- Configuração: Gerando {max_tokens} tokens ---")
        t = time.time()

        with torch.inference_mode():
            for i in range(N):
                # Variando levemente o prompt para evitar cache
                ids_teste = tok(prompt + str(i), return_tensors="pt").input_ids.to(
                    device
                )
                _ = model.generate(
                    ids_teste,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )

        dt = time.time() - t
        print(f"Resultado: {N} chamadas em {dt:.2f}s => {dt/N*1000:.1f} ms/chamada\n")


if __name__ == "__main__":
    executar_benchmark()
