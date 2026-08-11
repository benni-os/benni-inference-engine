#!/usr/bin/env python3
"""BIE VRAM Profiler — Mede uso de VRAM por camada, KV cache, throughput PCIe

Objetivo: Baseline para otimização de offload (32GB VRAM -> 8GB)
"""

import json
import subprocess
from pathlib import Path


def run_bie_benchmark(model: str = "Qwen3-8B-Q4_K_M", n_ctx: int = 8192):
    """Rodar benchmark do BIE e coletar métricas de VRAM."""
    # Usar CLI real do BIE (ajustar conforme implementação atual)
    cmd = ["python", "-m", "bie.serve", "bench", "--model", model, "--n-gpu-layers", "22", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", check=False)
    if result.returncode != 0:
        print(f"Erro no benchmark: {result.stderr}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"JSON inválido no benchmark: {result.stdout[:200]}")
        return None


def measure_vram_per_layer(model: str = "Qwen3-8B-Q4_K_M"):
    """Medir VRAM por camada usando llama.cpp + nvidia-smi."""
    layers = []
    model_path = f"bie/models/{model}.gguf"

    for layer_id in range(36):
        cmd = ["python", "-c", f"""
from llama_cpp import Llama
llm = Llama(model_path=r"{model_path}", n_gpu_layers={layer_id + 1}, n_ctx=8192)
llm.create_chat_completion(messages=[{{"role": "user", "content": "test"}}])
        """]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            vram = get_vram_usage()
            layers.append({"layer": layer_id, "vram_mb": vram})
            print(f"Layer {layer_id}: {vram} MB")

    return layers


def get_vram_usage():
    """Obter uso atual de VRAM via nvidia-smi."""
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else 0


def measure_pcie_throughput():
    """Medir throughput PCIe durante inferência."""
    cmd = ["nvidia-smi", "dmon", "-c", "10", "-s", "p"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout


def profile_kv_cache(model: str = "Qwen3-8B-Q4_K_M", contexts: list | None = None):
    """Medir KV cache por tamanho de contexto."""
    if contexts is None:
        contexts = [8192, 16384, 32768]

    results = []
    model_path = f"bie/models/{model}.gguf"

    for ctx in contexts:
        vram_before = get_vram_usage()

        cmd = ["python", "-c", f"""
from llama_cpp import Llama
llm = Llama(model_path=r"{model_path}", n_gpu_layers=22, n_ctx={ctx})
llm.create_chat_completion(messages=[{{"role": "user", "content": "test"}}])
        """]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            vram_after = get_vram_usage()
            results.append({"context": ctx, "kv_cache_mb": vram_after - vram_before})
            print(f"Context {ctx}: KV cache = {results[-1]['kv_cache_mb']} MB")

    return results


def main():
    print("=" * 60)
    print("BIE VRAM Profiler — Baseline para Otimização (32GB → 8GB)")
    print("=" * 60)

    print("\n[1/4] Benchmark baseline...")
    benchmark = run_bie_benchmark()
    if benchmark:
        print(f"Decode: {benchmark.get('decode_tok_s', 'N/A')} tok/s")
        print(f"VRAM pico: {benchmark.get('vram_peak_mib', 'N/A')} MiB")

    print("\n[2/4] VRAM por camada...")
    layers = measure_vram_per_layer()
    print(f"Camadas medidas: {len(layers)}")

    print("\n[3/4] KV cache por contexto...")
    kv_cache = profile_kv_cache()

    print("\n[4/4] Throughput PCIe...")
    pcie = measure_pcie_throughput()
    print(pcie[:500])

    output = {"benchmark": benchmark, "layers": layers, "kv_cache": kv_cache, "pcie_throughput": pcie[:1000]}
    output_path = Path("output/vram_profile_baseline.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Resultados salvos em {output_path}")


if __name__ == "__main__":
    main()