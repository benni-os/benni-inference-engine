#!/usr/bin/env python3
"""BIE VRAM Profiler — Mede VRAM por camada, KV cache, throughput PCIe.

Objetivo: Baseline para otimização de offload (32GB VRAM -> 8GB)
"""

import json
import subprocess
import sys
from pathlib import Path


def run_bie_benchmark(model: str = "Qwen3-8B-Q4_K_M", n_ctx: int = 8192):
    """Rodar benchmark do BIE e coletar métricas de VRAM."""
    cmd = [
        sys.executable, "-m", "bie.serve", "bench",
        "--model", model,
        "--model-path", f"bie/models/{model}.gguf",
        "--context", str(n_ctx),
        "--n-gpu-layers", "22",
        "--reps", "3",
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", check=False)
    if result.returncode != 0:
        print(f"Erro no benchmark: {result.stderr[:500]}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"JSON inválido no benchmark: {result.stdout[:200]}")
        return None


def _loaded_vram_mib(model_path: str, n_gpu_layers: int, n_ctx: int) -> int | None:
    """Carrega o modelo e reporta a VRAM usada enquanto o modelo está vivo."""
    code = f"""
import subprocess, time
from llama_cpp import Llama
llm = Llama(model_path=r"{model_path}", n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, verbose=False)
llm.create_chat_completion(messages=[{{"role": "user", "content": "test"}}], max_tokens=2)
time.sleep(0.3)
out = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    capture_output=True, text=True, check=False,
)
print(out.stdout.strip())
    """
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print(f"  erro load ngl={n_gpu_layers}: {result.stderr.strip()[:200]}")
        return None
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def measure_vram_per_layer(model: str = "Qwen3-8B-Q4_K_M"):
    """Medir VRAM usada por n_gpu_layers de 1..36 (load real do modelo)."""
    layers = []
    model_path = f"bie/models/{model}.gguf"
    previous = None
    for layer_id in range(36):
        used = _loaded_vram_mib(model_path, layer_id + 1, 8192)
        if used is None:
            continue
        delta = used - previous if previous is not None else used
        layers.append({"layer": layer_id, "ngl": layer_id + 1, "vram_used_mib": used, "vram_delta_mib": delta})
        print(f"Layer {layer_id} (ngl={layer_id + 1}): {used} MiB (delta +{delta})")
        previous = used
    return layers


def get_vram_usage():
    """Obter uso atual de VRAM via nvidia-smi."""
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else 0


def measure_pcie_throughput():
    """Medir throughput PCIe durante inferência (nvidia-smi dmon)."""
    cmd = ["nvidia-smi", "dmon", "-c", "10", "-s", "p"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout


def profile_kv_cache(model: str = "Qwen3-8B-Q4_K_M", contexts: list | None = None):
    """Medir VRAM por tamanho de contexto (KV cache), fixando ngl=22."""
    if contexts is None:
        contexts = [8192, 16384, 32768]
    results = []
    model_path = f"bie/models/{model}.gguf"
    reference = _loaded_vram_mib(model_path, 22, 2048)
    if reference is None:
        return results
    for ctx in contexts:
        used = _loaded_vram_mib(model_path, 22, ctx)
        if used is None:
            continue
        kv_mib = max(used - reference, 0)
        results.append({"context": ctx, "vram_used_mib": used, "kv_cache_mib": kv_mib})
        print(f"Context {ctx}: {used} MiB (KV vs ctx2048: +{kv_mib} MiB)")
    return results


def main():
    print("=" * 60)
    print("BIE VRAM Profiler — Baseline para Otimização (32GB -> 8GB)")
    print("=" * 60)

    print("\n[1/4] Benchmark baseline (bie.serve bench)...")
    benchmark = run_bie_benchmark()
    if benchmark:
        print(f"Decode: {benchmark.get('decode_tok_s', 'N/A')} tok/s")
        print(f"VRAM pico: {benchmark.get('vram_peak_mib', 'N/A')} MiB")
        print(f"VRAM delta: {benchmark.get('vram_delta_mib', 'N/A')} MiB")

    print("\n[2/4] VRAM por n_gpu_layers...")
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
    print(f"\nResultados salvos em {output_path}")


if __name__ == "__main__":
    main()