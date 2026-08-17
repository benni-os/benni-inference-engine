# Gate Qwen3.8-27B — Veredito Operacional (dense, multimodal)

**Data:** 2026-08-15 · RTX 4060 8GB · llama-cpp-python **0.3.34** (embarcado)
**Modelo testado:** `unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-Q4_K_M.gguf` (15,93 GB, arquitetura `qwen35`)
**Fonte:** `output/q38_ngl_{0,2,4,6,8,16,24,32,40,48}.json` + inspeção de load_tensors

## 1. Resultado: FAIL — decode abaixo do mínimo operacional

| Critério | Limite | Medido (ngl 0–48) | Status |
| :-- | :-- | :-- | :-- |
| Compatibilidade GGUF/runtime | carregar | ✅ carrega (39–115 s) | ✅ |
| RAM | ≤ 19.456 MiB | 16.171–16.350 MiB | ✅ |
| VRAM | ≤ 8.192 MiB | 2.651–2.797 MiB | ✅ |
| **Decode** | **≥ 3 tok/s** | **2,0–2,1 tok/s** | ❌ **FAIL** |
| OOM | ausente | não ocorreu | ✅ |

## 2. Causa raiz: offload GPU NÃO ocorre nesta arquitetura

- O GGUF usa arquitetura `qwen35` (Gated DeltaNet / Gated Attention / SSM conv + state, 65 layers, `nextn_predict_layers=1` MTP).
- Log de load com `n_gpu_layers=48`: **todas as camadas "assigned to device CPU"**, 0 na GPU.
- VRAM real medida via `nvidia-smi`: baseline 929 MiB → 2.483 MiB durante load de ngl=48 (+~1,5 GB ≈ overhead de buffers, não layers).
- RAM praticamente idêntica em todos os ngl (16,1–16,3 GB) — o modelo inteiro permanece residente na CPU.
- Conclusão: o runtime 0.3.34 **não possui kernels GPU para os blocks SSM/DeltaNet** do Qwen3.8; o parâmetro `n-gpu-layers` é ignorado para este arquivo.

## 3. Por que o decode trava em ~2,1 tok/s

- Modelo **denso**: ativa todos os 27B params a cada token (vs MoE com 3B ativos).
- Sem offload → execução 100% CPU + Gated DeltaNet sem aceleração GPU.
- p50 472 ms/tok, p95 494 ms/tok em todos os ngl.

## 4. Decisão

- **Qwen3.8-27B NÃO passa o gate** no hardware atual (decode 2,1 tok/s < 3,0) e **não é viável** enquanto o llama-cpp-python 0.3.34 não suportar offload da arquitetura `qwen35`.
- Caminhos possíveis (não executados): atualizar llama-cpp-python para versão com kernels GPU do Qwen3.8; ou aceitar execução CPU-only (decode ~2 tok/s) se um caso de uso tolerar.
- **Qwen3.5-35B-A3B (MoE) permanece o alvo principal** — decode medido ~8,5–9,6 tok/s, RAM 14,2 GB, VRAM ~2 GB. GGUF e runtime compatíveis.
- Qwen3-30B-A3B (Q3_K_M, legado) e Qwen3-8B-Q4_K_M permanecem como baseline.

## 5. Artefatos

- Benchmarks: `output/q38_ngl_*.json` (0, 2, 4, 6, 8, 16, 24, 32, 40, 48)
- GGUF: `bie/models/Qwen3.8-27B-Q4_K_M.gguf` (15,93 GB) — **mantido**, não é corrupção (carrega e gera)
