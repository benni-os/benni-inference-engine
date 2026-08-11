# Gate 30B em 8 GB VRAM — Plano (baseline medido em hardware)

**Baseline do profiler:** 2026-08-11, RTX 4060 8GB, CUDA 13.3, llama-cpp-python 0.3.34
**Fonte:** `bie/tools/vram_profiler.py` → `output/vram_profile_baseline.json`
**Perfil medido:** `bie/models/profiles/Qwen3-8B-Q4_K_M-offload-measured.json`

## 1. Dados reais medidos (Qwen3-8B-Q4_K_M, 5,03 GB / 36 layers)

| Métrica | Valor |
| :-- | :-- |
| VRAM por camada (curva 1..36) | **+130–160 MiB/layer** → `dense_layer_gb = 0,14` |
| ngl=6 (load) | 3.227 MiB |
| ngl=22 (load) | 5.326 MiB |
| ngl=36 (load, offload total) | 7.151 MiB |
| Benchmark ngl=22 ctx 8K | 15,1 tok/s, pico 5.519 MiB, p50 65,5 / p95 69,7 ms/tok |
| KV cache (ngl=22) | +776 MiB @8K · +1.998 MiB @16K · **32K → OOM** |

**Conclusões do baseline:**
- `dense_layer_gb = 0,14` validado por medição (o scheduler usava 0,14 e a curva confirma).
- **Offload total (ngl=36) cabe** no load (7.151 MiB), mas a geração em 8K fica perto do limite → `ngl=22` continua sendo o ponto equilibrado para o 8B.
- KV 32K não cabe com offload parcial nesta GPU.

## 2. Viabilidade Qwen3-30B-A3B-Q3_K_M (14,71 GB, 48 layers, MoE)

### Decisão revisada após medição
O **NO-GO anterior** assumia o offload padrão do scheduler (ngl≈20 → 28 layers em CPU, RAM estourada). O **offload agressivo (ngl=6)** muda a conta:

| Fator | Limite | Offload agressivo (ngl=6) | Status |
| :-- | :-- | :-- | :-- |
| VRAM | 8 GB | 42 layers em CPU → ~6 layers GPU ≈ **~2 GB** + KV | ✅ folga grande |
| RAM | ~19 GB livre | 42/48 × 14,71 = **12,9 GB** (page cache) + KV 8K (~1,5 GB) + overhead ≈ **15–16 GB** | ✅ sem swap (apertado) |
| Decode | — | CPU-bound (3B ativos em 12 threads): **estimativa 2–6 tok/s** | ⚠️ lento, medir no gate |
| Download | — | 14,71 GB (unsloth) | ⚠️ ~10 min @100 Mbps |

**Veredito: RAM-feasível com offload agressivo; qualidade de execução incerta (decode).**
Decisão do gate: **prosseguir somente se o operador aceitar decode 2–6 tok/s**; caso contrário, manter 8B como produção.

## 3. Plano de execução

1. **Download** `Qwen3-30B-A3B-Q3_K_M.gguf` (unsloth) → `bie/models/`
2. **SHA256 + metadados** (arquivo já é verificado depois):
   ```powershell
   Get-FileHash bie\models\Qwen3-30B-A3B-Q3_K_M.gguf -Algorithm SHA256
   python -m bie.core.scheduler_quant_aware   # via derive_from_gguf
   ```
3. **Benchmark com offload agressivo** (CLI real validado):
   ```powershell
   python -m bie.serve bench --model Qwen3-30B-A3B --model-path bie\models\Qwen3-30B-A3B-Q3_K_M.gguf --context 2048 --n-gpu-layers 6 --reps 3 --json
   ```
4. **Serving + endpoints**: `serve` → `/health`, `/ready`, chat, completion, SSE
5. **Soak** 6×64 tok com monitoramento de VRAM/RAM/swap
6. **Comparação** Q3_K_M (Q3_K_S se a RAM apertar) vs status quo 8B

## 4. Critérios de sucesso (Policy Gate)

- [ ] VRAM ≤ 8 GB (monitorar pico em tempo real)
- [ ] RAM ≤ 19 GB livre (sem swap; conferir `performance counter` de swap/commit)
- [ ] Load sem erro
- [ ] `/health` 200, `/ready` 200, chat/completion/SSE reais
- [ ] Soak 6×64 sem degradação progressiva
- [ ] Decode ≥ 3 tok/s (CPU-bound aceitável)

## 5. Riscos e mitigação

| Risco | Mitigação |
| :-- | :-- |
| Swap (RAM > 19 GB) | monitorar commit charge; se swap, abortar imediatamente (gate de parada) |
| Decode < 3 tok/s | aceitar só se operador validar; senão, manter 8B |
| OOM GPU | `ngl` reduzido (6) deixa >6 GB de folga; KV 8K em vez de 32K |
| Metadados errados | `derive_from_gguf` lê do arquivo real (arch/block_count) |

## 6. Histórico de decisão

- 2026-08-11: Gate 30B avaliado **antes do download** com tamanhos reais da HF API.
- Primeira análise (config default ngl≈20): **NO-GO** (RAM estourada, swap garantido).
- Após profiler medido: offload agressivo (ngl=6) torna o cenário **RAM-feasível**; decisão final depende do operador aceitar decode lento.
- O ModelProfile do 30B deve usar `dense_layer_gb` **quant-aware** (0,306/layer @Q3_K_M) via `scheduler_quant_aware.py`, não o valor de 0,16 do perfil original.