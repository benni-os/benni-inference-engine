# Gate Qwen3.8-27B — Veredito Operacional (dense, multimodal)

**Data:** 2026-08-15 (runtime embutido) · revisão 2026-08-26 (runtime nativo CUDA + adapter HTTP)
**Hardware:** RTX 4060 8GB
**Modelo:** `Qwen3.8-27B-Q4_K_M.gguf` (15,93 GB, arquitetura `qwen35`, denso 27B)
**Runtimes avaliados:** llama-cpp-python **0.3.34** (embarcado) · llama.cpp nativo CUDA (commit `ece963f`) + adapter HTTP `bie/serve/llama_server_adapter.py`

## Veredito final: EXPERIMENTAL

O modelo **passa** os critérios de decode, streaming e estabilidade no runtime nativo com offload GPU parcial, mas **excede o alvo de VRAM** (5.700–6.400 MiB) em todas as configurações viáveis. Não é candidato a produção; permanece disponível apenas via `--engine llama-server` (opt-in).

## 1. Resultados do runtime nativo (llama-server CUDA)

Offload GPU REAL confirmado: baseline de VRAM 1.018 MiB → ~7.47x MiB durante load; log `offloaded N/66 layers to GPU` (o runtime 0.3.34 não fazia offload nenhum — ver histórico abaixo).

| ngl | Decode | VRAM | Observação |
| :-- | :-- | :-- | :-- |
| 0 | 2,2 tok/s | 1.745 MiB | CPU-only |
| 20 | **3,1 tok/s** | 6.610 MiB | via adapter BIE (bench 3 reps) |
| 22 | **3,1 tok/s** | 6.732 MiB | ponto de referência recomendado |
| 24 | 3,2 tok/s | 7.468 MiB | acima do teto operacional (7.168) |
| 28 | 3,4 tok/s | 7.884 MiB | acima do teto |

### Bench detalhado ngl=20 (adapter + servidor real, porta 8090)

| Métrica | Valor |
| :-- | :-- |
| Decode | 3,1 tok/s (3 reps: 3,1 / 3,1 / 3,1 — estável) |
| TTFT | ~0,95 s |
| p50 / p95 ms/token | 320,9 / 332,0 |
| Prefill estimado | ~470 tok/s |
| VRAM pico | 6.610 MiB (baseline cliente 535 MiB) |
| Streaming SSE | ✅ funcional (chunks recebidos em tempo real) |
| Usage/timings do server | ✅ capturados pelo adapter |
| Tool calling | ⏳ pendente (requer `/v1/chat/completions`; adapter atual expõe só `/v1/completions`) |

## 2. Leitura dos critérios de gate

| Critério | Limite | Medido | Status |
| :-- | :-- | :-- | :-- |
| Decode | ≥ 3 tok/s | 3,1–3,4 tok/s (ngl ≥ 20) | ✅ |
| VRAM ideal | 5.700–6.400 MiB | 6.610–6.732 MiB nas configs viáveis | ❌ acima do alvo |
| VRAM teto absoluto | ≤ 7.168 MiB | 6.732 MiB (ngl=22) | ✅ (ngl ≥ 24 excede) |
| Streaming | funcional | ✅ SSE via adapter | ✅ |
| Repetibilidade | sem degradação | 3 reps idênticas | ✅ |
| Tool calling | funcional | ⏳ não testado | ⏳ |

Como o alvo de VRAM é excedido mesmo na config mais leve viável (ngl=20), o teste de refinamento ngl=21 **não altera o veredito** — ficou registrado como opcional para quando a máquina estiver ociosa.

## 3. Decisão

- **Qwen3.8-27B = EXPERIMENTAL**: utilizável via `python -m bie.serve --engine llama-server` com `llama-server -ngl 22 -c 4096`, mas fora do perfil de produção por VRAM.
- **Qwen3.5-35B-A3B (MoE) permanece PRIMARY** — decode ~8,5–9,6 tok/s, RAM 14,2 GB, VRAM ~2 GB, backend padrão inalterado.
- O adapter HTTP é opt-in explícito; o default de produção (`bie`) não mudou.
- Pendências opcionais: teste ngl=21; validação de tool calling via `/v1/chat/completions`.

## 4. Histórico: runtime embutido (llama-cpp-python 0.3.34) — FAIL arquivado

Sweep original (`output/q38_ngl_*.json`, ngl 0–48): decode 2,0–2,1 tok/s, RAM 16,1–16,3 GB, VRAM 2,65 GB.

Causa raiz: o GGUF usa arquitetura `qwen35` (Gated DeltaNet / Gated Attention / SSM conv + state, 66 layers). No 0.3.34, **todas as camadas ficavam "assigned to device CPU"** mesmo com `n_gpu_layers=48` — o runtime não possui kernels GPU para os blocos SSM/DeltaNet e ignora o parâmetro para este arquivo. Execução 100% CPU → p50 472 ms/tok.

Resolução parcial: build nativo do llama.cpp com CUDA resolveu o offload (seção 1), mas o custo de VRAM mantém o modelo fora do perfil de produção.

## 5. Artefatos

- Benchmarks antigos: `output/q38_ngl_*.json` (runtime 0.3.34)
- Bench adapter: executado 2026-08-26 contra llama-server nativo (`-ngl 20 -c 4096`, porta 8090)
- GGUF: `bie/models/Qwen3.8-27B-Q4_K_M.gguf` (15,93 GB) — **mantido**
- Runtime nativo: build local do llama.cpp (commit `ece963f`), binários em `%TEMP%\opencode\llama.cpp\build\bin\`
