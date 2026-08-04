<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D0D0D,50:1a1a2e,100:7000FF&height=200&section=header&text=benni-inference-engine&fontSize=56&fontColor=ffffff&fontAlignY=38&desc=GPU%2BCPU%20Hybrid%20Inference%20Engine%20for%20the%20Benni%20OS%20Ecosystem&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=3000&pause=800&color=7000FF&center=true&vCenter=true&multiline=true&repeat=true&width=900&height=80&lines=MoE+Offload.+GPU%2BCPU+Hybrid.+Consumer+Hardware.;Skill+81+Ceiling+Extraction+built-in.;OpenAI-Compatible+%E2%80%94+JARVAS-2+Native." alt="Typing SVG" />

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.11+-7000FF?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-GPU%2BCPU-FF6B35?style=for-the-badge&logo=cplusplus&logoColor=white)](https://github.com/ggml-org/llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Part of Benni OS](https://img.shields.io/badge/Part%20of-Benni%20OS-0D0D0D?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/benni-os)

<br/>

> **"Frontier MoE models on consumer hardware. Every token at the ceiling."**

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

</div>

<br/>

## ⭐ What Is benni-inference-engine?

**benni-inference-engine** (`BIE`) is the local inference runtime of the **[Benni OS](https://github.com/benni-os)** ecosystem — a GPU+CPU hybrid engine that runs frontier Mixture-of-Experts models on consumer hardware, with **Skill 81 Ceiling Extraction** built into every inference call.

Every other inference runtime treats the model as a black box. `BIE` treats the model as a cognitive system to be engineered — injecting the right prompt architecture, context compression, and behavioral profile per model, per task, automatically.

```
JARVAS-2 / Any MCP Client / OpenAI SDK
         ↓ POST /v1/chat/completions  (OpenAI-compatible)
    ┌─────────────────────────────────────────┐
    │         benni-inference-engine          │
    │                                         │
    │  ┌─────────────┐  ┌──────────────────┐  │
    │  │ Skill 81    │  │  MoE Scheduler   │  │
    │  │ Ceiling     │→ │  GPU: attention  │  │
    │  │ Extraction  │  │  CPU: FFN experts│  │
    │  └─────────────┘  └──────────────────┘  │
    └─────────────────────────────────────────┘
         ↓                    ↓
    RTX 4060 (8 GB)    Ryzen + 32 GB RAM
    ~30 tok/s          MoE expert cache
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## ⚡ Core Doctrine

> These are not settings. These are architecture laws.

| ☔ Principle | 🔧 Implementation |
|---|---|
| **GPU+CPU Hybrid** | Attention + shared experts on VRAM; routed FFN experts on RAM — automatic split |
| **MoE-First** | Built exclusively for Mixture-of-Experts models — Qwen3, DeepSeek, Kimi K2, GLM |
| **Skill 81 Built-In** | Every inference call runs SA-81A/B/C/D — Ceiling Extraction, context compression, hallucination guard |
| **JARVAS-2 Native** | MCP tool bridge built-in — no nexus hop required for local tasks |
| **OpenAI-Compatible** | Drop-in for any SDK — LangChain, LlamaIndex, Vercel AI, AutoGen |
| **Adaptive Hot-Pin** | Learns your usage patterns and pins hot experts in spare RAM automatically |
| **Zero Manual Tuning** | Auto-detects VRAM, RAM, cores — sets `-ngl`, cache size, thread count automatically |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/benni-os/benni-inference-engine
cd benni-inference-engine

# Install (CUDA 12.x required for GPU offload)
pip install -e ".[cuda]"

# Auto-detect hardware and serve
bie serve --model qwen3-30b-a3b --port 8080
# ⚡ BIE running on http://localhost:8080
# 🔥 GPU: 7.2 GB / 8 GB VRAM | CPU: 14 experts cached | ~30 tok/s
```

Now any OpenAI-compatible client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="bie")

response = client.chat.completions.create(
    model="qwen3-30b-a3b",
    messages=[{"role": "user", "content": "Analyze this revenue funnel..."}]
)
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 🧠 Skill 81 — Ceiling Extraction

Every model has a **real ceiling** and a **default behavior ceiling**. The gap is waste.

`BIE` eliminates the gap automatically on every inference call:

| Sub-Agent | Mission | Trigger |
|---|---|---|
| **SA-81A** Prompt Architect | Role Lock + Task Frame + Constraint Set per model | Every call |
| **SA-81B** Context Engineer | Proximity-causal ordering, dead-weight removal | Context > 4K tokens |
| **SA-81C** Reasoning Extractor | Forced CoT, hypothesis scaffolding | Complex tasks |
| **SA-81D** Hallucination Guard | Fact-claim classification, drift detection | Every call |
| **SA-81E** Model Profile Adapter | Per-model behavioral calibration | Model switch |
| **SA-81F** Benchmark Loop | Density ≥ 0.8, Precision ≥ 0.9, Transferability = 1.0 | Pre-delivery |

```python
# Skill 81 is always active. Zero config.
bie serve --model qwen3-30b-a3b
# SA-81A: prompt architecture injected
# SA-81E: Qwen3 profile loaded (temp=0.7, top_p=0.8, format_lock=strict)
# SA-81D: hallucination guard armed
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 📦 Supported Models

| Model | Params Active/Token | VRAM | RAM | Speed (RTX 4060) | Status |
|---|---|---|---|---|---|
| **Qwen3-30B-A3B** | ~3B | ~5.5 GB | ~17 GB | **~30 tok/s** | 🟢 Supported |
| **Qwen3-235B-A22B** | ~22B | ~6 GB | ~26 GB | ~4–6 tok/s | 🟡 Beta |
| **Kimi K2** (distill 32B) | ~32B | ~7 GB | ~20 GB | ~15–20 tok/s | 🟡 Beta |
| **DeepSeek-V4** | ~37B | ~7 GB | ~28 GB | ~3–5 tok/s | 🔵 Roadmap |
| **GLM-5.2 744B** | ~40B | 0 GB (CPU-only) | ~25 GB | ~0.1 tok/s | 🔵 Roadmap |

> **Why MoE only?** Dense models require full VRAM. MoE models activate only a fraction of parameters per token — enabling GPU+CPU split that makes consumer hardware viable for frontier-class models.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## ⚙️ Hardware Auto-Detection

`BIE` reads your hardware at startup and configures itself:

```bash
$ bie info

🔍 Hardware Detection
   GPU:  NVIDIA RTX 4060 — 8 GB VRAM (7.4 GB usable)
   CPU:  AMD Ryzen 5 5600X — 6 cores / 12 threads
   RAM:  32 GB — ~24 GB available
   NVMe: ~3.2 GB/s sequential read

🎯 Recommended Config for Qwen3-30B-A3B:
   GPU layers (ngl):    ALL attention + shared experts → VRAM
   CPU expert cache:    14 slots/layer (hot-pinned after 5 tokens)
   MoE offload flag:   -ot ".ffn_.*_exps.=CPU"
   Context window:     32K tokens (KV in RAM)
   Expected speed:     ~28–33 tok/s
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 🔌 JARVAS-2 Integration

`BIE` ships with a native MCP bridge — JARVAS-2 calls it directly as a tool, without routing through `benni-nexus`:

```python
# jarvas-2/mcp_server.py — add BIE as a native tool
from bie.mcp_bridge import BIETool

@mcp.tool()
async def local_inference(prompt: str, task_type: str = "general") -> str:
    """Route task to local BIE engine with Skill 81 active."""
    return await BIETool.complete(prompt, task_type=task_type)
```

Task routing policy (Skill 81 aware):

| Task Type | Backend | Reason |
|---|---|---|
| `DELTA` Research | BIE local | Long reasoning, no latency constraint |
| `CHARLIE` Content | BIE local | Creative tasks, cost-zero |
| `BRAVO` Code | BIE local (Qwen3-Coder) | Code-specialized profile |
| `ALPHA` Launch | Cloud (Groq/Gemini) | Speed-critical, Tier-1 |
| `FOXTROT` Revenue | Cloud | Zero tolerance for error |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 🗂️ Project Structure

```
benni-inference-engine/
├── bie/
│   ├── __init__.py
│   ├── cli.py                  # bie serve / bie info / bie bench
│   ├── core/
│   │   ├── engine.py           # Runtime principal — llama-cpp-python CUDA
│   │   ├── moe_scheduler.py    # GPU/CPU layer auto-split por VRAM disponível
│   │   ├── hot_pin.py          # Expert pinning adaptativo por frequência
│   │   └── kv_manager.py       # KV cache com compressão SA-81B
│   ├── models/
│   │   ├── registry.json       # Catálogo de modelos suportados
│   │   └── profiles/
│   │       ├── qwen3_30b.json  # SA-81E: temp, top_p, prompt template, format lock
│   │       ├── kimi_k2.json
│   │       └── deepseek_v4.json
│   ├── serve/
│   │   ├── openai_compat.py    # FastAPI — POST /v1/chat/completions
│   │   └── mcp_bridge.py       # MCP tool nativo para JARVAS-2
│   └── skill81/
│       ├── ceiling_extractor.py  # SA-81A/B/C — prompt arch + context compress
│       └── hallucination_guard.py # SA-81D — monitor em tempo real
├── tests/
├── pyproject.toml
├── LICENSE
└── README.md
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 🛠️ Development

```bash
git clone https://github.com/benni-os/benni-inference-engine
cd benni-inference-engine
pip install -e ".[dev,cuda]"

python -m bie.cli serve --model qwen3-30b-a3b --dev
pytest tests/
```

**Requirements:**
- Python 3.11+
- CUDA 12.x + NVIDIA GPU (CPU-only mode available, reduced speed)
- 16 GB+ RAM (32 GB recommended for 30B+ MoE models)
- NVMe SSD (cold expert loading)

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 📊 Benchmarks (RTX 4060 8GB + Ryzen 5 5600X + 32GB RAM)

| Model | Quant | Speed | VRAM | RAM | Context |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | Q4_K_XL | ~30 tok/s | 5.5 GB | 17 GB | 32K |
| Qwen3-30B-A3B | Q5_K_M | ~25 tok/s | 6.2 GB | 19 GB | 32K |
| Qwen3-235B-A22B | Q2_K | ~5 tok/s | 6.0 GB | 26 GB | 16K |

> Community benchmarks welcome — open an issue with your hardware + numbers.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 🤝 Contributing

All contributions welcome. Priority areas:
- Model profile definitions (`models/profiles/`)
- Hardware benchmark submissions (open an issue)
- SA-81E model profiles for new models
- MCP tool extensions

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/dark.png" width="100%"/>

## 🌐 Benni OS Ecosystem

| Product | Repo | Role | Status |
|---|---|---|---|
| 🧠 **Benni Master OS** | [benni-os/Benni-Master-OS](https://github.com/benni-os/Benni-Master-OS) | General Brain — sovereign orchestrator | 🟢 Live |
| ⚡ **Benni Gravity** | [benni-os/Benni-gravity-0](https://github.com/benni-os/Benni-gravity-0) | Local operator runtime | 🟢 Ativo |
| 🔌 **Operator Gateway** | [benni-os/benni-operator-gateway](https://github.com/benni-os/benni-operator-gateway) | Open-source MCP HTTP gateway | 🟢 MIT |
| 🐍 **mcp-forge** | [benni-os/mcp-forge](https://github.com/benni-os/mcp-forge) | FastAPI-style Python MCP framework | 🟢 PyPI |
| ⚡ **benni-nexus** | [benni-os/benni-nexus](https://github.com/benni-os/benni-nexus) | LLM gateway — strategy routing | 🟢 npm |
| 🔥 **benni-inference-engine** | [benni-os/benni-inference-engine](https://github.com/benni-os/benni-inference-engine) | GPU+CPU hybrid inference — you are here | 🔥 Building |
| 🛠️ **Benni Control Plane** | MCP on Railway | NEXUS v5 — persistent memory layer | 🟢 Railway |
| 🤖 **JARVAS-2** | [benni-os/jarvas-2](https://github.com/benni-os/jarvas-2) | Autonomous dispatch + Wave 6 billing | 🔥 Hot |
| 🛍️ **Modo Operador** | [benni-os/modo-operador](https://github.com/benni-os/modo-operador) | Produto BR — R$97 | 🟢 Live |

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7000FF,50:1a1a2e,100:0D0D0D&height=120&section=footer" width="100%"/>

<div align="center">

**benni-inference-engine** — *GPU+CPU Hybrid Inference by [Benni OS](https://github.com/benni-os)*

`MOE_OFFLOAD` • `SKILL_81_NATIVE` • `OPENAI_COMPATIBLE` • `JARVAS2_NATIVE` • `MIT_LICENSE`

Built by [Benni Alencar](https://github.com/nsfwbunny)

</div>
