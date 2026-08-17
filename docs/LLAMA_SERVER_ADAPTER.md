# LLama Server Adapter — Experimental

Adapter experimental para conectar o BIE a um **`llama-server` nativo**
(llama.cpp compilado com CUDA) via HTTP. Serve principalmente para validar
modelos que o `llama-cpp-python` embarcado não consegue offloadar na GPU —
ex.: o Qwen3.8-27B (arquitetura `qwen35` / Gated DeltaNet).

> ⚠️ **EXPERIMENTAL.** É opt-in e **nunca** faz fallback automático. O backend
> de produção (llama-cpp-python, `Qwen3.5-35B-A3B`) permanece inalterado.

---

## Instalação

Nenhuma dependência nova é necessária — o adapter usa `httpx`, já declarado
no `pyproject.toml`.

```bash
pip install -e ".[dev]"
```

## Inicialização do llama-server

Compile o llama.cpp com CUDA (uma vez):

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp\build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp\build --config Release -j --target llama-cli llama-server
```

Depois suba o servidor com o modelo alvo:

```bash
llama.cpp\build\bin\Release\llama-server.exe ^
  -m "bie\models\Qwen3.8-27B-Q4_K_M.gguf" ^
  -ngl 22 -c 4096 --host 127.0.0.1 --port 8080
```

Confirme com `curl http://127.0.0.1:8080/health` → `{"status":"ok"}`.

## Variáveis de ambiente

| Variável | Default | Descrição |
| :-- | :-- | :-- |
| `BIE_LLAMA_SERVER_URL` | `http://127.0.0.1:8080` | Base URL do llama-server |
| `BIE_LLAMA_SERVER_API_KEY` | (vazio) | Chave opcional enviada como `Authorization: Bearer`; **nunca** logada |

## Uso

Selecione o engine explicitamente (default continua `bie`):

```bash
# Health check
python -m bie.serve health --engine llama-server

# Serve via o mesmo contrato OpenAI-compat do BIE
python -m bie.serve serve --engine llama-server --model qwen3-8b --port 8081

# Bench (usa o fluxo existente do cmd_bench)
python -m bie.serve bench --engine llama-server --model qwen3-8b --reps 3
```

O adapter expõe o mesmo contrato interno do engine (`load`, `complete`,
`stream`, `unload`, `health`, `cancel`), então `openai_compat.py` e o `bench`
funcionam sem alteração de produção.

```python
from bie.serve.llama_server_adapter import LlamaServerAdapter

adapter = LlamaServerAdapter("qwen3-8b")
adapter.load()
print(adapter.complete("oi", temperature=0.0))
for chunk in adapter.stream("oi", max_tokens=32):
    print(chunk)
adapter.unload()
```

## Limitações

- **Experimental**: apenas para validação de modelos (ex.: Qwen3.8-27B);
  não é o backend de produção.
- `load()` faz apenas health check — as configurações de VRAM/RAM/decode do
  llama-server (flags `-ngl`, `-c`, etc.) são responsabilidade do processo
  externo e **não** são detectadas pelo BIE.
- Campos de geração são passados como whitelist (`temperature`, `max_tokens`,
  `top_p`, `top_k`, `min_p`, `repeat_penalty`, `stop`, `seed`, `n`); opções
  desconhecidas são ignoradas.
- Cancelamento (`cancel()`) aborta o streaming em andamento; não há suporte
  a cancelamento de geração síncrona no protocolo.
- `usage`/`timings` são repassados quando o llama-server os incluir no
  payload; não são calculados pelo adapter.

## Procedimento de benchmark

Padrão já existente (`cmd_bench`), apenas trocando o engine:

```bash
# baseline de VRAM antes de subir o servidor
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# suba o llama-server com o ngl desejado (ver seção acima)

python -m bie.serve bench --engine llama-server --model qwen3-8b ^
  --context 4096 --reps 3 --prompt-tokens 512 --max-tokens 128 --json
```

Compare `vram_peak_mib`/`vram_delta_mib` e `decode_tok_s` com o limite do
gate (VRAM ≤ 7.168 MiB, decode ≥ 3 tok/s). Para o Qwen3.8-27B, o ponto
referência atual é `-ngl 22` (~6,7 GB VRAM, ~3,1 tok/s).