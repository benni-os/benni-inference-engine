# benni-inference-engine — ROADMAP

## Phase 1 — Core Engine (BRAVO Formation)
> Target: 1 week | Priority: SHIP

- [ ] `engine.py` — llama-cpp-python CUDA load + forward
- [ ] `moe_scheduler.py` — auto VRAM/RAM detection + `-ngl` + `-ot` flag generation
- [ ] `openai_compat.py` — FastAPI server, streaming support
- [ ] `bie serve` CLI — working end-to-end with Qwen3-30B-A3B
- [ ] Hardware benchmark: RTX 4060 + Ryzen 5 5600X + 32GB RAM

## Phase 2 — Skill 81 + Hot Pin
> Target: 1 week after Phase 1

- [ ] `ceiling_extractor.py` — SA-81A/B/C full implementation
- [ ] `hallucination_guard.py` — SA-81D claim classifier
- [ ] `hot_pin.py` — expert frequency tracking + adaptive pinning
- [ ] `kv_manager.py` — sliding window compression
- [ ] SA-81E model profiles: Qwen3, Kimi K2, DeepSeek V4

## Phase 3 — JARVAS-2 Native + Open Source Launch
> Target: 2 weeks after Phase 2

- [ ] `mcp_bridge.py` — full JARVAS-2 integration
- [ ] Task routing policy (DELTA/CHARLIE/BRAVO → local; ALPHA/FOXTROT → cloud)
- [ ] PyPI publish: `pip install benni-inference-engine`
- [ ] Product Hunt launch alongside mcp-forge + operator-gateway
- [ ] Community benchmark submissions (open issue template)

## Phase 4 — Multi-Model + Advanced
> Roadmap

- [ ] DeepSeek V4 support
- [ ] Qwen3-235B-A22B full validation
- [ ] Speculative decoding (draft model)
- [ ] Web dashboard (inference stats, expert heat map)
- [ ] Windows / macOS support
