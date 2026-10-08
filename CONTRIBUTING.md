# Contributing to benni-inference-engine

Thank you for contributing to **benni-inference-engine** (`BIE`). We welcome contributions from developers, researchers, and AI engineers.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/benni-os/benni-inference-engine.git
   cd benni-inference-engine
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies in editable mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify the test suite:**
   ```bash
   pytest -v
   ```

---

## 🎯 High-Value Contribution Areas

We actively seek contributions in the following domains:

1. **Model Profiles (`bie/models/profiles/`):**
   - Adding profiles for new frontier MoE architectures (DeepSeek V3/V4, Kimi K2, GLM, MiniMax).
   - Documenting layer counts, active experts, total experts, attention dimensions, and GGUF quantizations.

2. **Hardware Benchmarks:**
   - Submit reproducible benchmarks using `bie bench` across different GPUs (RTX 3060, 4060, 4070, 4080, 4090, Apple Silicon).
   - Report token generation throughput (`tok/s`), VRAM allocated, and system RAM occupied.

3. **Inference Adapters:**
   - Enhancing the `llama-server` HTTP adapter or FastMCP endpoints for native tool use.

---

## 📜 Pull Request Process

1. Create a feature branch: `git checkout -b feat/your-feature-name`.
2. Format code and run linters: `ruff check .`.
3. Ensure **100% of tests pass**: `pytest -v`.
4. Write atomic, conventional commit messages: `feat:`, `fix:`, `docs:`, `perf:`, `test:`.
5. Open a Pull Request against `main`. All CI checks must pass before merging.
