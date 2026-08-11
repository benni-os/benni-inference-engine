"""BIE Core Engine — llama.cpp backend with explicit lifecycle and streaming."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .moe_scheduler import MoEScheduler


class BIEEngine:
    """Small, testable inference facade for the local BIE runtime."""

    def __init__(
        self,
        model_name: str,
        port: int = 8080,
        *,
        model_path: str | Path | None = None,
        n_ctx: int = 32768,
        n_threads: int | None = None,
        n_batch: int = 512,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.port = port
        self.model_path = Path(model_path) if model_path else None
        self.n_ctx = n_ctx
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self.n_batch = n_batch
        self.verbose = verbose
        self.model: Any | None = None
        self.runtime_config: dict[str, Any] | None = None

    def resolve_model_path(self) -> Path:
        """Resolve a GGUF path from an explicit path or BIE_MODEL_DIR."""
        candidate = self.model_path or Path(self.model_name)
        if candidate.is_file():
            return candidate
        model_dir = Path(os.getenv("BIE_MODEL_DIR", "models"))
        candidates = [model_dir / self.model_name]
        if not self.model_name.endswith(".gguf"):
            candidates.append(model_dir / f"{self.model_name}.gguf")
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Model not found: {self.model_name}. Set model_path or BIE_MODEL_DIR."
        )

    def load(self) -> dict[str, Any]:
        """Load the model using the hardware recommendation from MoEScheduler."""
        model_path = self.resolve_model_path()
        config = MoEScheduler(self.model_name).compute()
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required; install the CUDA build for GPU inference."
            ) from exc
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            n_gpu_layers=int(config["ngl"]),
            verbose=self.verbose,
        )
        self.runtime_config = {**config, "model_path": str(model_path)}
        return self.runtime_config

    def _require_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("BIEEngine is not loaded. Call load() first.")
        return self.model

    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a non-streaming completion."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        model = self._require_model()
        return model(prompt, **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield generated text chunks from llama.cpp."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        model = self._require_model()
        options = {**kwargs, "stream": True}
        for chunk in model(prompt, **options):
            choices = chunk.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
                if text:
                    yield text

    def unload(self) -> None:
        """Release the model reference and runtime configuration."""
        self.model = None
        self.runtime_config = None
