"""Experimental HTTP adapter for a native llama-server (llama.cpp).

This is an opt-in backend: it connects BIE to a ``llama-server`` process
exposing the OpenAI-compatible HTTP API. It exists mainly to validate
models that the bundled llama-cpp-python cannot offload to the GPU
(e.g. the ``qwen35`` / Gated DeltaNet architecture of Qwen3.8-27B).

It is **experimental**: it is selected only through explicit configuration
(``--engine llama-server`` on the CLI) and never falls back automatically.
The production backend (llama-cpp-python) is unchanged.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from typing import Any

import httpx

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_TEMPERATURE = 0.7

# Keys forwarded to llama-server /v1/completions. Unknown options are dropped
# so a newer server cannot reject the request for unexpected fields.
_SUPPORTED_OPTIONS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "stop",
    "seed",
    "n",
)


class LlamaServerError(RuntimeError):
    """Base error for llama-server adapter failures."""


class LlamaServerConnectionError(LlamaServerError):
    """Could not reach the llama-server process."""


class LlamaServerTimeout(LlamaServerError):
    """Request timed out while talking to llama-server."""


class LlamaServerHTTPError(LlamaServerError):
    """llama-server returned a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"llama-server HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class LlamaServerMalformedResponse(LlamaServerError):
    """Response body could not be parsed or lacks required fields."""


class LlamaServerAdapter:
    """llama-server backend normalized to the BIE engine contract.

    ``load()`` performs a health check only; the model weights are loaded
    inside the llama-server process. No model is downloaded or loaded here.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT_S,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = (
            base_url or os.environ.get("BIE_LLAMA_SERVER_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("BIE_LLAMA_SERVER_API_KEY")
        self._transport = transport
        self.model: Any | None = None
        self.runtime_config: dict[str, Any] | None = None
        self._client: httpx.Client | None = None
        self._cancel = threading.Event()

    def __repr__(self) -> str:
        # Never leak the API key in repr/str output.
        key = "set" if self._api_key else "unset"
        return (
            f"LlamaServerAdapter(model={self.model_name!r}, "
            f"base_url={self.base_url!r}, api_key={key}, timeout_s={self.timeout})"
        )

    def _client_for(self) -> httpx.Client:
        if self._client is None:
            headers: dict[str, str] = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                transport=self._transport,
            )
        return self._client

    def _require_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("LlamaServerAdapter is not loaded; call load() first.")

    def health(self) -> dict[str, Any]:
        """Check that llama-server is reachable and healthy."""
        try:
            response = self._client_for().get("/health")
        except httpx.TimeoutException as exc:
            raise LlamaServerTimeout(
                f"llama-server health check timed out after {self.timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LlamaServerConnectionError(f"llama-server unreachable: {exc}") from exc
        if response.status_code != 200:
            raise LlamaServerHTTPError(response.status_code, response.text[:200])
        return response.json()

    def load(self) -> dict[str, Any]:
        """Health-check llama-server and report runtime config."""
        self.health()
        self.model = object()
        self.runtime_config = {
            "backend": "llama-server",
            "base_url": self.base_url,
            "model": self.model_name,
            "timeout_s": self.timeout,
        }
        return self.runtime_config

    def unload(self) -> None:
        """Close the HTTP client and reset state."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self.model = None
        self.runtime_config = None

    def cancel(self) -> None:
        """Signal an in-flight streaming request to stop."""
        self._cancel.set()

    def _completion_payload(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model_name, "prompt": prompt}
        for name in _SUPPORTED_OPTIONS:
            if name in options:
                payload[name] = options[name]
        payload.setdefault("max_tokens", self.max_tokens)
        payload.setdefault("temperature", self.temperature)
        return payload

    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a non-streaming completion, normalized to BIE contract."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self._require_loaded()
        payload = self._completion_payload(prompt, kwargs)
        try:
            response = self._client_for().post("/v1/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LlamaServerTimeout(
                f"llama-server completion timed out after {self.timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LlamaServerConnectionError(f"llama-server request failed: {exc}") from exc
        if response.status_code != 200:
            raise LlamaServerHTTPError(response.status_code, response.text[:200])
        return self._normalize_completion(response)

    @staticmethod
    def _normalize_completion(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LlamaServerMalformedResponse("llama-server returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise LlamaServerMalformedResponse("llama-server returned a non-object payload")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LlamaServerMalformedResponse("llama-server response missing choices")
        text = choices[0].get("text", "")
        return {
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": data.get("usage"),
            "timings": data.get("timings"),
        }

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield text chunks from a streaming completion (SSE)."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self._require_loaded()
        self._cancel.clear()
        payload = self._completion_payload(prompt, kwargs)
        payload["stream"] = True
        try:
            with self._client_for().stream("POST", "/v1/completions", json=payload) as response:
                if response.status_code != 200:
                    body = response.read().decode("utf-8", "replace")
                    raise LlamaServerHTTPError(response.status_code, body[:200])
                for line in response.iter_lines():
                    if self._cancel.is_set():
                        break
                    text = self._parse_sse_text(line)
                    if text:
                        yield text
        except httpx.TimeoutException as exc:
            raise LlamaServerTimeout(
                f"llama-server stream timed out after {self.timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LlamaServerConnectionError(f"llama-server stream failed: {exc}") from exc

    @staticmethod
    def _parse_sse_text(line: str) -> str | None:
        if not line.startswith("data:"):
            return None
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return None
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise LlamaServerMalformedResponse("llama-server sent malformed SSE payload") from exc
        if not isinstance(event, dict):
            return None
        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            text = choices[0].get("text", "")
            return text or None
        return None