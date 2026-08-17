"""Unit tests for the experimental llama-server HTTP adapter.

All tests use httpx.MockTransport, so no server, GPU or model is required
and no network calls are made.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from bie.serve.llama_server_adapter import (
    LlamaServerAdapter,
    LlamaServerConnectionError,
    LlamaServerHTTPError,
    LlamaServerMalformedResponse,
    LlamaServerTimeout,
)

BASE_URL = "http://127.0.0.1:8080"

COMPLETION_BODY = {
    "id": "cmpl-1",
    "object": "text_completion",
    "choices": [{"text": "Hello BIE", "index": 0, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    "timings": {"predicted_per_second": 3.1},
}

SSE_BODY = (
    'data: {"id":"cmpl-1","object":"text_completion",'
    '"choices":[{"text":"Hello ","index":0,"finish_reason":null}]}\n\n'
    'data: {"id":"cmpl-1","object":"text_completion",'
    '"choices":[{"text":"BIE","index":0,"finish_reason":null}]}\n\n'
    "data: [DONE]\n\n"
)


def _handler_for(
    request: httpx.Request,
    *,
    completion: dict | None = None,
    sse: str | None = None,
    status: int = 200,
    body: bytes | str = b"",
    raise_exc: Exception | None = None,
    fail_path: str | None = "/v1/completions",
    health_status: int = 200,
    record: list[httpx.Request] | None = None,
) -> httpx.Response:
    if record is not None:
        record.append(request)
    if raise_exc is not None and request.url.path == fail_path:
        raise raise_exc
    if request.url.path == "/health":
        if health_status != 200:
            return httpx.Response(health_status, content=body)
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/completions":
        if request.content:
            payload = json.loads(request.content.decode("utf-8"))
            if payload.get("stream") and sse is not None:
                return httpx.Response(status, text=sse, headers={"content-type": "text/event-stream"})
        if status != 200:
            return httpx.Response(status, content=body)
        if body:
            return httpx.Response(status, content=body)
        return httpx.Response(status, json=completion or COMPLETION_BODY)
    return httpx.Response(404)


def _adapter(handler, **kwargs) -> LlamaServerAdapter:
    transport = httpx.MockTransport(handler)
    adapter = LlamaServerAdapter("qwen3-27b", BASE_URL, transport=transport, **kwargs)
    return adapter


@pytest.fixture()
def adapter():
    return _adapter(lambda request: _handler_for(request, sse=SSE_BODY))


def test_health_ok(adapter):
    assert adapter.health() == {"status": "ok"}


def test_load_reports_runtime_config(adapter):
    config = adapter.load()
    assert config["backend"] == "llama-server"
    assert config["base_url"] == BASE_URL
    assert adapter.model is not None


def test_load_http_error():
    adapter = _adapter(
        lambda request: _handler_for(request, health_status=503, body=b"down")
    )
    with pytest.raises(LlamaServerHTTPError) as exc:
        adapter.load()
    assert exc.value.status_code == 503


def test_load_connection_error():
    adapter = _adapter(
        lambda request: _handler_for(request, raise_exc=httpx.ConnectError("refused"), fail_path="/health")
    )
    with pytest.raises(LlamaServerConnectionError):
        adapter.load()


def test_load_health_timeout():
    adapter = _adapter(
        lambda request: _handler_for(request, raise_exc=httpx.ReadTimeout("slow"), fail_path="/health"),
        timeout=0.001,
    )
    with pytest.raises(LlamaServerTimeout):
        adapter.load()


def test_unload_closes_client_and_resets_state():
    adapter = _adapter(lambda request: _handler_for(request))
    adapter.load()
    client = adapter._client
    adapter.unload()
    assert client is not None and client.is_closed
    assert adapter.model is None
    assert adapter.runtime_config is None


def test_complete_requires_load():
    adapter = _adapter(lambda request: _handler_for(request))
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.complete("hi")


def test_complete_rejects_invalid_prompt(adapter):
    adapter.model = object()
    with pytest.raises(ValueError):
        adapter.complete("   ")
    with pytest.raises(ValueError):
        adapter.complete(123)


def test_complete_normalizes_text_and_usage(adapter):
    adapter.load()
    result = adapter.complete("hi", temperature=0.0, max_tokens=8)
    assert result["choices"][0]["text"] == "Hello BIE"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 7
    assert result["timings"]["predicted_per_second"] == 3.1


def test_complete_timeout():
    adapter = _adapter(
        lambda request: _handler_for(request, raise_exc=httpx.ReadTimeout("slow")), timeout=0.001
    )
    adapter.load()
    with pytest.raises(LlamaServerTimeout):
        adapter.complete("hi")


def test_complete_connection_error():
    adapter = _adapter(
        lambda request: _handler_for(request, raise_exc=httpx.ConnectError("reset"))
    )
    adapter.load()
    with pytest.raises(LlamaServerConnectionError):
        adapter.complete("hi")


def test_complete_http_error():
    adapter = _adapter(lambda request: _handler_for(request, status=500, body=b"boom"))
    adapter.load()
    with pytest.raises(LlamaServerHTTPError) as exc:
        adapter.complete("hi")
    assert exc.value.status_code == 500


def test_complete_malformed_json():
    adapter = _adapter(lambda request: _handler_for(request, body=b"not-json"))
    adapter.load()
    with pytest.raises(LlamaServerMalformedResponse):
        adapter.complete("hi")


def test_complete_malformed_shape():
    adapter = _adapter(lambda request: _handler_for(request, body=b'{"nope": 1}'))
    adapter.load()
    with pytest.raises(LlamaServerMalformedResponse):
        adapter.complete("hi")


def test_complete_non_object_payload():
    adapter = _adapter(lambda request: _handler_for(request, body=b"[1, 2, 3]"))
    adapter.load()
    with pytest.raises(LlamaServerMalformedResponse, match="non-object"):
        adapter.complete("hi")


def test_stream_yields_text_chunks(adapter):
    adapter.load()
    assert list(adapter.stream("hi", temperature=0.0)) == ["Hello ", "BIE"]


def test_stream_rejects_invalid_prompt(adapter):
    adapter.load()
    with pytest.raises(ValueError):
        list(adapter.stream("   "))


def test_stream_http_error():
    adapter = _adapter(lambda request: _handler_for(request, sse=SSE_BODY, status=500, body=b"boom"))
    adapter.load()
    with pytest.raises(LlamaServerHTTPError) as exc:
        list(adapter.stream("hi"))
    assert exc.value.status_code == 500


def test_stream_connection_error():
    adapter = _adapter(
        lambda request: _handler_for(request, sse=SSE_BODY, raise_exc=httpx.ConnectError("reset"))
    )
    adapter.load()
    with pytest.raises(LlamaServerConnectionError):
        list(adapter.stream("hi"))


def test_stream_sse_non_object_event():
    adapter = _adapter(
        lambda request: _handler_for(request, sse='data: [1, 2]\n\ndata: [DONE]\n\n')
    )
    adapter.load()
    assert list(adapter.stream("hi")) == []


def test_stream_sse_missing_choices():
    adapter = _adapter(
        lambda request: _handler_for(request, sse='data: {"id": "x"}\n\ndata: [DONE]\n\n')
    )
    adapter.load()
    assert list(adapter.stream("hi")) == []


def test_stream_malformed_sse():
    adapter = _adapter(lambda request: _handler_for(request, sse="data: not-json\n\ndata: [DONE]\n\n"))
    adapter.load()
    with pytest.raises(LlamaServerMalformedResponse):
        list(adapter.stream("hi"))


def test_stream_cancel_stops_iteration():
    adapter = _adapter(lambda request: _handler_for(request, sse=SSE_BODY))
    adapter.load()
    gen = adapter.stream("hi")
    assert next(gen) == "Hello "
    adapter.cancel()
    with pytest.raises(StopIteration):
        next(gen)


def test_stream_sends_stream_true():
    requests: list[httpx.Request] = []

    def handler(request):
        return _handler_for(request, sse=SSE_BODY, record=requests)

    adapter = _adapter(handler)
    adapter.load()
    list(adapter.stream("hi"))
    payload = json.loads(requests[1].content.decode("utf-8"))
    assert payload["stream"] is True
    assert payload["prompt"] == "hi"


def test_stream_timeout():
    adapter = _adapter(
        lambda request: _handler_for(request, sse=SSE_BODY, raise_exc=httpx.ReadTimeout("slow")),
        timeout=0.001,
    )
    adapter.load()
    with pytest.raises(LlamaServerTimeout):
        list(adapter.stream("hi"))


def test_base_url_and_headers():
    requests: list[httpx.Request] = []

    def handler(request):
        return _handler_for(request, sse=SSE_BODY, record=requests)

    adapter = _adapter(handler, api_key="sk-secret-123")
    adapter.load()
    adapter.complete("hi")
    adapter.stream("hi")
    for request in requests:
        assert str(request.url).startswith(BASE_URL)
    completions = [r for r in requests if r.url.path == "/v1/completions"]
    assert completions
    for request in completions:
        assert request.headers["Authorization"] == "Bearer sk-secret-123"


def test_no_secret_leak_in_repr():
    adapter = _adapter(lambda request: _handler_for(request), api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(adapter)
    assert "sk-super-secret" not in str(adapter)
    assert "api_key=set" in repr(adapter)


def test_no_secret_leak_in_error_path(caplog):
    with caplog.at_level(logging.DEBUG):
        adapter = _adapter(
            lambda request: _handler_for(request, raise_exc=httpx.ConnectError("nope"), fail_path="/health"),
            api_key="sk-leak-check",
        )
        with pytest.raises(LlamaServerConnectionError):
            adapter.load()
    assert "sk-leak-check" not in caplog.text


def test_cancel_is_opt_in_no_fallback():
    adapter = _adapter(lambda request: _handler_for(request))
    assert adapter.cancel is not None
    adapter.cancel()  # no-op before stream; must not raise