"""OpenAI-compatible serve layer — POST /v1/chat/completions."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="benni-inference-engine", version="0.1.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-30b-a3b"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint.
    Skill 81 Ceiling Extraction is applied automatically before forwarding to engine.
    TODO: Phase 1 — wire BIEEngine + ceiling_extractor
    """
    return {
        "id": "bie-placeholder",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "message": {"role": "assistant", "content": "Phase 1 in progress."},
            "finish_reason": "stop",
            "index": 0
        }]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "benni-inference-engine", "version": "0.1.0"}
