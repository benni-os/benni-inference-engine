"""MCP Bridge — JARVAS-2 native tool integration. No nexus hop required."""

class BIETool:
    """
    Native MCP tool for JARVAS-2.
    Usage in jarvas-2/mcp_server.py:

        from bie.mcp_bridge import BIETool

        @mcp.tool()
        async def local_inference(prompt: str, task_type: str = "general") -> str:
            return await BIETool.complete(prompt, task_type=task_type)
    """

    BIE_BASE_URL = "http://localhost:8080"

    @classmethod
    async def complete(cls, prompt: str, task_type: str = "general", model: str = "qwen3-30b-a3b") -> str:
        """
        Route to local BIE engine with Skill 81 active.
        task_type maps to JARVAS-2 Formation types: DELTA, CHARLIE, BRAVO, etc.
        TODO: Phase 1 — implement HTTP call to openai_compat.py
        """
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{cls.BIE_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
                timeout=120.0
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
