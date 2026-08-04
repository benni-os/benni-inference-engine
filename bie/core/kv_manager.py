"""KV Cache Manager — context compression via SA-81B (Benni OS Skill 81)."""

class KVManager:
    """
    Manages KV cache lifecycle:
    - Compresses context when approaching window limit (SA-81B)
    - Removes dead-weight tokens (no causal influence on current output)
    - Tracks proximity-causal ordering for optimal attention
    """

    def __init__(self, ctx_size: int = 32768):
        self.ctx_size = ctx_size
        self.current_len = 0

    def should_compress(self, threshold: float = 0.8) -> bool:
        """Returns True when context exceeds threshold of window."""
        return self.current_len > self.ctx_size * threshold

    def compress(self, messages: list[dict]) -> list[dict]:
        """
        SA-81B: Remove dead-weight, preserve causal decisions.
        TODO: Phase 2 — implement sliding window + causal summary
        """
        return messages
