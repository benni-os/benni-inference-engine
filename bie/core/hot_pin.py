"""Hot Pin — adaptive expert pinning. Learns usage patterns and pins hot experts in RAM."""
# Inspired by colibri (JustVugg/colibri, Apache 2.0) hot-pinning logic.
# Adapted for llama-cpp-python context.

class HotPinManager:
    """
    Tracks expert activation frequency across inference calls.
    After warmup_tokens, pins the top-N most-used experts per layer in RAM.
    Persists pinning state to .bie_hot_pin.json next to the model.
    """

    def __init__(self, n_layers: int, n_experts: int, hot_n: int = 14, warmup_tokens: int = 32):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.hot_n = hot_n
        self.warmup_tokens = warmup_tokens
        self.freq = [[0] * n_experts for _ in range(n_layers)]
        self.token_count = 0
        self.pinned = False

    def record(self, layer: int, expert_ids: list[int]):
        """Record expert activations for a token."""
        for eid in expert_ids:
            self.freq[layer][eid] += 1
        self.token_count += 1
        if not self.pinned and self.token_count >= self.warmup_tokens:
            self._pin()

    def _pin(self):
        """Pin top-N experts per layer."""
        self.pinned = True
        # TODO: Phase 1 — integrate with llama-cpp-python cache control
