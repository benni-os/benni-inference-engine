"""MoE Scheduler — auto-computes GPU/CPU layer split based on available VRAM and RAM."""
import psutil

class MoEScheduler:
    """
    Given a model profile and current hardware state, computes:
    - n_gpu_layers (ngl): how many layers to offload to GPU
    - moe_offload_pattern: regex for llama.cpp -ot flag to route FFN experts to CPU
    - expert_cache_slots: number of expert slots to keep in RAM per layer
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def detect_vram(self) -> float:
        """Return available VRAM in GB. Returns 0.0 if no CUDA GPU found."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            mb = float(result.stdout.strip().split("\n")[0])
            return mb / 1024.0
        except Exception:
            return 0.0

    def detect_ram(self) -> float:
        """Return available RAM in GB."""
        return psutil.virtual_memory().available / 1e9

    def compute(self) -> dict:
        """
        Returns recommended inference config for the detected hardware.
        """
        vram_gb = self.detect_vram()
        ram_gb = self.detect_ram()

        # TODO: Phase 1 — load model profile from registry.json
        # and compute optimal split based on layer sizes
        return {
            "vram_available_gb": round(vram_gb, 2),
            "ram_available_gb": round(ram_gb, 2),
            "ngl": 99,  # all non-expert layers to GPU
            "moe_offload_pattern": r"\.ffn_.*_exps\.=CPU",
            "expert_cache_slots": 14,
        }
