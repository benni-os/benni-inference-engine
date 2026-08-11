"""Deterministic hardware-aware planning for local MoE inference."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from math import floor
from typing import Any

import psutil


@dataclass(frozen=True)
class ModelProfile:
    """Hardware-relevant model facts; values must come from model metadata."""

    total_layers: int
    dense_layer_gb: float
    expert_cache_gb_per_slot: float
    expert_slots: int
    moe_offload_pattern: str = r"\.ffn_.*_exps\.=CPU"


@dataclass(frozen=True)
class HardwareBudget:
    """Measured memory and safety reserves used by the planner."""

    vram_available_gb: float
    ram_available_gb: float
    vram_safety_margin: float = 0.85
    vram_reserve_gb: float = 1.0
    ram_reserve_gb: float = 2.0
    kv_reserve_gb: float = 1.5


DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "qwen3-30b-a3b": ModelProfile(
        total_layers=48,
        dense_layer_gb=0.16,
        expert_cache_gb_per_slot=0.12,
        expert_slots=128,
    ),
}


class MoEScheduler:
    """Plan GPU/CPU placement without silently overcommitting hardware."""

    def __init__(
        self,
        model_name: str,
        *,
        profile: ModelProfile | None = None,
        budget: HardwareBudget | None = None,
    ) -> None:
        self.model_name = model_name
        self.profile = profile or self._profile_for(model_name)
        self._budget = budget

    def _profile_for(self, model_name: str) -> ModelProfile:
        key = model_name.lower().removesuffix(".gguf")
        for name, profile in DEFAULT_PROFILES.items():
            if name in key:
                return profile
        return ModelProfile(
            total_layers=0,
            dense_layer_gb=0.0,
            expert_cache_gb_per_slot=0.0,
            expert_slots=0,
        )

    def detect_vram(self) -> float:
        """Return free VRAM in GiB; return zero when nvidia-smi is unavailable."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            value = result.stdout.strip().splitlines()[0]
            return max(0.0, float(value) / 1024.0)
        except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
            return 0.0

    def detect_ram(self) -> float:
        """Return available system RAM in GiB."""
        return max(0.0, psutil.virtual_memory().available / (1024**3))

    def _hardware_budget(self) -> HardwareBudget:
        return self._budget or HardwareBudget(
            vram_available_gb=self.detect_vram(),
            ram_available_gb=self.detect_ram(),
        )

    @staticmethod
    def _env_int(name: str) -> int | None:
        value = os.getenv(name)
        if value is None:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        return max(0, parsed)

    def compute(self) -> dict[str, Any]:
        """Return an auditable, conservative placement plan."""
        budget = self._hardware_budget()
        warnings: list[str] = []
        gpu_override = self._env_int("BIE_GPU_LAYERS")
        cache_override = self._env_int("BIE_EXPERT_CACHE_SLOTS")

        if self.profile.total_layers <= 0:
            warnings.append("unknown_model_profile: GPU placement disabled")
        usable_vram = max(
            0.0,
            (budget.vram_available_gb - budget.vram_reserve_gb - budget.kv_reserve_gb)
            * budget.vram_safety_margin,
        )
        calculated_ngl = (
            min(self.profile.total_layers, floor(usable_vram / self.profile.dense_layer_gb))
            if self.profile.dense_layer_gb > 0
            else 0
        )
        ngl = min(self.profile.total_layers, gpu_override) if gpu_override is not None else calculated_ngl
        if budget.vram_available_gb <= 0 and ngl > 0:
            warnings.append("no_cuda_vram: GPU placement forced to zero")
            ngl = 0

        usable_ram = max(0.0, budget.ram_available_gb - budget.ram_reserve_gb)
        calculated_slots = (
            floor(usable_ram / self.profile.expert_cache_gb_per_slot)
            if self.profile.expert_cache_gb_per_slot > 0
            else 0
        )
        slots = min(self.profile.expert_slots, calculated_slots)
        if cache_override is not None:
            slots = min(self.profile.expert_slots, cache_override)

        return {
            "vram_available_gb": round(budget.vram_available_gb, 2),
            "ram_available_gb": round(budget.ram_available_gb, 2),
            "ngl": max(0, ngl),
            "moe_offload_pattern": self.profile.moe_offload_pattern,
            "expert_cache_slots": max(0, slots),
            "model_profile": "known" if self.profile.total_layers else "unknown",
            "planner_source": "override" if gpu_override is not None else "calculated",
            "warnings": warnings,
        }
