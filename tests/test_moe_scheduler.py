"""Tests for MoE Scheduler deterministic hardware-aware planning."""
import pytest

from bie.core.moe_scheduler import HardwareBudget, ModelProfile, MoEScheduler


def _budget(vram: float, ram: float) -> HardwareBudget:
    return HardwareBudget(vram_available_gb=vram, ram_available_gb=ram)


def test_detect_ram():
    scheduler = MoEScheduler("qwen3-30b-a3b")
    ram = scheduler.detect_ram()
    assert ram > 0, "RAM detection should return positive value"


def test_known_model_full_offload():
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=16, ram=64))
    config = scheduler.compute()
    assert config["ngl"] == 48
    assert config["expert_cache_slots"] == 128
    assert config["model_profile"] == "known"
    assert config["planner_source"] == "calculated"
    assert config["moe_offload_pattern"].startswith(r"\.ffn_")
    assert config["warnings"] == []


def test_ngl_is_conservative_with_low_vram():
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=4, ram=64))
    config = scheduler.compute()
    assert config["ngl"] == 7


def test_ngl_never_exceeds_total_layers():
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=1024, ram=1024))
    config = scheduler.compute()
    assert config["ngl"] == 48


def test_expert_cache_slots_limited_by_ram():
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=16, ram=4))
    config = scheduler.compute()
    assert config["expert_cache_slots"] == 16


def test_qwen3_8b_profile_is_dense():
    scheduler = MoEScheduler("qwen3-8b", budget=_budget(vram=8, ram=16))
    config = scheduler.compute()
    assert config["model_profile"] == "known"
    assert config["ngl"] == 33
    assert config["expert_cache_slots"] == 0
    assert config["planner_source"] == "calculated"


def test_profile_and_budget_injection():
    profile = ModelProfile(
        total_layers=32,
        dense_layer_gb=0.5,
        expert_cache_gb_per_slot=0.2,
        expert_slots=64,
    )
    budget = _budget(vram=20, ram=16)
    scheduler = MoEScheduler("custom-model", profile=profile, budget=budget)
    config = scheduler.compute()
    assert config["ngl"] == 29
    assert config["expert_cache_slots"] == 64
    assert config["moe_offload_pattern"] == profile.moe_offload_pattern


def test_unknown_model_disables_gpu_placement():
    scheduler = MoEScheduler("made-up-model-900b", budget=_budget(vram=32, ram=128))
    config = scheduler.compute()
    assert config["model_profile"] == "unknown"
    assert config["ngl"] == 0
    assert config["expert_cache_slots"] == 0
    assert any("unknown_model_profile" in w for w in config["warnings"])


def test_no_gpu_plan_keeps_ngl_zero():
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=0, ram=64))
    config = scheduler.compute()
    assert config["vram_available_gb"] == 0
    assert config["ngl"] == 0
    assert config["planner_source"] == "calculated"


def test_no_gpu_override_warns_and_forces_zero(monkeypatch):
    monkeypatch.setenv("BIE_GPU_LAYERS", "48")
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=0, ram=64))
    config = scheduler.compute()
    assert config["ngl"] == 0
    assert config["planner_source"] == "override"
    assert any("no_cuda_vram" in w for w in config["warnings"])


def test_gpu_layers_override(monkeypatch):
    monkeypatch.setenv("BIE_GPU_LAYERS", "20")
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=16, ram=64))
    config = scheduler.compute()
    assert config["ngl"] == 20
    assert config["planner_source"] == "override"


def test_expert_cache_slots_override(monkeypatch):
    monkeypatch.setenv("BIE_EXPERT_CACHE_SLOTS", "64")
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=16, ram=64))
    config = scheduler.compute()
    assert config["expert_cache_slots"] == 64


def test_invalid_override_raises(monkeypatch):
    monkeypatch.setenv("BIE_GPU_LAYERS", "lots")
    scheduler = MoEScheduler("qwen3-30b-a3b", budget=_budget(vram=16, ram=64))
    with pytest.raises(ValueError, match="must be an integer"):
        scheduler.compute()
