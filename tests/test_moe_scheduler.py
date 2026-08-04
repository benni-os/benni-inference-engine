"""Tests for MoE Scheduler hardware detection."""
from bie.core.moe_scheduler import MoEScheduler

def test_detect_ram():
    scheduler = MoEScheduler("qwen3-30b-a3b")
    ram = scheduler.detect_ram()
    assert ram > 0, "RAM detection should return positive value"

def test_compute_returns_config():
    scheduler = MoEScheduler("qwen3-30b-a3b")
    config = scheduler.compute()
    assert "ngl" in config
    assert "moe_offload_pattern" in config
    assert "ram_available_gb" in config
    assert config["ngl"] == 99
