"""Unit tests for the Gate 30B measurement runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "gate_30b_runner.py"
SPEC = importlib.util.spec_from_file_location("gate_30b_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_gate_rejects_unmeasured_criteria() -> None:
    result = runner.evaluate([runner.not_measured("ram", 19456, "RSS unavailable")])
    assert result["approved"] is False
    assert result["status"] == "not_measured"


def test_gate_approves_only_measured_passes() -> None:
    result = runner.evaluate([runner.measured("load", True, True, True)])
    assert result["approved"] is True
    assert result["status"] == "approved"


def test_gate_rejects_measured_failure() -> None:
    result = runner.evaluate([runner.measured("decode", 1.0, 3.0, False)])
    assert result["approved"] is False
    assert result["status"] == "rejected"


def test_degradation_is_calculated_from_first_and_last() -> None:
    assert runner.degradation(10.0, 9.0) == 10.0
    assert runner.degradation(10.0, 8.0) == 20.0


def test_degradation_returns_none_without_valid_pair() -> None:
    assert runner.degradation(None, 8.0) is None
    assert runner.degradation(0.0, 8.0) is None


def test_rss_mib_returns_none_for_missing_pid() -> None:
    assert runner.rss_mib(999_999_999) is None


def test_metric_reads_supported_decode_keys() -> None:
    assert runner.metric({"decode_tok_s": 3.5}, ("decode_tok_s", "tok_s")) == 3.5
    assert runner.metric({"tok_s": 2.5}, ("decode_tok_s", "tok_s")) == 2.5
    assert runner.metric({}, ("decode_tok_s",)) is None


def test_parse_json_accepts_only_objects() -> None:
    assert runner.parse_json('{"decode_tok_s": 3.5}') == {"decode_tok_s": 3.5}
    assert runner.parse_json("[]") is None
    assert runner.parse_json("not-json") is None


def test_command_uses_current_python_and_json() -> None:
    args = type("Args", (), {
        "model_path": Path("model.gguf"),
        "context": 8192,
        "n_gpu_layers": 6,
        "reps": 3,
        "prompt_tokens": 512,
        "max_tokens": 128,
    })()
    command = runner.command(args)
    assert command[:3] == [runner.sys.executable, "-m", "bie.serve"]
    assert "--json" in command
    assert "--n-gpu-layers" in command
