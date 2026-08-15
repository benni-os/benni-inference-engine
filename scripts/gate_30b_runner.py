#!/usr/bin/env python3
"""Gate 30B runner — measurement only.

No model download, package installation, or system service mutation.
Missing measurements never pass the gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VRAM_LIMIT_MIB = 8192
RAM_LIMIT_MIB = 19 * 1024
DECODE_MIN_TOK_S = 3.0
SOAK_MAX_DEGRADATION_PCT = 10.0


@dataclass
class Criterion:
    name: str
    status: str
    passed: bool | None
    value: Any = None
    limit: Any = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measured(name: str, value: Any, limit: Any, passed: bool, detail: str = "") -> Criterion:
    return Criterion(name, "measured", passed, value, limit, detail)


def not_measured(name: str, limit: Any, detail: str) -> Criterion:
    return Criterion(name, "not_measured", None, None, limit, detail)


def evaluate(criteria: list[Criterion]) -> dict[str, Any]:
    if any(item.passed is False for item in criteria):
        status = "rejected"
        approved = False
    elif any(item.status != "measured" or item.passed is not True for item in criteria):
        status = "not_measured"
        approved = False
    else:
        status = "approved"
        approved = True
    return {"status": status, "approved": approved, "criteria": [item.as_dict() for item in criteria]}


def rss_mib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def vram_mib(gpu_index: int) -> tuple[int | None, str | None]:
    cmd = ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode != 0 or not result.stdout.strip():
        return None, (result.stderr or "nvidia-smi failed").strip()
    try:
        return int(result.stdout.strip().splitlines()[0]), None
    except ValueError:
        return None, "unparseable nvidia-smi output"


class Monitor:
    def __init__(self, pid: int, gpu_index: int) -> None:
        self.pid = pid
        self.gpu_index = gpu_index
        self.stop = threading.Event()
        self.ram_peak_mib: int | None = None
        self.vram_peak_mib: int | None = None
        self.vram_error: str | None = None
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def _poll(self) -> None:
        while not self.stop.is_set():
            ram = rss_mib(self.pid)
            if ram is not None:
                self.ram_peak_mib = max(self.ram_peak_mib or 0, ram)
            vram, error = vram_mib(self.gpu_index)
            if vram is not None:
                self.vram_peak_mib = max(self.vram_peak_mib or 0, vram)
            elif error:
                self.vram_error = error
            self.stop.wait(0.25)


def run_process(command: list[str], timeout_s: int, gpu_index: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc), "error": str(exc)}
    monitor = Monitor(process.pid, gpu_index)
    monitor.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
    monitor.finish()
    return {
        "returncode": process.returncode,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "timed_out": timed_out,
        "error": "timeout" if timed_out else None,
        "duration_s": round(time.perf_counter() - started, 3),
        "ram_peak_mib": monitor.ram_peak_mib,
        "gpu_vram_peak_mib_global": monitor.vram_peak_mib,
        "gpu_vram_error": monitor.vram_error,
    }


def parse_json(stdout: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(stdout.strip())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def metric(payload: dict[str, Any] | None, names: tuple[str, ...]) -> float | None:
    if not payload:
        return None
    for name in names:
        value = payload.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def degradation(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or first <= 0:
        return None
    return round((first - last) / first * 100, 2)


def command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-m", "bie.serve", "bench",
        "--model-path", str(args.model_path),
        "--context", str(args.context),
        "--n-gpu-layers", str(args.n_gpu_layers),
        "--reps", str(args.reps),
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-tokens", str(args.max_tokens),
        "--json",
    ]


def run_cycle(args: argparse.Namespace) -> dict[str, Any]:
    result = run_process(command(args), args.timeout_s, args.gpu_index)
    payload = parse_json(result["stdout"])
    decode = metric(payload, ("decode_tok_s", "tok_s", "tokens_per_second"))
    result["payload"] = payload
    result["decode_tok_s"] = decode
    result["load_ok"] = result["returncode"] == 0 and not result["timed_out"] and decode is not None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gate_30b_runner")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-gpu-layers", type=int, default=6)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--soak-cycles", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    path = Path(args.model_path).expanduser().resolve()
    if not path.is_file():
        print(json.dumps({"status": "not_measured", "error": "model_not_found"}, indent=2))
        return 1

    benchmark = run_cycle(args)
    cycles = []
    rates = []
    for _ in range(args.soak_cycles):
        cycle = run_cycle(args)
        cycles.append({
            "decode_tok_s": cycle.get("decode_tok_s"),
            "returncode": cycle.get("returncode"),
            "timed_out": cycle.get("timed_out"),
            "ram_peak_mib": cycle.get("ram_peak_mib"),
            "gpu_vram_peak_mib_global": cycle.get("gpu_vram_peak_mib_global"),
            "error": cycle.get("error"),
        })
        if cycle.get("decode_tok_s") is not None:
            rates.append(cycle["decode_tok_s"])
    soak_degradation = degradation(rates[0], rates[-1]) if len(rates) >= 2 else None
    decode = benchmark.get("decode_tok_s")
    error_text = str(benchmark.get("error") or "").lower()
    criteria = [
        measured("load", benchmark.get("load_ok"), True, bool(benchmark.get("load_ok"))),
        measured("decode_tok_s", decode, DECODE_MIN_TOK_S, decode >= DECODE_MIN_TOK_S) if decode is not None else not_measured("decode_tok_s", DECODE_MIN_TOK_S, "decode not reported"),
        measured("ram_peak_mib", benchmark.get("ram_peak_mib"), RAM_LIMIT_MIB, benchmark["ram_peak_mib"] <= RAM_LIMIT_MIB) if benchmark.get("ram_peak_mib") is not None else not_measured("ram_peak_mib", RAM_LIMIT_MIB, "RSS unavailable"),
        measured("gpu_vram_peak_mib_global", benchmark.get("gpu_vram_peak_mib_global"), VRAM_LIMIT_MIB, benchmark["gpu_vram_peak_mib_global"] <= VRAM_LIMIT_MIB) if benchmark.get("gpu_vram_peak_mib_global") is not None else not_measured("gpu_vram_peak_mib_global", VRAM_LIMIT_MIB, benchmark.get("gpu_vram_error") or "VRAM unavailable"),
        measured("oom", False, False, "oom" not in error_text and "out of memory" not in error_text),
        measured("repeated_run_degradation_pct", soak_degradation, SOAK_MAX_DEGRADATION_PCT, soak_degradation <= SOAK_MAX_DEGRADATION_PCT) if soak_degradation is not None else not_measured("repeated_run_degradation_pct", SOAK_MAX_DEGRADATION_PCT, "fewer than two valid cycles"),
    ]
    evaluation = evaluate(criteria)
    result = {
        "trace_id": "bennos-pr1-owner-takeover-20260814T2230",
        "model_path": str(path),
        "configuration": vars(args),
        "benchmark": {key: value for key, value in benchmark.items() if key != "stdout"},
        "soak_test": {"mode": "repeated_run", "cycles": cycles, "degradation_pct": soak_degradation},
        "gate_evaluation": evaluation,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if evaluation["approved"] else 1


if __name__ == "__main__":
    sys.exit(main())
