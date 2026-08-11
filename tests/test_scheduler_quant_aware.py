"""Tests for BIE Scheduler Quant-Aware."""

import pytest

from bie.core.scheduler_quant_aware import (
    QUANT_BITS,
    derive_dense_layer_gb,
    quant_bits,
    vram_estimate_gb,
)


def test_quant_bits_known():
    assert quant_bits("Q4_K_M") == 4.5
    assert quant_bits("Q8_0") == 8.0


def test_quant_bits_unknown_raises():
    with pytest.raises(ValueError):
        quant_bits("Q99_XYZ")


def test_derive_q4km_matches_measurement():
    # medição real: 5.03 GB / 36 layers = 0.1397 (Q4_K_M, escala 1.0)
    assert derive_dense_layer_gb(5.03, 36, "Q4_K_M") == pytest.approx(0.1397, abs=1e-4)


def test_derive_scales_by_bits():
    # Q8_0: 0.1397 * (8.0 / 4.5)
    assert derive_dense_layer_gb(5.03, 36, "Q8_0") == pytest.approx(0.2484, abs=1e-4)
    # Q3_K_M: 0.1397 * (3.5 / 4.5)
    assert derive_dense_layer_gb(5.03, 36, "Q3_K_M") == pytest.approx(0.1087, abs=1e-4)


def test_derive_invalid_inputs():
    with pytest.raises(ValueError):
        derive_dense_layer_gb(0, 36, "Q4_K_M")
    with pytest.raises(ValueError):
        derive_dense_layer_gb(5.03, 0, "Q4_K_M")


def test_vram_estimate():
    assert vram_estimate_gb(0.14, 6) == pytest.approx(0.84)
    assert vram_estimate_gb(0.14, 22, kv_cache_gb=0.76) == pytest.approx(3.84)


def test_quant_bits_table_has_all_families():
    assert "Q3_K_S" in QUANT_BITS
    assert "Q3_K_M" in QUANT_BITS
    assert "Q4_K_S" in QUANT_BITS
    assert "Q6_K" in QUANT_BITS