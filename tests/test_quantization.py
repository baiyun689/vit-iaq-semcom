"""quantization.py 单测（spec: iaq-source-coding / Patch 均匀量化器）。"""

import numpy as np
import pytest

from vit_iaq_semcom.quantization import (
    bits_to_index,
    bitstream_to_indices,
    index_to_bits,
    quantize_to_bitstream,
    uniform_dequantize,
    uniform_quantize,
)


def test_roundtrip_error_within_step():
    """往返最大误差 <= 量化步长 (umax-umin)/2^m。"""
    rng = np.random.default_rng(0)
    umin, umax = -2.0, 3.0
    vals = rng.uniform(umin, umax, size=2000)
    for m in (1, 2, 4, 6, 8):
        idx = uniform_quantize(vals, m, umin, umax)
        rec = uniform_dequantize(idx, m, umin, umax)
        step = (umax - umin) / (1 << m)
        assert np.max(np.abs(rec - vals)) <= step + 1e-9


def test_more_bits_not_worse_mse():
    """m+1 比特的重建 MSE 不大于 m 比特。"""
    rng = np.random.default_rng(1)
    umin, umax = 0.0, 1.0
    vals = rng.uniform(umin, umax, size=5000)
    prev = np.inf
    for m in range(1, 9):
        rec = uniform_dequantize(uniform_quantize(vals, m, umin, umax), m, umin, umax)
        mse = float(np.mean((rec - vals) ** 2))
        assert mse <= prev + 1e-12
        prev = mse


def test_index_bits_roundtrip():
    for m in range(1, 9):
        for idx in range(1 << m):
            bits = index_to_bits(idx, m)
            assert bits.shape == (m,)
            assert bits_to_index(bits) == idx


def test_bitstream_roundtrip():
    rng = np.random.default_rng(2)
    umin, umax = -1.0, 1.0
    vals = rng.uniform(umin, umax, size=37)
    m = 5
    stream = quantize_to_bitstream(vals, m, umin, umax)
    assert stream.shape == (37 * m,)
    idx = bitstream_to_indices(stream, m, count=37)
    np.testing.assert_array_equal(idx, uniform_quantize(vals, m, umin, umax))


def test_m_zero_no_bits_midpoint():
    umin, umax = -4.0, 6.0
    vals = np.array([0.0, 1.0, -2.0])
    assert quantize_to_bitstream(vals, 0, umin, umax).size == 0
    assert index_to_bits(0, 0).size == 0
    rec = uniform_dequantize(uniform_quantize(vals, 0, umin, umax), 0, umin, umax)
    np.testing.assert_allclose(rec, np.full(3, (umin + umax) / 2.0))


def test_constant_patch_no_crash():
    rec = uniform_dequantize(uniform_quantize(np.zeros(5), 4, 2.0, 2.0), 4, 2.0, 2.0)
    np.testing.assert_allclose(rec, 2.0)


def test_bitstream_understep_pads_zero():
    """信道失步导致比特不足时，补 0 索引、不崩溃。"""
    idx = bitstream_to_indices(np.array([1, 0, 1], dtype=np.uint8), m=4, count=3)
    assert idx.shape == (3,)
    assert idx[0] == 0b1010  # 前 4 位：1,0,1,(补0)
