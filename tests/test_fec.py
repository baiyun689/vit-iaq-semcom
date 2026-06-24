"""fec.py 单测（spec: iaq-fec）。"""

import numpy as np

from vit_iaq_semcom.fec import (
    repetition_decode,
    repetition_decoded_ber,
    repetition_encode,
)


def test_noiseless_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=1000).astype(np.uint8)
    for r in (1, 3, 5):
        out = repetition_decode(repetition_encode(bits, r), r)
        np.testing.assert_array_equal(out, bits)


def test_bit_inflation():
    bits = np.zeros(137, dtype=np.uint8)
    for r in (1, 3, 5, 7):
        assert repetition_encode(bits, r).size == 137 * r


def test_corrects_minority_errors():
    """r=5 时每组翻 ≤2 个仍能纠正。"""
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    r = 5
    enc = repetition_encode(bits, r).copy()
    # 第 0 组(全1)翻 2 个 → 仍多数 1；第 1 组(全0)翻 1 个 → 仍多数 0。
    enc[0] ^= 1
    enc[1] ^= 1
    enc[5] ^= 1
    out = repetition_decode(enc, r)
    np.testing.assert_array_equal(out, bits)


def test_empty():
    empty = np.array([], dtype=np.uint8)
    assert repetition_encode(empty, 3).size == 0
    assert repetition_decode(empty, 3).size == 0


def test_estimate_monotone_in_r():
    p = 0.1
    bers = [repetition_decoded_ber(p, r) for r in (1, 3, 5, 7, 9)]
    for a, b in zip(bers, bers[1:]):
        assert b <= a + 1e-12


def test_estimate_matches_montecarlo():
    """闭式估计 ≈ BSC 翻转 + 多数表决实测。"""
    rng = np.random.default_rng(7)
    L = 200_000
    bits = rng.integers(0, 2, size=L).astype(np.uint8)
    for p, r in ((0.1, 3), (0.2, 5), (0.05, 5)):
        enc = repetition_encode(bits, r)
        flip = rng.random(enc.size) < p
        dec = repetition_decode((enc ^ flip).astype(np.uint8), r)
        emp = float((dec != bits).mean())
        theory = repetition_decoded_ber(p, r)
        assert abs(emp - theory) < max(0.1 * theory, 2e-3)
