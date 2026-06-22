"""modulation.py 单测（spec: iaq-modulation）。"""

import math

import numpy as np

from vit_iaq_semcom.modulation import awgn, bpsk_awgn, bpsk_demodulate, bpsk_modulate


def _q(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def test_noiseless_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=1000).astype(np.uint8)
    out = bpsk_demodulate(bpsk_modulate(bits))
    np.testing.assert_array_equal(out, bits)


def test_high_snr_near_errorfree():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=10_000).astype(np.uint8)
    out = bpsk_awgn(bits, ebn0_db=12.0, rng=rng)
    ber = float((out != bits).mean())
    assert ber < 1e-3


def test_uncoded_ber_matches_theory():
    """未编码 BER ≈ Q(√(2·Eb/N0))。"""
    rng = np.random.default_rng(2)
    bits = np.zeros(400_000, dtype=np.uint8)
    for ebn0_db in (0.0, 2.0, 4.0, 6.0):
        out = bpsk_awgn(bits, ebn0_db, rng)
        ber = float(out.mean())
        ebn0_lin = 10.0 ** (ebn0_db / 10.0)
        theory = _q(math.sqrt(2 * ebn0_lin))
        assert abs(ber - theory) < max(0.1 * theory, 5e-4)


def test_inf_snr_lossless():
    rng = np.random.default_rng(3)
    sym = bpsk_modulate(np.array([0, 1, 0, 1], dtype=np.uint8))
    np.testing.assert_array_equal(awgn(sym, float("inf"), rng), sym)


def test_reproducible():
    bits = np.random.default_rng(4).integers(0, 2, size=2000).astype(np.uint8)
    a = bpsk_awgn(bits, 3.0, np.random.default_rng(9))
    b = bpsk_awgn(bits, 3.0, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
