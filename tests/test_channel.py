"""channel.py 单测（spec: iaq-channel）。"""

import numpy as np

from vit_iaq_semcom.channel import (
    Packet,
    apply_channel,
    awgn_channel,
    bsc,
    pack_metadata,
    unpack_metadata,
)


def _packet(rng, n=196, vpp=8, m_max=8):
    m_map = rng.integers(0, m_max + 1, size=n)
    payload = rng.integers(0, 2, size=int(m_map.sum()) * vpp).astype(np.uint8)
    return Packet(payload, m_map, umin=-2.0, umax=3.0, values_per_patch=vpp, m_max=m_max)


def test_bsc_zero_lossless():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=1000).astype(np.uint8)
    np.testing.assert_array_equal(bsc(bits, 0.0, rng), bits)


def test_bsc_flip_ratio_near_mu():
    rng = np.random.default_rng(1)
    bits = np.zeros(200_000, dtype=np.uint8)
    for mu in (0.01, 0.1, 0.3):
        out = bsc(bits, mu, rng)
        ratio = out.mean()
        assert abs(ratio - mu) < 0.01


def test_bsc_reproducible():
    bits = np.random.default_rng(2).integers(0, 2, size=500).astype(np.uint8)
    a = bsc(bits, 0.2, np.random.default_rng(7))
    b = bsc(bits, 0.2, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_metadata_pack_roundtrip():
    rng = np.random.default_rng(3)
    m_map = rng.integers(0, 9, size=196)
    bits = pack_metadata(m_map, -2.0, 3.0, m_max=8)
    m2, umin, umax = unpack_metadata(bits, n=196, m_max=8)
    np.testing.assert_array_equal(m2, m_map)
    assert abs(umin - (-2.0)) < 1e-3 and abs(umax - 3.0) < 1e-3


def test_switch_off_metadata_lossless():
    """开关关闭：mu>0 下元信息逐位不变，仅载荷可能被翻。"""
    rng = np.random.default_rng(4)
    pkt = _packet(rng)
    out = apply_channel(pkt, mu=0.1, metadata_through_channel=False, rng=rng)
    np.testing.assert_array_equal(out.m_map, pkt.m_map)
    assert out.umin == pkt.umin and out.umax == pkt.umax
    assert not np.array_equal(out.payload_bits, pkt.payload_bits)  # 载荷被翻


def test_switch_on_metadata_can_change():
    """开关开启：mu 较大时元信息会被翻。"""
    rng = np.random.default_rng(5)
    pkt = _packet(rng)
    out = apply_channel(pkt, mu=0.2, metadata_through_channel=True, rng=rng)
    assert not np.array_equal(out.m_map, pkt.m_map) or out.umin != pkt.umin


def test_awgn_high_snr_near_lossless():
    """高 Eb/N0：载荷与元信息近无损。"""
    rng = np.random.default_rng(6)
    pkt = _packet(rng)
    out = awgn_channel(pkt, ebn0_db=15.0, metadata_through_channel=True, rng=rng)
    np.testing.assert_array_equal(out.m_map, pkt.m_map)
    assert (out.payload_bits != pkt.payload_bits).mean() < 1e-2


def test_awgn_switch_off_metadata_lossless():
    """开关关闭：低 Eb/N0 下元信息逐位不变，仅载荷可能误码。"""
    rng = np.random.default_rng(7)
    pkt = _packet(rng)
    out = awgn_channel(pkt, ebn0_db=2.0, metadata_through_channel=False, rng=rng)
    np.testing.assert_array_equal(out.m_map, pkt.m_map)
    assert out.umin == pkt.umin and out.umax == pkt.umax
