"""a1.py 单测（spec: iaq-a1-metadata-free）。全合成数据,不加载模型。"""

import dataclasses

import numpy as np

from vit_iaq_semcom.a1 import (
    A1Packet,
    a1_decode,
    a1_encode,
    derive_allocation,
    quantize_attention,
)
from vit_iaq_semcom.quantization import uniform_dequantize, uniform_quantize


def _rand_patches(n=8, p=12, seed=0):
    return np.random.default_rng(seed).random((n, p))


def test_derive_allocation_deterministic_and_budget():
    a = np.random.default_rng(1).random(8)
    d1 = derive_allocation(a, b_enh=10, m_max=4)
    d2 = derive_allocation(a, b_enh=10, m_max=4)
    np.testing.assert_array_equal(d1, d2)          # 确定性:同 a_i 同结果
    assert d1.sum() <= 10                           # 预算
    assert d1.max() <= 4                            # m_max


def test_attention_quant_absorbs_small_drift():
    a = np.random.default_rng(2).random(16)
    a2 = a + 1e-7 * np.random.default_rng(3).standard_normal(16)  # 微小漂移
    np.testing.assert_array_equal(quantize_attention(a), quantize_attention(a2))
    np.testing.assert_array_equal(
        derive_allocation(a, 20, 4), derive_allocation(a2, 20, 4))


def test_packet_carries_no_metadata_map():
    """A1Packet 不含 M_i 图 / 段长度元信息(核心:不传脆弱元信息)。"""
    fields = {f.name for f in dataclasses.fields(A1Packet)}
    assert "m_map" not in fields and "seg_len" not in fields and "lengths" not in fields


def test_base_layer_is_embedded():
    """基础层比特 == 直接 M_base bit 量化(嵌入式性质)。"""
    patches = _rand_patches()
    a = np.random.default_rng(4).random(patches.shape[0])
    pkt = a1_encode(patches, a, 0.0, 1.0, m_base=1, b_enh=8, m_max=4)
    n, p, mb = patches.shape[0], patches.shape[1], 1
    from vit_iaq_semcom.quantization import quantize_to_bitstream
    expect = np.concatenate(
        [quantize_to_bitstream(patches[i], mb, 0.0, 1.0) for i in range(n)])
    np.testing.assert_array_equal(pkt.base_bits, expect)


def test_lossless_roundtrip_same_attention():
    """两端同 a_i_base、无信道:重建 == 各 patch (M_base+ΔM_i) bit 直接量化。"""
    patches = _rand_patches(seed=5)
    a = np.random.default_rng(6).random(patches.shape[0])
    mb, b_enh, m_max = 1, 12, 4
    pkt = a1_encode(patches, a, 0.0, 1.0, mb, b_enh, m_max)
    recon = a1_decode(pkt, a)
    delta = derive_allocation(a, b_enh, m_max)
    for i in range(patches.shape[0]):
        t = mb + int(delta[i])
        exp = uniform_dequantize(
            uniform_quantize(patches[i], t, 0.0, 1.0), t, 0.0, 1.0)
        np.testing.assert_allclose(recon[i], exp)


def test_enhancement_boundary_auto_aligns():
    """收端只靠派生 ΔM_i 切增强层,无段长度信息也对齐(重建高保真)。"""
    patches = _rand_patches(n=16, p=24, seed=7)
    a = np.random.default_rng(8).random(16)
    pkt = a1_encode(patches, a, 0.0, 1.0, m_base=1, b_enh=20, m_max=4)
    recon = a1_decode(pkt, a)
    # 高保真:重建误差应远小于 1-bit 基础层格宽(0.5)
    assert np.abs(recon - patches).mean() < 0.2


def test_graceful_floor_no_crater():
    """增强层全毁、基础层完好 → 退化到基础层质量,逐 patch 误差有界,无崩盘。"""
    patches = _rand_patches(n=16, p=24, seed=9)
    a = np.random.default_rng(10).random(16)
    mb = 1
    pkt = a1_encode(patches, a, 0.0, 1.0, mb, b_enh=20, m_max=4)
    # 把增强层整段替换为随机比特(模拟极低 SNR)
    rng = np.random.default_rng(11)
    garbled = dataclasses.replace(
        pkt, enh_bits=rng.integers(0, 2, size=pkt.enh_bits.size).astype(np.uint8))
    recon = a1_decode(garbled, a)
    bin_w = 1.0 / (2 ** mb)               # 基础层格宽 = 0.5
    # MSB(基础层)完好 → 每 patch 落在正确基础格内 → 逐标量误差 < 格宽(无失步崩盘)
    assert np.abs(recon - patches).max() < bin_w + 1e-9
    assert np.isfinite(recon).all()


def test_base_fec_roundtrip_corrects_minority():
    """基础层重复码 r=3,每组至多 1 错(可纠)→ 基础层逐 bit 恢复 → 全解码与无信道一致。"""
    patches = _rand_patches(seed=12)
    a = np.random.default_rng(13).random(patches.shape[0])
    clean = a1_decode(a1_encode(patches, a, 0.0, 1.0, 1, 8, 4), a)
    pkt = a1_encode(patches, a, 0.0, 1.0, 1, 8, 4, base_fec_r=3)
    # 每个 r=3 组内随机翻 1 位(多数表决保证可纠)
    flip = np.zeros(pkt.base_bits.size, dtype=np.uint8)
    groups = flip.reshape(-1, 3)
    rng = np.random.default_rng(14)
    groups[np.arange(groups.shape[0]), rng.integers(0, 3, groups.shape[0])] = 1
    noisy = dataclasses.replace(
        pkt, base_bits=(pkt.base_bits ^ flip).astype(np.uint8))
    recon = a1_decode(noisy, a)
    np.testing.assert_allclose(recon, clean)
