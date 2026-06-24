"""bucket.py 单测（spec: iaq-bucket-coding）。全合成数据,不加载模型(纯 CPU)。"""

import dataclasses

import numpy as np

from vit_iaq_semcom.bucket import (
    BucketPacket,
    bit_depth_map,
    bits_per_rank,
    bucket_alloc,
    bucket_decode,
    bucket_decode_base,
    bucket_encode,
    bucket_index,
)
from vit_iaq_semcom.quantization import (
    quantize_to_bitstream,
    uniform_dequantize,
    uniform_quantize,
)


def _rand_patches(n=16, p=12, seed=0):
    return np.random.default_rng(seed).random((n, p))


# b 段桶配置:m_base=1,top-4 到 3bit、次-4 到 2bit、其余到 1bit
BUCKETS = [(4, 3), (4, 2), (8, 1)]
MB, MMAX = 1, 8


def test_packet_carries_no_metadata_map():
    """BucketPacket 不含 M_i 图 / 段长度元信息(核心:不传脆弱元信息)。"""
    fields = {f.name for f in dataclasses.fields(BucketPacket)}
    assert "m_map" not in fields and "seg_len" not in fields and "lengths" not in fields


def test_bits_per_rank_independent_of_attention():
    """每秩位宽只依赖桶配置与 n,与具体注意力无关(定长定位的根基)。"""
    pr = bits_per_rank(BUCKETS, 16)
    assert pr.tolist() == [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    assert int(pr.sum()) == 4 * 3 + 4 * 2 + 8 * 1  # ΣM_i 守恒于桶预算


def test_bucket_alloc_budget_conserved():
    a = np.random.default_rng(1).random(16)
    m = bucket_alloc(a, BUCKETS)
    assert int(m.sum()) == bits_per_rank(BUCKETS, 16).sum()
    assert m.max() <= 3 and m.min() >= 1


def test_base_layer_decode_independent_of_ranking():
    """基础层按固定位置解码:换任意注意力,基础层重建完全不变(不依赖排序)。"""
    patches = _rand_patches()
    a1 = np.random.default_rng(2).random(16)
    a2 = np.random.default_rng(3).random(16)        # 完全不同的排序
    pkt = bucket_encode(patches, a1, 0.0, 1.0, MB, BUCKETS, MMAX)
    base1 = bucket_decode_base(pkt)
    # 基础层比特与排序无关 → 用 a2 编码得到的 base_bits 应一致
    pkt2 = bucket_encode(patches, a2, 0.0, 1.0, MB, BUCKETS, MMAX)
    np.testing.assert_array_equal(pkt.base_bits, pkt2.base_bits)
    np.testing.assert_allclose(base1, bucket_decode_base(pkt2))


def test_base_layer_is_embedded():
    """基础层比特 == 直接 m_base bit 量化(嵌入式高位)。"""
    patches = _rand_patches()
    a = np.random.default_rng(4).random(16)
    pkt = bucket_encode(patches, a, 0.0, 1.0, MB, BUCKETS, MMAX)
    expect = np.concatenate(
        [quantize_to_bitstream(patches[i], MB, 0.0, 1.0) for i in range(16)])
    np.testing.assert_array_equal(pkt.base_bits, expect)


def test_lossless_roundtrip_same_attention():
    """两端同 a_i、无信道:重建 == 各 patch 按其总比特 M_i 直接量化(嵌入式重组无损)。"""
    patches = _rand_patches(seed=5)
    a = np.random.default_rng(6).random(16)
    pkt = bucket_encode(patches, a, 0.0, 1.0, MB, BUCKETS, MMAX)
    recon = bucket_decode(pkt, a)
    m = bucket_alloc(a, BUCKETS)
    for i in range(16):
        t = int(m[i])
        exp = uniform_dequantize(
            uniform_quantize(patches[i], t, 0.0, 1.0), t, 0.0, 1.0)
        np.testing.assert_allclose(recon[i], exp)


def test_enhancement_fixed_length_alignment():
    """增强层按定长秩槽读取,无段长信息也对齐(高保真重建)。"""
    patches = _rand_patches(n=16, p=24, seed=7)
    a = np.random.default_rng(8).random(16)
    pkt = bucket_encode(patches, a, 0.0, 1.0, MB, BUCKETS, MMAX)
    recon = bucket_decode(pkt, a)
    assert np.abs(recon - patches).mean() < 0.2     # 远小于 1-bit 基础格宽 0.5


def test_base_corruption_does_not_cascade():
    """基础层翻若干 bit:仅对应 patch 粗值变化(局部),不发生游标失步/全图崩。"""
    patches = _rand_patches(n=16, p=24, seed=9)
    a = np.random.default_rng(10).random(16)
    pkt = bucket_encode(patches, a, 0.0, 1.0, MB, BUCKETS, MMAX)
    clean = bucket_decode(pkt, a)
    # 翻转基础层中段几个 bit(模拟基础层受损)
    corrupt = pkt.base_bits.copy()
    flip_pos = [10, 11, 50, 200]
    corrupt[flip_pos] ^= 1
    noisy = dataclasses.replace(pkt, base_bits=corrupt)
    # 注意:派生排序用未受损的 a(隔离"基础层比特错"这一路);收端仍对齐读增强层
    recon = bucket_decode(noisy, a)
    # 受影响 patch 数有限(局部),绝大多数 patch 与无误码完全一致
    changed = np.array([not np.allclose(recon[i], clean[i]) for i in range(16)])
    assert changed.sum() <= 4                        # 远小于 N=16,无级联
    assert np.isfinite(recon).all()


def test_derived_mismatch_is_local():
    """收端派生排序与发端不一致 → 仅边界少数 patch 错配,码流对齐不变、不崩。"""
    patches = _rand_patches(n=16, p=24, seed=11)
    a_tx = np.random.default_rng(12).random(16)
    pkt = bucket_encode(patches, a_tx, 0.0, 1.0, MB, BUCKETS, MMAX)
    # 收端排序:在发端注意力上加噪 → 仅相邻秩偶发对调(边界错配)
    a_rx = a_tx + 0.02 * np.random.default_rng(13).standard_normal(16)
    recon = bucket_decode(pkt, a_rx)
    # 解码不崩(有限值);桶号不同的 patch 数有限
    assert np.isfinite(recon).all()
    bo = bucket_index(a_tx, BUCKETS)
    br = bucket_index(a_rx, BUCKETS)
    assert (bo != br).sum() <= 8                     # 局部错配,非全体
    # 严重跨档(≥2)应极少
    assert (np.abs(bo - br) >= 2).sum() <= 2


def test_bit_depth_map_exportable():
    """比特深度图可导出,等于解码各 patch 实得总比特(供修补网络作可靠性输入)。"""
    a = np.random.default_rng(14).random(16)
    bdm = bit_depth_map(a, BUCKETS, MMAX)
    np.testing.assert_array_equal(bdm, np.minimum(bucket_alloc(a, BUCKETS), MMAX))
    assert bdm.shape == (16,)


def test_base_fec_roundtrip_corrects_minority():
    """基础层重复码 r=3,每组至多 1 错 → 基础层逐 bit 恢复 → 全解码与无信道一致。"""
    patches = _rand_patches(seed=15)
    a = np.random.default_rng(16).random(16)
    clean = bucket_decode(bucket_encode(patches, a, 0.0, 1.0, MB, BUCKETS, MMAX), a)
    pkt = bucket_encode(patches, a, 0.0, 1.0, MB, BUCKETS, MMAX, base_fec_r=3)
    flip = np.zeros(pkt.base_bits.size, dtype=np.uint8)
    groups = flip.reshape(-1, 3)
    rng = np.random.default_rng(17)
    groups[np.arange(groups.shape[0]), rng.integers(0, 3, groups.shape[0])] = 1
    noisy = dataclasses.replace(pkt, base_bits=(pkt.base_bits ^ flip).astype(np.uint8))
    recon = bucket_decode(noisy, a)
    np.testing.assert_allclose(recon, clean)
