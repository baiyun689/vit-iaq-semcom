"""非级联粗桶源编码（spec: iaq-bucket-coding）。

把比特分两层，**仍保留 IAQ 的动态重要性分配（底色）**，但换掉 A1 的变长游标
（级联崩的根源），改用定长定位结构，使信道误码只造成**局部值错配、绝不级联失步**：

  基础层 base    : 每 patch 固定 m_base bit 均匀量化，**定长、固定位置**，解码不依赖
                   任何排序/元信息。双用：粗图先验 + 收端派生排序的来源。
  增强层 enhance : 按注意力降序把 patch 填入**固定大小、固定比特**的桶
                   （如 top-K1 加到 b1 bit、次-K2 加到 b2 bit……），增强码流按
                   **桶/秩位置定长排布**，每个秩槽的位宽由桶配置唯一确定。

非级联的关键（与 A1 变长游标对比）：
  增强码流的切分**只由桶配置 + 秩位置决定，与"哪个 patch 落在该秩"无关**。收发两端
  的游标推进完全相同（即使派生排序不一致也一样推进），故永不失步。派生排序若与发端
  不一致，只是把某个秩槽的细节值贴给了相邻 patch（**局部错配**），不可能雪崩。

嵌入式性质（同 quantization 约定）：均匀量化在同一 [umin,umax] 下，T bit 量化索引的
高 m_base 位 == m_base bit 量化索引。故"基础层高位 + 增强层低位"重组即得 T=m_base+d
bit 精度，无需重复传高位。

本模块纯 numpy：注意力 a_i 由调用方（pipeline 用 ViT 在基础层重建图上算）传入。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .fec import repetition_decode, repetition_encode
from .modulation import bpsk_awgn
from .quantization import (
    bitstream_to_indices,
    quantize_to_bitstream,
    uniform_dequantize,
)

# 桶配置约定：list[(cnt, bits)]，bits = 该桶 patch 的**总** M_i（含基础层 m_base）。
# 例 b392：m_base=1，buckets=[(49,4),(49,2),(98,1)] → ΣM_i = 49·4+49·2+98·1 = 392。
Buckets = list[tuple[int, int]]


def bits_per_rank(buckets: Buckets, n: int) -> np.ndarray:
    """每个秩位置（0=最重要）的**总比特** M_i，长度 n。

    桶未覆盖满 n 时，多余秩沿用最后一档（与探针 bucket_alloc 约定一致）。该数组
    只依赖桶配置与 n，**与派生排序是否正确无关** —— 这是增强层定长定位的依据。
    """
    out: list[int] = []
    for cnt, bits in buckets:
        out.extend([int(bits)] * int(cnt))
    if len(out) < n:
        out.extend([int(buckets[-1][1])] * (n - len(out)))
    return np.asarray(out[:n], dtype=np.int64)


def bucket_alloc(a_i: np.ndarray, buckets: Buckets) -> np.ndarray:
    """按注意力降序入桶，返回每 patch 的总比特 M_i（patch 顺序）。"""
    a_i = np.asarray(a_i, dtype=np.float64)
    n = a_i.shape[0]
    order = np.argsort(-a_i, kind="stable")          # 重要性降序
    per_rank = bits_per_rank(buckets, n)
    m = np.zeros(n, dtype=np.int64)
    m[order] = per_rank                               # 秩 r 的总比特给 order[r]
    return m


def bucket_index(a_i: np.ndarray, buckets: Buckets) -> np.ndarray:
    """每 patch 落在第几号桶（0=最重要档）。供量化错配/可视化。"""
    a_i = np.asarray(a_i, dtype=np.float64)
    n = a_i.shape[0]
    order = np.argsort(-a_i, kind="stable")
    bidx = np.full(n, len(buckets) - 1, dtype=np.int64)
    rank = 0
    for bi, (cnt, _bits) in enumerate(buckets):
        sel = order[rank:rank + cnt]
        bidx[sel] = bi
        rank += cnt
    return bidx


@dataclass
class BucketPacket:
    """粗桶比特包——只含基础层 + 增强层，**不含 M_i 图 / 段长度元信息**。"""

    base_bits: np.ndarray       # 基础层比特（已含 FEC，若 base_fec_r>1）
    enh_bits: np.ndarray        # 增强层比特流（按秩/桶定长排布）
    umin: float
    umax: float
    values_per_patch: int       # 每 patch 标量数 P
    n_patches: int
    m_base: int
    buckets: Buckets
    m_max: int
    base_fec_r: int             # 基础层重复码速率分母（1=不保护）


def bucket_encode(
    patches: np.ndarray,
    a_i_base: np.ndarray,
    umin: float,
    umax: float,
    m_base: int,
    buckets: Buckets,
    m_max: int,
    base_fec_r: int = 1,
) -> BucketPacket:
    """发端：分层编码一张图。

    patches: (N,P) 该图 patch 张量；a_i_base: (N,) 在**基础层重建图**上算的注意力。
    增强码流按发端排序 order_tx 的秩顺序写入：秩 r 的低 (M_i−m_base) 位放进流里，
    位宽由 bits_per_rank（桶配置）唯一确定。
    """
    patches = np.asarray(patches, dtype=np.float64)
    n, p = patches.shape
    order = np.argsort(-np.asarray(a_i_base, dtype=np.float64), kind="stable")
    per_rank = bits_per_rank(buckets, n)              # 每秩总比特

    # 基础层：每 patch m_base 高位，**patch 顺序、定长**（解码不依赖排序）
    base_bits = np.concatenate(
        [quantize_to_bitstream(patches[i], m_base, umin, umax) for i in range(n)]
    ).astype(np.uint8) if n else np.zeros(0, dtype=np.uint8)

    # 增强层：按秩顺序，每秩写其 (总比特−m_base) 低位（嵌入式取低位）
    enh_segs: list[np.ndarray] = []
    for rank in range(n):
        i = int(order[rank])
        t = int(min(per_rank[rank], m_max))
        d = t - m_base
        if d > 0:
            full = quantize_to_bitstream(patches[i], t, umin, umax).reshape(p, t)
            enh_segs.append(full[:, m_base:].ravel())  # 低 d 位 = 增强层
    enh_bits = (
        np.concatenate(enh_segs).astype(np.uint8) if enh_segs
        else np.zeros(0, dtype=np.uint8)
    )

    if base_fec_r > 1:
        base_bits = repetition_encode(base_bits, base_fec_r)

    return BucketPacket(
        base_bits=base_bits, enh_bits=enh_bits, umin=umin, umax=umax,
        values_per_patch=p, n_patches=n, m_base=m_base, buckets=list(buckets),
        m_max=m_max, base_fec_r=base_fec_r,
    )


def bucket_through_channel(
    pkt: BucketPacket, ebn0_db: float, rng: np.random.Generator
) -> BucketPacket:
    """基础层（已含 FEC）与增强层各自经 BPSK→AWGN→解调，返回受损 packet。

    基础层 FEC 解码在 bucket_decode_base 内完成。增强层 v1 不加 FEC —— 极低 SNR 被毁
    时由基础层兜底（局部值错，无失步崩盘）。
    """
    return dataclasses.replace(
        pkt,
        base_bits=bpsk_awgn(pkt.base_bits, ebn0_db, rng),
        enh_bits=bpsk_awgn(pkt.enh_bits, ebn0_db, rng),
    )


def bucket_decode_base(pkt: BucketPacket) -> np.ndarray:
    """收端第一步：FEC 解基础层 → 基础层重建 patch (N,P)。

    **不依赖任何排序**（固定位置，每 patch m_base bit）。供调用方过 ViT 算 a_i_base。
    """
    base_raw = (
        repetition_decode(pkt.base_bits, pkt.base_fec_r)
        if pkt.base_fec_r > 1 else pkt.base_bits
    )
    p, n, mb = pkt.values_per_patch, pkt.n_patches, pkt.m_base
    base_patches = np.empty((n, p), dtype=np.float64)
    for i in range(n):
        seg = base_raw[i * p * mb:(i + 1) * p * mb]
        idx = bitstream_to_indices(seg, mb, count=p)
        base_patches[i] = uniform_dequantize(idx, mb, pkt.umin, pkt.umax)
    return base_patches


def bucket_decode(pkt: BucketPacket, a_i_base: np.ndarray) -> np.ndarray:
    """收端第二步：用基础层重建图上算的 a_i_base 派生排序，按**定长秩槽**读增强层，
    与各 patch 基础层高位重组反量化 → 重建 patch (N,P)。

    游标推进只看 bits_per_rank（桶配置），与派生排序是否正确无关 → **永不失步**。
    派生与发端不一致时，仅把某秩槽细节贴给相邻 patch（局部错配），不级联。
    """
    base_raw = (
        repetition_decode(pkt.base_bits, pkt.base_fec_r)
        if pkt.base_fec_r > 1 else pkt.base_bits
    )
    p, n, mb = pkt.values_per_patch, pkt.n_patches, pkt.m_base
    order = np.argsort(-np.asarray(a_i_base, dtype=np.float64), kind="stable")
    per_rank = bits_per_rank(pkt.buckets, n)

    out = np.empty((n, p), dtype=np.float64)
    cursor = 0
    for rank in range(n):
        i = int(order[rank])
        t = int(min(per_rank[rank], pkt.m_max))
        d = t - mb
        base_seg = base_raw[i * p * mb:(i + 1) * p * mb].reshape(p, mb)
        if d > 0:
            nbits = p * d
            enh_seg = pkt.enh_bits[cursor:cursor + nbits]
            if enh_seg.size < nbits:                 # 防御:不足补 0(对齐时不应发生)
                enh_seg = np.concatenate(
                    [enh_seg, np.zeros(nbits - enh_seg.size, dtype=np.uint8)])
            cursor += nbits
            full = np.concatenate([base_seg, enh_seg.reshape(p, d)], axis=1).ravel()
            idx = bitstream_to_indices(full, t, count=p)
            out[i] = uniform_dequantize(idx, t, pkt.umin, pkt.umax)
        else:
            idx = bitstream_to_indices(base_seg.ravel(), mb, count=p)
            out[i] = uniform_dequantize(idx, mb, pkt.umin, pkt.umax)
    return out


def bit_depth_map(a_i_base: np.ndarray, buckets: Buckets, m_max: int) -> np.ndarray:
    """收端派生的每 patch 总比特深度 M_i（=解码副产品）。

    供修补网络作**可靠性通道**输入：低比特/边界 patch 更不可信、更依赖上下文修复。
    与 bucket_decode 用同一派生排序，故就是解码时各 patch 实得的总比特，无额外计算。
    """
    return np.minimum(bucket_alloc(a_i_base, buckets), int(m_max))
