"""A1 元信息-free 信源链路（spec: iaq-a1-metadata-free）。

把比特分成两层,**不传 M_i 分配图**:
  基础层 base    : 每 patch 固定 M_base bit 均匀量化(双用:粗图 + 派生分配来源)
  增强层 enhance : 剩余预算按派生 ΔM_i 增量分配,只对重要 patch 加细节

核心机制(消除元信息悬崖):
  均匀标量量化在同一 [umin,umax] 下是**嵌入式**的——T bit 量化索引的高 M_base 位
  恰好等于 M_base bit 量化索引。于是
    每 patch 总比特 T_i = M_base + ΔM_i,高 M_base 位=基础层,低 ΔM_i 位=增强层。
  收端解出基础层(经 FEC 逐 bit 正确)→ 在基础层重建图上算注意力 → 量化到 16 档 →
  跑同一增量分配 → 派生出与发端**逐 bit 相同**的 ΔM_i → 增强层边界自动对齐。
  没有「传错 M_i 长度」这个失步源,故无③式灾难崩盘。

诚实边界:极低 SNR 增强层被毁、基础层经 FEC 仍正确时,A1 退化到基础层(M_base bit)
质量量级——不出现失步崩盘,但该底可能低于同信源预算的 2-bit 均匀基线 ①。

本模块纯 numpy:注意力 a_i 由调用方(pipeline 用 ViT)在基础层重建图上算好后传入。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .bit_allocation import incremental_allocation
from .fec import repetition_decode, repetition_encode
from .modulation import bpsk_awgn
from .quantization import (
    bitstream_to_indices,
    quantize_to_bitstream,
    uniform_dequantize,
    uniform_quantize,
)

ATTN_LEVELS = 16  # 注意力量化档数(锁两端一致,见 spec)


def base_reconstruct(patches_or_img: np.ndarray, m_base: int, umin: float, umax: float) -> np.ndarray:
    """对给定张量做 M_base bit 均匀量化再反量化,得基础层重建(同形状)。

    umin/umax 用每图全局范围(与 pipeline 约定一致)。两端在此重建图上算注意力。
    """
    idx = uniform_quantize(patches_or_img, m_base, umin, umax)
    return uniform_dequantize(idx, m_base, umin, umax)


def quantize_attention(a_i: np.ndarray, levels: int = ATTN_LEVELS) -> np.ndarray:
    """把注意力 a_i 量化到 levels 档(按本图 min-max 归一)。

    两端浮点前向的微小差异落入同档,使派生分配确定性一致。返回档位索引(float,
    单调保序即可喂增量分配)。
    """
    a = np.asarray(a_i, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    q = np.floor((a - lo) / (hi - lo) * levels)
    return np.clip(q, 0, levels - 1)


def derive_allocation(
    a_i: np.ndarray, b_enh: int, m_max: int, levels: int = ATTN_LEVELS
) -> np.ndarray:
    """从注意力派生增强层分配 ΔM_i(确定性:同 a_i 必得同结果)。

    先量化 a_i 到 levels 档,再在增强预算 b_enh 下跑增量分配。
    """
    qa = quantize_attention(a_i, levels)
    return incremental_allocation(qa, b_enh, m_max)


@dataclass
class A1Packet:
    """A1 比特包——只含基础层 + 增强层,**不含 M_i 图 / 段长度元信息**。"""

    base_bits: np.ndarray       # 基础层比特(已含 FEC,若 base_fec_r>1)
    enh_bits: np.ndarray        # 增强层比特流(各 patch 按 ΔM_i 拼接)
    umin: float
    umax: float
    values_per_patch: int       # 每 patch 标量数 P
    n_patches: int
    m_base: int
    b_enh: int
    m_max: int
    base_fec_r: int             # 基础层重复码速率分母(1=不保护)


def a1_encode(
    patches: np.ndarray,
    a_i_base: np.ndarray,
    umin: float,
    umax: float,
    m_base: int,
    b_enh: int,
    m_max: int,
    base_fec_r: int = 1,
) -> A1Packet:
    """发端:分层编码一张图。

    patches: (N, P) 该图的 patch 张量;a_i_base: (N,) 在**基础层重建图**上算的注意力。
    """
    patches = np.asarray(patches, dtype=np.float64)
    n, p = patches.shape
    delta = derive_allocation(a_i_base, b_enh, m_max)  # (N,)

    base_segs, enh_segs = [], []
    for i in range(n):
        t = m_base + int(delta[i])
        # 基础层:嵌入式高 M_base 位 == 直接 M_base bit 量化(见模块说明)
        base_segs.append(quantize_to_bitstream(patches[i], m_base, umin, umax))
        if delta[i] > 0:
            full = quantize_to_bitstream(patches[i], t, umin, umax).reshape(p, t)
            enh_segs.append(full[:, m_base:].ravel())   # 低 ΔM_i 位 = 增强层

    base_bits = (
        np.concatenate(base_segs).astype(np.uint8) if base_segs
        else np.zeros(0, dtype=np.uint8)
    )
    enh_bits = (
        np.concatenate(enh_segs).astype(np.uint8) if enh_segs
        else np.zeros(0, dtype=np.uint8)
    )
    if base_fec_r > 1:
        base_bits = repetition_encode(base_bits, base_fec_r)

    return A1Packet(
        base_bits=base_bits, enh_bits=enh_bits, umin=umin, umax=umax,
        values_per_patch=p, n_patches=n, m_base=m_base, b_enh=b_enh,
        m_max=m_max, base_fec_r=base_fec_r,
    )


def a1_through_channel(
    pkt: A1Packet, ebn0_db: float, rng: np.random.Generator
) -> A1Packet:
    """基础层(已含 FEC)与增强层各自经 BPSK→AWGN→解调,返回受损 packet。

    基础层的 FEC 解码在 a1_decode_base 内完成(对称于 a1_encode 的 FEC 编码)。
    增强层 v1 不加 FEC——极低 SNR 被毁时由基础层兜底(有底降级,无失步崩盘)。
    """
    return dataclasses.replace(
        pkt,
        base_bits=bpsk_awgn(pkt.base_bits, ebn0_db, rng),
        enh_bits=bpsk_awgn(pkt.enh_bits, ebn0_db, rng),
    )


def a1_decode_base(pkt: A1Packet) -> tuple[np.ndarray, np.ndarray]:
    """收端第一步:FEC 解基础层 → 还原基础层比特 + 基础层重建 patch (N,P)。

    返回 (base_bits_raw, base_patches)。base_patches 供调用方过 ViT 算 a_i_base。
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
    return base_raw, base_patches


def a1_decode(pkt: A1Packet, a_i_base: np.ndarray) -> np.ndarray:
    """收端第二步:用在基础层重建图上算的 a_i_base 派生 ΔM_i,切增强层 → 重建 patch (N,P)。

    a_i_base 与发端一致时(基础层逐 bit 正确 → 注意力逐 bit 一致),ΔM_i 逐 patch 相同,
    增强层边界自动对齐。
    """
    base_raw, _ = a1_decode_base(pkt)
    delta = derive_allocation(a_i_base, pkt.b_enh, pkt.m_max)  # 与发端同
    p, n, mb = pkt.values_per_patch, pkt.n_patches, pkt.m_base

    out = np.empty((n, p), dtype=np.float64)
    cursor = 0
    for i in range(n):
        d = int(delta[i])
        t = mb + d
        base_seg = base_raw[i * p * mb:(i + 1) * p * mb].reshape(p, mb)
        if d > 0:
            nbits = p * d
            enh_seg = pkt.enh_bits[cursor:cursor + nbits]
            if enh_seg.size < nbits:  # 防御:增强层不足补 0(不应发生于对齐时)
                enh_seg = np.concatenate(
                    [enh_seg, np.zeros(nbits - enh_seg.size, dtype=np.uint8)])
            enh_seg = enh_seg.reshape(p, d)
            full = np.concatenate([base_seg, enh_seg], axis=1).ravel()
            cursor += nbits
        else:
            full = base_seg.ravel()
        idx = bitstream_to_indices(full, t, count=p)
        out[i] = uniform_dequantize(idx, t, pkt.umin, pkt.umax)
    return out
