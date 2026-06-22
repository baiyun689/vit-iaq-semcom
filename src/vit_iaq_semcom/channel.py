"""Sec. IV — BSC 信道 + 元信息打包。

载荷比特流恒过 BSC；元信息（M_i 图 + umin/umax）按 ``metadata_through_channel``
开关决定是否也过 BSC：
- 关（默认）：复现论文——元信息无损直达，仅载荷受误码。
- 开：暴露悬崖——M_i 被翻 → 收端切分载荷失步 → 整图崩；umin/umax 被翻 → 全局
  反量化尺度错。

元信息编码（与论文式 12 一致的定长帧头）：
- M_i 图：每 patch ⌈log2(M_max+1)⌉ 比特定长编码。
- umin/umax：各 16 比特，在固定范围 META_RANGE 内均匀量化（归一化 ViT 输入
  量级约 ±2.5，取 ±10 留余量；定长保证不失步、但翻位会致尺度错）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .quantization import (
    bits_to_index,
    index_to_bits,
    uniform_dequantize,
    uniform_quantize,
)

META_LO, META_HI = -10.0, 10.0
META_BITS = 16  # umin/umax 各 16 比特


@dataclass
class Packet:
    """一次传输的载荷 + 元信息。"""

    payload_bits: np.ndarray  # uint8，各 patch 量化比特拼接
    m_map: np.ndarray         # (N,) int，每 patch 比特数 M_i
    umin: float
    umax: float
    values_per_patch: int     # 每 patch 量化的标量个数（用于收端切分）
    m_max: int


def _bits_per_m(m_max: int) -> int:
    return max(1, math.ceil(math.log2(m_max + 1)))


def bsc(bits: np.ndarray, mu: float, rng: np.random.Generator) -> np.ndarray:
    """二进制对称信道：以概率 mu 独立翻转每一位。mu=0 恒等。"""
    bits = np.asarray(bits, dtype=np.uint8)
    if mu <= 0 or bits.size == 0:
        return bits.copy()
    flips = (rng.random(bits.size) < mu).astype(np.uint8)
    return np.bitwise_xor(bits, flips)


def pack_metadata(m_map: np.ndarray, umin: float, umax: float, m_max: int) -> np.ndarray:
    """M_i 图 + umin/umax -> 定长比特序列。"""
    bpm = _bits_per_m(m_max)
    parts = [index_to_bits(int(m), bpm) for m in np.asarray(m_map).ravel()]
    lo = int(uniform_quantize(np.array([umin]), META_BITS, META_LO, META_HI)[0])
    hi = int(uniform_quantize(np.array([umax]), META_BITS, META_LO, META_HI)[0])
    parts.append(index_to_bits(lo, META_BITS))
    parts.append(index_to_bits(hi, META_BITS))
    return np.concatenate(parts).astype(np.uint8)


def unpack_metadata(meta_bits: np.ndarray, n: int, m_max: int):
    """定长比特序列 -> (m_map, umin, umax)。M_i 翻位后 clamp 到 [0, m_max]。"""
    bpm = _bits_per_m(m_max)
    meta_bits = np.asarray(meta_bits, dtype=np.uint8)
    m_map = np.empty(n, dtype=np.int64)
    for i in range(n):
        seg = meta_bits[i * bpm:(i + 1) * bpm]
        m_map[i] = min(bits_to_index(seg), m_max)
    off = n * bpm
    lo_idx = bits_to_index(meta_bits[off:off + META_BITS])
    hi_idx = bits_to_index(meta_bits[off + META_BITS:off + 2 * META_BITS])
    umin = float(uniform_dequantize(np.array([lo_idx]), META_BITS, META_LO, META_HI)[0])
    umax = float(uniform_dequantize(np.array([hi_idx]), META_BITS, META_LO, META_HI)[0])
    return m_map, umin, umax


def apply_channel(
    packet: Packet,
    mu: float,
    metadata_through_channel: bool,
    rng: np.random.Generator,
) -> Packet:
    """对 packet 施加 BSC，返回新的（可能受损的）packet。"""
    payload = bsc(packet.payload_bits, mu, rng)

    if metadata_through_channel and mu > 0:
        meta = pack_metadata(packet.m_map, packet.umin, packet.umax, packet.m_max)
        meta = bsc(meta, mu, rng)
        m_map, umin, umax = unpack_metadata(meta, packet.m_map.shape[0], packet.m_max)
    else:  # 元信息无损直达
        m_map, umin, umax = packet.m_map.copy(), packet.umin, packet.umax

    return Packet(
        payload_bits=payload,
        m_map=m_map,
        umin=umin,
        umax=umax,
        values_per_patch=packet.values_per_patch,
        m_max=packet.m_max,
    )
