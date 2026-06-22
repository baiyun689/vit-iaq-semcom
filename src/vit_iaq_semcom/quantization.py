"""Sec. III-A — patch-wise 均匀量化器。

对每个 patch 按给定比特数 M_i 做均匀标量量化（式 5），并提供量化索引 <-> 比特
序列互转（式 6）。量化作用在 ViT 输入张量上（已归一化），范围由 [umin, umax] 给定。

约定：
- M_i >= 1：把 [umin, umax] 均分为 2^M_i 个区间，索引取区间号，反量化取区间中心。
- M_i == 0：不产生载荷比特，反量化以区间中点 (umin+umax)/2 填充。
"""

from __future__ import annotations

import numpy as np


def uniform_quantize(
    values: np.ndarray, m: int, umin: float, umax: float
) -> np.ndarray:
    """均匀量化为整数索引（式 5）。

    返回与 ``values`` 同形状的 int 数组，取值 [0, 2^m - 1]。m==0 时全为 0。
    """
    values = np.asarray(values, dtype=np.float64)
    if m <= 0:
        return np.zeros(values.shape, dtype=np.int64)
    levels = 1 << m  # 2^m
    rng = umax - umin
    if rng <= 0:  # 常数 patch：退化为全 0
        return np.zeros(values.shape, dtype=np.int64)
    scaled = (values - umin) / rng * levels
    idx = np.floor(scaled).astype(np.int64)
    return np.clip(idx, 0, levels - 1)


def uniform_dequantize(
    idx: np.ndarray, m: int, umin: float, umax: float
) -> np.ndarray:
    """索引 -> 重建值，取区间中心。m==0 时返回区间中点。"""
    rng = umax - umin
    if m <= 0 or rng <= 0:
        mid = umin + rng / 2.0
        return np.full(np.shape(idx), mid, dtype=np.float64)
    levels = 1 << m
    idx = np.asarray(idx, dtype=np.float64)
    return umin + (idx + 0.5) * rng / levels


def index_to_bits(idx: int, m: int) -> np.ndarray:
    """整数索引 -> 长度 m 的比特序列（MSB 优先，自然二进制）。m==0 返回空。"""
    if m <= 0:
        return np.zeros(0, dtype=np.uint8)
    idx = int(idx)
    bits = [(idx >> (m - 1 - k)) & 1 for k in range(m)]
    return np.asarray(bits, dtype=np.uint8)


def bits_to_index(bits: np.ndarray) -> int:
    """长度 m 的比特序列（MSB 优先）-> 整数索引。空序列返回 0。"""
    idx = 0
    for b in np.asarray(bits, dtype=np.uint8).tolist():
        idx = (idx << 1) | int(b)
    return idx


def quantize_to_bitstream(
    values: np.ndarray, m: int, umin: float, umax: float
) -> np.ndarray:
    """量化一组值并展平为比特流（每值 m 比特，MSB 优先）。m==0 返回空。"""
    idx = uniform_quantize(values, m, umin, umax).ravel()
    if m <= 0 or idx.size == 0:
        return np.zeros(0, dtype=np.uint8)
    # 向量化：对每个索引取第 k 位（MSB 优先）
    shifts = (m - 1 - np.arange(m)).reshape(1, m)
    bits = ((idx.reshape(-1, 1) >> shifts) & 1).astype(np.uint8)
    return bits.ravel()


def bitstream_to_indices(bits: np.ndarray, m: int, count: int) -> np.ndarray:
    """比特流 -> count 个索引（每 m 比特一个，MSB 优先）。

    不足部分（信道失步导致比特不够）以 0 索引补齐，保证流程不崩溃。
    """
    if m <= 0:
        return np.zeros(count, dtype=np.int64)
    bits = np.asarray(bits, dtype=np.int64).ravel()
    need = count * m
    if bits.size < need:  # 失步：补 0
        bits = np.concatenate([bits, np.zeros(need - bits.size, dtype=np.int64)])
    bits = bits[:need].reshape(count, m)
    weights = (1 << (m - 1 - np.arange(m))).reshape(m, 1)
    return (bits @ weights).ravel()
