"""前向纠错砌块 — 重复码（spec: iaq-fec）。

最朴素的 FEC：每个信息比特重复 r 次（r 为奇数），解码用多数表决。速率 1/r，
误码率有二项尾闭式，便于在「同信道带宽」对照口径下精确分配预算。

用途：
- A1 基础层保护（命门，逐 bit 无误则增强层自动对齐，见 iaq-a1-metadata-free）；
- EEP/UEP 对照基线（对元信息 / 载荷施加不同强度保护）。

不上 LDPC/Turbo：本课题只需一个预算可闭式计算的对照砌块。
"""

from __future__ import annotations

from math import comb

import numpy as np


def repetition_encode(bits: np.ndarray, r: int) -> np.ndarray:
    """以速率 1/r 编码：每个信息比特重复 r 次。输出长度 = len(bits)·r。"""
    if r < 1:
        raise ValueError(f"重复次数 r 须 ≥ 1，得到 {r}")
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return bits.copy()
    return np.repeat(bits, r).astype(np.uint8)


def repetition_decode(bits: np.ndarray, r: int) -> np.ndarray:
    """多数表决解码：把 r 个副本一组取多数（r 奇数无平票）。"""
    if r < 1:
        raise ValueError(f"重复次数 r 须 ≥ 1，得到 {r}")
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return bits.copy()
    if bits.size % r != 0:
        raise ValueError(f"比特数 {bits.size} 不是 r={r} 的整数倍")
    groups = bits.reshape(-1, r)
    # 多数：每组 1 的个数是否超过半数（r 奇数，阈值 r/2）。
    return (groups.sum(axis=1) * 2 > r).astype(np.uint8)


def repetition_decoded_ber(p: float, r: int) -> float:
    """给定链路翻转概率 p 与重复次数 r，返回多数表决后信息比特误码率。

    误码当 r 个独立副本中被翻转数 k ≥ ⌈r/2⌉（r 奇数即 (r+1)/2）。
    闭式 = Σ_{k=⌈r/2⌉}^{r} C(r,k) p^k (1−p)^{r−k}。
    """
    if r < 1:
        raise ValueError(f"重复次数 r 须 ≥ 1，得到 {r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"翻转概率 p 须在 [0,1]，得到 {p}")
    k_min = (r + 1) // 2  # ⌈r/2⌉
    return float(
        sum(comb(r, k) * p**k * (1.0 - p) ** (r - k) for k in range(k_min, r + 1))
    )
