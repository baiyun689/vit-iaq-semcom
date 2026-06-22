"""数字调制层 — BPSK + AWGN（论文 Sec. IV 的物理化信道）。

把量化比特经 BPSK 调制成 ±1 符号、加按 Eb/N0 标定的高斯噪声、再硬判决解调还原
比特。横轴由抽象翻转概率 μ 升级为物理 Eb/N0(dB)。

约定：b=0 → +1，b=1 → −1（单位符号能量，Eb=1）。未编码 BER 闭式为
Q(√(2·Eb/N0))，作为实现正确性的金标准（见 tests）。

分层独立（调制 / 加噪 / 解调三个纯函数）是为后续在"调制↔信道"之间插 FEC/UEP
编码器留接口（缩减元信息影响那一步）。
"""

from __future__ import annotations

import numpy as np


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """比特 -> BPSK 符号：b=0→+1，b=1→−1。"""
    bits = np.asarray(bits, dtype=np.int8)
    return (1 - 2 * bits).astype(np.float64)


def awgn(symbols: np.ndarray, ebn0_db: float, rng: np.random.Generator) -> np.ndarray:
    """对符号加高斯噪声，σ² = 1/(2·Eb/N0_lin)。ebn0_db=inf 时无噪。"""
    symbols = np.asarray(symbols, dtype=np.float64)
    if np.isinf(ebn0_db):
        return symbols.copy()
    ebn0_lin = 10.0 ** (ebn0_db / 10.0)
    sigma = np.sqrt(1.0 / (2.0 * ebn0_lin))
    return symbols + rng.normal(0.0, sigma, size=symbols.shape)


def bpsk_demodulate(r: np.ndarray) -> np.ndarray:
    """硬判决解调：r<0 → 1，否则 0（与 b=0→+1 约定一致）。"""
    r = np.asarray(r, dtype=np.float64)
    return (r < 0).astype(np.uint8)


def bpsk_awgn(bits: np.ndarray, ebn0_db: float, rng: np.random.Generator) -> np.ndarray:
    """便捷组合：比特 → BPSK → AWGN → 解调 → 比特。"""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return bits.copy()
    return bpsk_demodulate(awgn(bpsk_modulate(bits), ebn0_db, rng))
