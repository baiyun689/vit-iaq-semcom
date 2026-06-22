"""Sec. III-B / III-C — 重要性感知比特分配。

把注意力重要性 a_i 转成每 patch 的量化比特数 M_i（分配图），在总预算
``b_target``（对 sum_i M_i 的约束）下，用增量法逐比特分配给边际收益最大的 patch。

边际收益模型（式 7-9）：patch i 的加权量化误差 ∝ w(a_i) · 4^{-M_i}；给 i 再加
1 比特使误差按因子 1/4 下降，边际下降 ∝ w(a_i) · 4^{-M_i}。每步贪心选该值最大者，
即 P1 的离散最优解（Sec. III-B）。注水法（water_filling）留作后续对照。
"""

from __future__ import annotations

import heapq

import numpy as np


def weight_fn(a_i: np.ndarray, weight: str = "attention") -> np.ndarray:
    """重要性权重 w(a_i)：注意力分数的递增函数。

    - ``"attention"`` / ``"linear"``：恒等（a_i 已非负且和=1，单调即可）。
    """
    a = np.asarray(a_i, dtype=np.float64)
    if weight in ("attention", "linear"):
        return a
    raise ValueError(f"未知 weight={weight!r}")


def incremental_allocation(
    a_i: np.ndarray, b_target: int, m_max: int, weight: str = "attention"
) -> np.ndarray:
    """增量法分配，返回长度 N 的整数 M_i 图。

    满足 sum_i M_i <= b_target 且 0 <= M_i <= m_max。
    """
    w = weight_fn(a_i, weight)
    n = w.shape[0]
    m = np.zeros(n, dtype=np.int64)
    budget = int(b_target)
    if budget <= 0 or m_max <= 0:
        return m

    # 最大堆（用负值）：键 = w_i · 4^{-M_i}，初始 M_i=0 → 键=w_i
    heap = [(-float(w[i]), i) for i in range(n)]
    heapq.heapify(heap)

    used = 0
    while used < budget and heap:
        _, i = heapq.heappop(heap)
        if m[i] >= m_max:
            continue
        m[i] += 1
        used += 1
        if m[i] < m_max:
            gain = float(w[i]) * (4.0 ** (-int(m[i])))
            heapq.heappush(heap, (-gain, i))
    return m


def water_filling(*_args, **_kwargs):  # noqa: D401 — 占位
    """注水法（Theorem 1，式 16-17）。后续 change 实现，作增量法对照。"""
    raise NotImplementedError("water_filling 留作后续对照，本 change 用 incremental")


def allocate(a_i: np.ndarray, b_target: int, m_max: int, cfg: dict | None = None):
    """按 config 的 strategy 分发。默认 incremental。"""
    cfg = cfg or {}
    strategy = cfg.get("strategy", "incremental")
    weight = cfg.get("weight", "attention")
    if strategy == "incremental":
        return incremental_allocation(a_i, b_target, m_max, weight=weight)
    if strategy == "water_filling":
        return water_filling(a_i, b_target, m_max, weight=weight)
    raise ValueError(f"未知 strategy={strategy!r}")
