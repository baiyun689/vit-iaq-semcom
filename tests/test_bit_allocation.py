"""bit_allocation.py 单测（spec: iaq-source-coding / 重要性感知增量比特分配）。"""

import numpy as np
import pytest

from vit_iaq_semcom.bit_allocation import (
    allocate,
    incremental_allocation,
    water_filling,
)


def _normed(rng, n):
    a = rng.uniform(0, 1, size=n)
    return a / a.sum()


def test_budget_and_bounds():
    rng = np.random.default_rng(0)
    a = _normed(rng, 196)
    for b in (100, 588, 1568, 100000):
        m = incremental_allocation(a, b_target=b, m_max=8)
        assert m.shape == (196,)
        assert m.sum() <= b
        assert m.min() >= 0 and m.max() <= 8


def test_important_patch_gets_no_fewer_bits():
    """w(a_p) > w(a_q) ⇒ M_p >= M_q（增量收敛后单调）。"""
    rng = np.random.default_rng(1)
    a = _normed(rng, 64)
    m = incremental_allocation(a, b_target=200, m_max=8)
    order = np.argsort(a)  # 升序
    m_sorted = m[order]
    # 重要性升序 → 比特数应非递减
    assert np.all(np.diff(m_sorted) >= 0)


def test_budget_exhausted_stops():
    a = np.ones(10) / 10
    m = incremental_allocation(a, b_target=7, m_max=8)
    assert m.sum() == 7  # 均匀重要性下应恰好用满小预算


def test_budget_capped_by_mmax():
    a = np.ones(4) / 4
    m = incremental_allocation(a, b_target=10_000, m_max=3)
    assert np.all(m == 3)  # 预算远超时全部到 m_max


def test_allocate_dispatch_and_water_filling_placeholder():
    a = np.ones(8) / 8
    m = allocate(a, b_target=16, m_max=8, cfg={"strategy": "incremental"})
    assert m.sum() == 16
    with pytest.raises(NotImplementedError):
        water_filling(a, 16, 8)
