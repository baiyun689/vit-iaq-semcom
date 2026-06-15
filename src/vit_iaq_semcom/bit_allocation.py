"""Sec. III-B / III-C — 重要性感知比特分配。

待实现 (Step 4):
- 加权量化误差最小化问题 P1，权重 w(a_i) 为注意力分数的递增函数 (式 9-11)。
- incremental_allocation: 增量分配法，P1 的最优解 (Sec. III-B)。
- water_filling: 注水法，松弛为凸问题后用 KKT 求闭式解 (Theorem 1, 式 16-17)。
"""
