"""Sec. IV — 真实数字通信信道建模。

待实现 (Step 6):
- 将 调制->衰落->均衡->检测->解调 等效为并行 BSC，翻转概率 mu (Sec. IV-A)。
- 同时考虑量化误差 + 通信误码的失真 D(Q_i; mu) (式 49)。
- 据此改造 incremental / water-filling 比特分配 (Sec. IV-B/C)。
"""
