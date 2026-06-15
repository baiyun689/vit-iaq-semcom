"""端到端 IAQ 流程编排 (Sec. II 图 1)。

待实现 (Step 5+):
device: 图像 -> ViT 注意力 -> 重要性 a_i -> 比特分配 M_i -> patch 量化 -> 比特流 b
channel: b -> (BSC) -> b_hat
server: b_hat -> 反量化 -> 重建图像 -> ViT 分类
"""
