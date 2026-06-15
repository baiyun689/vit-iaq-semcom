"""Sec. II — ViT encoder 与注意力分数提取。

待实现 (Step 2):
- 加载预训练 ViT (timm)，作为 device 端轻量 encoder / server 端复杂 encoder。
- 前向得到分类结果。
- 提取最后一层 class-token 对各 patch 的多头注意力分数，多头求平均得到
  每个 patch 的重要性 a_i (论文式 3-4)。
"""
