# vit-iaq-semcom

复现论文 **《Vision Transformer-based Semantic Communications with Importance-Aware Quantization》**
(J. Park et al., POSTECH, arXiv:2412.06038, 2024)。

核心思想：用**预训练 ViT** 的注意力分数衡量图像各 patch 的重要性，给重要 patch 分配更多量化比特、
不重要 patch 分配更少比特，从而在无线图像传输中以更低的通信开销保住下游分类任务性能——**无需端到端训练**。

## 论文 → 代码模块对照

| 论文章节 | 内容 | 代码模块 |
|---------|------|---------|
| Sec. II | ViT encoder + class-token 注意力分数提取 (式 3-4) | `vit_encoder.py` |
| Sec. III-A | patch-wise 均匀量化器 (式 5-8) | `quantization.py` |
| Sec. III-B | 增量比特分配 (incremental allocation, 求解 P1) | `bit_allocation.py` |
| Sec. III-C | 注水比特分配 (water-filling, Theorem 1) | `bit_allocation.py` |
| Sec. IV | 真实数字通信：并行 BSC 建模 + 误码失真 (式 49) | `channel.py` |
| Sec. V | 端到端流程 + 实验 (CIFAR-100 / MIRO / MVP-N) | `pipeline.py`, `scripts/` |

## 复现路线图（逐步进行）

- [ ] **Step 1 — 项目骨架**（当前）
- [ ] Step 2 — 加载预训练 ViT，提取 class-token→patch 注意力分数，复现 Fig.3 注意力热图
- [ ] Step 3 — patch-wise 均匀量化器 + 量化误差验证
- [ ] Step 4 — 重要性感知比特分配：增量分配法 + 注水法
- [ ] Step 5 — 无误码场景端到端：分类精度 vs. 通信开销曲线
- [ ] Step 6 — BSC 信道建模，扩展到有误码场景
- [ ] Step 7 — 多视图任务 (MIRO / MVP-N) + 对比基线

## 环境

推荐使用 conda 环境 `signal_Gen`（torch 2.6.0+cu118, CUDA）：

```powershell
conda activate signal_Gen
pip install -r requirements.txt
```

## 数据集

- CIFAR-100（单视图分类，torchvision 自动下载）
- MIRO / MVP-N（多视图分类，后续步骤接入）

数据放在 `data/`（已 gitignore）。
