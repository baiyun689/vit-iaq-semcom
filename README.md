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

- [x] **Step 1 — 项目骨架**
- [x] Step 2 — 加载预训练 ViT，提取 class-token→patch 注意力分数，复现 Fig.3 注意力热图
- [x] Step 3 — patch-wise 均匀量化器 + 量化误差验证
- [x] Step 4 — 重要性感知比特分配：增量分配法（注水法留作后续对照）
- [x] Step 5 — 无误码场景端到端：分类精度 vs. 通信开销（干净信道 b_target=980，掉点 <1%）
- [x] Step 6a — BSC 信道建模 + **创新点 A 元信息性能悬崖**（E1/E2，`outputs/cliff_metadata.png`）
- [x] Step 6b — AWGN 数字信道（BPSK）+ **三线元信息影响实验**（均匀/IAQ理想/IAQ过信道，vs Eb/N0）
- [~] Step 6c — 创新点 B 探路「A1 元信息-free」**已证伪**：A1 增强层仍是变长游标，基础层
      错 1 bit 即级联失步；用 KL 蒸馏干净图注意力的微调反而摊平注意力、准确率倒退。
      保留 `src/.../a1.py`、`scripts/run_a1_*.py` 作反面对照与动机（变长+免传+抗误码不可兼得）。
- [ ] Step 6d — **创新点 B（定稿）：非级联粗桶 + 学习式语义修补（AI-centric）**
      （change `iaq-learned-semantic-repair`）：
      - 源编码改定长定位的**粗桶**（均匀基础层 + 固定大小固定比特的重要性桶），保住动态
        重要性分配（底色），信道误码只造成**局部值错配、绝不级联崩**（`src/.../bucket.py`）；
      - 收端从基础层**派生排序**入桶，副产「比特深度=可靠性图」零额外传输；
      - **收端学习式语义修补网络**（创新内核，`src/.../repair.py`）：轻量 patch-Transformer，
        输入受损重建 + 可靠性图，输出残差修正（残差兜底=最坏≈不修），分类任务损失跨 SNR 训；
      - 六线同总信道比特、双场景对照（匹配诚实呈现经典 FEC≈理想 / 失配为杀招），
        `scripts/run_repair_sixline_experiment.py`、训练 `scripts/run_repair_train.py`。
- [ ] Step 6e — 创新点 C：收端 token 级语义错误隐藏（E4，备选）
- [ ] Step 7 — 多视图任务 (MIRO / MVP-N) + 对比基线

## 环境

推荐使用 conda 环境 `vit-iaq-semcom`（Python 3.11, torch 2.11.0+cu128, CUDA）：

```powershell
conda create -n vit-iaq-semcom python=3.11 -y
conda activate vit-iaq-semcom
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

> torch 走 cu128 轮子（最新驱动 CUDA 13.x 向后兼容可用）；无 GPU 时可改用 CPU 版 torch。

## 数据集

- CIFAR-100（单视图分类，torchvision 自动下载）
- MIRO / MVP-N（多视图分类，后续步骤接入）

数据放在 `data/`（已 gitignore）。
