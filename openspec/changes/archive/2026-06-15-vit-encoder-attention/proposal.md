## Why

论文 IAQ 方法的整个比特分配链条都建立在「每个 patch 的重要性 \(a_i\)」之上，而 \(a_i\) 来自预训练 ViT 最后一层 class-token 对各 patch 的注意力分数（论文式 3-4）。没有这一步，后续的量化（Sec. III-A）和重要性感知比特分配（Sec. III-B/C）都无从谈起。Step 2 提供这个上游能力，并通过复现论文 Fig.3 的注意力热图来验证提取逻辑正确。

## What Changes

- 新增 ViT encoder 加载能力：基于 `timm` 加载预训练 ViT，支持「device 端轻量 encoder / server 端复杂 encoder」两种角色（论文图 1）。
- CIFAR-100 默认使用 timm 原生权重 `hf_hub:edadaltocg/vit_base_patch16_224_in21k_ft_cifar100`（93.2% top-1，224×224，与现有 timm pipeline 直接对接）；ImageNet 场景使用 `vit_base_patch16_224.augreg2_in21k_ft_in1k`。
- 实现前向分类：输入图像 → ViT → logits / 预测类别。
- 实现 class-token → patch 注意力分数提取：取最后一层多头注意力中 class-token（query）对各 patch（key）的权重，多头求平均，得到每个 patch 的重要性 \(a_i\)（式 3-4）。
- 新增注意力热图可视化与 Fig.3 复现脚本，作为提取正确性的验证手段。
- 更新 `configs/default.yaml`，落地具体权重 id 与 encoder 角色配置。

## Capabilities

### New Capabilities
- `vit-encoder`: 加载预训练 ViT、执行图像分类、提取 class-token→patch 注意力分数 \(a_i\)，并提供注意力热图可视化。

### Modified Capabilities
<!-- 无：openspec/specs/ 当前为空，本 change 不修改已有 spec 行为 -->

## Impact

- 代码：`src/vit_iaq_semcom/vit_encoder.py`（实现）、`configs/default.yaml`（权重 id 与角色）、新增可视化脚本（`scripts/` 或 `notebooks/`）、新增 `tests/test_vit_encoder.py`。
- 依赖：`timm`、`torch`、`torchvision`（CIFAR-100 加载）；需联网下载 HF 权重与数据集。
- 下游：`quantization.py`（Step 3）、`bit_allocation.py`（Step 4）将消费本步骤输出的 \(a_i\) 与 patch 表示。
- 无 BREAKING：此前 `vit_encoder.py` 仅为占位文档。
