# iaq-pipeline Specification

## Purpose

编排 IAQ 语义通信的端到端流程（device → channel → server，论文 Sec. II 图 1），以降质重建图像的下游分类准确率为唯一指标，并提供扫 BSC 翻转概率 \(\mu\) 的元信息性能悬崖对照实验（创新点 A 的 E1/E2）。

## Requirements

### Requirement: 端到端 IAQ 编排

系统 SHALL 提供端到端流程：device（图像 → ViT 注意力 \(a_i\) → 增量比特分配 \(M_i\) → patch 均匀量化 → 载荷比特流 + 元信息）→ channel（BSC）→ server（反量化 → 重建图像 → ViT 分类 → Top-1 预测）。分类器 SHALL 使用现成 CIFAR-100 微调权重纯推理，本流程不训练任何网络。

#### Scenario: 干净信道端到端可分类
- **WHEN** \(\mu = 0\) 且预算 \(B_{target}\) 足够大
- **THEN** pipeline SHALL 对一个 batch 输出形状 \((B, 100)\) 的 logits 与 Top-1 预测，且准确率接近无量化时的分类基线

#### Scenario: 预算降低准确率单调不升
- **WHEN** 在 \(\mu = 0\) 下逐步降低 \(B_{target}\)
- **THEN** 分类准确率 SHALL 总体不随预算降低而上升（率失真趋势）

### Requirement: 元信息性能悬崖实验

系统 SHALL 提供脚本，在固定 \(B_{target}\) 下扫一组 BSC 翻转概率 \(\mu\)，分别在 `metadata_through_channel` 关闭与开启两种模式下测分类准确率，并产出对照曲线（创新点 A 的 E1/E2）。

#### Scenario: 产出两条对照曲线
- **WHEN** 运行悬崖实验脚本
- **THEN** 输出 SHALL 包含「元信息无损」与「元信息过信道」两条「准确率 vs \(\mu\)」曲线，并保存为图片到 `outputs/`

#### Scenario: 元信息过信道呈现悬崖
- **WHEN** \(\mu\) 增大到非零量级
- **THEN** 「元信息过信道」曲线的准确率 SHALL 比「元信息无损」曲线下降显著更快（暴露论文乐观假设）
