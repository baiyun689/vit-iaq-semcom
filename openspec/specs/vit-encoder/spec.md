# vit-encoder Specification

## Purpose

提供基于预训练 ViT 的图像编码与分类能力，并支持从最后一层多头自注意力中提取 class-token→patch 注意力分数及其可视化，作为 IAQ 语义通信中语义重要性度量的基础。

## Requirements

### Requirement: 加载预训练 ViT encoder
系统 SHALL 通过 `timm` 加载预训练 ViT 模型，并支持以 device 端轻量 encoder 或 server 端复杂 encoder 两种角色实例化。模型名、是否预训练、device（cpu/cuda）SHALL 可经配置指定。CIFAR-100 默认权重 SHALL 为 `hf_hub:edadaltocg/vit_base_patch16_224_in21k_ft_cifar100`。

#### Scenario: 按默认配置加载 CIFAR-100 ViT
- **WHEN** 以默认配置（`vit_base_patch16_224` / CIFAR-100 权重）实例化 encoder
- **THEN** 返回一个处于 eval 模式的 ViT 模型，patch 数为 196（14×14）、分类头输出 100 类

#### Scenario: 指定设备
- **WHEN** 配置 `device: cuda` 且 CUDA 可用
- **THEN** 模型参数位于 CUDA 设备上；当 CUDA 不可用时回退到 cpu 而非报错

### Requirement: 图像分类前向
系统 SHALL 对输入图像批次执行 ViT 前向，输出每张图的分类 logits，并能据此给出 top-1 预测类别。输入图像 SHALL 按模型期望的分辨率（224×224）与归一化预处理。

#### Scenario: 单批图像前向
- **WHEN** 输入形状为 `(B, 3, 224, 224)` 的预处理图像批次
- **THEN** 返回形状为 `(B, num_classes)` 的 logits，且 `argmax` 给出每张图的预测类别

### Requirement: 提取 class-token→patch 注意力分数
系统 SHALL 从 ViT 最后一层多头自注意力中提取 class-token（query）对各 patch（key）的注意力权重，对多头求平均，得到每个 patch 的重要性分数 \(a_i\)（论文式 3-4）。输出 SHALL 排除 class-token 自身，仅保留 patch 维度，且每张图的 \(a_i\) 在 patch 维度上非负、求和为 1（注意力概率分布）。

#### Scenario: 提取单图注意力分数
- **WHEN** 对一张 224×224 图像执行带注意力提取的前向
- **THEN** 返回长度为 196 的非负向量 \(a_i\)，其和约等于 1（数值容差内）

#### Scenario: 多头平均
- **WHEN** 模型最后一层有 H 个注意力头
- **THEN** 输出的 \(a_i\) 等于 H 个头各自 class-token→patch 注意力的算术平均

### Requirement: 注意力热图可视化
系统 SHALL 提供将 patch 重要性 \(a_i\) 重排为 14×14 网格并叠加到原图上的可视化能力，用于复现论文 Fig.3。

#### Scenario: 生成注意力热图
- **WHEN** 给定一张输入图像与其 \(a_i\)
- **THEN** 生成一张将 \(a_i\) 上采样到原图尺寸的热图叠加图像，重要 patch 区域高亮
