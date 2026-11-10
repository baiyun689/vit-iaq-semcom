## ADDED Requirements

### Requirement: 实验系统描述
系统 SHALL 用连贯段落描述实验验证平台的整体架构：ViT 编码器提取注意力 → 增量比特分配差异化分配比特 → 均匀量化器 pack/unpack → 数字信道传输（BSC/AWGN/BPSK）→ 收端反量化重建 → 冻结分类器分类。SHALL NOT 逐行罗列代码文件名，而应描述各模块的功能角色与模块间数据流。

#### Scenario: 完整管线描述
- **WHEN** 读者阅读 2.1 节
- **THEN** 能从文字中理解输入图像如何经注意力提取→比特分配→量化→信道→反量化→分类的完整流程

### Requirement: 干净信道基线结果
系统 SHALL 呈现干净信道（无误码）下的差异化分配 vs 均匀分配对比，证明显式重要性感知路径的性能潜力。数据点 SHALL 标注 b_target 与 n 值。

#### Scenario: 基线对比
- **WHEN** 呈现干净信道结果
- **THEN** 给出差异化分配 vs 均匀分配的 Top-1 准确率差异（≈14pt），并说明该增益来自重要性差异化而非编码器改进

### Requirement: 注意力热图
系统 SHALL 插入 ViT 注意力热图（`outputs/attention_fig3.png`），说明"重要性从哪来"，使读者直观理解注意力分数如何体现 patch 对分类的贡献差异。

#### Scenario: 热图展示
- **WHEN** 阅读 2.1 节
- **THEN** 可见一张原图-注意力热图并排对比，标注所用 ViT 模型与输入图像来源
