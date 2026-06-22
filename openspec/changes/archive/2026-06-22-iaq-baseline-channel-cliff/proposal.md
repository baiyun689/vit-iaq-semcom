## Why

Step 2 已能从 ViT 提取 patch 重要性 \(a_i\)，但量化、比特分配、信道、端到端编排（`quantization.py` / `bit_allocation.py` / `channel.py` / `pipeline.py`）仍是占位文档，导致**任何信道实验都无法仿真验证**。本 change 打通最小可跑链路，并直接产出毕设创新点 A 的招牌结果——**元信息性能悬崖**：论文（Sec. IV）的 BSC 误码只施加在像素载荷上，元信息（\(M_i\) 分配图、\(u_{min}/u_{max}\)）被默认无损到达（Sec. III，error-free）。把元信息也路由进 BSC 即可暴露这个未言明的乐观假设带来的「整图崩」性能悬崖。

## What Changes

- 新增 **patch 均匀量化器**：按每 patch 比特数 \(M_i\) 做均匀标量量化（式 5），并实现量化值 ↔ 比特序列互转（式 6）。
- 新增 **重要性感知比特分配（增量法）**：以注意力权重 \(w(a_i)\) 为序，在总预算 \(B_{target}\) 约束下逐比特分配，产出 \(M_i\) 分配图（Sec. III-B，式 9-11，P1 最优解）。注水法不在本 change 范围。
- 新增 **BSC 信道**：按翻转概率 \(\mu\) 翻转载荷比特；并提供 `metadata_through_channel` 开关——关闭时元信息无损（复现论文），开启时 \(M_i\) 图与 \(u_{min}/u_{max}\) 也过 BSC（暴露悬崖）。
- 新增 **端到端 IAQ pipeline**：device（图像→\(a_i\)→\(M_i\)→量化→比特流）→ channel → server（反量化→重建图像→ViT 分类），输出 Top-1 准确率。
- 新增 **悬崖实验脚本**：扫 BSC 翻转概率 \(\mu\)，画「元信息无损 vs 元信息过信道」两条准确率曲线（创新点 A 的 E1/E2）。
- 提速决策：分类用现成 CIFAR-100 微调权重纯推理，**本 change 不训练任何网络**（量化/分配/信道均为确定性、无可学习参数）。
- 更新 `configs/default.yaml`：填充 `quantization`（`b_target`、`weight`）与 `channel`（`enabled`、`bsc_flip_prob`、`metadata_through_channel`）字段；server 端分类器权重切到 CIFAR-100 微调权重以保证曲线可读。

## Capabilities

### New Capabilities
- `iaq-source-coding`: patch 均匀量化（式 5-6）+ 重要性感知增量比特分配（式 9-11），将图像与 \(a_i\) 编码为载荷比特流 + \(M_i\) 分配图 + \(u_{min}/u_{max}\) 元信息。
- `iaq-channel`: BSC 比特翻转信道（参数 \(\mu\)），含「元信息是否过信道」开关，支持复现论文（元信息无损）与暴露悬崖（元信息过信道）两种模式。
- `iaq-pipeline`: 端到端 device→channel→server 编排，输出降质重建图像的分类准确率；提供扫 \(\mu\) 的悬崖对照实验。

### Modified Capabilities
<!-- 无：本 change 不改变 vit-encoder 的已有 spec 行为，仅作为上游消费其 a_i 输出。 -->

## Impact

- 代码：`src/vit_iaq_semcom/quantization.py`、`bit_allocation.py`、`channel.py`、`pipeline.py`（四处实现）；新增 `scripts/run_cliff_experiment.py`；新增 `tests/test_quantization.py`、`test_bit_allocation.py`、`test_channel.py`、`test_pipeline.py`。
- 配置：`configs/default.yaml` 填充 quantization / channel 字段；server 分类器权重切到 CIFAR-100 微调权重。
- 依赖：复用现有 `timm` / `torch` / `torchvision`；无新增第三方依赖。
- 下游：创新点 A 的 FEC/CRC 修复（E3）与创新点 C 的语义错误隐藏将构建在本 change 的 channel 与 pipeline 之上。
- 无 BREAKING：四个模块此前仅为占位文档。
