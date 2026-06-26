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

### Requirement: 可选信道与量化模式

系统 SHALL 允许端到端流程选择信道类型（`awgn` | `bsc`）与比特分配策略（`incremental` | `uniform`），其余编排不变。AWGN 模式下以 Eb/N0 为信道参数。

#### Scenario: AWGN + IAQ 端到端
- **WHEN** 选择 `awgn` 信道 + `incremental` 分配，在高 Eb/N0 下运行
- **THEN** pipeline SHALL 输出 \((B,100)\) logits 与 Top-1 预测，准确率接近无量化基线

#### Scenario: AWGN + 均匀端到端
- **WHEN** 选择 `awgn` 信道 + `uniform` 分配
- **THEN** pipeline SHALL 正常输出分类结果，且不传输 \(M_i\) 图（均匀分配常数已知）

### Requirement: AWGN 三线元信息影响实验

系统 SHALL 提供脚本，扫一组 Eb/N0，产出三条「准确率 vs SNR」曲线：① 均匀量化 ② IAQ·元信息理想（无损） ③ IAQ·元信息过信道，并保存图与结果 JSON（支持 `--replot` 零 GPU 重画）。

#### Scenario: 产出三条对照曲线
- **WHEN** 运行三线实验脚本
- **THEN** 输出 SHALL 含三条「准确率 vs Eb/N0」曲线并保存图片与 JSON 到 `outputs/`

#### Scenario: 可量化收益与元信息影响
- **WHEN** 比较三条曲线
- **THEN** 「②−①」SHALL 体现重要性感知的收益，「②−③」SHALL 体现元信息过信道对收益的侵蚀，且二者可在同一 SNR 轴上读出

### Requirement: A1 链路与 EEP/UEP 对照分支

`IAQPipeline` SHALL 支持在同一工作点运行多种元信息处理模式：A1(元信息-free,派生分配)、
EEP(元信息与载荷等强 FEC)、UEP(元信息强 FEC、载荷弱 FEC),以及既有的裸传/理想/均匀模式。
各模式 SHALL 共享同一 ViT、量化器与信道,仅在元信息处理与 FEC 分配上不同。

#### Scenario: A1 模式端到端跑通
- **WHEN** 以 A1 模式对一批图编码→过 AWGN+BPSK 信道→解码→分类
- **THEN** 输出 SHALL 含 top-1 预测与准确率,且全程不传输 \(M_i\) 图

#### Scenario: EEP/UEP 模式传输并保护元信息
- **WHEN** 以 EEP 或 UEP 模式编码
- **THEN** 元信息 SHALL 按所选速率重复码保护后随载荷一同过信道

### Requirement: 同 ρ 六线对照实验

系统 SHALL 提供六线对照实验:在固定 ρ=0.25(b392)下扫 SNR,输出六条准确率曲线——
① 均匀 ② IAQ 理想 ③ 裸传 ④ EEP ⑤ UEP ⑥ A1。六线 SHALL 对齐**同信道带宽**(FEC 膨胀
计入总预算)。结果 SHALL 写入 JSON 供复现与绘图。

#### Scenario: 图1 匹配场景
- **WHEN** 每个 SNR 都按当前 SNR 设计 FEC 保护强度
- **THEN** ④⑤⑥ SHALL 在高 SNR 收敛到 ② 附近("都能修")

#### Scenario: 图2 失配场景(杀招)
- **WHEN** 按固定设计点(如 8dB)选定 FEC 强度后再扫真实 SNR
- **THEN** 真实 SNR 低于设计点时 ⑤ UEP SHALL 出现再悬崖,而 ⑥ A1 曲线 SHALL 保持有底平滑
  (不低于 ① 均匀)

#### Scenario: 结果可复现
- **WHEN** 给定相同种子、样本数与 SNR 列表重复运行
- **THEN** 两次输出曲线 SHALL 一致

### Requirement: 1 bit 基础层注意力一致性验证

系统 SHALL 提供验证实验:对同一批图分别用「原图」与「1 bit 基础层重建图」计算注意力 \(a_i\),
度量两者派生比特分配的一致性(如排序相关/重叠度)与最终准确率差异,用于判定 A1 走零训练还是
退路。该实验 SHALL 可在 CPU 上以小样本运行。

#### Scenario: 输出一致性度量
- **WHEN** 对一批图运行该验证实验
- **THEN** SHALL 输出原图与基础层重建图两者派生分配的一致性指标及准确率差

