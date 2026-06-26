## ADDED Requirements

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
