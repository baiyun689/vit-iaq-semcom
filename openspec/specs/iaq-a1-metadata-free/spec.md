# iaq-a1-metadata-free Specification

## Purpose
TBD - created by archiving change iaq-a1-metadata-free. Update Purpose after archive.
## Requirements
### Requirement: 基础层/增强层两层切分

系统 SHALL 在总信源预算 \(B_{target}\) 下把比特分成两层：基础层对每个 patch 固定分配
\(M_{base}\) bit 做均匀量化；增强层把剩余预算 \(B_{target} - N \cdot M_{base}\) bit 按派生
分配 \(\Delta M_i\) 增量分配给重要 patch。每 patch 总比特 SHALL 为 \(M_{base} + \Delta M_i\)。
工作点 b392 取 \(M_{base}=1\)（基础层 196 bit + 增强层 196 bit）。

#### Scenario: 两层比特预算守恒
- **WHEN** 给定 \(B_{target}=392\)、\(N=196\)、\(M_{base}=1\)
- **THEN** 基础层 SHALL 占 \(196\) bit，增强层 \(\sum_i \Delta M_i\) SHALL 不超过 \(196\) bit

#### Scenario: 基础层独立可解
- **WHEN** 仅用基础层比特(不含增强层)反量化
- **THEN** SHALL 得到一张完整的粗重建图(每 patch \(M_{base}\) bit 均匀量化)

### Requirement: 两端确定性派生分配

系统 SHALL 在发端与收端各自、独立地从**同一份基础层重建图**派生比特分配：把基础层重建图
喂入共享 ViT 得到注意力 \(a_i\)，量化后喂增量分配算法得到 \(\Delta M_i\)。当两端基础层重建
逐 bit 相同时，两端派生的 \(\Delta M_i\) SHALL 逐 patch 完全相同。系统 SHALL NOT 传输
\(M_i\) 分配图或增强层各段长度。

#### Scenario: 相同基础层派生相同分配
- **WHEN** 发端与收端拿到逐 bit 相同的基础层重建图、用同一份 ViT 权重派生分配
- **THEN** 两端的 \(\Delta M_i\) SHALL 逐 patch 完全相同

#### Scenario: 不传元信息
- **WHEN** A1 链路对一张图编码
- **THEN** 输出比特流 SHALL 只含基础层(含其 FEC)与增强层载荷，SHALL NOT 含 \(M_i\) 图或
  段长度元信息

### Requirement: 注意力量化锁一致性

系统 SHALL 在把注意力 \(a_i\) 送入增量分配算法前，将其量化到固定档数(默认 16 档)。该量化
SHALL 使两端浮点前向的微小数值差异落入同一档，从而保证派生分配确定性一致。

#### Scenario: 微小漂移不改变分配
- **WHEN** 两端 \(a_i\) 仅有小于一个量化档间隔的数值差异
- **THEN** 量化后的 \(a_i\) SHALL 逐 patch 相同，派生 \(\Delta M_i\) SHALL 相同

### Requirement: 增强层边界自动对齐

系统 SHALL 让收端按自己派生的 \(\Delta M_i\) 切分增强层比特流，无需任何显式段长度信息。
当两端 \(\Delta M_i\) 一致时，收端 SHALL 正确还原每个 patch 的增强比特,不发生失步。

#### Scenario: 派生一致则无失步
- **WHEN** 基础层逐 bit 无误、两端 \(\Delta M_i\) 一致
- **THEN** 收端按派生 \(\Delta M_i\) 切分增强层 SHALL 逐 patch 对齐，重建与发端一致

#### Scenario: 有底降级,无灾难崩盘
- **WHEN** 信道极差导致增强层大量误码,但基础层经 FEC 仍逐 bit 正确
- **THEN** A1 重建质量 SHALL 退化到基础层(\(M_{base}\) bit)量级,SHALL NOT 出现裸传式
  失步崩盘(跌到随机猜水平)。诚实声明:该底可能低于同信源预算的 2-bit 均匀基线 ①——
  这是分层把半数比特押在增强层的代价;A1 的价值在消除灾难崩盘与信道失配鲁棒,不在底部绝对值。

### Requirement: 基础层 FEC 保护

系统 SHALL 用重复码(见 iaq-fec)保护基础层比特,使其在目标 SNR 区间逐 bit 正确恢复。
基础层 FEC 的比特开销 SHALL 计入总信道带宽预算(同带宽对照口径)。

#### Scenario: 基础层在目标 SNR 无误恢复
- **WHEN** 在目标 SNR 下用所选速率 \(1/r\) 重复码保护基础层并过 AWGN+BPSK 信道
- **THEN** 解码后基础层比特 SHALL 与发端逐 bit 相同(目标 BER 阈内)

