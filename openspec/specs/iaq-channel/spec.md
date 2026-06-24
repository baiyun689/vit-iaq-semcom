# iaq-channel Specification

## Purpose

对 IAQ 语义通信的比特流建模二进制对称信道（BSC，论文 Sec. IV）。载荷比特恒过信道；元信息（\(M_i\) 图、\(u_{min}/u_{max}\)）是否过信道由开关控制，用以对照论文「元信息无损」的乐观假设与「元信息也过信道」导致的性能悬崖。

## Requirements

### Requirement: BSC 比特翻转信道

系统 SHALL 提供二进制对称信道（BSC）：以翻转概率 \(\mu \in [0, 0.5]\) 独立翻转输入比特序列的每一位（论文 Sec. IV，Assumption 1）。\(\mu = 0\) 时输出 SHALL 与输入完全一致。

#### Scenario: 零误码无失真
- **WHEN** \(\mu = 0\)
- **THEN** 输出比特序列 SHALL 与输入逐位相等

#### Scenario: 翻转比例近似 mu
- **WHEN** 对足够长的比特序列施加 \(\mu > 0\) 的 BSC
- **THEN** 实际翻转比特比例 SHALL 在 \(\mu\) 附近（统计容差内）

#### Scenario: 可复现
- **WHEN** 给定相同随机种子与相同输入
- **THEN** 两次 BSC 输出 SHALL 完全一致

### Requirement: 元信息是否过信道开关

系统 SHALL 提供 `metadata_through_channel` 开关。关闭时（默认复现论文）仅载荷比特过 BSC，元信息（\(M_i\) 图、\(u_{min}/u_{max}\)）无损直达；开启时元信息也过同一 \(\mu\) 的 BSC。

#### Scenario: 关闭时元信息无损
- **WHEN** `metadata_through_channel = false` 且 \(\mu > 0\)
- **THEN** 接收端拿到的 \(M_i\) 图与 \(u_{min}/u_{max}\) SHALL 与发送端完全一致，仅载荷比特可能被翻转

#### Scenario: 开启时元信息可被翻转
- **WHEN** `metadata_through_channel = true` 且 \(\mu > 0\)
- **THEN** 接收端的 \(M_i\) 图或 \(u_{min}/u_{max}\) SHALL 可能与发送端不同

#### Scenario: M_i 被翻导致载荷切分错位
- **WHEN** 元信息过信道且某 patch 的 \(M_i\) 值被翻转为 \(M_i'\)
- **THEN** 接收端按 \(M_i'\) 切分载荷 SHALL 导致该 patch 之后的载荷比特发生错位（失步），体现「整图崩」机制

### Requirement: AWGN 数字信道

系统 SHALL 提供 AWGN 数字信道 `awgn_channel`：载荷与（按开关的）元信息比特经 BPSK 调制 → AWGN 加噪（按 Eb/N0）→ 解调还原比特。SHALL 复用现有 Packet 结构与 `metadata_through_channel` 开关语义（载荷恒过信道；元信息仅在开关开启时过同一 Eb/N0 的信道）。原 BSC 路径 SHALL 保留以作对照。

#### Scenario: 高 SNR 端到端近无损
- **WHEN** Eb/N0 取很大值
- **THEN** 接收 packet 的载荷与元信息 SHALL 近似与发送一致

#### Scenario: 元信息开关在 AWGN 下同样生效
- **WHEN** `metadata_through_channel = false`，即使 Eb/N0 较低
- **THEN** 接收端 \(M_i\) 图与 \(u_{min}/u_{max}\) SHALL 与发送端完全一致，仅载荷可能误码

#### Scenario: 横轴为物理 SNR
- **WHEN** 以 Eb/N0(dB)为自变量扫描
- **THEN** 信道误码强度 SHALL 由 Eb/N0 决定，而非抽象翻转概率
