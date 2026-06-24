## ADDED Requirements

### Requirement: AWGN 数字信道

系统 SHALL 提供 AWGN 数字信道 `awgn_channel`:载荷与(按开关的)元信息比特经 BPSK 调制 → AWGN 加噪(按 Eb/N0)→ 解调还原比特。SHALL 复用现有 Packet 结构与 `metadata_through_channel` 开关语义(载荷恒过信道;元信息仅在开关开启时过同一 Eb/N0 的信道)。原 BSC 路径 SHALL 保留以作对照。

#### Scenario: 高 SNR 端到端近无损
- **WHEN** Eb/N0 取很大值
- **THEN** 接收 packet 的载荷与元信息 SHALL 近似与发送一致

#### Scenario: 元信息开关在 AWGN 下同样生效
- **WHEN** `metadata_through_channel = false`,即使 Eb/N0 较低
- **THEN** 接收端 \(M_i\) 图与 \(u_{min}/u_{max}\) SHALL 与发送端完全一致,仅载荷可能误码

#### Scenario: 横轴为物理 SNR
- **WHEN** 以 Eb/N0(dB)为自变量扫描
- **THEN** 信道误码强度 SHALL 由 Eb/N0 决定,而非抽象翻转概率
