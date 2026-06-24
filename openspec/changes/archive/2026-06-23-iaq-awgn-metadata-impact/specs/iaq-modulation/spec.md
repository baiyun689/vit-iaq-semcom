## ADDED Requirements

### Requirement: BPSK 数字调制与解调

系统 SHALL 提供 BPSK 调制:把比特 \(b \in \{0,1\}\) 映射为单位能量符号 \(s \in \{+1,-1\}\)(约定 \(b=0 \to +1\),\(b=1 \to -1\));并提供硬判决解调:由接收实数 \(r\) 判回比特(\(r > 0 \to 0\),否则 \(1\))。

#### Scenario: 调制解调无噪声往返
- **WHEN** 对任意比特序列先 BPSK 调制、不加噪声、再解调
- **THEN** 解调输出 SHALL 与原比特逐位相等

### Requirement: AWGN 加噪(按 Eb/N0)

系统 SHALL 对 BPSK 符号加独立同分布高斯噪声 \(n \sim \mathcal{N}(0, \sigma^2)\),其中 \(\sigma^2 = 1/(2 \cdot \mathrm{Eb/N0}_{\text{linear}})\),\(\mathrm{Eb/N0}\) 以 dB 输入。给定随机种子 SHALL 可复现。

#### Scenario: 高 SNR 趋于无误码
- **WHEN** Eb/N0 取很大值(如 ≥ 12 dB)
- **THEN** 解调后误码率 SHALL 趋近 0

#### Scenario: 未编码 BER 符合理论
- **WHEN** 对足够长的比特序列在给定 Eb/N0 下做 BPSK→AWGN→解调
- **THEN** 实测误码率 SHALL 近似 \(Q(\sqrt{2 \cdot \mathrm{Eb/N0}_{\text{linear}}})\)(统计容差内)

#### Scenario: 可复现
- **WHEN** 给定相同种子、相同输入与相同 Eb/N0
- **THEN** 两次输出比特 SHALL 完全一致
