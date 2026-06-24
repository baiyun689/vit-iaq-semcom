## Why

上一个 change（iaq-baseline-channel-cliff）用抽象 BSC（自由翻转概率 μ）验证了元信息性能悬崖,但 μ 不对应任何物理量。本 change 把信道升级为**数字传输过高斯（AWGN）信道**——量化比特经 BPSK 调制、加 AWGN 噪声、解调还原——使横轴变为真实 **Eb/N0(SNR)**,并把"调制↔信道↔解调"分层独立,为后续 FEC/UEP 留接口。

实验目标:用三条「准确率 vs SNR」曲线量化 **引入元信息对分类收益的影响**——重要性感知量化（IAQ）相对均匀量化的收益值不值,以及元信息过信道后这份收益被侵蚀多少。后续再研究如何用保护手段缩减影响。

## What Changes

- 新增 **数字调制层**(BPSK):比特→符号、AWGN 加噪(按 Eb/N0)、解调还原比特;未编码 BER SHALL ≈ Q(√(2·Eb/N0))。
- **信道升级为 AWGN**:新增 `awgn_channel`,内部走 调制→AWGN→解调;**保留现有 Packet 与 `metadata_through_channel` 开关**(载荷恒过,元信息按开关)。旧 BSC 函数保留作对照,不删除。
- 新增 **均匀量化基线**:`uniform_allocation` 每 patch 固定 `M = B_target // N`,与 IAQ 同总码率;均匀**无 M_i 图**,仅需 umin/umax——构成"无重要性元信息"的参照系。
- 新增 **三线实验脚本**:扫 Eb/N0,出三条曲线——① 均匀量化 ② IAQ·元信息理想(无损) ③ IAQ·元信息过信道;沿用 `--replot`+JSON,改样式零 GPU。
- 横轴从抽象 μ 改为物理 Eb/N0(dB)。

## Capabilities

### New Capabilities
- `iaq-modulation`: BPSK 数字调制/解调 + AWGN 加噪,按 Eb/N0 把比特流送过高斯信道,未编码 BER 符合理论 Q 函数。

### Modified Capabilities
- `iaq-channel`: 在 BSC 之外新增 AWGN 数字信道路径(经 iaq-modulation),保留元信息开关语义;横轴由 μ 扩展到 Eb/N0。
- `iaq-source-coding`: 比特分配新增 `uniform` 策略(固定 M_i,与增量法同总码率),作为无重要性元信息的对照基线。
- `iaq-pipeline`: 端到端流程支持 AWGN 信道与均匀/IAQ 双量化模式;新增三线「准确率 vs SNR」对照实验(替代/补充原悬崖实验)。

## Impact

- 代码:新增 `src/vit_iaq_semcom/modulation.py`;改 `channel.py`(awgn_channel)、`bit_allocation.py`(uniform_allocation)、`pipeline.py`(信道/量化模式选择);新增 `scripts/run_awgn_metadata_experiment.py`;新增 `tests/test_modulation.py`,扩 `test_channel.py`/`test_bit_allocation.py`/`test_pipeline.py`。
- 配置:`configs/default.yaml` channel 段新增 `type: awgn|bsc`、`ebn0_db`;quantization 段新增 `alloc: incremental|uniform`。
- 依赖:复用 numpy/torch,无新增第三方。
- 下游:FEC/UEP 缩减元信息影响(下一个 change）将构建在 iaq-modulation 分层之上。
- 无 BREAKING:BSC 路径与既有 specs 保留;新增能力与修改均向后兼容。
