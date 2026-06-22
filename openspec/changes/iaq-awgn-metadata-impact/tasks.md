## 1. 配置

- [x] 1.1 `configs/default.yaml` channel 段新增 `type: awgn`(默认)、`ebn0_db`;保留 `bsc_flip_prob` 供 bsc 路径
- [x] 1.2 `configs/default.yaml` quantization 段新增 `alloc: incremental`(默认),可选 `uniform`

## 2. 数字调制层（modulation.py，新）

- [x] 2.1 `bpsk_modulate(bits)`：b∈{0,1}→s∈{+1,−1}（b=0→+1）
- [x] 2.2 `bpsk_demodulate(r)`：硬判决 r>0→0
- [x] 2.3 `awgn(symbols, ebn0_db, rng)`：加 σ²=1/(2·Eb/N0_lin) 高斯噪声，种子可复现
- [x] 2.4 `tests/test_modulation.py`：无噪往返无损、高 SNR 近无误码、未编码 BER≈Q(√(2·Eb/N0))、种子复现

## 3. AWGN 信道（channel.py，改）

- [x] 3.1 `awgn_channel(packet, ebn0_db, metadata_through_channel, rng)`：载荷恒过 调制→AWGN→解调；元信息按开关过同一 Eb/N0
- [x] 3.2 保留 bsc 路径不动；元信息 M_i 翻位仍 clamp [0,m_max]
- [x] 3.3 扩 `tests/test_channel.py`：高 Eb/N0 近无损、开关关闭时元信息逐位不变

## 4. 均匀量化基线（bit_allocation.py，改）

- [x] 4.1 `uniform_allocation(n, b_target, m_max)`：每 patch M=⌊B_target/N⌋（与 a_i 无关）
- [x] 4.2 `allocate` 分发支持 `strategy="uniform"`
- [x] 4.3 扩 `tests/test_bit_allocation.py`：全 patch 等比特、与增量法同总码率（差≤取整余数）

## 5. pipeline 信道/量化模式选择（pipeline.py，改）

- [x] 5.1 `run_batch` 支持 `channel_type` + `ebn0_db`（awgn|bsc）与 `alloc`（incremental|uniform）
- [x] 5.2 均匀模式下不打包/不传 M_i 图（常数已知）
- [x] 5.3 扩 `tests/test_pipeline.py`：AWGN 高 SNR 近无损分类、均匀模式正常出分类

## 6. 三线实验与验证

- [x] 6.1 新增 `scripts/run_awgn_metadata_experiment.py`：扫 Eb/N0，跑三线（均匀 / IAQ 元信息理想 / IAQ 元信息过信道），结果存 JSON
- [x] 6.2 画三条「准确率 vs Eb/N0」曲线，保存 `outputs/awgn_metadata_impact.png`；支持 `--replot`（零 GPU）
- [ ] 6.3 与用户约定样本数/时长后跑一次（默认 n=128），肉眼核对：高 SNR 三线收敛、低 SNR 拉开、③ 相对 ② 的侵蚀可见
- [x] 6.4 `conda run -n vit-iaq-semcom pytest -m "not network"` 全绿
- [x] 6.5 更新 README 路线图（Step 6 信道：AWGN 三线元信息影响）
