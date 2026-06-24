## Context

iaq-baseline-channel-cliff 已用抽象 BSC 验证元信息悬崖,但 μ 无物理意义。本 change 把信道换成数字传输过 AWGN(BPSK 起步),横轴变 Eb/N0,并把调制/信道/解调分层,为后续 FEC/UEP 留接口。现有模块(quantization/bit_allocation/channel/pipeline/ViTEncoder)直接复用,只在信道段与分配段做加法。

约束:conda env `vit-iaq-semcom`;**用户机器持续 GPU 满载会过热关机**,三线实验比之前更吃 GPU,须默认小样本、可调 batch、结果落盘 + `--replot`。

## Goals / Non-Goals

**Goals:**
- BPSK + AWGN 数字链路,未编码 BER ≈ Q(√(2·Eb/N0))(有理论自检)。
- 保留 Packet 与 `metadata_through_channel` 开关;BSC 路径不删。
- 均匀量化基线(同总码率,无 M_i 图)作"无重要性元信息"参照。
- 三线「准确率 vs Eb/N0」:均匀 / IAQ 元信息理想 / IAQ 元信息过信道,量化收益与元信息影响。

**Non-Goals:**
- FEC / 纠错码 / 元信息 UEP 保护(下一个 change,即"缩减影响")。
- QAM 高阶调制、衰落/多径信道、软判决/LLR。
- 重训分类器或在降质数据上微调。
- 把 Eb/N0 严格对齐论文 Sec.IV 的失真公式 D(Q_i;μ)。

## Decisions

**D1 — BPSK 起步,硬判决。** 约定 b=0→+1, b=1→−1;AWGN σ²=1/(2·Eb/N0_lin)(单位符号能量,Eb=1);解调 r>0→0。未编码 BER 闭式 = Q(√(2·Eb/N0)),作单测金标准。QAM/软判决留后续。

**D2 — 新增 `modulation.py` 分层,而非把 AWGN 塞进 bsc()。** 调制/加噪/解调是独立纯函数;`channel.awgn_channel` 组合它们。理由:后续 FEC/UEP 要在"调制↔信道"之间插编码器,分层不返工。代价:多一个小模块(可接受)。

**D3 — awgn_channel 复用 Packet + 元信息开关。** 与 bsc 版同构:载荷比特恒过 调制→AWGN→解调;`metadata_through_channel` 开启时元信息(打包成比特)也过同一 Eb/N0 链路再解包。M_i 翻位后仍 clamp 到 [0,m_max](沿用既有失步语义)。

**D4 — 均匀基线同总码率对齐。** `uniform_allocation` 取 M=⌊B_target/N⌋,使均匀与 IAQ 在同一 payload 码率下比较,差异只来自"比特分配是否按重要性"。均匀的 M_i 是常数、先验已知,**不计入脆弱元信息**(只需 umin/umax),这正是与 IAQ 的关键差异点。

**D5 — 三线的信道处理保持一致可比。** 三条线都过 AWGN 载荷;区别仅:① 均匀分配(无 M_i 图)② IAQ 且元信息无损 ③ IAQ 且元信息过信道。均匀线与 IAQ-理想线的 umin/umax 同等理想处理,保证"收益=②−①"只反映分配策略、不混入元信息脆弱性。

**D6 — 配置开关而非分叉代码。** `channel.type: awgn|bsc`、`channel.ebn0_db`、`quantization.alloc: incremental|uniform`;pipeline 按配置选路,单一代码路径。

## Risks / Trade-offs

- **[AWGN 数字+硬判决本质仍是 BSC(μ=Q(...))]** → 已对用户说明;增量价值在于物理 SNR 轴 + 为 FEC/UEP/调制阶数铺路,而非改变比特出错机制。本 change 接受这一点。
- **[三线更吃 GPU → 过热]** → 默认 n=128、batch 可调、JSON 落盘、`--replot` 零 GPU 重画;跑前与用户约样本数/时长。
- **[均匀 vs IAQ 同码率取整余数不等]** → 余数级差异,记入实验说明,不影响定性结论。
- **[低 Eb/N0 下准确率噪声大]** → 每点足够样本 + 固定种子;曲线追求定性趋势。

## Open Questions

- Eb/N0 扫描范围(先 -4~12 dB 看三线交叉/悬崖位置再细化)。
- B_target 复用上个 change 的 980(同码率对比),还是另挑一档使三线区分更明显——实现时扫一下定。
