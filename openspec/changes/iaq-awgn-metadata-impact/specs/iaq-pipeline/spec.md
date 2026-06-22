## ADDED Requirements

### Requirement: 可选信道与量化模式

系统 SHALL 允许端到端流程选择信道类型(`awgn` | `bsc`)与比特分配策略(`incremental` | `uniform`),其余编排不变。AWGN 模式下以 Eb/N0 为信道参数。

#### Scenario: AWGN + IAQ 端到端
- **WHEN** 选择 `awgn` 信道 + `incremental` 分配,在高 Eb/N0 下运行
- **THEN** pipeline SHALL 输出 \((B,100)\) logits 与 Top-1 预测,准确率接近无量化基线

#### Scenario: AWGN + 均匀端到端
- **WHEN** 选择 `awgn` 信道 + `uniform` 分配
- **THEN** pipeline SHALL 正常输出分类结果,且不传输 \(M_i\) 图(均匀分配常数已知)

### Requirement: AWGN 三线元信息影响实验

系统 SHALL 提供脚本,扫一组 Eb/N0,产出三条「准确率 vs SNR」曲线:① 均匀量化 ② IAQ·元信息理想(无损) ③ IAQ·元信息过信道,并保存图与结果 JSON(支持 `--replot` 零 GPU 重画)。

#### Scenario: 产出三条对照曲线
- **WHEN** 运行三线实验脚本
- **THEN** 输出 SHALL 含三条「准确率 vs Eb/N0」曲线并保存图片与 JSON 到 `outputs/`

#### Scenario: 可量化收益与元信息影响
- **WHEN** 比较三条曲线
- **THEN** 「②−①」SHALL 体现重要性感知的收益,「②−③」SHALL 体现元信息过信道对收益的侵蚀,且二者可在同一 SNR 轴上读出
