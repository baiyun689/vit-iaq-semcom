## ADDED Requirements

### Requirement: 形式化定义
系统 SHALL 给出元信息级联脆弱性的形式化数学定义。设信源被分为 N 个区域，第 i 个区域分配 M_i bit，元信息向量 M = [M_1, ..., M_N]。过噪声信道后收端获得 M̂ = [M̂_1, ..., M̂_N]。载荷码流总长度 L = Σ M_i。收端游标在第 k 个区域前的累积偏移 Δ_k = Σ_{i=1}^{k−1} (M̂_i − M_i)。若 ∃k 使得 Δ_k ≠ 0，则第 k 至 N 区域均从偏移后的错误位置读取比特，定义为**级联失步**。崩塌严重度 = 受影响区域数 / N。

#### Scenario: 形式化定义可读
- **WHEN** 读者阅读 2.2 节的形式化部分
- **THEN** 能理解 M_i 与游标偏移的因果关系，以及"单点差错→全局崩塌"的结构必然性

### Requirement: 崩塌实验（AWGN）
系统 SHALL 呈现 AWGN/BPSK 信道下元信息过信道的三线对比实验：理想（元信息无损）、均匀（无元信息）、Ideal（差异化+裸传元信息，即悬崖基线）。横轴 Eb/N0 (dB)，纵轴 Top-1 Accuracy (%)。SHALL 插入 `outputs/awgn_metadata_impact_b392.png`。数据条件 SHALL 标注 n、b_target、信道类型。

#### Scenario: 三线准确率曲线
- **WHEN** 阅读 2.2 节
- **THEN** 可见三条曲线（理想/均匀/裸传元信息）的准确率 vs Eb/N0 对比，裸传线在低 SNR 区显著低于均匀线

### Requirement: 崩塌实验（SSIM + 接收图对比）
系统 SHALL 呈现低 SNR（0dB / 3dB）下差异化裸传元信息 vs 粗桶的接收图像并排对比（`outputs/cliff_vs_bucket_n256_snr0.png` 或同等图），辅以 SSIM 分布统计：均值、最小值、负值率。SHALL 强调 SSIM 负值（反相关）是级联崩塌区别于普通噪声损伤的诊断特征。

#### Scenario: 结构崩塌可视化
- **WHEN** 读者看到接收图对比
- **THEN** 能直观感知：左侧（差异化裸传）图像空间结构被系统性颠覆，右侧（粗桶）保留结构完整性；SSIM 分布表格量化差异

### Requirement: 传统应对方案分析
系统 SHALL 简述传统通信三类应对方案（CRC+ARQ、UEP/强FEC、固定码率/均匀分配）在语义通信场景下的局限，点明"概率缓解 vs 结构消除"的本质区别。SHALL NOT 展开为独立章节，仅作为脆弱性分析的自然延伸。

#### Scenario: 传统方案对比
- **WHEN** 阅读 2.2 节末尾
- **THEN** 理解为何概率性方案不能从根本上解决结构性矛盾，引出粗桶方案的结构性免疫思路
