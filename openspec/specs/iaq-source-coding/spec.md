# iaq-source-coding Specification

## Purpose

将图像与其 patch 重要性 a_i 编码为可传输的载荷比特流 + 元信息：对每个 patch 按分得的比特数 M_i 做均匀标量量化（式 5-6），并用增量法在总预算约束下产出重要性感知的 M_i 分配图（式 9-11），作为 IAQ 语义通信发端的信源编码环节。

## Requirements

### Requirement: Patch 均匀量化器

系统 SHALL 对每个 patch 按给定比特数 \(M_i\) 做均匀标量量化（论文式 5），并 SHALL 提供量化值与比特序列的互转（式 6）。量化范围由该 patch（或全局）的 \(u_{min}/u_{max}\) 确定。

#### Scenario: 按 M_i 量化并反量化
- **WHEN** 输入一个取值落在 \([u_{min}, u_{max}]\) 的 patch 与比特数 \(M_i \ge 1\)
- **THEN** 量化→反量化后的最大绝对误差 SHALL 不超过量化步长 \((u_{max}-u_{min})/2^{M_i}\)

#### Scenario: M_i 越大误差越小
- **WHEN** 对同一 patch 分别用 \(M_i\) 与 \(M_i+1\) 比特量化
- **THEN** \(M_i+1\) 的重建均方误差 SHALL 不大于 \(M_i\) 的均方误差

#### Scenario: 量化值与比特序列互转
- **WHEN** 将量化索引编码为长度 \(M_i\) 的比特序列后再解码
- **THEN** 解码得到的索引 SHALL 与原索引完全相等

#### Scenario: M_i = 0 不传该 patch
- **WHEN** 某 patch 分得 \(M_i = 0\) 比特
- **THEN** 该 patch SHALL 不产生载荷比特，反量化时 SHALL 以约定默认值（如区间中点）填充

### Requirement: 重要性感知增量比特分配

系统 SHALL 以注意力权重的递增函数 \(w(a_i)\) 为重要性度量，在总载荷预算 \(B_{target}\) 约束下，用增量法（逐比特分配给当前边际收益最大的 patch，Sec. III-B 式 9-11）产出每个 patch 的比特数 \(M_i\)（即 \(M_i\) 分配图）。单 patch 比特数 SHALL 不超过 \(M_{max}\)。

#### Scenario: 总比特数满足预算
- **WHEN** 给定 \(N\) 个 patch 的 \(a_i\) 与预算 \(B_{target}\)
- **THEN** 分配结果满足 \(\sum_i M_i \le B_{target}\) 且每个 \(0 \le M_i \le M_{max}\)

#### Scenario: 重要 patch 分得不少于次要 patch
- **WHEN** patch p 的重要性 \(w(a_p)\) 大于 patch q 的 \(w(a_q)\)
- **THEN** 在增量分配收敛后 \(M_p \ge M_q\)

#### Scenario: 预算耗尽即停止
- **WHEN** 已分配比特数达到 \(B_{target}\)
- **THEN** 分配 SHALL 停止，不再追加比特

### Requirement: 元信息打包

系统 SHALL 将一次传输的元信息（\(M_i\) 分配图 + \(u_{min}/u_{max}\)）与载荷比特流分离地表示，使下游信道可对两者独立施加误码。\(M_i\) 图 SHALL 用定长编码（每 patch \(\lceil \log_2(M_{max}+1) \rceil\) 比特）。

#### Scenario: 元信息与载荷可分离
- **WHEN** pipeline 对一张图编码
- **THEN** 输出 SHALL 包含可独立访问的「载荷比特流」与「元信息（\(M_i\) 图、\(u_{min}/u_{max}\)）」两部分

### Requirement: 均匀比特分配基线

系统 SHALL 提供 `uniform` 分配策略：给每个 patch 分配相同比特数 \(M = \lfloor B_{target} / N \rfloor\)，与增量法**同总码率**，作为"无重要性元信息"的对照基线。均匀分配 SHALL 不依赖注意力 \(a_i\)，且其 \(M_i\) 图为常数（先验已知、无需作为脆弱元信息传输）。

#### Scenario: 同总码率
- **WHEN** 给定 \(N\) 个 patch 与预算 \(B_{target}\)
- **THEN** 均匀分配的 \(\sum_i M_i\) SHALL 与同预算下增量法的总码率相当（差异不超过取整余数）

#### Scenario: 全 patch 等比特
- **WHEN** 使用 `uniform` 策略
- **THEN** 所有 patch 的 \(M_i\) SHALL 相等（常数 \(M\)），与 \(a_i\) 无关
