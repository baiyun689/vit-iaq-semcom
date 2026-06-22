## ADDED Requirements

### Requirement: 均匀比特分配基线

系统 SHALL 提供 `uniform` 分配策略:给每个 patch 分配相同比特数 \(M = \lfloor B_{target} / N \rfloor\),与增量法**同总码率**,作为"无重要性元信息"的对照基线。均匀分配 SHALL 不依赖注意力 \(a_i\),且其 \(M_i\) 图为常数(先验已知、无需作为脆弱元信息传输)。

#### Scenario: 同总码率
- **WHEN** 给定 \(N\) 个 patch 与预算 \(B_{target}\)
- **THEN** 均匀分配的 \(\sum_i M_i\) SHALL 与同预算下增量法的总码率相当(差异不超过取整余数)

#### Scenario: 全 patch 等比特
- **WHEN** 使用 `uniform` 策略
- **THEN** 所有 patch 的 \(M_i\) SHALL 相等(常数 \(M\)),与 \(a_i\) 无关
