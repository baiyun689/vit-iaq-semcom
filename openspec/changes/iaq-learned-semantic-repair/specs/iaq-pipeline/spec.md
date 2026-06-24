## ADDED Requirements

### Requirement: 粗桶与修补管线模式
`IAQPipeline` SHALL 支持 `bucket`（粗桶非级联源编码、无修补）与 `bucket+repair`
（粗桶 + 收端学习式修补）两种模式。`bucket+repair` MUST 在 `bucket` 的解码结果上加修补网络，
两者过信道的比特完全相同（修补在收端、零额外带宽）。既有 `iaq` 模式行为 MUST 保持不变。

#### Scenario: bucket+repair 零额外带宽
- **WHEN** 同一批图分别以 `bucket` 与 `bucket+repair` 运行
- **THEN** 两者标称信道比特相同；差异仅来自收端修补网络（纯算力）

#### Scenario: 默认模式向后兼容
- **WHEN** 不指定新模式调用 `run_batch`
- **THEN** 管线行为与既有 `iaq` 模式一致，回归不破坏

### Requirement: 六线同带宽对照评估
系统 SHALL 提供六线对照实验（⓪理想 / ①均匀 / ③裸传元信息 / ⑥IAQ+保护元信息 /
④粗桶 / ⑤粗桶+AI修补），扫 Eb/N0 出 CIFAR-100 Top-1 曲线。**全线总信道比特 MUST 相等**：
③⑥ 的元信息（及其 FEC）从预算内扣（ΣM_i 让位），⓪ 与 ③ 同信源预算、仅差元信息是否过信道。

#### Scenario: 同总信道比特记账
- **WHEN** 生成六线结果
- **THEN** 每条线记录其标称总信道比特，且 ①④⑤ 与（③⑥ 信源+元信息）在同一总比特口径下可比

#### Scenario: 差异隔离
- **WHEN** 解读曲线
- **THEN** ⓪−③ 隔离元信息悬崖、④−① 隔离重要性红利、⑤−④ 隔离 AI 修补增益（零额外带宽）

### Requirement: 匹配与失配双场景
系统 SHALL 支持匹配场景（FEC 按当前 SNR 设计）与失配场景（FEC 按固定设计 SNR 配死后扫
真实 SNR）。失配场景 MUST 能展示经典"保护元信息"线（⑥）在真实 SNR 低于设计点时再次悬崖，
而粗桶+修补线（⑤）平滑退化。

#### Scenario: 失配杀招
- **WHEN** FEC 按高设计 SNR 配死，真实 SNR 低于设计点
- **THEN** ⑥（受保护元信息）再次悬崖，⑤（粗桶+AI修补）平滑有底不崩

#### Scenario: 匹配场景诚实呈现
- **WHEN** FEC 按当前 SNR 设计
- **THEN** 系统如实呈现 ⑥ 近似 ⓪ 理想（经典 FEC 在匹配场景即可填平悬崖），不隐藏
