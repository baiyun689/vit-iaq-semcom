## ADDED Requirements

### Requirement: 收端学习式语义修补网络
系统 SHALL 提供一个收端修补网络（轻量 patch-Transformer），输入**受损重建 patch 网格**
与**比特深度可靠性图**，输出对重建图的**残差修正**，再交冻结分类器分类。网络 MUST 在
像素/patch 域工作，复用冻结的分类裁判，不修改分类器权重。

#### Scenario: 残差兜底不帮倒忙
- **WHEN** 修补网络对某受损重建输出残差修正
- **THEN** 修正加回原重建；最坏情形下分类准确率 SHALL NOT 低于不修补（粗图地板）

#### Scenario: 可靠性图作为输入
- **WHEN** 修补网络前向
- **THEN** 系统将比特深度图作为可靠性通道输入，使网络对低比特/边界 patch 更依赖上下文修复

### Requirement: 任务目标训练（跨 SNR）
修补网络 SHALL 用**分类交叉熵为主损失**（可加重建 MSE 为辅）训练，训练数据为 CIFAR-100
过完整粗桶管线 + 随机 SNR 的 AWGN 受损样本。系统 MUST 冻结分类裁判与 patch_embed，只更新
修补网络。训练 MUST NOT 以"逼近干净图注意力"等代理目标为主（已证会摊平注意力、倒退准确率）。

#### Scenario: 只训修补网络
- **WHEN** 训练一个 step
- **THEN** 梯度仅更新修补网络参数；分类器与 patch_embed 参数保持冻结不变

#### Scenario: 跨 SNR 泛化
- **WHEN** 训练样本在一段 Eb/N0 区间内随机取信道
- **THEN** 单个修补网络 SHALL 能处理该区间内不同损伤档位，无需逐 SNR 训练单独模型

### Requirement: 极低 SNR 优雅退化
当损伤过广（大部分 patch 受损、上下文耗尽）时，系统 SHALL 优雅退化到粗图地板而非崩盘，
不产生灾难性的随机猜级准确率。

#### Scenario: 损伤过广时退回地板
- **WHEN** 极低 SNR 下大多数 patch 损坏
- **THEN** 修补网络借残差兜底退回基础层粗图质量级别，准确率平滑下降、无悬崖
