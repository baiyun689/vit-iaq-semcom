# vit-iaq-semcom

用于硕士论文实验的原生图像语义通信框架，协议版本 `native-rgb-v1`。

在 CIFAR-100 的 32×32 或 STL-10 的 96×96 RGB 图像上进行量化、编码和信道传输，接收后修补；分类器单独完成缩放与归一化。提供统一的配置、训练、冻结评价、校准、断点恢复、统计和绘图入口。

**完整操作说明：[实验框架使用说明](docs/实验框架使用说明.md)**。历史代码说明见 [归档目录](archive/README.md)。

交付检查：121项测试通过，已完成CPU离线、冻结测试、跨信道及真实CIFAR-100 GPU微型训练。见 [验证记录](docs/框架交付验证记录_20261008.md)。

## 快速使用

在项目根目录打开终端。使用已有的 `vit-iaq-semcom` 环境：

```powershell
conda activate vit-iaq-semcom
python -m pip install -e ".[dev]"
python -m vit_iaq_semcom doctor
python -m vit_iaq_semcom run --config configs/experiments/smoke.yaml --resume
```

最后一条命令用 CPU 和合成数据跑通三种传输方法、两种修补模型及无修补对照，输出在 `outputs/experiments/native_smoke/`。它检验流程，不能作为论文性能结果。第一次与相同代码下的恢复运行都可加 `--resume`；配置或源码变更后请换输出目录。

冻结已训练模型后，可换测试集、SNR 和信道种子，独立评价：

```powershell
python -m vit_iaq_semcom evaluate --config configs/experiments/smoke_test.yaml --checkpoints-from outputs/experiments/native_smoke --resume
```

查看 `summary/report.md`、`summary/metrics.csv` 和 `summary/curves_*.png`。安装后也可用 `semcom` 代替 `python -m vit_iaq_semcom`。

## 已提供的能力

- **数据和分类器**：CIFAR-100、STL-10、离线合成数据；五个独立用途的数据划分；可微 ViT 分类与注意力到通信网格的映射；分类器头训练或整模型微调；本地权重和预训练权重加载。
- **通信链路**：均匀量化、带受保护位深表的 IAQ、带显式位置和 CRC 的固定槽增强传输；BPSK、AWGN、平坦块瑞利衰落；重复码与 Hamming(7,4) 软译码；头部、CRC、填充、纠错冗余均纳入总 bit 预算。
- **修补和消融**：无修补、图像、位深、聚合 LLR、逐比特证据、不确定性、可信前缀七种变体；CNN/Transformer；任务损失、像素损失、前缀软约束和干净图像保持损失。
- **实验管理**：单卡串行任务矩阵；训练轮次恢复与评价批次恢复；配置、源码、数据和权重核验；冻结模型最终测试；校准集上的置信度和区间评价。
- **结果**：逐图配对预测、准确率与净收益、救回/修坏比例、MSE/PSNR/SSIM、实际开销、运行耗时；图像级 bootstrap 区间和单独报告的训练种子标准差；CSV、JSON、Markdown 和曲线图。

## 按用途选择配置

| 配置 | 用途 |
|---|---|
| `smoke.yaml` / `smoke_test.yaml` | CPU 离线流程和冻结测试验证 |
| `rayleigh_hamming.yaml` | 瑞利信道、短码和软证据的离线验证 |
| `fit_smoke_classifier.yaml` | 分类器训练接口的离线验证 |
| `cifar100_check.yaml` | 真实数据与预训练分类器的小样本接通检查 |
| `cifar100_gpu_check.yaml` | 单卡上的真实数据修补训练与反向传播检查 |
| `cifar100_baselines.yaml` | 三种传输基线，不训练修补网络 |
| `cifar100_repair.yaml` | 一个预算下的显式传输修补消融，三个训练种子 |
| `cifar100_test.yaml` | 从修补实验冻结权重后评价官方测试集 |
| `cifar100_rayleigh_test.yaml` / `cifar100_mismatch_test.yaml` | 冻结模型的跨信道和噪声估计失配评价，需显式打开 `--allow-channel-shift` |
| `smoke_shift.yaml` | 上述泛化接口的离线接通检查 |
| `cifar100.yaml` | 完整实验矩阵模板；计算量大，运行前先查看计划 |
| `fit_cifar100_classifier.yaml` / `fit_stl10_classifier.yaml` | 在本项目划分上训练分类器 |
| `cifar100_local_classifier.yaml` / `stl10.yaml` | 使用已训练好的本地分类器开展实验 |

真实数据配置默认不下载数据。需先放置数据集，或显式设置 `data.download: true`。STL-10 配置要求先准备本地分类器。已有 CIFAR-100 预训练分类器适合接通和通信研究，但它可能见过完整官方训练集；严格的数据隔离实验应使用本项目划分训练的分类器，并报告权重来源。

## 常用命令

```powershell
# 检查配置、列出任务，不开始训练
python -m vit_iaq_semcom validate --config configs/experiments/cifar100_repair.yaml
python -m vit_iaq_semcom plan --config configs/experiments/cifar100_repair.yaml

# 正式运行和恢复
python -m vit_iaq_semcom run --config configs/experiments/cifar100_repair.yaml --resume

# 校准只读取 calibration 划分；不会自动改写配置阈值
python -m vit_iaq_semcom calibrate --config configs/experiments/cifar100_repair.yaml

# 锁定方法后，使用已有最佳模型评价官方测试集
python -m vit_iaq_semcom evaluate --config configs/experiments/cifar100_test.yaml --checkpoints-from outputs/experiments/cifar100_repair --resume

# 泛化评价：保持原模型，显式允许改用瑞利信道
python -m vit_iaq_semcom evaluate --config configs/experiments/cifar100_rayleigh_test.yaml --checkpoints-from outputs/experiments/cifar100_repair --allow-channel-shift --resume

# 根据已保存的逐图预测重新绘图、生成统计表
python -m vit_iaq_semcom summarize --run-dir outputs/experiments/cifar100_test --bootstrap-samples 10000

# 在项目划分上训练分类器，默认只训练分类头
python -m vit_iaq_semcom fit-classifier --config configs/experiments/fit_stl10_classifier.yaml --output checkpoints/stl10_classifier.pt --resume

# 验证框架
python -m pytest -q
```

## 代码布局

```text
src/vit_iaq_semcom/framework/
  config.py               配置、继承和校验
  data.py                 原生数据与划分指纹
  classifier.py           分类器、可微预处理、注意力映射
  classifier_training.py  分类器训练和导出
  transport.py            预算、打包、编码、信道、软证据
  models.py               修补模型和前缀约束
  runner.py               训练、冻结评价、校准、断点恢复
  metrics.py              统计表与图
  cli.py                  唯一命令入口
configs/experiments/      可直接修改的实验配置
tests/                   当前框架测试
docs/                    使用说明及既有论文材料
archive/                 已核验的旧代码备份
```

## 使用边界

框架搭建完成不代表论文的两个创新点已经得到验证。当前 IAQ 和显式传输是可比较的基线；新的风险分配算法仍需要按研究设计实现并开展消融。可信前缀使用软惩罚，可能高置信地译错，也不保证修补永不变差。

瑞利模式假设接收端知道衰落增益，尚未计入导频和信道估计开销。Hamming(7,4) 是短码对照，不能代替现代长码强基线。运行时记录的耗时包括当前执行条件，不等同于经过预热和同步控制的硬件基准测试。小样本接通检查与完整训练结果应分别报告。

旧协议源文件、旧实验脚本、旧测试和旧默认配置已从主目录清理；数据、权重、论文文件和历史输出保留。详见 [归档清单](archive/legacy_code_20261008_manifest.json)。旧结果不得与新协议结果直接混合统计。
