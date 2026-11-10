# 旧代码归档

`legacy_code_20261008.zip` 保存清理前的 69 个文件，包括工作区中尚未提交的修改。对应清单 `legacy_code_20261008_manifest.json` 记录逐文件 SHA-256、字节数和是否从主目录移除，并记录压缩包 SHA-256。

移除的 65 个文件包括旧源模块、旧实验/文档生成脚本、旧测试和 `configs/default.yaml`。旧 README、依赖清单、项目元数据和包版本也归档，以便解释历史结果。清理前已验证压缩包每个条目的内容哈希与原文件一致。

数据集、训练权重、`outputs/` 历史实验、论文文档和 OpenSpec 历史记录均保留。新的实验入口见根目录 README。

需要回查时，先解压到独立目录，避免覆盖当前框架：

```powershell
Expand-Archive -LiteralPath archive/legacy_code_20261008.zip -DestinationPath legacy_review
```

旧代码采用不同的图像域和协议，不能将旧结果直接并入 `native-rgb-v1` 的汇总表。归档用于恢复和追溯，不是第二套受维护的实验入口。
