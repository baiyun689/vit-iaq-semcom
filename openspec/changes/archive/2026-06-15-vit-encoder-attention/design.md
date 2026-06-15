## Context

论文用预训练 ViT（无需端到端训练）的注意力作为 patch 重要性来源。Step 1 已搭好包骨架，`vit_encoder.py` 目前仅为占位文档。本步骤要在 `timm` 之上实现 encoder 加载、分类前向与注意力分数提取，并产出 Fig.3 复现热图。约束：使用 `signal_Gen` conda 环境（torch 2.6.0+cu118）；需联网下载 HF 权重与 CIFAR-100。

## Goals / Non-Goals

**Goals:**
- 一个可配置的 ViT encoder 封装：加载预训练权重、分类前向、提取 class-token→patch 注意力 \(a_i\)。
- 多头注意力平均得到的 \(a_i\) 与论文式 3-4 一致，且为合法概率分布。
- 复现 Fig.3 注意力热图作为正确性验证。
- 单测覆盖 shape、概率归一化、多头平均、设备回退。

**Non-Goals:**
- 量化、比特分配、信道（留给 Step 3-6）。
- 自行训练 / fine-tune ViT（直接用现成权重）。
- 多视图数据集 MIRO / MVP-N（Step 7）。

## Decisions

**D1 — 用 timm 而非 HuggingFace transformers 加载。**
项目既有 config 与 pipeline 均以 timm 为前提。timm 的 `vit_base_patch16_224` 与 `hf_hub:edadaltocg/...` 加载方式统一，且 CIFAR-100 权重里 timm 原生那个精度最高（93.2%）。备选 HF transformers 的 `ViTForImageClassification` 注意力接口（`output_attentions=True`）与 timm 不同，会引入第二套分支，故不选。

**D2 — 注意力提取用 forward hook，而非依赖 `attn_drop` 或改模型源码。**
timm ViT 的 `Attention` 模块默认不返回注意力矩阵（用了 fused SDPA）。在最后一个 block 的 `Attention` 上注册 forward hook 拿不到 softmax 后权重时，改为设置 `model.blocks[-1].attn.fused_attn = False` 并 hook，或重算：取 hook 捕获的 `qkv` 输出，手动计算 `softmax(q·kᵀ/√d)`。决策：优先关闭 `fused_attn` 后从重写的注意力路径取权重，保证拿到真实 softmax 概率；备选手动重算 qkv（更脆弱，依赖内部张量布局）。实现时以前者为主、后者为 fallback，并用「\(a_i\) 求和≈1」断言守护。

**D3 — \(a_i\) 定义：最后一层、class-token 行、去掉 class-token 列、多头算术平均。**
即取注意力矩阵 `A ∈ (H, N+1, N+1)`，取 `A[:, 0, 1:]`（class-token 作 query 对各 patch），再 `mean(axis=0)` 得到长度 N=196 的向量。这与式 3-4 的「class-token 对 patch 的多头平均注意力」一致。是否跨多层聚合（attention rollout）属备选，论文用单层（最后一层），故默认单层。

**D4 — device/server 双角色用同一封装、不同模型名参数实现。**
论文图 1 区分轻量（device）与复杂（server）encoder。用同一个 `ViTEncoder` 类，靠传入不同 `model.name` 区分，避免重复代码。Step 2 默认两者可相同；真正差异化模型留待后续按需配置。

**D5 — 可视化放 `scripts/`（脚本）+ 函数放模块。**
热图叠加逻辑（`a_i` → 14×14 → 上采样 → overlay）作为模块内纯函数便于测试；Fig.3 复现作为 `scripts/` 下可运行脚本调用该函数并保存图片。

## Risks / Trade-offs

- **[timm 版本差异导致 hook 拿不到注意力]** → 在测试中断言 `a_i` 形状与归一化；若 `fused_attn` 路径失效，用手动重算 qkv 作 fallback，并在文档注明 timm 版本。
- **[CIFAR-100 32×32 与 ViT 224×224 不匹配]** → 预处理统一 resize 到 224 并用模型自带的归一化常数；注意力热图按原始/上采样尺寸叠加。
- **[联网下载失败（HF 权重 / 数据集）]** → 加载封装支持本地缓存路径；测试用小批量、允许跳过需下载的用例（mark 网络依赖）。
- **[`edadaltocg` 权重仅训 10 epochs]** → 精度够复现注意力提取与 Fig.3；若后续精度敏感，可在 config 中切换权重或自行 fine-tune（非本步骤）。

## Open Questions

- device 端「轻量」encoder 具体用哪个更小的 ViT 变体（如 `vit_small`/`vit_tiny`）？Step 2 先用同一 base，差异化推迟到比特分配实验需要时再定。
- Fig.3 复现选用哪几张 CIFAR-100 样例图以最接近论文展示？实现时挑选若干高/低注意力对比明显的样本即可。
