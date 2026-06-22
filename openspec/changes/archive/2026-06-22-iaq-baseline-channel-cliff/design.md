## Context

Step 2 已交付 `ViTEncoder`（注意力 \(a_i\) 提取 + 分类前向）。本 change 实现其下游的 Sec. III（量化、比特分配）+ Sec. IV（信道）+ Sec. II 图 1 端到端编排，把四个占位模块变成可跑链路，并产出创新点 A 的招牌结果——元信息性能悬崖。

关键架构事实（来自 `pipeline.py` 既有约定）：**传输的是量化后的 patch 像素，不是深层特征**；server 端「反量化 → 重建降质图像 → 一个普通 ViT 分类」。因此分类器可直接用现成 CIFAR-100 微调权重纯推理，本 change 无任何训练。

约束：conda 环境 `vit-iaq-semcom`（timm 1.0.27 / torch 2.11.0+cu128，CUDA 可用；git-bash 的 `python` 非该环境，须 `conda run -n vit-iaq-semcom python`）。

## Goals / Non-Goals

**Goals:**
- 打通 device→channel→server 最小端到端链路，干净信道下分类准确率接近无量化基线。
- 增量法比特分配 + patch 均匀量化 + BSC，三者各为独立可测单元。
- 元信息（\(M_i\) 图、\(u_{min}/u_{max}\)）与载荷比特流可分离，信道可对两者独立施误码。
- 产出「元信息无损 vs 元信息过信道」准确率-\(\mu\) 悬崖对照图（E1/E2）。

**Non-Goals:**
- 注水法比特分配（增量法已是 P1 最优解，注水法留作后续对照）。
- FEC / CRC / UEP 保护（创新点 A 的 E3 修复，下一个 change）。
- 收端语义错误隐藏（创新点 C，后续 change）。
- DeiT-Tiny/-Small 微调与正统 edge-server 骨干（提速期用现成 CIFAR-100 权重，正统骨干后续切换）。
- Lloyd–Max / 非均匀量化（暂搁置，定位未决）。

## Decisions

**D1 — 提速优先：分类器用现成 CIFAR-100 微调权重纯推理。**
config 现为 `imagenet` 权重，对 32×32 上采样图 OOD、准确率低、曲线难看。server 端分类器切到旧 Step2 验证过的 `hf_hub:edadaltocg/vit_base_patch16_224_in21k_ft_cifar100`（93.2% top-1），只推理不训练。device 端算 \(a_i\) 用 ImageNet DeiT-Tiny 即可（重要性排序不需高精度）。备选「先微调 DeiT」会阻塞当前急需的仿真，推迟。

**D2 — 比特分配用增量法，不做注水法。**
增量法（每次把 1 比特发给边际加权误差下降最大的 patch，直到 \(\sum M_i = B_{target}\)）是 P1 离散最优解且 ~20 行可实现；注水法需 KKT + 连续解取整，结果近似相同。写成可切换 `strategy` 参数，注水法留作后续对照。对悬崖图而言分配策略不敏感（悬崖来自元信息误码），但增量法是论文方法、不浪费。

**D3 — 元信息与载荷分离表示,信道独立施误码。**
量化输出为一个结构体：`payload_bits`（拼接的各 patch 量化比特）+ `meta`（\(M_i\) 图、\(u_{min}/u_{max}\)）。\(M_i\) 图定长编码（每 patch \(\lceil\log_2(M_{max}+1)\rceil\) 比特）。channel 按 `metadata_through_channel` 决定是否把 `meta` 也过 BSC。这样 E1（关）与 E2（开）只差一个开关，便于对照。

**D4 — server 端按接收到的 \(M_i\) 图切分载荷。**
反量化时严格用「接收到的」\(M_i\) 图逐 patch 切 `payload_bits`。当 \(M_i\) 被信道翻转，切分长度错→后续 patch 失步→重建图崩，这正是悬崖的机制（与 spec 场景对齐）。失步时按可用比特尽力解析、不足部分填区间中点,保证流程不崩溃只是图质崩。

**D5 — patch 像素量化的范围 \(u_{min}/u_{max}\) 取全局每图统计。**
对一张图统一取所有 patch 像素的 min/max 作为量化区间（论文式 12 也是每图传一对 16bit umin/umax）。简单且与论文元信息开销模型一致。

**D6 — 量化作用在「ViT 输入张量」上,而非原始 0-255 像素。**
device 与 server 共用 `ViTEncoder` 的预处理（resize 224 + 归一化）。在归一化后的张量上按 16×16 patch 量化，server 直接喂量化后的张量给分类器，避免再走一遍预处理引入不一致。

## Risks / Trade-offs

- **[现成 ViT-Base 权重非论文 DeiT 骨干]** → 当前目标是「跑通链路 + 出悬崖图」,骨干正确性是后续 change 的事;config 留 `server_weights` 字段,切换无需改代码。
- **[失步后准确率噪声大、曲线不平滑]** → 每个 \(\mu\) 点用足够多测试图 + 固定种子多次平均;悬崖只需定性显著,不追求平滑。
- **[量化作用在归一化张量上,数值范围因通道而异]** → 每图全局 min/max 已覆盖;若单图悬殊可后续改 per-channel,本 change 不做。
- **[BSC 直接翻「比特序列」需要明确比特表示]** → 量化索引用自然二进制定长编码;Gray 码 / COVQ 是 Lloyd–Max 点的事,本 change 用自然二进制即可。
- **[CIFAR-100 / 权重需联网下载]** → 复用 Step2 已下载缓存;测试对网络依赖打 `network` mark 离线跳过。

## Open Questions

- 悬崖实验的 \(B_{target}\) 取值（决定干净基线准确率与悬崖落差对比度）→ 实现时先扫几档预算挑一个干净准确率合理的点固定。
- \(\mu\) 扫描范围与采样点 → 先用对数档（如 1e-4 ~ 1e-1）看出悬崖位置再细化。
