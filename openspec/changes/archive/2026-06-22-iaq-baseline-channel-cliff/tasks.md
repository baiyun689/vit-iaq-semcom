## 1. 配置与脚手架

- [x] 1.1 `configs/default.yaml`：填充 `quantization`（`m_max`、`b_target`、`weight=attention`）与 `channel`（`enabled`、`bsc_flip_prob`、新增 `metadata_through_channel: false`）字段
- [x] 1.2 `configs/default.yaml`：`server_weights` 切到 `hf:edadaltocg/vit_base_patch16_224_in21k_ft_cifar100`（提速纯推理），device 端保持 DeiT-Tiny/imagenet 算 \(a_i\)
- [x] 1.3 确认 `vit-iaq-semcom` 环境下 server 权重可加载（一次性 smoke，复用 Step2 缓存）

## 2. 均匀量化器（quantization.py）

- [x] 2.1 实现 `uniform_quantize(patch, m, umin, umax)`：均匀标量量化返回整数索引（式 5），`m=0` 不产比特
- [x] 2.2 实现 `uniform_dequantize(idx, m, umin, umax)`：索引→重建值，`m=0` 填区间中点
- [x] 2.3 实现索引 ↔ 定长比特序列互转（自然二进制，长度 \(m\)，式 6）
- [x] 2.4 `tests/test_quantization.py`：往返误差 ≤ 步长、\(m\) 越大 MSE 不增、索引↔比特无损往返、`m=0` 行为

## 3. 增量比特分配（bit_allocation.py）

- [x] 3.1 实现 `weight_fn(a_i)`：注意力分数→重要性权重 \(w(a_i)\)（递增函数，先用恒等/线性）
- [x] 3.2 实现 `incremental_allocation(a_i, b_target, m_max)`：贪心逐比特分配，返回长度 \(N\) 的 \(M_i\) 图，满足 \(\sum M_i \le B_{target}\)、\(0 \le M_i \le M_{max}\)
- [x] 3.3 预留 `strategy` 参数（`incremental` 默认；`water_filling` 占位 NotImplemented，留后续对照）
- [x] 3.4 `tests/test_bit_allocation.py`：预算约束、重要 patch 比特不少于次要 patch、预算耗尽即停

## 4. 元信息打包与 BSC 信道（channel.py）

- [x] 4.1 实现元信息打包：`M_i` 图定长编码（每 patch \(\lceil\log_2(M_{max}+1)\rceil\) 比特）+ \(u_{min}/u_{max}\)，与 `payload_bits` 分离
- [x] 4.2 实现 `bsc(bits, mu, rng)`：按 \(\mu\) 独立翻转，\(\mu=0\) 恒等，支持种子复现
- [x] 4.3 实现 `apply_channel(packet, mu, metadata_through_channel, rng)`：载荷恒过 BSC；元信息按开关决定是否过 BSC
- [x] 4.4 `tests/test_channel.py`：\(\mu=0\) 无失真、翻转比例≈\(\mu\)、种子可复现、开关关闭时元信息逐位不变

## 5. 端到端 pipeline（pipeline.py）

- [x] 5.1 实现 device 段：图像 → `ViTEncoder.attention_scores` → 量化区间 \(u_{min}/u_{max}\) → 增量分配 \(M_i\) → 逐 patch 量化 → 打包（payload + meta）
- [x] 5.2 实现 server 段：按**接收到的** \(M_i\) 图切分 payload → 反量化重建张量（失步时尽力解析+填中点）→ `ViTEncoder.classify` → Top-1
- [x] 5.3 实现 `run_batch(images, cfg)`：串起 device→channel→server，返回 logits/预测/准确率
- [x] 5.4 `tests/test_pipeline.py`：\(\mu=0\) 且大 \(B_{target}\) 时输出 \((B,100)\) 且准确率接近无量化基线；降 \(B_{target}\) 准确率总体不升

## 6. 悬崖实验与验证

- [x] 6.1 新增 `scripts/run_cliff_experiment.py`：固定 \(B_{target}\)，扫一组 \(\mu\)，分别跑 `metadata_through_channel` 关/开，记录准确率
- [x] 6.2 画「准确率 vs \(\mu\)」两条曲线（无损 vs 过信道），保存到 `outputs/cliff_metadata.png`
- [x] 6.3 跑通实验，肉眼核对：干净信道准确率合理、元信息过信道呈现显著悬崖
- [x] 6.4 `conda run -n vit-iaq-semcom pytest -m "not network"` 全绿
- [x] 6.5 更新 README 路线图：勾选 Step 3-5 与信道悬崖（创新点 A 的 E1/E2）
