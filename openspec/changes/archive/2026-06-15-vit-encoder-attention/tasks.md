## 1. 配置与依赖

- [x] 1.1 在 `configs/default.yaml` 落地权重 id：CIFAR-100 用 `hf_hub:edadaltocg/vit_base_patch16_224_in21k_ft_cifar100`，新增 encoder 角色字段（device/server）
- [x] 1.2 确认 `requirements.txt` 含 `timm`、`torch`、`torchvision`、`matplotlib`（热图）；缺则补充
- [x] 1.3 在 `vit-iaq-semcom` 环境验证可联网加载该权重（一次性 smoke）

## 2. ViTEncoder 封装（vit_encoder.py）

- [x] 2.1 实现 `ViTEncoder` 类：按配置（arch / weights / device）经 timm 加载，eval 模式，device 不可用时回退 cpu。weights 支持 random/imagenet/hf:<repo>（hf: 走手动 load_state_dict）
- [x] 2.2 实现图像预处理：用 timm `resolve_data_config`+`create_transform`（resize→224、模型归一化常数）
- [x] 2.3 实现 `classify(images)`：前向返回 `(B, num_classes)` logits 与 top-1 预测
- [x] 2.4 实现注意力提取：关闭最后一层 `fused_attn` + forward hook (`attn_drop`) 取 softmax 注意力
- [x] 2.5 实现 `attention_scores(images)`：取最后一层 `A[:, 0, prefix:]`、多头平均，返回长度 196 的 \(a_i\)，归一化后非负且和=1

## 3. 注意力热图可视化

- [x] 3.1 实现纯函数 `attention_heatmap`：`a_i` → 14×14 → 上采样到原图 → overlay（模块内，已测）
- [x] 3.2 新增 `scripts/visualize_attention.py`：对若干 CIFAR-100 样例生成并保存注意力热图，复现 Fig.3

## 4. 测试

- [x] 4.1 `tests/test_vit_encoder.py`：classify 输出 shape `(B, 100)`、argmax 给出类别
- [x] 4.2 注意力 \(a_i\) 测试：长度 196、非负、求和≈1（数值容差）
- [x] 4.3 多头平均测试：输出等于各头 class-token→patch 注意力的算术平均
- [x] 4.4 设备回退测试：CUDA 不可用时不报错、落到 cpu
- [x] 4.5 网络依赖用例打 `network` mark（离线跳过）

## 5. 验证与收尾

- [x] 5.1 运行全部测试通过（8 passed）
- [x] 5.2 跑 Fig.3 脚本，肉眼核对热图：重要 patch（主体物）高亮、背景低（手动下载 CIFAR-100，MD5 校验通过；outputs/attention_fig3.png 已生成并核对）
- [x] 5.3 更新 README 路线图，勾选 Step 2
