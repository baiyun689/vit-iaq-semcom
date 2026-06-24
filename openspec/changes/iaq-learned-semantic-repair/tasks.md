## 1. 粗桶源编码（iaq-bucket-coding）

- [x] 1.1 新增 `src/vit_iaq_semcom/bucket.py`：均匀基础层编解码（定长、固定位置）
- [x] 1.2 实现固定桶增强层编码：按注意力排序入"固定大小固定比特"桶，定长码流；
      利用均匀量化嵌入式性质（高 m_base 位=基础层、低位=增强层）
- [x] 1.3 实现收端派生排序入桶：解基础层→算注意力→排序→桶号/比特深度图（解码副产品）
- [x] 1.4 实现增强层定长定位解码 + 与基础层重组反量化（错配只局部、不级联）
- [x] 1.5 单测：基础层解码不依赖排序、基础层受损不级联、增强层定长对齐、嵌入式重组无损、
      派生不一致仅局部错配、比特深度图可导出（tests/test_bucket.py，11 passed）

## 2. 桶配置标定（探针已有，补低码率真实基线）

- [ ] 2.1 用 `scripts/run_bucket_probe.py` 以 **m_base=1** 重测 b392 派生真实代价（钉死，
      非探针的 2-bit 乐观值），记录结论
- [ ] 2.2 小样本扫桶档位（档数/各桶大小/比特），选定 b392 工作点的桶配置，记录

## 3. 学习式语义修补网络（iaq-semantic-repair）

- [x] 3.1 新增 `src/vit_iaq_semcom/repair.py`：轻量 patch-Transformer，输入受损重建网格 +
      比特深度可靠性图，输出残差修正（残差兜底：最坏≈不修）
- [x] 3.2 实现训练：冻结分类裁判 + patch_embed，只训修补器；数据=CIFAR-100 过粗桶管线 +
      随机 SNR AWGN；损失=分类 CE（主）+ 重建 MSE（辅）；跨 SNR
      （scripts/run_repair_train.py；CPU n=8 冒烟通过梯度回流，真训练待 GPU 机器）
- [x] 3.3 实现推理接口：给受损重建 + 比特深度图 → 修正重建（repair_reconstruction）
- [x] 3.4 单测（mock/小网络，CPU）：残差兜底不低于不修、可靠性图作为输入通道、
      只训修补器（分类器冻结）、损伤过广退回地板有限无 NaN（tests/test_repair.py，6 passed）

## 4. 管线模式分支（iaq-pipeline）

- [x] 4.1 `IAQPipeline` 增加 `bucket` 模式（粗桶非级联源编码，无修补）
- [x] 4.2 增加 `bucket+repair` 模式（在 bucket 解码上加修补网络，过信道比特与 bucket 相同）
- [x] 4.3 单测（mock encoder，CPU）：bucket 端到端跑通、bucket+repair 零额外带宽、
      iaq 默认模式回归不破坏（tests/test_pipeline_modes.py，全套 83 passed）

## 5. 六线对照实验（双场景、同带宽）

- [x] 5.1 新增六线实验脚本（⓪理想/①均匀/③裸传/⑥IAQ+保护元信息/④粗桶/⑤粗桶+修补），
      同总信道比特记账：③⑥ 元信息+FEC 从预算内扣、⓪与③同信源预算
      （scripts/run_repair_sixline_experiment.py；CPU n=4 管路冒烟六线跑通 + --replot 出图）
- [ ] 5.2 匹配场景：每 SNR 按当前 SNR 设计 FEC（诚实呈现 ⑥≈⓪）（待 GPU 跑）
- [ ] 5.3 失配场景（杀招）：FEC 按固定高设计 SNR 配死再扫真实 SNR → ⑥ 再悬崖、⑤ 平滑
      （待 GPU 跑）
- [ ] 5.4 消融：AI 修补开/关、可靠性图开/关、粗桶+修补 vs 均匀+修补、跨 SNR vs 单点训
- [ ] 5.5 绘图：六线曲线（标注 ρ/场景/开销）+ 修补前后可视化；--replot 纯 CPU 出图

## 6. 收尾

- [ ] 6.1 自审 + `openspec validate`；同步 delta specs 到主 specs 并归档（先同步再归档）
- [x] 6.2 更新 README 路线图（创新点 B：非级联粗桶 + 学习式语义修补）
- [x] 6.3 更新记忆进度
