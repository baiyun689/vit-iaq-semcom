## Context

单GPU研究环境，存在未提交的旧实验代码与结果，必须保留可恢复记录。框架完成不等于两项算法的新颖性或收益已证明。

## Goals / Non-Goals

Goals: 一次交付可执行框架，覆盖原生数据、软接收、三类基线、多预算多种子、修补消融、可恢复训练评价、校准、汇总、说明与迁移。
Non-Goals: 本次不启动耗时的正式GPU训练，不制造论文性能结论；短码对照不冒充现代LDPC强基线，未实现算法不能静默回退。

## Decisions

模块位于framework/。最终依赖审计确认新模块不再使用任何旧根级算法模块，因此旧源模块、运行入口和对应旧测试均在逐文件备份校验后移除；只保留包初始化和新CLI入口。
所有图像为原生RGB float32 CHW [0,1]；通信块由transmission.patch_size定义。分类器独立做可微插值和标准化；注意力映射按面积重叠守恒。
每幅图严格核算载荷、位置、CRC、编码冗余和填充。均匀/IAQ/显式方案使用同一预算上限。接收网络只能使用译码证据，仿真真值独立记录。
数据划分保存稳定ID；训练、选型、校准、审查相互隔离，官方测试单独保留。随机/合成分类器显式标记为流程验证。
每个实验配置、源码版本和数据划分决定运行指纹。训练按轮次恢复，评价按完整批次原子保存并按ID跳过；不同配置不复用旧目录。
统计以图像为抽样单位，训练种子变异另报。预算不足明确报错，禁止伪装完成。任务失败保存状态并可继续。

## Module contracts

- config.py（root）: load_config(path)->dict，严格默认合并/校验；schema_version,name,output_dir,device,data,classifier,transmission,experiment,repair,training。
- data.py（agent models）: build_datasets(cfg, output_dir)->dict[split,Dataset]，每项(image:TensorCHW,label:int,sample_id:str)；splits=train,selection,calibration,audit,test；持久splits.json。classifier.py/models.py同agent。
- ClassifierAdapter(cfg): .forward(native_tensor)->logits；.importance(native_tensor,patch_size)->numpy(B,N)；.fingerprint，.device；build_classifier(cfg)。支持tiny流程验证、timm ViT以及本地/HF分类权重，随机权重真实数据需显式允许。
- transport.py（agent transport）: transmit(image:npCHW,scores:npN,method:str,budget:int,snr:float,seed:int,config:dict)->dict；config是transmission段。返回image,hard,soft为CHW，depth为1HW实际位深，uncertainty/prefix_lower/prefix_upper/prefix_confidence为CHW，llr/bit_prob/bit_mask为(3*m_max,H,W)，counts含total及所有分项，diagnostics仅离线真值。禁止将diagnostics送给网络。缺失位mask=0而非可靠0。方法uniform/iaq/addressed。信道awgn/rayleigh（接收理想CSI明确标记），编码repetition/hamming74。
- models.py: build_repair(cfg,variant)->torch.nn.Module；forward(image,evidence_dict)->image。variant=image/depth/llr/bitwise/uncertainty/prefix，同一结构按条件屏蔽；prefix_penalty(output,evidence)函数。原型允许soft prefix penalty，不能保证不修坏。
- runner.py（agent runner）: plan(cfg)->dict; run(cfg,resume=True)->dict; evaluate(cfg,checkpoint_root,resume=False,allow_channel_shift=False)->dict；依赖上面接口。冻结评价默认仅允许评价设置变化，显式放宽时只多允许信道类型和噪声估计失配。支持每方法/预算/训练种子/修补变体训练，none不训练，审查/测试多SNR多信道种子评估；训练随机状态和优化器恢复；记录损失、checkpoint、逐图rows，训练仅train、选择仅selection。
- metrics.py（root）: summarize(run_dir,bootstrap_samples=1000)->dict，读predictions.jsonl列。结果行约定method,variant,budget,snr,train_seed,channel_seed,sample_id,label,before,after,clean,correct,before_correct,mse,psnr,total_bits，counts/diagnostics可附加。runner每job/predictions.jsonl；root只读取单个manifest对应plan列出的job，拒绝嵌套实验混合汇总。不把多信道种子当独立图像。
- cli.py（root）: validate/plan/run/evaluate/calibrate/fit-classifier/summarize/doctor；计划不用加载数据/模型，运行才执行；--config必须明确，--resume恢复相同指纹。

## Configuration fields

data: name(synthetic/cifar100/stl10),root,download,native_size,num_classes,train_size,selection_size,calibration_size,audit_size,test_size,split_seed。
classifier: kind(tiny/vit),arch,weights,input_size,mean,std,checkpoint,allow_random。
transmission: patch_size,m_max,m_base,header_repetition,payload_repetition,base_repetition,code,channel,noise_mismatch_db,decode,prefix_threshold。
experiment: methods,budgets,snrs,channel_seeds,training_seeds,split,bootstrap_samples。
repair: variants(none/image/depth/llr/bitwise/uncertainty/prefix),dim,depth,heads,architecture(transformer/cnn)。
training: epochs,batch_size,lr,snr_min,snr_max,lambda_task,lambda_mse,lambda_prefix,lambda_identity,grad_accum。

## Risks / Trade-offs

- 小样本及随机分类器结果 → 输出标注smoke，不可用于论文结论。
- 单卡长任务 → 串行运行、断点、先离线和极小真实数据验证。
- 新软证据与旧结果不可混比 → 协议版本、hash、完整预算与原生分辨率写入元数据。
- 清理误删 → 精确文件清单，验证在仓库内，先归档哈希校验再删除。

## Final verification

2026-10-08：121项测试通过；六组端到端运行全部完成，共608条逐图评价记录。含CPU合成数据、冻结测试、跨信道失配、瑞利Hamming、真实CIFAR100 CPU及RTX4060 8GB上的微型修补训练。恢复检查确认权重及预测不变；69份旧文件精确归档，65份旧入口移出主目录。详情见docs/框架交付验证记录_20261008.md。
