"""粗桶档位标定扫描（change iaq-learned-semantic-repair，task 2.2）。

在**同总预算（默认 b392）**约束下,扫一批"桶配置",在 **clean（无误码,高 SNR）**
下评 ④粗桶 的**纯源编码质量**,据此选出"源编码损失最小 + 重要性红利够"的工作点。

为什么只评 clean:换桶配置会改变比特深度图分布 → 现有修补网 repair_net.pt 失配,
不能直接拿来评 ⑤。所以本脚本**不碰修补网、不过信道噪声**,只衡量桶配置本身的源编码
代价（recon_mse 主、clean 准确率次,均与修补/信道解耦）。选定桶后再单独重训修补网。

参照线:②理想（incremental,吃满重要性）/ ①均匀（同预算均匀 2bit）在同 clean 下各算一次,
给出每个桶配置离"天花板"还差多少、有没有甩开均匀。

指标解读:
  - acc(clean) 越高 → 重要性分配把**任务相关**信息保得越好 → **首选排序依据**(任务对齐)。
    需较大 n 让准确率稳(GPU 机器 n=256;小 n 噪声大,只看大小关系)。
  - recon_mse 是**全局像素**失真,偏向均匀分配(均匀全局 MSE 最低却吃不到重要性,
    与任务目标方向相反),**仅作参考、不可用来选桶**。
  - 真实预算 = bits_per_rank(buckets, N).sum()（桶没覆盖满 N 时尾秩沿用末档）,
    脚本会校验是否 == b_target,偏离会告警。

⚠️ ViT-Base 前向在 CPU 上较慢:n=24 约十几分钟。本机过热,**优先在 GPU 机器秒级跑完**;
非要本机就用小 n（默认 24）纯 CPU,别开 GPU。

用法:
    python scripts/run_bucket_sweep.py --n 24 --device cpu
    python scripts/run_bucket_sweep.py --n 256 --device cuda            # GPU 机器,更稳
    python scripts/run_bucket_sweep.py --configs "49:4,49:2,98:1;28:5,42:3,126:1"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

N_PATCH = 196


def parse_one(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        cnt, bits = part.split(":")
        out.append((int(cnt), int(bits)))
    return out


def parse_configs(s: str) -> list[list[tuple[int, int]]]:
    return [parse_one(c) for c in s.split(";") if c.strip()]


# 默认候选:同预算(=392)、覆盖"陡↔平"谱系。陡=重要性红利强但低比特 patch 多(源编码损失大);
# 平=接近均匀(损失小但红利少)。每组 Σcnt=196、Σcnt·bits=392。
DEFAULT_CONFIGS = [
    [(49, 4), (49, 2), (98, 1)],            # A 现状(3 档,中陡)
    [(28, 5), (42, 3), (126, 1)],           # B 更陡(尖峰)
    [(49, 5), (147, 1)],                    # C 2 档极陡
    [(98, 3), (98, 1)],                     # D 更平(2 档)
    [(60, 3), (76, 2), (60, 1)],            # E 平滑 3 档
    [(28, 4), (28, 3), (56, 2), (84, 1)],   # F 4 档细
]


def fmt_buckets(bk: list[tuple[int, int]]) -> str:
    return "[" + ",".join(f"{c}×{b}b" for c, b in bk) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=24, help="样本数(CPU 慢,默认小)")
    ap.add_argument("--b_target", type=int, default=392)
    ap.add_argument("--m_base", type=int, default=1)
    ap.add_argument("--clean-snr", type=float, default=100.0,
                    help="评 clean 用的高 Eb/N0(dB),≈无误码")
    ap.add_argument("--configs", default=None,
                    help='候选桶,";"分隔多组,每组 "cnt:bits,cnt:bits";缺省用内置 6 组')
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cpu", help="默认 cpu(本机过热,别开 GPU)")
    ap.add_argument("--out", default="outputs/bucket_sweep.json")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchvision
    import yaml

    from vit_iaq_semcom.bucket import bits_per_rank
    from vit_iaq_semcom.pipeline import IAQPipeline

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["device"] = args.device
    pipe = IAQPipeline.from_config(cfg)
    pipe.b_target = args.b_target
    n_patch = pipe.gh * pipe.gw

    configs = parse_configs(args.configs) if args.configs else DEFAULT_CONFIGS
    # 候选桶须容纳得下 m_max（避免比特溢出）
    pipe.m_max = max(pipe.m_max, max(b for bk in configs for _, b in bk))

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform)
    xs, ys = zip(*[ds[i] for i in range(args.n)])
    x, y = torch.stack(xs), torch.tensor(ys)
    base_acc = float((pipe.enc.classify(x)[1].cpu() == y).float().mean())
    snr = args.clean_snr
    print(f"encoder={pipe.enc.arch} n={args.n} b_target={args.b_target} "
          f"m_base={args.m_base} clean_snr={snr} 无量化基线={base_acc:.3f}")

    def ev(**kw):
        out = pipe.run_batch(x, labels=y, channel_type="awgn", ebn0_db=snr, seed=0, **kw)
        return out["accuracy"], out["recon_mse"]

    # 参照线(各算一次):②理想 incremental / ①均匀
    ideal = ev(mode="iaq", alloc="incremental")
    uniform = ev(mode="iaq", alloc="uniform")
    print(f"② 理想(incremental): acc={ideal[0]:.3f} mse={ideal[1]:.4f}")
    print(f"① 均匀(uniform)    : acc={uniform[0]:.3f} mse={uniform[1]:.4f}")
    print("-" * 72)

    rows = []
    for bk in configs:
        budget = int(bits_per_rank(bk, n_patch).sum())
        warn = "" if budget == args.b_target else f"  ⚠ 预算={budget}≠{args.b_target}"
        acc, mse = ev(mode="bucket", m_base=args.m_base, buckets=bk, base_fec_r=1)
        rows.append({"buckets": bk, "budget": budget, "acc": acc, "mse": mse})
        print(f"④ {fmt_buckets(bk):<34} acc={acc:.3f} mse={mse:.4f}"
              f"  Δacc_vs均匀={acc - uniform[0]:+.3f}{warn}")

    # 按 clean 准确率降序(任务对齐;recon_mse 偏向均匀,只列不用来排)
    rows_sorted = sorted(rows, key=lambda r: (-r["acc"], r["mse"]))
    print("-" * 72)
    print("按 clean 准确率降序(任务对齐排序;mse 仅参考):")
    for r in rows_sorted:
        print(f"  acc={r['acc']:.3f} mse={r['mse']:.4f} {fmt_buckets(r['buckets'])}")
    best = rows_sorted[0]
    print(f"\n任务最优候选: {fmt_buckets(best['buckets'])} "
          f"(acc={best['acc']:.3f}, mse={best['mse']:.4f})")

    res = {
        "n": args.n, "b_target": args.b_target, "m_base": args.m_base,
        "clean_snr": snr, "arch": pipe.enc.arch, "base_acc": base_acc,
        "ref": {"ideal": {"acc": ideal[0], "mse": ideal[1]},
                "uniform": {"acc": uniform[0], "mse": uniform[1]}},
        "configs": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out.resolve()}")


if __name__ == "__main__":
    main()
