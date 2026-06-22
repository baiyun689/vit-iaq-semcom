"""元信息对分类收益的影响 —— AWGN 三线实验。

数字传输过高斯(AWGN)信道(BPSK),扫一组 Eb/N0,出三条「准确率 vs SNR」曲线：
  ① 均匀量化(无重要性元信息,参照系)
  ② IAQ·元信息理想(无损,论文乐观上界)
  ③ IAQ·元信息过信道(诚实值)
读法:收益 = ②−①(重要性值不值);元信息影响 = ②−③(脆弱性侵蚀);
并看 ③ 是否仍压过 ①(扣掉元信息代价后净值)。

用法(在 vit-iaq-semcom 环境)：
    python scripts/run_awgn_metadata_experiment.py --n 128 --out outputs/awgn_metadata_impact.png
改样式/字体零 GPU 重画：
    python scripts/run_awgn_metadata_experiment.py --replot outputs/awgn_metadata_impact.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from vit_iaq_semcom.pipeline import IAQPipeline

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_results(results: dict, out: Path) -> None:
    """从结果字典画三线图（纯 CPU，供 --replot 复用）。"""
    snr = results["ebn0_db"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(results["base_acc"], color="gray", ls=":", lw=1,
               label=f"无量化基线 ({results['base_acc']:.2f})")
    ax.plot(snr, results["acc_uniform"], "^-", color="tab:green", label="① 均匀量化（无重要性元信息）")
    ax.plot(snr, results["acc_iaq_ideal"], "o-", color="tab:blue", label="② IAQ·元信息理想")
    ax.plot(snr, results["acc_iaq_channel"], "s-", color="tab:red", label="③ IAQ·元信息过信道")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("CIFAR-100 Top-1 准确率")
    ax.set_title("元信息对分类收益的影响（AWGN 数字信道）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out.resolve()}")


def evaluate(pipe, x, y, ebn0_db, alloc, metadata_through_channel, batch=64, seed=0):
    correct = total = 0
    for s in range(0, x.shape[0], batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        out = pipe.run_batch(
            xb, labels=yb, channel_type="awgn", ebn0_db=ebn0_db, alloc=alloc,
            metadata_through_channel=metadata_through_channel, seed=seed + s,
        )
        correct += out["accuracy"] * xb.shape[0]
        total += xb.shape[0]
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=128, help="评测样本数（吃 GPU，默认小）")
    parser.add_argument("--out", default="outputs/awgn_metadata_impact.png")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--snr", default="-4,-2,0,2,4,6,8,10,12",
                        help="逗号分隔的 Eb/N0(dB) 扫描点")
    parser.add_argument("--replot", default=None,
                        help="只从结果 JSON 重画图（纯 CPU，不加载模型/不上 GPU）")
    args = parser.parse_args()

    if args.replot:
        results = json.loads(Path(args.replot).read_text(encoding="utf-8"))
        plot_results(results, Path(args.out))
        return

    import torch
    import torchvision
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pipe = IAQPipeline.from_config(cfg)
    print(f"encoder={pipe.enc.arch} grid={pipe.enc.grid_size} b_target={pipe.b_target}")

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform
    )
    xs, ys = zip(*[ds[i] for i in range(args.n)])
    x, y = torch.stack(xs), torch.tensor(ys)

    base_acc = float((pipe.enc.classify(x)[1].cpu() == y).float().mean())
    print(f"无量化直接分类基线: {base_acc:.4f}")

    snrs = [float(s) for s in args.snr.split(",")]
    acc_u, acc_i, acc_c = [], [], []
    for snr in snrs:
        au = evaluate(pipe, x, y, snr, alloc="uniform", metadata_through_channel=False)
        ai = evaluate(pipe, x, y, snr, alloc="incremental", metadata_through_channel=False)
        ac = evaluate(pipe, x, y, snr, alloc="incremental", metadata_through_channel=True)
        acc_u.append(au); acc_i.append(ai); acc_c.append(ac)
        print(f"Eb/N0={snr:<5g}  ①均匀={au:.4f}  ②IAQ理想={ai:.4f}  ③IAQ过信道={ac:.4f}")

    results = {
        "ebn0_db": snrs, "acc_uniform": acc_u,
        "acc_iaq_ideal": acc_i, "acc_iaq_channel": acc_c,
        "base_acc": base_acc, "n": args.n,
        "b_target": pipe.b_target, "m_max": pipe.m_max, "arch": pipe.enc.arch,
    }
    json_out = Path(args.out).with_suffix(".json")
    json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {json_out.resolve()}")
    plot_results(results, Path(args.out))


if __name__ == "__main__":
    main()
