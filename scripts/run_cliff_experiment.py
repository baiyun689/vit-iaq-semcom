"""创新点 A — 元信息性能悬崖实验（E1/E2）。

固定载荷预算 B_target，扫一组 BSC 翻转概率 mu，分别在「元信息无损」(论文假设)
与「元信息也过信道」两种模式下测 CIFAR-100 分类准确率，画两条对照曲线。

用法（在 vit-iaq-semcom 环境）：
    python scripts/run_cliff_experiment.py --n 256 --out outputs/cliff_metadata.png

说明：server 端用 CIFAR-100 微调 ViT-Base 纯推理（见 configs/default.yaml）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from vit_iaq_semcom.pipeline import IAQPipeline

# Windows 中文字体，避免标签渲染成方框
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_results(results: dict, out: Path) -> None:
    """从结果字典画悬崖图（纯 CPU，不依赖模型/GPU；供 --replot 复用）。"""
    mus = results["mus"]
    xs_plot = [m if m > 0 else 1e-5 for m in mus]  # mu=0 在对数轴上用极小值代替
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(results["base_acc"], color="gray", ls=":", lw=1,
               label=f"无量化基线 ({results['base_acc']:.2f})")
    ax.plot(xs_plot, results["acc_safe"], "o-", color="tab:blue", label="元信息无损（论文假设）")
    ax.plot(xs_plot, results["acc_cliff"], "s-", color="tab:red", label="元信息也过信道（本工作）")
    ax.set_xscale("log")
    ax.set_xlabel("BSC 翻转概率 μ")
    ax.set_ylabel("CIFAR-100 Top-1 准确率")
    ax.set_title("元信息性能悬崖：论文乐观假设 vs 元信息裸传")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out.resolve()}")


def evaluate(pipe, x, y, mu, metadata_through_channel, batch=64, seed=0):
    """分批跑，返回整体准确率。"""
    correct = total = 0
    for s in range(0, x.shape[0], batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        out = pipe.run_batch(
            xb, labels=yb, mu=mu,
            metadata_through_channel=metadata_through_channel, seed=seed + s,
        )
        correct += out["accuracy"] * xb.shape[0]
        total += xb.shape[0]
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=256, help="评测样本数")
    parser.add_argument("--out", default="outputs/cliff_metadata.png")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--mus", default="0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1",
        help="逗号分隔的 BSC 翻转概率扫描点",
    )
    parser.add_argument(
        "--replot", default=None,
        help="只从已存的结果 JSON 重画图（纯 CPU，不加载模型/不上 GPU）",
    )
    args = parser.parse_args()

    # --replot：零 GPU 路径，仅重画样式/字体
    if args.replot:
        results = json.loads(Path(args.replot).read_text(encoding="utf-8"))
        plot_results(results, Path(args.out))
        return

    # 重负载路径才导入 torch / 加载模型
    import torch
    import torchvision
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pipe = IAQPipeline.from_config(cfg)
    print(f"encoder: arch={pipe.enc.arch} grid={pipe.enc.grid_size} "
          f"b_target={pipe.b_target} m_max={pipe.m_max}")

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform
    )
    xs, ys = zip(*[ds[i] for i in range(args.n)])
    x, y = torch.stack(xs), torch.tensor(ys)

    base_acc = float((pipe.enc.classify(x)[1].cpu() == y).float().mean())
    print(f"无量化直接分类基线准确率: {base_acc:.4f}")

    mus = [float(s) for s in args.mus.split(",")]
    acc_safe, acc_cliff = [], []
    for mu in mus:
        a_safe = evaluate(pipe, x, y, mu, metadata_through_channel=False)
        a_cliff = evaluate(pipe, x, y, mu, metadata_through_channel=True)
        acc_safe.append(a_safe)
        acc_cliff.append(a_cliff)
        print(f"mu={mu:<8g}  元信息无损={a_safe:.4f}  元信息过信道={a_cliff:.4f}")

    results = {
        "mus": mus, "acc_safe": acc_safe, "acc_cliff": acc_cliff,
        "base_acc": base_acc, "n": args.n,
        "b_target": pipe.b_target, "m_max": pipe.m_max, "arch": pipe.enc.arch,
    }
    # 结果落盘：以后改样式/字体用 --replot 从此文件 CPU 重画，无需再上 GPU
    json_out = Path(args.out).with_suffix(".json")
    json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {json_out.resolve()}")
    plot_results(results, Path(args.out))


if __name__ == "__main__":
    main()
