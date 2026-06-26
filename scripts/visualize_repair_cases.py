"""答辩展示图:语义修补网络「输入 → 输出 → 准确率提升」工作流案例。

每行一个案例,列依次:
    受损图像(输入①·像素) | 基础层粗重建(输入源) | 比特深度图(输入②·可靠性) |
    修补后输出(网络输出) | 原图(真值参考)
底部:本批 N 张图「修补前 vs 修补后」CIFAR-100 Top-1 准确率对比柱。

说明(如实标注):修补网真实的两路输入是 (a) 受损重建图(像素) 与 (b) 比特深度图;
后者由基础层派生(收端解基础层→算注意力→入桶),故「基础层」是经由比特深度图进场的。
默认 SNR=6dB(修补增益最大);案例默认自动挑「修补前✗→修补后✓」的救活样例。

用法(强制 CPU,符合本机控温):
    python scripts/visualize_repair_cases.py --snr 6 --n-stat 128 --n-show 3
    python scripts/visualize_repair_cases.py --indices 5,11,23 --snr 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from vit_iaq_semcom.a1 import base_reconstruct
from vit_iaq_semcom.pipeline import IAQPipeline
from vit_iaq_semcom.repair import repair_reconstruction


def parse_buckets(s: str) -> list[tuple[int, int]]:
    return [(int(c), int(b)) for c, b in (p.split(":") for p in s.split(","))]


def norm_const(transform):
    for t in getattr(transform, "transforms", [transform]):
        if t.__class__.__name__ == "Normalize":
            return (np.array(t.mean, np.float32).reshape(3, 1, 1),
                    np.array(t.std, np.float32).reshape(3, 1, 1))
    return np.zeros((3, 1, 1), np.float32), np.ones((3, 1, 1), np.float32)


def denorm(x, m, s):
    return np.clip(x * s + m, 0.0, 1.0)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--repair-ckpt", default="outputs/六线图2/repair_net.pt")
    ap.add_argument("--snr", type=float, default=6.0)
    ap.add_argument("--n-stat", type=int, default=128, help="算准确率柱的样本数")
    ap.add_argument("--n-show", type=int, default=3, help="展示案例行数")
    ap.add_argument("--indices", default=None, help="手动指定案例下标,如 5,11,23")
    ap.add_argument("--buckets", default="49:4,49:2,98:1")
    ap.add_argument("--m_base", type=int, default=1)
    ap.add_argument("--base-fec-r", type=int, default=1)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="outputs/repair_cases_b392.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["device"] = "cpu"                                   # 控温:强制 CPU
    pipe = IAQPipeline.from_config(cfg)
    buckets = parse_buckets(args.buckets)
    pipe.m_max = max(pipe.m_max, max(b for _, b in buckets))
    gh, gw = pipe.gh, pipe.gw
    pipe.load_repair_net(
        args.repair_ckpt, grid=(gh, gw), patch_dim=(3, 224 // gh, 224 // gw),
        m_max=pipe.m_max, dim=args.dim, depth=args.depth, heads=args.heads)

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform)
    classes = ds.classes
    nstat = args.n_stat
    xs, ys = zip(*[ds[i] for i in range(nstat)])
    x, y = torch.stack(xs), torch.tensor(ys)
    print(f"encoder={pipe.enc.arch} buckets={buckets} SNR={args.snr} n_stat={nstat}")

    mean, std = norm_const(pipe.enc.transform)
    rng = np.random.default_rng(args.seed)

    # 受损重建 + 比特深度图(同一损伤实例),再仅收端修补
    recon_b, bdm = pipe.bucket_corrupt(
        x.numpy(), args.snr, args.m_base, args.base_fec_r, buckets, rng, torch.float32)
    recon_a = repair_reconstruction(pipe.repair_net, recon_b, bdm, torch.float32)
    # 基础层粗重建(输入源;m_base=1bit 的粗图)
    xnp = x.numpy()
    base = np.stack([base_reconstruct(xnp[k], args.m_base, float(xnp[k].min()),
                                      float(xnp[k].max())) for k in range(nstat)])

    with torch.no_grad():
        pb = pipe.enc.classify(torch.from_numpy(recon_b).float())[1].cpu().numpy()
        pa = pipe.enc.classify(torch.from_numpy(recon_a).float())[1].cpu().numpy()
    yn = y.numpy()
    acc_b, acc_a = float((pb == yn).mean()), float((pa == yn).mean())
    print(f"准确率 修补前={acc_b:.3f} 修补后={acc_a:.3f} (+{(acc_a-acc_b)*100:.1f}pt, n={nstat})")

    # 选案例:优先「修补前✗→修补后✓」的救活样例
    if args.indices:
        sel = [int(i) for i in args.indices.split(",")]
    else:
        flipped = [i for i in range(nstat) if pb[i] != yn[i] and pa[i] == yn[i]]
        sel = (flipped + [i for i in range(nstat) if i not in flipped])[:args.n_show]
    sel = sel[:args.n_show]

    od = denorm(xnp, mean, std)
    bd = denorm(recon_b, mean, std)
    ad = denorm(recon_a, mean, std)
    based = denorm(base, mean, std)

    # 比特深度图离散配色(值=每patch总比特,如 1/2/4)
    uvals = sorted(set(int(v) for v in np.unique(bdm)))
    bcolors = {1: "#2c7bb6", 2: "#fdae61", 4: "#d7191c", 8: "#7b3294"}
    cmap = ListedColormap([bcolors.get(v, "#999999") for v in uvals])
    norm = BoundaryNorm([uvals[0] - 0.5] + [v + 0.5 for v in uvals], len(uvals))

    ns = len(sel)
    fig = plt.figure(figsize=(13.0, 3.0 * ns + 2.0))
    gs = fig.add_gridspec(ns + 1, 5, height_ratios=[1] * ns + [0.95], hspace=0.32,
                          wspace=0.06)
    col_titles = ["受损图像\n(输入①·像素)", "基础层粗重建\n(输入源·1bit)",
                  "比特深度图\n(输入②·可靠性)", "修补后输出\n(网络输出)",
                  "原图\n(真值参考)"]

    for r, i in enumerate(sel):
        truth = classes[yn[i]]
        ok_b = pb[i] == yn[i]
        ok_a = pa[i] == yn[i]
        bdm_grid = bdm[i].reshape(gh, gw)
        cells = [
            (bd[i].transpose(1, 2, 0), f"{'✓' if ok_b else '✗'} {classes[pb[i]]}", None),
            (based[i].transpose(1, 2, 0), f"PSNR {psnr(od[i], based[i]):.1f}dB", None),
            (bdm_grid, "红4/橙2/蓝1 bit", (cmap, norm)),
            (ad[i].transpose(1, 2, 0),
             f"{'✓' if ok_a else '✗'} {classes[pa[i]]}", None),
            (od[i].transpose(1, 2, 0), f"真值: {truth}", None),
        ]
        for c, (img, sub, cm) in enumerate(cells):
            ax = fig.add_subplot(gs[r, c])
            if cm is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cm[0], norm=cm[1], interpolation="nearest")
            color = "green" if (c == 3 and ok_a and not ok_b) else "black"
            ax.set_xlabel(sub, fontsize=8.5, color=color)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9.5)

    # 底部:准确率对比柱
    axb = fig.add_subplot(gs[ns, 1:4])
    bars = axb.bar(["修补前\n(粗桶解码)", "修补后\n(+AI修补)"], [acc_b, acc_a],
                   color=["#888888", "black"], width=0.55)
    for rect, v in zip(bars, [acc_b, acc_a]):
        axb.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", fontsize=11, fontweight="bold")
    axb.set_ylim(0, max(acc_a, acc_b) * 1.25 + 0.05)
    axb.set_ylabel("CIFAR-100 Top-1")
    axb.set_title(f"修补前后准确率  +{(acc_a-acc_b)*100:.1f}pt  (n={nstat}, SNR={args.snr:g}dB)",
                  fontsize=10)
    axb.grid(axis="y", alpha=0.3)

    flips = int(((pb != yn) & (pa == yn)).sum())
    fig.suptitle(
        f"语义修补网络工作流 · b392/ρ0.25 · SNR={args.snr:g}dB · 零额外带宽 · "
        f"本批救活 {flips}/{nstat} 张(✗→✓)",
        fontsize=13, y=0.998)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out.resolve()}  cases={sel}")


if __name__ == "__main__":
    main()
