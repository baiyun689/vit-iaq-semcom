"""可视化「注意力 → 排名 → 分桶」：每个 patch 落进哪个桶（染该桶的比特深度色）。

对应第二创新点(粗桶非级联源编码)的分配过程:不看注意力数值,只按**排名**把
patch 塞进固定大小/固定比特的桶。默认 b392 桶配置 [(49,4),(49,2),(98,1)]:
  top-49 重要 patch → 4 bit；次-49 → 2 bit；其余 98 → 1 bit。

用法(vit-iaq-semcom 环境;强制 CPU 即可,vit_base 前向 3 张图秒级):
    python scripts/visualize_buckets.py --n 3 --role server --out outputs/bucket_assignment_vitbase.png
    python scripts/visualize_buckets.py --buckets 49:4,49:2,98:1

每行三栏:
    原图 | 原图叠加桶分配(离散色块) | 桶分配图(离散色 + 比特深度图例)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from vit_iaq_semcom.bucket import bucket_index
from vit_iaq_semcom.vit_encoder import ViTEncoder

# 桶配色:从重要(高比特)到次要(低比特),暖→冷,离散。
BUCKET_COLORS = ["#d7191c", "#fdae61", "#2c7bb6", "#7b3294", "#1a9641"]


def parse_buckets(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        cnt, bits = part.split(":")
        out.append((int(cnt), int(bits)))
    return out


def _overlay(ax, raw: np.ndarray, bidx_grid: np.ndarray, cmap, norm) -> None:
    """中栏:原图 + 按 patch 块叠加桶颜色(最近邻不插值) + 网格线。"""
    h, w = raw.shape[:2]
    gh, gw = bidx_grid.shape
    ax.imshow(raw, extent=(0, w, h, 0))
    ax.imshow(bidx_grid, cmap=cmap, norm=norm, alpha=0.55,
              extent=(0, w, h, 0), interpolation="nearest")
    for c in range(1, gw):
        ax.axvline(c * w / gw, color="white", lw=0.4, alpha=0.6)
    for r in range(1, gh):
        ax.axhline(r * h / gh, color="white", lw=0.4, alpha=0.6)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=3, help="可视化样例数")
    parser.add_argument("--buckets", default="49:4,49:2,98:1",
                        help="桶配置 cnt:总比特,...(默认 b392)")
    parser.add_argument("--role", default="server", choices=["device", "server"],
                        help="算注意力的编码器(默认 server=vit_base,与链路一致)")
    parser.add_argument("--out", default="outputs/bucket_assignment_vitbase.png")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    buckets = parse_buckets(args.buckets)
    n_bucket = len(buckets)
    cmap = ListedColormap(BUCKET_COLORS[:n_bucket])
    norm = BoundaryNorm(np.arange(-0.5, n_bucket + 0.5), n_bucket)
    # 图例:桶号 → "X bit (top-K)" 标注
    legend_handles = [
        Patch(facecolor=BUCKET_COLORS[i], edgecolor="k",
              label=f"{bits} bit  ×{cnt}")
        for i, (cnt, bits) in enumerate(buckets)
    ]

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    enc = ViTEncoder.from_config(cfg, role=args.role)
    print(f"encoder: arch={enc.arch} heads={enc.num_heads} grid={enc.grid_size} "
          f"buckets={buckets}")

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=enc.transform)
    raw_ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True,
        transform=torchvision.transforms.ToTensor())

    gh, gw = enc.grid_size
    fig, axes = plt.subplots(args.n, 3, figsize=(10.0, 3.0 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for i in range(args.n):
        x, _ = ds[i]
        a = enc.attention_scores(x.unsqueeze(0))[0].cpu().numpy()   # (196,) 和为 1
        bidx = bucket_index(a, buckets)                             # (196,) 桶号(0=最重要)
        bidx_grid = bidx.reshape(gh, gw)
        raw = raw_ds[i][0].permute(1, 2, 0).numpy()

        axes[i, 0].imshow(raw)
        axes[i, 0].set_title("Original Image", fontsize=9)
        axes[i, 0].axis("off")

        _overlay(axes[i, 1], raw, bidx_grid, cmap, norm)
        axes[i, 1].set_title("Patch → Bucket (overlay)", fontsize=9)

        axes[i, 2].imshow(bidx_grid, cmap=cmap, norm=norm, interpolation="nearest")
        axes[i, 2].set_title("Bit-depth Allocation", fontsize=9)
        axes[i, 2].axis("off")
        axes[i, 2].legend(handles=legend_handles, loc="center left",
                          bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out.resolve()}")


if __name__ == "__main__":
    main()
