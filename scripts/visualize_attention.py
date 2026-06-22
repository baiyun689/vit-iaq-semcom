"""复现论文 Fig.3：CIFAR-100 样例的 class-token→patch 注意力图。

用法（在 vit-iaq-semcom 环境）：
    python scripts/visualize_attention.py --n 3 --out outputs/attention_fig3.png

对齐论文 Fig.3 的三栏布局（每行一个样例）：
    原图 | patch 网格叠注意力 | 原始 14×14 注意力热图(hot 配色 + colorbar)

要点（与论文协议对齐）：
- 骨干默认 DeiT-Tiny（论文设置，3 头 192 维），由 config 指定。
- 第三栏直接画原始 a_i（归一化到和为 1，~1/196 量级，与论文 colorbar 范围一致），
  不做 per-image min-max 拉伸，避免把背景噪声染色放大。
- 第三栏用 'hot' 配色 + 离散 14×14 块（不插值），与论文一致。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import yaml

from vit_iaq_semcom.vit_encoder import ViTEncoder


def _grid_overlay(ax, raw: np.ndarray, a_grid: np.ndarray, cmap: str = "jet") -> None:
    """中栏：原图 + 按 patch 着色的注意力叠加 + 网格线（对齐论文 Fig.3 中栏）。"""
    h, w = raw.shape[:2]
    gh, gw = a_grid.shape
    ax.imshow(raw, extent=(0, w, h, 0))
    # 注意力按 patch 块叠加（最近邻，不插值），半透明
    ax.imshow(
        a_grid, cmap=cmap, alpha=0.55, extent=(0, w, h, 0), interpolation="nearest"
    )
    # patch 网格线
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
    parser.add_argument("--out", default="outputs/attention_fig3.png")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    enc = ViTEncoder.from_config(cfg, role="device")
    print(f"encoder: arch={enc.arch} heads={enc.num_heads} grid={enc.grid_size}")

    # CIFAR-100 测试集；encoder 预处理 transform（resize→224 + 归一化）喂模型
    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=enc.transform
    )
    # 原图（[0,1]）用于显示
    raw_ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    gh, gw = enc.grid_size
    fig, axes = plt.subplots(args.n, 3, figsize=(9.5, 3.0 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for i in range(args.n):
        x, _ = ds[i]
        a = enc.attention_scores(x.unsqueeze(0))[0].cpu().numpy()  # (196,) 和为 1
        a_grid = a.reshape(gh, gw)
        raw = raw_ds[i][0].permute(1, 2, 0).numpy()  # (32,32,3) [0,1]

        axes[i, 0].imshow(raw)
        axes[i, 0].set_title("Original Image", fontsize=9)
        axes[i, 0].axis("off")

        _grid_overlay(axes[i, 1], raw, a_grid)
        axes[i, 1].set_title("Image with Attention Scores", fontsize=9)

        # 第三栏：原始 a_i，hot 配色，不做 min-max 拉伸（imshow 按本图 min/max 自适应，
        # colorbar 显示真实数值，与论文一致）
        im = axes[i, 2].imshow(a_grid, cmap="hot", interpolation="nearest")
        axes[i, 2].set_title("Attention Heatmap", fontsize=9)
        axes[i, 2].axis("off")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out.resolve()}")


if __name__ == "__main__":
    main()
