"""复现论文 Fig.3：CIFAR-100 样例的 class-token→patch 注意力热图。

用法（在 vit-iaq-semcom 环境）：
    python scripts/visualize_attention.py --n 6 --out outputs/attention_fig3.png

加载 config 中的 device encoder（默认 CIFAR-100 fine-tuned 权重），对若干测试集
样例提取 a_i 并叠加热图，左原图 / 右热图并排保存。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml

from vit_iaq_semcom.vit_encoder import ViTEncoder, attention_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=6, help="可视化样例数")
    parser.add_argument("--out", default="outputs/attention_fig3.png")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    enc = ViTEncoder.from_config(cfg, role="device")

    # CIFAR-100 测试集；用 encoder 的预处理 transform（resize→224 + 归一化）
    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=enc.transform
    )
    # 原图（PIL→[0,1] tensor）用于叠加显示
    raw_ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    fig, axes = plt.subplots(2, args.n, figsize=(2.2 * args.n, 4.6))
    for j in range(args.n):
        x, _ = ds[j]
        a = enc.attention_scores(x.unsqueeze(0))[0].cpu().numpy()

        raw = raw_ds[j][0].permute(1, 2, 0).numpy()  # (32,32,3) [0,1]
        overlay = attention_heatmap(raw, a, enc.grid_size)

        axes[0, j].imshow(raw)
        axes[0, j].set_title("input", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(overlay)
        axes[1, j].set_title("attention", fontsize=8)
        axes[1, j].axis("off")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out.resolve()}")


if __name__ == "__main__":
    main()
