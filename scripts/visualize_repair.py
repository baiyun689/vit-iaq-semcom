"""答辩展示图:语义修补「前 vs 后」直观对比(零额外带宽的收端 AI 修补)。

同一损伤分布下并排展示:
    原图 | 修补前(粗桶解码) | 修补后(粗桶+修补) | 修补残差(AI 补在哪)
并标注每张图的 PSNR 与 CIFAR-100 分类是否正确(✗→✓ 即修补救活)。

工作点(与六线实验一致):b392 / ρ0.25,桶 [(49,4),(49,2),(98,1)],m_base=1,
base_fec_r=1,AWGN+BPSK。默认 SNR=6 dB(修补增益最大:粗桶 0.145→修补 0.453)。

用法(强制 CPU,vit_base 前向 + 修补网前向均秒级,符合本机控温):
    python scripts/visualize_repair.py --snr 6 --n 6 --repair-ckpt outputs/六线图2/repair_net.pt
    python scripts/visualize_repair.py --indices 3,7,12,18 --snr 0
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

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from vit_iaq_semcom.pipeline import IAQPipeline
from vit_iaq_semcom.repair import repair_reconstruction


def parse_buckets(s: str) -> list[tuple[int, int]]:
    return [(int(c), int(b)) for c, b in (p.split(":") for p in s.split(","))]


def norm_const(transform) -> tuple[np.ndarray, np.ndarray]:
    """从 timm transform 里抽出 Normalize 的 mean/std(用于反归一化显示)。"""
    for t in getattr(transform, "transforms", [transform]):
        if t.__class__.__name__ == "Normalize":
            m = np.array(t.mean, dtype=np.float32).reshape(3, 1, 1)
            s = np.array(t.std, dtype=np.float32).reshape(3, 1, 1)
            return m, s
    return np.zeros((3, 1, 1), np.float32), np.ones((3, 1, 1), np.float32)


def denorm(x: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(B,C,H,W) 归一化空间 → [0,1] 显示空间。"""
    return np.clip(x * s + m, 0.0, 1.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """两张 [0,1] 图之间的 PSNR(dB)。"""
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--repair-ckpt", default="outputs/六线图2/repair_net.pt")
    ap.add_argument("--snr", type=float, default=6.0, help="工作 Eb/N0 (dB)")
    ap.add_argument("--n", type=int, default=6, help="展示样例数(--indices 优先)")
    ap.add_argument("--indices", default=None, help="指定样例下标,如 3,7,12")
    ap.add_argument("--buckets", default="49:4,49:2,98:1")
    ap.add_argument("--m_base", type=int, default=1)
    ap.add_argument("--base-fec-r", type=int, default=1)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="outputs/repair_before_after_b392.png")
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
    print(f"修补网: {args.repair_ckpt}  encoder={pipe.enc.arch} buckets={buckets} "
          f"m_max={pipe.m_max} SNR={args.snr}")

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform)
    classes = ds.classes
    idx = ([int(i) for i in args.indices.split(",")] if args.indices
           else list(range(args.n)))
    xs, ys = zip(*[ds[i] for i in idx])
    x, y = torch.stack(xs), torch.tensor(ys)

    mean, std = norm_const(pipe.enc.transform)
    rng = np.random.default_rng(args.seed)

    # 同一损伤实例:先拿受损重建+比特深度图,再仅在收端做修补(零额外带宽)
    recon_before, bdm = pipe.bucket_corrupt(
        x.numpy(), args.snr, args.m_base, args.base_fec_r, buckets, rng, torch.float32)
    recon_after = repair_reconstruction(pipe.repair_net, recon_before, bdm, torch.float32)

    # 分类(归一化空间)→ 看修补是否把误分类救回
    with torch.no_grad():
        pred_b = pipe.enc.classify(torch.from_numpy(recon_before).float())[1].cpu().numpy()
        pred_a = pipe.enc.classify(torch.from_numpy(recon_after).float())[1].cpu().numpy()

    orig_d = denorm(x.numpy(), mean, std)
    before_d = denorm(recon_before, mean, std)
    after_d = denorm(recon_after, mean, std)
    resid = np.abs(after_d - before_d).sum(axis=1)            # (B,H,W) 修补改了哪

    n = len(idx)
    fig, axes = plt.subplots(n, 4, figsize=(11.0, 2.7 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["原图", "修补前(粗桶解码)", "修补后(粗桶+AI修补)", "修补残差|after−before|"]

    for r in range(n):
        truth = classes[int(y[r])]
        ok_b = "✓" if pred_b[r] == y[r] else "✗"
        ok_a = "✓" if pred_a[r] == y[r] else "✗"
        panels = [
            (orig_d[r].transpose(1, 2, 0), f"真值: {truth}", None),
            (before_d[r].transpose(1, 2, 0),
             f"PSNR {psnr(orig_d[r], before_d[r]):.1f}dB  {ok_b} {classes[pred_b[r]]}", None),
            (after_d[r].transpose(1, 2, 0),
             f"PSNR {psnr(orig_d[r], after_d[r]):.1f}dB  {ok_a} {classes[pred_a[r]]}", None),
            (resid[r], "AI 补回的细节(亮=改动大)", "magma"),
        ]
        for c, (img, sub, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap)
            ax.set_xlabel(sub, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)

    flips = int(((pred_b != y.numpy()) & (pred_a == y.numpy())).sum())
    fig.suptitle(
        f"语义修补前后对比 · b392/ρ0.25 · SNR={args.snr:g}dB · "
        f"修补救活 {flips}/{n} 张(✗→✓)· 零额外带宽",
        fontsize=13, y=0.997)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out.resolve()}  (修补救活 {flips}/{n})")


if __name__ == "__main__":
    main()
