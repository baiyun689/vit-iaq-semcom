"""A1 修复头探针 —— 学 base 注意力→原图注意力的校正,看能否补回零训练派生缺口。

一次性退路评估(change iaq-a1-metadata-free §2.4):§2.4 闸门发现零训练 A1 在
m_base=1/b392 下派生分配掉 ~15pt(收端在 1bit 基础层上算的注意力跟原图偏太多)。
本探针训一个小修复头 f:a_base -> a_orig,在最糟工作点上量三档准确率:
    acc(原图派生)   acc(base 派生,零训练)   acc(修复头派生)
看缺口能补回多少 → 决定要不要上大规模微调。

关键:修复头是确定性的,收发两端跑同一个头、同一 base 输入 → 派生仍逐位一致,
**不破坏 metadata-free**(不传 M_i 图)。头输出走 softmax 成正注意力分布,语义对齐
incremental_allocation(其边际收益依赖正幅度,不能喂带负的 z-score)。

纯 CPU、轻量训练,不碰 GPU。用法(vit-iaq-semcom 环境):
    python scripts/run_a1_repair_head_probe.py --n-train 512 --n-test 128 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vit_iaq_semcom.a1 import base_reconstruct as _base_recon_one
from vit_iaq_semcom.pipeline import IAQPipeline, _from_patches


def base_reconstruct(x: np.ndarray, m_base: int = 1) -> np.ndarray:
    """逐图全局 umin/umax 的 m_base 比特均匀量化重建(复用 a1.base_reconstruct)。"""
    out = np.empty_like(x)
    for k in range(x.shape[0]):
        out[k] = _base_recon_one(x[k], m_base, float(x[k].min()), float(x[k].max()))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 1.0


def compute_attn(pipe, imgs, batch: int = 32) -> np.ndarray:
    outs = []
    for s in range(0, imgs.shape[0], batch):
        outs.append(pipe.enc.attention_scores(imgs[s:s + batch]).cpu().numpy())
    return np.concatenate(outs, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--b_target", type=int, default=392)
    ap.add_argument("--m_base", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/a1_repair_head_probe.json")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import yaml

    torch.manual_seed(0)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["device"] = args.device
    pipe = IAQPipeline.from_config(cfg)
    pipe.b_target = args.b_target
    print(f"encoder={pipe.enc.arch} b_target={pipe.b_target} m_base={args.m_base} "
          f"n_train={args.n_train} n_test={args.n_test}")

    tf = pipe.enc.transform
    tr = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=tf)

    def load(ds, n):
        xs, ys = zip(*[ds[i] for i in range(n)])
        return torch.stack(xs), torch.tensor(ys)

    xtr, _ = load(tr, args.n_train)
    xte, yte = load(te, args.n_test)

    def attn_pair(x):
        a_o = compute_attn(pipe, x)
        xb = base_reconstruct(x.cpu().numpy(), args.m_base)
        a_b = compute_attn(pipe, torch.from_numpy(xb).to(x.dtype))
        return a_o, a_b

    print("computing train attention (ViT forward, CPU)...")
    a_o_tr, a_b_tr = attn_pair(xtr)
    print("computing test attention...")
    a_o_te, a_b_te = attn_pair(xte)

    def zrow(a):  # 输入特征标准化(只做输入,输出走 softmax)
        m = a.mean(1, keepdims=True)
        s = a.std(1, keepdims=True) + 1e-8
        return (a - m) / s

    def norm(a):  # 归一成分布(KL 目标)
        return a / a.sum(1, keepdims=True)

    N = a_o_tr.shape[1]
    Xtr = torch.from_numpy(zrow(a_b_tr)).float()
    Ttr = torch.from_numpy(norm(a_o_tr)).float()
    Xte = torch.from_numpy(zrow(a_b_te)).float()

    head = nn.Sequential(
        nn.Linear(N, args.hidden), nn.ReLU(),
        nn.Linear(args.hidden, args.hidden), nn.ReLU(),
        nn.Linear(args.hidden, N),
    )
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    head.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        pred_log = F.log_softmax(head(Xtr), dim=1)
        loss = F.kl_div(pred_log, Ttr, reduction="batchmean")
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep + 1}/{args.epochs} kl={loss.item():.5f}")
    head.eval()
    with torch.no_grad():
        a_rep_te = F.softmax(head(Xte), dim=1).cpu().numpy()  # 正注意力分布

    x_np = xte.cpu().numpy()
    c, h, w = x_np.shape[1:]
    ph, pw = h // pipe.gh, w // pipe.gw

    def acc_for(a_src):
        recon = np.empty_like(x_np)
        for k in range(x_np.shape[0]):
            recon[k] = _from_patches(
                pipe.decode(pipe.encode(x_np[k], a_src[k])), c, pipe.gh, pipe.gw, ph, pw)
        preds = pipe.enc.classify(torch.from_numpy(recon).to(xte.dtype))[1].cpu()
        return float((preds == yte).float().mean())

    acc_o = acc_for(a_o_te)
    acc_b = acc_for(a_b_te)
    acc_r = acc_for(a_rep_te)

    sp_b = float(np.mean([_spearman(a_o_te[k], a_b_te[k]) for k in range(len(a_o_te))]))
    sp_r = float(np.mean([_spearman(a_o_te[k], a_rep_te[k]) for k in range(len(a_o_te))]))

    res = {
        "n_train": args.n_train, "n_test": args.n_test, "b_target": args.b_target,
        "m_base": args.m_base, "epochs": args.epochs, "hidden": args.hidden,
        "arch": pipe.enc.arch,
        "spearman_base": sp_b, "spearman_repaired": sp_r,
        "acc_alloc_orig": acc_o, "acc_alloc_base": acc_b, "acc_alloc_repaired": acc_r,
        "acc_diff_base": acc_o - acc_b, "acc_diff_repaired": acc_o - acc_r,
        "gap_closed_frac": ((acc_r - acc_b) / (acc_o - acc_b)) if acc_o > acc_b else None,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
