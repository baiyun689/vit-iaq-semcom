"""A1 头号风险验证 —— 1 bit 基础层注意力一致性（change: iaq-a1-metadata-free §2）。

A1 要成立,前提是收发两端在「1 bit 基础层重建图」上各自算注意力、派生出与原图
**足够一致**的比特分配。本脚本量化这个前提:

  对同一批图,分别用
    (orig) 原图              -> a_i^orig -> 派生 M^orig
    (base) 1bit 基础层重建图 -> a_i^base -> 派生 M^base
  度量:
    - 注意力排序相关(Spearman / Kendall τ)
    - 分配一致性(逐 patch M 相等比例 / 分配图 TV 距离)
    - 最终准确率差(用各自分配在无信道下重建再分类)

据此判定 A1 走零训练,还是退路(b588 换工作点 / 小修复头 / 蒸馏)。

另含 `--drift-check`:纯 CPU 量「同权重同输入跑两遍 attention_scores 的浮点漂移」,
确认注意力量化(16 档)是否足够锁两端一致(spec: 注意力量化锁一致性)。

用法(vit-iaq-semcom 环境):
    python scripts/run_a1_attention_consistency.py --n 64 --b_target 392
    python scripts/run_a1_attention_consistency.py --drift-check --n 4 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vit_iaq_semcom.a1 import base_reconstruct as _base_recon_one
from vit_iaq_semcom.bit_allocation import incremental_allocation
from vit_iaq_semcom.pipeline import IAQPipeline, _from_patches


def base_reconstruct(x: np.ndarray, m_base: int = 1) -> np.ndarray:
    """1 bit(默认)基础层重建:逐图全局 umin/umax 量化再反量化(复用 a1.base_reconstruct)。"""
    out = np.empty_like(x)
    for k in range(x.shape[0]):
        out[k] = _base_recon_one(x[k], m_base, float(x[k].min()), float(x[k].max()))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """两向量的 Spearman 秩相关(无 scipy 依赖)。"""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 1.0


def _kendall_tau(a: np.ndarray, b: np.ndarray, max_n: int = 196) -> float:
    """Kendall τ-b(O(n²),n=196 可接受)。"""
    n = min(len(a), max_n)
    a, b = a[:n], b[:n]
    conc = disc = 0
    for i in range(n):
        da = a[i] - a[i + 1 :]
        db = b[i] - b[i + 1 :]
        s = np.sign(da) * np.sign(db)
        conc += int((s > 0).sum())
        disc += int((s < 0).sum())
    total = conc + disc
    return float((conc - disc) / total) if total > 0 else 1.0


def drift_check(pipe: IAQPipeline, x, n: int) -> dict:
    """同权重同输入跑两遍 attention_scores,量浮点漂移与量化后是否一致。"""
    a1 = pipe.enc.attention_scores(x[:n]).cpu().numpy()
    a2 = pipe.enc.attention_scores(x[:n]).cpu().numpy()
    max_abs = float(np.abs(a1 - a2).max())
    bit_identical = bool(np.array_equal(a1, a2))
    # 量化到 16 档(按每图 min-max 归一)后是否逐 patch 一致
    def q16(a):
        lo = a.min(axis=1, keepdims=True)
        hi = a.max(axis=1, keepdims=True)
        rng = np.where(hi > lo, hi - lo, 1.0)
        return np.clip(np.floor((a - lo) / rng * 16), 0, 15).astype(int)
    q_consistent = bool(np.array_equal(q16(a1), q16(a2)))
    return {
        "n": n,
        "max_abs_drift": max_abs,
        "bit_identical": bit_identical,
        "quant16_consistent": q_consistent,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=64, help="样本数(吃算力,默认小)")
    parser.add_argument("--b_target", type=int, default=392, help="A1 工作点总预算")
    parser.add_argument("--m_base", type=int, default=1, help="基础层每 patch 比特")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default=None, help="覆盖 config 的 device(如 cpu)")
    parser.add_argument("--drift-check", action="store_true", help="只做浮点漂移检查")
    parser.add_argument("--out", default="outputs/a1_attention_consistency.json")
    args = parser.parse_args()

    import torch
    import torchvision
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    pipe = IAQPipeline.from_config(cfg)
    pipe.b_target = args.b_target
    print(f"encoder={pipe.enc.arch} device={pipe.enc.device} "
          f"b_target={pipe.b_target} m_base={args.m_base}")

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform
    )
    xs, ys = zip(*[ds[i] for i in range(args.n)])
    x, y = torch.stack(xs), torch.tensor(ys)

    if args.drift_check:
        res = drift_check(pipe, x, min(args.n, 8))
        print(f"漂移检查: {res}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_name("a1_drift_check.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    x_np = x.cpu().numpy()
    x_base = base_reconstruct(x_np, m_base=args.m_base)

    a_orig = pipe.enc.attention_scores(x).cpu().numpy()                        # (B,N)
    a_base = pipe.enc.attention_scores(
        torch.from_numpy(x_base).to(x.dtype)).cpu().numpy()

    n_img = x_np.shape[0]
    spear, kend, m_agree, m_tv = [], [], [], []
    recon_o = np.empty_like(x_np)
    recon_b = np.empty_like(x_np)
    c, h, w = x_np.shape[1:]
    ph, pw = h // pipe.gh, w // pipe.gw
    for k in range(n_img):
        mo = incremental_allocation(a_orig[k], pipe.b_target, pipe.m_max)
        mb = incremental_allocation(a_base[k], pipe.b_target, pipe.m_max)
        spear.append(_spearman(a_orig[k], a_base[k]))
        kend.append(_kendall_tau(a_orig[k], a_base[k]))
        m_agree.append(float((mo == mb).mean()))
        m_tv.append(float(np.abs(mo - mb).sum()) / (2 * pipe.b_target))
        # 用各自分配在无信道下重建(allocation from a_orig / a_base,载荷为原图 patch)
        recon_o[k] = _from_patches(
            pipe.decode(pipe.encode(x_np[k], a_orig[k])), c, pipe.gh, pipe.gw, ph, pw)
        recon_b[k] = _from_patches(
            pipe.decode(pipe.encode(x_np[k], a_base[k])), c, pipe.gh, pipe.gw, ph, pw)

    acc_o = float((pipe.enc.classify(torch.from_numpy(recon_o).to(x.dtype))[1].cpu()
                   == y).float().mean())
    acc_b = float((pipe.enc.classify(torch.from_numpy(recon_b).to(x.dtype))[1].cpu()
                   == y).float().mean())

    results = {
        "n": n_img, "b_target": pipe.b_target, "m_base": args.m_base,
        "arch": pipe.enc.arch,
        "spearman_mean": float(np.mean(spear)),
        "kendall_tau_mean": float(np.mean(kend)),
        "alloc_agree_mean": float(np.mean(m_agree)),
        "alloc_tv_mean": float(np.mean(m_tv)),
        "acc_alloc_orig": acc_o,
        "acc_alloc_base": acc_b,
        "acc_diff": acc_o - acc_b,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
