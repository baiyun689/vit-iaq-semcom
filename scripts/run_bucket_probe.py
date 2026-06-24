"""粗桶分配可行性探针（brainstorm: 非级联重要性分配 + AI 修补，验证假设）。

纯推理、不训练、不建修补网络。回答两个未知数:
  (a) 粗桶留住多少 IAQ 红利? —— 干净信道下比 均匀 / 增量(理想IAQ) / 粗桶 三种分配的准确率。
  (b) 派生错几个 patch? —— 用基础层重建图派生的"桶分配" vs 用原图派生的,量错配比例
      (=AI 修补的工作量),并看据此重建的准确率掉多少。

粗桶 = 按注意力排序入"固定大小、固定比特"的桶(非级联的关键);默认 b588:
  49×6 + 49×4 + 98×1 = 588。排序入桶取代增量分配。

用法(vit-iaq-semcom 环境,建议 CPU 小样本):
    python scripts/run_bucket_probe.py --n 128 --device cpu
    python scripts/run_bucket_probe.py --n 128 --buckets 98:4,98:2 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vit_iaq_semcom.a1 import base_reconstruct as _base_recon_one
from vit_iaq_semcom.bit_allocation import incremental_allocation, uniform_allocation
from vit_iaq_semcom.bucket import bucket_alloc, bucket_index  # 单一真源(src/bucket.py)
from vit_iaq_semcom.pipeline import IAQPipeline, _from_patches, _to_patches
from vit_iaq_semcom.quantization import uniform_dequantize, uniform_quantize


def parse_buckets(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        cnt, bits = part.split(":")
        out.append((int(cnt), int(bits)))
    return out


def reconstruct(img: np.ndarray, m_map: np.ndarray, umin: float, umax: float,
                gh: int, gw: int) -> np.ndarray:
    """干净信道:按 M_i 逐 patch 均匀量化再反量化(m=0 → 中点)。"""
    c, h, w = img.shape
    ph, pw = h // gh, w // gw
    patches = _to_patches(img, gh, gw)
    out = np.empty_like(patches)
    mid = 0.5 * (umin + umax)
    for i in range(patches.shape[0]):
        m = int(m_map[i])
        if m <= 0:
            out[i] = mid
        else:
            q = uniform_quantize(patches[i], m, umin, umax)
            out[i] = uniform_dequantize(q, m, umin, umax)
    return _from_patches(out, c, gh, gw, ph, pw)


def acc_for(pipe, x_np, y, m_maps, dtype, batch=64):
    """给每图一组 M_i,重建→分类→准确率。"""
    import torch
    recon = np.empty_like(x_np)
    for k in range(x_np.shape[0]):
        umin, umax = float(x_np[k].min()), float(x_np[k].max())
        recon[k] = reconstruct(x_np[k], m_maps[k], umin, umax, pipe.gh, pipe.gw)
    correct = total = 0
    for s in range(0, recon.shape[0], batch):
        xb = torch.from_numpy(recon[s:s + batch]).to(dtype)
        preds = pipe.enc.classify(xb)[1].cpu().numpy()
        correct += int((preds == y[s:s + batch]).sum())
        total += xb.shape[0]
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--b_target", type=int, default=588)
    ap.add_argument("--m_base", type=int, default=2, help="派生排序用的基础层比特")
    ap.add_argument("--buckets", default="49:6,49:4,98:1")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/bucket_probe.json")
    args = ap.parse_args()

    import torch
    import torchvision
    import yaml

    buckets = parse_buckets(args.buckets)
    bsum = sum(c * b for c, b in buckets)
    bcnt = sum(c for c, _ in buckets)
    print(f"buckets={buckets} 总比特={bsum} (b_target={args.b_target}) 覆盖patch={bcnt}")
    if bsum != args.b_target:
        print(f"⚠️ 桶总比特 {bsum} != b_target {args.b_target},预算不完全对齐,解读注意")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    pipe = IAQPipeline.from_config(cfg)
    pipe.b_target = args.b_target
    print(f"encoder={pipe.enc.arch} device={pipe.enc.device} b_target={pipe.b_target} "
          f"m_base={args.m_base}")

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform)
    xs, ys = zip(*[ds[i] for i in range(args.n)])
    x, y = torch.stack(xs), np.array(ys)
    x_np = x.cpu().numpy()
    n_patch = pipe.gh * pipe.gw

    base_acc = float((pipe.enc.classify(x)[1].cpu().numpy() == y).mean())

    # 原图注意力
    a_orig = pipe.enc.attention_scores(x).cpu().numpy()
    # 基础层重建图注意力(派生用)
    x_base = np.stack([
        _base_recon_one(x_np[k], args.m_base, float(x_np[k].min()), float(x_np[k].max()))
        for k in range(x_np.shape[0])])
    a_base = pipe.enc.attention_scores(torch.from_numpy(x_base).to(x.dtype)).cpu().numpy()

    # ---- (a) 干净信道:均匀 / 增量 / 粗桶(均由原图注意力派生)----
    M_uni = [uniform_allocation(n_patch, args.b_target, pipe.m_max) for _ in range(x_np.shape[0])]
    M_inc = [incremental_allocation(a_orig[k], args.b_target, pipe.m_max) for k in range(x_np.shape[0])]
    M_buk = [bucket_alloc(a_orig[k], buckets) for k in range(x_np.shape[0])]

    acc_uni = acc_for(pipe, x_np, y, M_uni, x.dtype)
    acc_inc = acc_for(pipe, x_np, y, M_inc, x.dtype)
    acc_buk = acc_for(pipe, x_np, y, M_buk, x.dtype)

    # ---- (b) 派生定位:桶分配 原图 vs 基础层 ----
    Bo = np.stack([bucket_index(a_orig[k], buckets) for k in range(x_np.shape[0])])
    Bb = np.stack([bucket_index(a_base[k], buckets) for k in range(x_np.shape[0])])
    mis_frac = float((Bo != Bb).mean())                       # 任意错配(含相邻边界抖动)
    severe_frac = float((np.abs(Bo - Bb) >= 2).mean())        # 严重错配(跨≥2档,如 0↔2)
    # 据基础层派生的粗桶重建准确率(=收端实际能拿到的)
    M_buk_base = [bucket_alloc(a_base[k], buckets) for k in range(x_np.shape[0])]
    acc_buk_base = acc_for(pipe, x_np, y, M_buk_base, x.dtype)

    res = {
        "n": int(x_np.shape[0]), "b_target": args.b_target, "m_base": args.m_base,
        "buckets": buckets, "bucket_bits_sum": bsum, "arch": pipe.enc.arch,
        "base_acc_no_quant": base_acc,
        "a_clean_uniform": acc_uni,
        "a_clean_incremental": acc_inc,
        "a_clean_bucket": acc_buk,
        "bucket_keeps_of_iaq_gain": (
            (acc_buk - acc_uni) / (acc_inc - acc_uni) if acc_inc > acc_uni else None),
        "b_misbucket_frac": mis_frac,
        "b_severe_misbucket_frac": severe_frac,
        "b_acc_bucket_from_base": acc_buk_base,
        "b_acc_drop_vs_orig": acc_buk - acc_buk_base,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
