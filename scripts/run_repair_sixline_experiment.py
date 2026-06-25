"""粗桶 + 学习式修补 六线对照实验（change iaq-learned-semantic-repair，task 5.1）。

同总信道比特(口径 = 768×b_target)扫 SNR,出六条 CIFAR-100 Top-1 曲线:
  ⓪ 理想(oracle 上界)      = IAQ 增量,元信息无损直达(不过信道)
  ① 均匀(metadata-free 基线) = IAQ 均匀分配
  ③ 裸传元信息(悬崖)         = IAQ 增量,元信息过信道**不保护**(预算内扣元信息位)
  ⑥ IAQ+保护元信息(经典修法) = IAQ 增量,元信息强 FEC、载荷不保护(预算内扣元信息+FEC 膨胀)
  ④ 粗桶(无修补)            = bucket 非级联源编码
  ⑤ 粗桶+AI 修补(本方案)     = bucket+repair,零额外带宽(修补在收端)

同带宽记账:③⑥ 把元信息(及其 FEC)从信源预算内扣(ΣM_i 让位),⓪ 与 ③ 同信源预算、
唯一差别是元信息过不过信道 → **⓪−③ 隔离纯悬崖**;④−① 隔离重要性红利;
**⑤−④ 隔离 AI 增益(零额外带宽)**。

双场景(沿用 A1 六线约定):
  - 匹配(--design-snr 不给):每 SNR 按当前 SNR 设计 FEC → ⑥ 高 SNR 收敛到 ⓪(诚实:经典也能修)。
  - 失配(--design-snr 8):FEC 按固定设计点配死再扫真实 SNR → ⑥ 真实<设计时再悬崖、⑤ 平滑(杀招)。

⚠️ 跑全扫吃 GPU(ViT-Base 前向),本机过热 —— 请在另一台机器跑。⑤ 需 --repair-ckpt
(run_repair_train.py 产出);缺省时 ⑤ 退化为 ④ 并告警。
纯出图(--replot a.json)可在本机 CPU。

用法(目标机器):
    python scripts/run_repair_sixline_experiment.py --n 256 --repair-ckpt outputs/repair_net.pt
    python scripts/run_repair_sixline_experiment.py --n 256 --design-snr 8 --repair-ckpt ...  # 失配
    python scripts/run_repair_sixline_experiment.py --replot outputs/repair_sixline_matched_b392.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

N_PATCH, P_SCALARS, META_BITS = 196, 768, 800  # 元信息 = 4·196 + 16


def parse_buckets(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        cnt, bits = part.split(":")
        out.append((int(cnt), int(bits)))
    return out


def matched_rate(snr: float) -> int:
    """重复码速率分母 r(SNR 越低保护越强,奇数)。"""
    if snr < 0:
        return 5
    if snr < 4:
        return 3
    return 1


def deducted_b_target(b_full: int, meta_r: int, with_meta: bool) -> int:
    """同带宽:把元信息(及 FEC 膨胀)从信源预算内扣,返回该线的信源 b_target。

    768·b_src + META·meta_r = 768·b_full  →  b_src = b_full − ⌈META·meta_r/768⌉。
    with_meta=False(⓪①④⑤)不扣;③ meta_r=1;⑥ meta_r=r。
    """
    if not with_meta:
        return b_full
    return b_full - math.ceil(META_BITS * meta_r / P_SCALARS)


def nominal_channel_bits(line: str, b_full: int, r: int) -> int:
    """每条线标称总信道比特(同带宽口径,应全部≈768·b_full)。"""
    if line in ("ideal", "uniform", "bucket", "bucket_repair"):
        return P_SCALARS * b_full
    if line == "raw":
        return P_SCALARS * deducted_b_target(b_full, 1, True) + META_BITS
    if line == "protected":
        return P_SCALARS * deducted_b_target(b_full, r, True) + META_BITS * r
    raise ValueError(line)


def evaluate(pipe, x, y, snr, *, mode, alloc=None, mtc=False, meta_fec_r=1,
             payload_fec_r=1, base_fec_r=1, m_base=1, buckets=None,
             b_target=None, batch=64, seed=0):
    if b_target is not None:
        pipe.b_target = b_target
    correct = total = 0
    for s in range(0, x.shape[0], batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        out = pipe.run_batch(
            xb, labels=yb, channel_type="awgn", ebn0_db=snr, mode=mode, alloc=alloc,
            metadata_through_channel=mtc, meta_fec_r=meta_fec_r,
            payload_fec_r=payload_fec_r, base_fec_r=base_fec_r, m_base=m_base,
            buckets=buckets, seed=seed + s)
        correct += out["accuracy"] * xb.shape[0]
        total += xb.shape[0]
    return correct / total


def plot_results(res: dict, out: Path) -> None:
    snr = res["snr"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.axhline(res["base_acc"], color="gray", ls=":", lw=1,
               label=f"无量化基线 ({res['base_acc']:.2f})")
    style = {
        "uniform": ("^-", "tab:green", "均匀量化"),
        "ideal": ("o-", "tab:blue", "IAQ·元信息无损"),
        "raw": ("s-", "tab:red", "IAQ·元信息裸传"),
        "protected": ("D-", "tab:purple", "IAQ·元信息FEC保护"),
        "bucket": ("v-", "tab:orange", "粗桶分层"),
        "bucket_repair": ("*-", "black", "粗桶分层+语义修补"),
    }
    for line, (mk, col, lab) in style.items():
        if line in res["acc"]:
            ax.plot(snr, res["acc"][line], mk, color=col, label=lab)
    rho = res["b_target"] / 1568
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("CIFAR-100 Top-1 准确率")
    ax.set_title(f"比特预算 B={res['b_target']} (ρ={rho:.2f})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=256, help="样本数(吃 GPU,默认小)")
    ap.add_argument("--b_target", type=int, default=392)
    ap.add_argument("--m_base", type=int, default=1)
    ap.add_argument("--buckets", default="49:4,49:2,98:1", help="粗桶配置 cnt:总比特,...")
    ap.add_argument("--snr", default="-4,-2,0,2,4,6,8,10,12")
    ap.add_argument("--design-snr", type=float, default=None,
                    help="失配场景固定设计点;不给=匹配场景")
    ap.add_argument("--repair-ckpt", default=None, help="修补网络 .pt(缺省 ⑤ 退化为 ④)")
    ap.add_argument("--bucket-base-fec-r", type=int, default=1,
                    help="④⑤ 基础层 FEC(默认 1=不保护,最干净同带宽)")
    # 修补网络结构(须与训练一致)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replot", default=None)
    args = ap.parse_args()

    if args.replot:
        res = json.loads(Path(args.replot).read_text(encoding="utf-8"))
        plot_results(res, Path(args.out or Path(args.replot).with_suffix(".png")))
        return

    import torch
    import torchvision
    import yaml

    from vit_iaq_semcom.pipeline import IAQPipeline

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    pipe = IAQPipeline.from_config(cfg)
    pipe.m_max = max(pipe.m_max, max(b for _, b in parse_buckets(args.buckets)))
    buckets = parse_buckets(args.buckets)
    b_full = args.b_target

    if args.repair_ckpt:
        gh, gw = pipe.gh, pipe.gw
        pipe.load_repair_net(
            args.repair_ckpt, grid=(gh, gw), patch_dim=(3, 224 // gh, 224 // gw),
            m_max=pipe.m_max, dim=args.dim, depth=args.depth, heads=args.heads)
        print(f"⑤ 修补网络: {args.repair_ckpt}")
    else:
        print("⚠️ 未给 --repair-ckpt,⑤ 退化为 ④(纯粗桶)")

    scen = "mismatch" if args.design_snr is not None else "matched"
    out = Path(args.out or f"outputs/repair_sixline_{scen}_b{b_full}.png")
    print(f"encoder={pipe.enc.arch} b_full={b_full} m_base={args.m_base} "
          f"buckets={buckets} 场景={scen} design_snr={args.design_snr}")

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform)
    xs, ys = zip(*[ds[i] for i in range(args.n)])
    x, y = torch.stack(xs), torch.tensor(ys)
    base_acc = float((pipe.enc.classify(x)[1].cpu() == y).float().mean())
    print(f"无量化基线: {base_acc:.4f}")

    snrs = [float(s) for s in args.snr.split(",")]
    acc = {k: [] for k in
           ("ideal", "uniform", "raw", "protected", "bucket", "bucket_repair")}
    chan_bits = {}
    for snr in snrs:
        r = matched_rate(args.design_snr if args.design_snr is not None else snr)
        b_raw = deducted_b_target(b_full, 1, True)        # ③ 扣元信息
        b_prot = deducted_b_target(b_full, r, True)       # ⑥ 扣元信息+FEC
        acc["ideal"].append(evaluate(
            pipe, x, y, snr, mode="iaq", alloc="incremental", b_target=b_full))
        acc["uniform"].append(evaluate(
            pipe, x, y, snr, mode="iaq", alloc="uniform", b_target=b_full))
        acc["raw"].append(evaluate(
            pipe, x, y, snr, mode="iaq", alloc="incremental", mtc=True, b_target=b_raw))
        acc["protected"].append(evaluate(
            pipe, x, y, snr, mode="uep", meta_fec_r=r, payload_fec_r=1, b_target=b_prot))
        acc["bucket"].append(evaluate(
            pipe, x, y, snr, mode="bucket", m_base=args.m_base, buckets=buckets,
            base_fec_r=args.bucket_base_fec_r))
        acc["bucket_repair"].append(evaluate(
            pipe, x, y, snr, mode="bucket+repair", m_base=args.m_base, buckets=buckets,
            base_fec_r=args.bucket_base_fec_r))
        chan_bits[f"{snr}"] = {ln: nominal_channel_bits(ln, b_full, r) for ln in acc}
        print(f"SNR={snr:<5g} r={r} " + " ".join(f"{k}={acc[k][-1]:.3f}" for k in acc))

    res = {
        "snr": snrs, "acc": acc, "base_acc": base_acc, "n": args.n,
        "b_target": b_full, "m_base": args.m_base, "buckets": buckets,
        "design_snr": args.design_snr, "repair_ckpt": args.repair_ckpt,
        "bucket_base_fec_r": args.bucket_base_fec_r, "arch": pipe.enc.arch,
        "channel_bits": chan_bits,
    }
    out_json = out.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_json.resolve()}")
    plot_results(res, out)


if __name__ == "__main__":
    main()
