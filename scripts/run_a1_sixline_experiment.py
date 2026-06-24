"""A1 六线对照实验（change: iaq-a1-metadata-free §5）。

同 ρ=0.25(b392)扫 SNR,出六条准确率曲线:
  ① 均匀(无元信息)  ② IAQ 理想(元信息无损上界)  ③ 裸传(元信息过信道不保护)
  ④ EEP(元信息&载荷等强 FEC)  ⑤ UEP(元信息强、载荷弱)  ⑥ A1(元信息-free)

两种场景:
  - 匹配(--design-snr 不给):每个 SNR 按当前 SNR 选 FEC 强度 → ④⑤⑥ 高 SNR 收敛到 ②。
  - 失配(--design-snr 8):按固定设计点选 FEC 再扫真实 SNR → ⑤ 在真实<设计时再悬崖,
    ⑥ A1 有底平滑(杀招)。

同信道带宽口径(口径 B):记录每条线的标称信道比特(信源 + FEC 膨胀),供归一化/开销表。
诚实声明:A1 不省带宽;其底可能低于 ①(分层代价),价值在消除崩盘 + 失配鲁棒。

用法(vit-iaq-semcom 环境):
    python scripts/run_a1_sixline_experiment.py --n 64 --design-snr 8   # 失配(杀招)
    python scripts/run_a1_sixline_experiment.py --n 64                  # 匹配
    python scripts/run_a1_sixline_experiment.py --replot outputs/a1_sixline_mismatch.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

N_PATCH, P_SCALARS, META_BITS = 196, 768, 800  # 4·196+16


def matched_rate(snr: float) -> int:
    """匹配场景的重复码速率分母 r(SNR 越低保护越强)。奇数。"""
    if snr < 0:
        return 5
    if snr < 4:
        return 3
    return 1


def nominal_channel_bits(line: str, b_target: int, r: int, weak_r: int = 1,
                         m_base: int = 1) -> int:
    """每条线标称信道比特(信源 + FEC 膨胀),用于同带宽开销记账。"""
    payload = P_SCALARS * b_target          # incremental 用满预算
    uni = P_SCALARS * (N_PATCH * (b_target // N_PATCH))
    base_src = P_SCALARS * (N_PATCH * m_base)
    enh_src = P_SCALARS * (b_target - N_PATCH * m_base)
    return {
        "uniform": uni,
        "ideal": payload + META_BITS,
        "raw": payload + META_BITS,
        "eep": (payload + META_BITS) * r,
        "uep": payload * weak_r + META_BITS * r,
        "a1": base_src * r + enh_src,
    }[line]


def plot_results(res: dict, out: Path) -> None:
    snr = res["snr"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(res["base_acc"], color="gray", ls=":", lw=1,
               label=f"无量化基线 ({res['base_acc']:.2f})")
    style = {
        "uniform": ("^-", "tab:green", "① 均匀(无元信息)"),
        "ideal": ("o-", "tab:blue", "② IAQ 理想(上界)"),
        "raw": ("s-", "tab:red", "③ 裸传(元信息过信道)"),
        "eep": ("D-", "tab:purple", "④ EEP"),
        "uep": ("v-", "tab:orange", "⑤ UEP"),
        "a1": ("*-", "black", "⑥ A1(元信息-free)"),
    }
    for line, (mk, col, lab) in style.items():
        ax.plot(snr, res["acc"][line], mk, color=col, label=lab)
    scen = f"失配(设计点 {res['design_snr']}dB)" if res["design_snr"] is not None else "匹配"
    rho = res["b_target"] / 1568
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("CIFAR-100 Top-1 准确率")
    ax.set_title(f"A1 六线对照 · {scen} · ρ={rho:.3f} (b{res['b_target']})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out.resolve()}")


def evaluate(pipe, x, y, snr, *, mode, alloc=None, mtc=False,
             meta_fec_r=1, payload_fec_r=1, base_fec_r=1, m_base=1, batch=64, seed=0):
    correct = total = 0
    for s in range(0, x.shape[0], batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        out = pipe.run_batch(
            xb, labels=yb, channel_type="awgn", ebn0_db=snr, mode=mode, alloc=alloc,
            metadata_through_channel=mtc, meta_fec_r=meta_fec_r,
            payload_fec_r=payload_fec_r, base_fec_r=base_fec_r, m_base=m_base, seed=seed + s)
        correct += out["accuracy"] * xb.shape[0]
        total += xb.shape[0]
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=64, help="样本数(吃 GPU,默认小)")
    ap.add_argument("--b_target", type=int, default=392)
    ap.add_argument("--snr", default="-4,-2,0,2,4,6,8,10,12")
    ap.add_argument("--design-snr", type=float, default=None,
                    help="失配场景固定设计点;不给=匹配场景(按当前 SNR 设计)")
    ap.add_argument("--m_base", type=int, default=1, help="A1 基础层每 patch 比特(工作点)")
    ap.add_argument("--attn-ckpt", default=None,
                    help="§2.5 微调后 student 注意力网络 .pt;给了则 A1 派生用它(分类仍冻结)")
    ap.add_argument("--weak-r", type=int, default=1, help="UEP 载荷弱保护速率分母")
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
    pipe.b_target = args.b_target
    if args.attn_ckpt:
        pipe.load_attn_encoder(args.attn_ckpt, cfg)
        print(f"A1 派生注意力用微调 student: {args.attn_ckpt}")
    scen = "mismatch" if args.design_snr is not None else "matched"
    tag = f"_tuned" if args.attn_ckpt else ""
    out = Path(args.out or f"outputs/a1_sixline_{scen}_b{args.b_target}{tag}.png")
    print(f"encoder={pipe.enc.arch} b_target={pipe.b_target} m_base={args.m_base} "
          f"场景={scen} design_snr={args.design_snr} attn_ckpt={args.attn_ckpt}")

    ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=pipe.enc.transform)
    xs, ys = zip(*[ds[i] for i in range(args.n)])
    x, y = torch.stack(xs), torch.tensor(ys)
    base_acc = float((pipe.enc.classify(x)[1].cpu() == y).float().mean())
    print(f"无量化基线: {base_acc:.4f}")

    snrs = [float(s) for s in args.snr.split(",")]
    acc = {k: [] for k in ("uniform", "ideal", "raw", "eep", "uep", "a1")}
    chan_bits = {}
    for snr in snrs:
        # 匹配=按当前 SNR 选 r;失配=固定按 design_snr 选 r
        r = matched_rate(args.design_snr if args.design_snr is not None else snr)
        acc["uniform"].append(evaluate(pipe, x, y, snr, mode="iaq", alloc="uniform"))
        acc["ideal"].append(evaluate(pipe, x, y, snr, mode="iaq", alloc="incremental"))
        acc["raw"].append(evaluate(pipe, x, y, snr, mode="iaq", alloc="incremental", mtc=True))
        acc["eep"].append(evaluate(pipe, x, y, snr, mode="eep", meta_fec_r=r, payload_fec_r=r))
        acc["uep"].append(evaluate(pipe, x, y, snr, mode="uep", meta_fec_r=r, payload_fec_r=args.weak_r))
        acc["a1"].append(evaluate(pipe, x, y, snr, mode="a1", base_fec_r=r, m_base=args.m_base))
        chan_bits[f"{snr}"] = {
            ln: nominal_channel_bits(ln, args.b_target, r, args.weak_r, args.m_base)
            for ln in acc}
        print(f"SNR={snr:<5g} r={r} " + " ".join(
            f"{k}={acc[k][-1]:.3f}" for k in acc))

    res = {
        "snr": snrs, "acc": acc, "base_acc": base_acc, "n": args.n,
        "b_target": args.b_target, "design_snr": args.design_snr,
        "m_base": args.m_base, "attn_ckpt": args.attn_ckpt,
        "weak_r": args.weak_r, "arch": pipe.enc.arch,
        "channel_bits": chan_bits,
    }
    out_json = out.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_json.resolve()}")
    plot_results(res, out)


if __name__ == "__main__":
    main()
