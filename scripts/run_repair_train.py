"""训练收端语义修补网络（change iaq-learned-semantic-repair，task 3.2）。

数据 = CIFAR-100 过**完整粗桶非级联管线 + 随机 SNR 的 AWGN** 得到的受损重建 + 比特深度图；
目标 = **分类任务**(交叉熵为主 + 重建 MSE 为辅),跨 SNR 训一个网络。

冻结：分类裁判(含其 patch_embed)全程不动；只训修补网络。梯度经 repaired 图回流到
修补器(分类器权重冻结、但输入可微)。**用任务目标训练**,不重蹈 §2.5 代理目标(逼近干净
图注意力)摊平注意力的覆辙。

⚠️ GPU 重活(CIFAR-100 + ViT-Base 前向 × 每步)。本机 GPU 满载会过热关机 —— **请在另一台
机器上跑**。本机仅可 `--device cpu --n-train 64 --epochs 1` 极小样本冒烟验证管路。

用法(目标机器)：
    python scripts/run_repair_train.py --n-train 50000 --epochs 10 --batch 64 \
        --m_base 1 --buckets 49:4,49:2,98:1 --snr-lo 0 --snr-hi 12 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_buckets(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        cnt, bits = part.split(":")
        out.append((int(cnt), int(bits)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--b_target", type=int, default=392)
    ap.add_argument("--m_base", type=int, default=1)
    ap.add_argument("--buckets", default="49:4,49:2,98:1", help="桶配置 cnt:总比特,...")
    ap.add_argument("--base-fec-r", type=int, default=1, help="基础层重复码(1=不保护)")
    ap.add_argument("--snr-lo", type=float, default=0.0, help="训练 Eb/N0 区间下界(dB)")
    ap.add_argument("--snr-hi", type=float, default=12.0, help="训练 Eb/N0 区间上界(dB)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-mse", type=float, default=0.1, help="重建 MSE 辅损权重")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-snrs", default="0,3,6,9,12")
    ap.add_argument("--out", default="outputs/repair_train.json")
    ap.add_argument("--ckpt", default="outputs/repair_net.pt")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import torchvision
    import yaml

    from vit_iaq_semcom.pipeline import IAQPipeline
    from vit_iaq_semcom.repair import SemanticRepairNet

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["device"] = args.device
    buckets = parse_buckets(args.buckets)

    pipe = IAQPipeline.from_config(cfg)
    pipe.b_target = args.b_target
    clf = pipe.enc                                    # 冻结分类裁判
    for p in clf.model.parameters():
        p.requires_grad_(False)
    dev = clf.device

    gh, gw = pipe.gh, pipe.gw
    # patch 维度由 224/grid 推出(deit/vit patch16 → 16×16×3)
    ph, pw = 224 // gh, 224 // gw
    net = SemanticRepairNet(
        grid=(gh, gw), patch_dim=(3, ph, pw), m_max=pipe.m_max,
        dim=args.dim, depth=args.depth, heads=args.heads).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    n_param = sum(p.numel() for p in net.parameters())
    print(f"encoder={clf.arch} device={dev} b_target={pipe.b_target} m_base={args.m_base} "
          f"buckets={buckets} 修补参数={n_param/1e6:.2f}M snr=[{args.snr_lo},{args.snr_hi}]dB")

    tf = clf.transform
    tr = torchvision.datasets.CIFAR100(args.data_root, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(args.data_root, train=False, download=True, transform=tf)

    def load(ds, n):
        xs, ys = zip(*[ds[i] for i in range(min(n, len(ds)))])
        return torch.stack(xs), torch.tensor(ys)

    xtr, ytr = load(tr, args.n_train)
    xte, yte = load(te, args.n_test)
    nb = xtr.shape[0]

    # ------------------------------------------------------------------ 训练
    for ep in range(args.epochs):
        perm = torch.randperm(nb)
        run_ce = run_mse = 0.0
        for s in range(0, nb, args.batch):
            idx = perm[s:s + args.batch]
            xb = xtr[idx]
            yb = ytr[idx].to(dev)
            ebn0 = float(rng.uniform(args.snr_lo, args.snr_hi))   # 跨 SNR:每步随机
            recon_np, bdm_np = pipe.bucket_corrupt(
                xb.numpy(), ebn0, args.m_base, args.base_fec_r, buckets, rng, xb.dtype)
            corrupted = torch.from_numpy(recon_np).to(xb.dtype).to(dev)
            bdm = torch.from_numpy(bdm_np).to(dev)
            clean = xb.to(dev)

            repaired = net(corrupted, bdm)                        # 可微(梯度只进修补器)
            logits = clf.model(repaired)                          # 冻结分类器,输入可微
            ce = F.cross_entropy(logits, yb)
            mse = F.mse_loss(repaired, clean)
            loss = ce + args.lambda_mse * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_ce += float(ce.detach()) * xb.shape[0]
            run_mse += float(mse.detach()) * xb.shape[0]
        print(f"epoch {ep+1}/{args.epochs} ce={run_ce/nb:.4f} mse={run_mse/nb:.5f}")

    # ------------------------------------------------------------------ 评估
    net.eval()
    eval_snrs = [float(v) for v in args.eval_snrs.split(",")]

    @torch.no_grad()
    def acc_at(ebn0, repair):
        correct = total = 0
        for s in range(0, xte.shape[0], args.batch):
            xb = xte[s:s + args.batch]
            yb = yte[s:s + args.batch]
            recon_np, bdm_np = pipe.bucket_corrupt(
                xb.numpy(), ebn0, args.m_base, args.base_fec_r, buckets, rng, xb.dtype)
            x_hat = torch.from_numpy(recon_np).to(xb.dtype).to(dev)
            if repair:
                x_hat = net(x_hat, torch.from_numpy(bdm_np).to(dev))
            preds = clf.model(x_hat).argmax(1).cpu()
            correct += int((preds == yb).sum())
            total += xb.shape[0]
        return correct / total

    curve = {}
    for snr in eval_snrs:
        a_bucket = acc_at(snr, repair=False)          # ④ 粗桶无修补
        a_repair = acc_at(snr, repair=True)           # ⑤ 粗桶 + 修补
        curve[snr] = {"bucket": a_bucket, "bucket_repair": a_repair,
                      "ai_gain": a_repair - a_bucket}
        print(f"SNR={snr:>5}dB  ④bucket={a_bucket:.4f}  ⑤repair={a_repair:.4f}  "
              f"AI增益={a_repair - a_bucket:+.4f}")

    res = {
        "n_train": int(nb), "n_test": int(xte.shape[0]), "b_target": args.b_target,
        "m_base": args.m_base, "buckets": buckets, "base_fec_r": args.base_fec_r,
        "snr_train": [args.snr_lo, args.snr_hi], "epochs": args.epochs,
        "lr": args.lr, "lambda_mse": args.lambda_mse, "arch": clf.arch,
        "repair_params_M": n_param / 1e6, "curve": curve,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(net.state_dict(), args.ckpt)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"saved metrics: {Path(args.out).resolve()}")
    print(f"saved repair ckpt: {Path(args.ckpt).resolve()}")


if __name__ == "__main__":
    main()
