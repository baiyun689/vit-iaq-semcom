"""A1 压缩图注意力提取器微调(路 B,change iaq-a1-metadata-free §2.4 退路·大规模版)。

§2.4 闸门发现:零训练 A1 在 base 压缩图上算注意力 = 跑到分布外(网络在干净
CIFAR-100 上微调,没见过 1bit/2bit base 重建图),派生分配掉 ~15pt。修复头探针
(196→196 标量 MLP)只补回 ~1/3,因为它够不着图像信息。

本脚本做正解:复制一份 CIFAR-100 微调过的 vit_base 当**专职注意力提取器**,
在 **base 压缩重建图**上微调,**目标=蒸馏干净图注意力**:
    teacher = 冻结原 vit_base 在原图的 class-token 注意力 a_orig(分布,detach)
    student = 可训副本在压缩图上算的注意力 a_stu(可微)
    loss    = KL(a_orig || a_stu)
让 attention(压缩) ≈ attention(原图) → 收端在 base 上派生出接近理想的分配。

要点:
- 分类裁判 = 冻结原 vit_base,全程不动(不挪评分尺子)。
- 收发两端共用这个微调后的压缩图注意力网络 → 派生仍逐位一致,**metadata-free 不破**。
- 微调是针对某个 m_base 压缩分布的;换工作点需重训或跨 m_base 一起训。

⚠️ 这是 GPU 重活(全量 50k + ViT-Base 微调),本机 GPU 满载会过热关机 ——
**请在另一台机器上跑**。本机仅可 --device cpu 极小样本冒烟。

用法(目标机器):
    python scripts/run_a1_finetune_attention.py --n-train 50000 --n-test 1000 \
        --m_base 1 --trainable-blocks 2 --epochs 5 --batch 64 --device cuda
评估三档:acc(原图派生)上界 / acc(base 零训练) / acc(微调 student 派生),报缺口补回比例。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vit_iaq_semcom.a1 import base_reconstruct as _base_recon_one
from vit_iaq_semcom.pipeline import IAQPipeline, _from_patches
from vit_iaq_semcom.vit_encoder import ViTEncoder


def base_reconstruct(x: np.ndarray, m_base: int) -> np.ndarray:
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


def student_attention(model, images, num_prefix, normalize=True):
    """可微的 class-token→patch 注意力(不 no_grad、不 detach),供反传训练。"""
    attn_module = model.blocks[-1].attn
    prev_fused = attn_module.fused_attn
    attn_module.fused_attn = False
    store: dict = {}

    def _hook(_m, inp, _out):
        store["attn"] = inp[0]  # (B,H,N,N) softmax 后权重,保留计算图

    handle = attn_module.attn_drop.register_forward_hook(_hook)
    try:
        model(images)
    finally:
        handle.remove()
        attn_module.fused_attn = prev_fused

    cls_to_patch = store["attn"][:, :, 0, num_prefix:]  # (B,H,P)
    a = cls_to_patch.mean(dim=1)                         # (B,P)
    if normalize:
        a = a / a.sum(dim=-1, keepdim=True)
    return a


def set_trainable(model, n_blocks: int):
    """冻结全部,再解冻最后 n_blocks 个 transformer block + 末层 norm。n_blocks<0 = 全解冻。"""
    for p in model.parameters():
        p.requires_grad_(False)
    if n_blocks < 0:
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for blk in model.blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in model.norm.parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--b_target", type=int, default=392)
    ap.add_argument("--m_base", type=int, default=1)
    ap.add_argument("--trainable-blocks", type=int, default=2,
                    help="解冻最后几个 block(-1=全微调)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/a1_finetune_attention.json")
    ap.add_argument("--ckpt", default="outputs/a1_student_attn.pt")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import torchvision
    import yaml

    torch.manual_seed(0)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["device"] = args.device

    # teacher / 裁判 = 冻结原 vit_base(server 角色);student = 同权重可训副本
    pipe = IAQPipeline.from_config(cfg)
    pipe.b_target = args.b_target
    teacher = pipe.enc
    student = ViTEncoder.from_config(cfg, role="server")
    dev = student.device
    print(f"encoder={teacher.arch} device={dev} b_target={pipe.b_target} "
          f"m_base={args.m_base} trainable_blocks={args.trainable_blocks}")

    params = set_trainable(student.model, args.trainable_blocks)
    n_param = sum(p.numel() for p in params)
    student.model.train()
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"可训参数: {n_param/1e6:.2f}M")

    tf = teacher.transform
    tr = torchvision.datasets.CIFAR100(args.data_root, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(args.data_root, train=False, download=True, transform=tf)

    def load(ds, n):
        xs, ys = zip(*[ds[i] for i in range(min(n, len(ds)))])
        return torch.stack(xs), torch.tensor(ys)

    xtr, _ = load(tr, args.n_train)
    xte, yte = load(te, args.n_test)
    np_prefix = student.num_prefix_tokens

    # -------------------------------------------------------------- 训练
    nb = xtr.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(nb)
        run = 0.0
        for s in range(0, nb, args.batch):
            idx = perm[s:s + args.batch]
            xb = xtr[idx]
            x_base = torch.from_numpy(
                base_reconstruct(xb.numpy(), args.m_base)).to(xb.dtype).to(dev)
            with torch.no_grad():
                a_tea = teacher.attention_scores(xb.to(dev))             # (B,P) 分布,detach
            a_stu = student_attention(student.model, x_base, np_prefix)  # 可微
            loss = F.kl_div(a_stu.clamp_min(1e-12).log(), a_tea, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.detach()) * xb.shape[0]
        print(f"epoch {ep+1}/{args.epochs} kl={run/nb:.5f}")

    # -------------------------------------------------------------- 评估
    student.model.eval()
    x_np = xte.numpy()
    x_base_te = base_reconstruct(x_np, args.m_base)
    with torch.no_grad():
        a_orig = teacher.attention_scores(xte.to(dev)).cpu().numpy()              # 上界
        a_base = teacher.attention_scores(
            torch.from_numpy(x_base_te).to(xte.dtype).to(dev)).cpu().numpy()      # 零训练
        a_stu = []
        for s in range(0, x_np.shape[0], args.batch):
            xb = torch.from_numpy(x_base_te[s:s + args.batch]).to(xte.dtype).to(dev)
            a_stu.append(student_attention(student.model, xb, np_prefix).cpu().numpy())
        a_stu = np.concatenate(a_stu, 0)                                          # 微调派生

    c, h, w = x_np.shape[1:]
    ph, pw = h // pipe.gh, w // pipe.gw

    def acc_for(a_src):
        recon = np.empty_like(x_np)
        for k in range(x_np.shape[0]):
            recon[k] = _from_patches(
                pipe.decode(pipe.encode(x_np[k], a_src[k])), c, pipe.gh, pipe.gw, ph, pw)
        preds = teacher.classify(torch.from_numpy(recon).to(xte.dtype))[1].cpu()
        return float((preds == yte).float().mean())

    acc_o, acc_b, acc_s = acc_for(a_orig), acc_for(a_base), acc_for(a_stu)
    sp_b = float(np.mean([_spearman(a_orig[k], a_base[k]) for k in range(len(a_orig))]))
    sp_s = float(np.mean([_spearman(a_orig[k], a_stu[k]) for k in range(len(a_orig))]))

    res = {
        "n_train": xtr.shape[0], "n_test": x_np.shape[0], "b_target": args.b_target,
        "m_base": args.m_base, "trainable_blocks": args.trainable_blocks,
        "epochs": args.epochs, "lr": args.lr, "arch": teacher.arch,
        "trainable_params_M": n_param / 1e6,
        "spearman_base": sp_b, "spearman_tuned": sp_s,
        "acc_alloc_orig": acc_o, "acc_alloc_base": acc_b, "acc_alloc_tuned": acc_s,
        "acc_diff_base": acc_o - acc_b, "acc_diff_tuned": acc_o - acc_s,
        "gap_closed_frac": ((acc_s - acc_b) / (acc_o - acc_b)) if acc_o > acc_b else None,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(student.model.state_dict(), args.ckpt)
    print(f"saved metrics: {Path(args.out).resolve()}")
    print(f"saved student ckpt: {Path(args.ckpt).resolve()}")


if __name__ == "__main__":
    main()
