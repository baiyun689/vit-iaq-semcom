"""repair.py 单测（spec: iaq-semantic-repair）。小网络、CPU、不加载真实分类器。"""

import numpy as np
import torch

from vit_iaq_semcom.repair import (
    SemanticRepairNet,
    patchify,
    repair_reconstruction,
    unpatchify,
)


def _net(grid=(4, 4), patch_dim=(3, 2, 2), m_max=4):
    torch.manual_seed(0)
    return SemanticRepairNet(grid=grid, patch_dim=patch_dim, m_max=m_max,
                             dim=24, depth=2, heads=2)


def _img(b=2, c=3, h=8, w=8, seed=0):
    return torch.from_numpy(
        np.random.default_rng(seed).standard_normal((b, c, h, w))).float()


def test_patchify_roundtrip():
    img = _img()
    p = patchify(img, 4, 4)
    assert p.shape == (2, 16, 3 * 2 * 2)
    back = unpatchify(p, 3, 4, 4, 2, 2)
    torch.testing.assert_close(back, img)


def test_residual_fallback_is_identity_at_init():
    """head 零初始化 → 残差=0 → 输出恒等于输入(最坏≈不修,不帮倒忙)。"""
    net = _net()
    img = _img(seed=1)
    bd = torch.randint(0, 5, (2, 16))
    out = net(img, bd)
    torch.testing.assert_close(out, img, atol=1e-6, rtol=0)


def test_reliability_map_is_an_input():
    """比特深度图确实作为输入通道:扰动 head 使非恒等后,改 bit_depth → 输出改变。"""
    net = _net()
    torch.nn.init.normal_(net.head.weight, std=0.1)   # 解除零初始化,让网络真的输出残差
    img = _img(seed=2)
    bd1 = torch.zeros(2, 16, dtype=torch.long)
    bd2 = torch.full((2, 16), 4, dtype=torch.long)
    out1, out2 = net(img, bd1), net(img, bd2)
    assert (out1 - out2).abs().max().item() > 1e-6


def test_only_repair_params_present_and_trainable():
    """网络自身参数可训(分类器冻结在训练脚本层面;此处验证修补器参数齐备可训)。"""
    net = _net()
    assert all(p.requires_grad for p in net.parameters())
    img = _img(seed=3)
    bd = torch.randint(0, 5, (2, 16))
    loss = net(img, bd).pow(2).mean()
    loss.backward()
    # 残差路径有梯度(patch_embed / depth_embed / blocks 都接到输出)
    assert net.depth_embed.weight.grad is not None
    assert net.patch_embed.weight.grad is not None


def test_broad_damage_finite_no_nan():
    """损伤过广(全图随机大噪)→ 前向有限、无 NaN(优雅退化,残差兜底)。"""
    net = _net()
    torch.nn.init.normal_(net.head.weight, std=0.1)
    img = _img(seed=4) * 50.0                          # 夸张损伤幅度
    bd = torch.zeros(2, 16, dtype=torch.long)          # 全低比特(最不可信)
    out = net(img, bd)
    assert torch.isfinite(out).all()


def test_inference_interface_single_and_batch():
    """推理接口接受 (B,C,H,W) 与 (C,H,W),输出同形 numpy;未训练时兜底恒等。"""
    net = _net()
    recon = np.random.default_rng(5).standard_normal((2, 3, 8, 8)).astype(np.float32)
    bd = np.random.default_rng(6).integers(0, 5, (2, 16))
    fixed = repair_reconstruction(net, recon, bd)
    assert fixed.shape == recon.shape
    np.testing.assert_allclose(fixed, recon, atol=1e-6)  # 零初始化兜底
    one = repair_reconstruction(net, recon[0], bd[0])
    assert one.shape == (1, 3, 8, 8)
