"""pipeline.py 单测（spec: iaq-pipeline）。

核心逻辑用随机权重 DeiT-Tiny（不联网）验证；干净信道准确率对照用真权重 + 真
数据，打 network mark，离线跳过。
"""

import numpy as np
import pytest
import torch

from vit_iaq_semcom.pipeline import IAQPipeline, _from_patches, _to_patches
from vit_iaq_semcom.vit_encoder import ViTEncoder


def test_patch_reshape_inverse():
    """_to_patches / _from_patches 互逆（与量化无关的纯重排正确性）。"""
    rng = np.random.default_rng(0)
    img = rng.normal(size=(3, 224, 224))
    patches = _to_patches(img, 14, 14)
    assert patches.shape == (196, 3 * 16 * 16)
    back = _from_patches(patches, c=3, gh=14, gw=14, ph=16, pw=16)
    np.testing.assert_allclose(back, img)


@pytest.fixture(scope="module")
def rand_pipe():
    enc = ViTEncoder(arch="deit_tiny_patch16_224", weights="random",
                     num_classes=100, device="cpu")
    return IAQPipeline(enc, m_max=8, b_target=10_000)  # 大预算 → 近无损


def test_clean_channel_shapes_and_near_lossless(rand_pipe):
    x = torch.randn(2, 3, 224, 224)
    out = rand_pipe.run_batch(x, mu=0.0, metadata_through_channel=False)
    assert out["logits"].shape == (2, 100)
    assert out["preds"].shape == (2,)
    assert out["recon_mse"] < 1e-3  # 8-bit 量化、干净信道 ≈ 无损


def test_lower_budget_higher_mse():
    """预算降低 → 重建 MSE 升（率失真趋势）。"""
    enc = ViTEncoder(arch="deit_tiny_patch16_224", weights="random",
                     num_classes=100, device="cpu")
    x = torch.randn(2, 3, 224, 224)
    hi = IAQPipeline(enc, m_max=8, b_target=10_000).run_batch(x, mu=0.0)["recon_mse"]
    lo = IAQPipeline(enc, m_max=8, b_target=300).run_batch(x, mu=0.0)["recon_mse"]
    assert lo > hi


def test_metadata_cliff_increases_mse(rand_pipe):
    """同一 mu 下，元信息过信道的重建 MSE 显著高于元信息无损（悬崖机制）。"""
    x = torch.randn(2, 3, 224, 224)
    safe = rand_pipe.run_batch(x, mu=0.05, metadata_through_channel=False, seed=1)
    cliff = rand_pipe.run_batch(x, mu=0.05, metadata_through_channel=True, seed=1)
    assert cliff["recon_mse"] > safe["recon_mse"] * 2


def test_awgn_high_snr_near_lossless(rand_pipe):
    """AWGN 高 Eb/N0：重建近无损、输出形状正确。"""
    x = torch.randn(2, 3, 224, 224)
    out = rand_pipe.run_batch(x, channel_type="awgn", ebn0_db=15.0)
    assert out["logits"].shape == (2, 100)
    assert out["recon_mse"] < 1e-2


def test_uniform_mode_runs(rand_pipe):
    """均匀分配模式正常出分类，且 M_i 图为常数。"""
    x = torch.randn(2, 3, 224, 224)
    out = rand_pipe.run_batch(x, channel_type="awgn", ebn0_db=20.0, alloc="uniform")
    assert out["preds"].shape == (2,)
    # 均匀模式下编码出的 M_i 应全等
    a = rand_pipe.enc.attention_scores(x).cpu().numpy()
    pkt = rand_pipe.encode(x.numpy()[0], a[0], {"strategy": "uniform"})
    assert len(np.unique(pkt.m_map)) == 1


@pytest.mark.network
def test_clean_baseline_accuracy_close():
    """真权重 + 真数据：干净信道大预算下准确率接近直接分类基线。"""
    import torchvision
    import torchvision.transforms as T

    cfg = {
        "model": {
            "arch": "deit_tiny_patch16_224",
            "num_classes": 100,
            "device_weights": "imagenet",
            "server_arch": "vit_base_patch16_224",
            "server_weights": "hf:edadaltocg/vit_base_patch16_224_in21k_ft_cifar100",
        },
        "device": "cuda",
        "quantization": {"m_max": 8, "b_target": 10_000},
    }
    pipe = IAQPipeline.from_config(cfg)
    ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                       transform=pipe.enc.transform)
    xs, ys = zip(*[ds[i] for i in range(16)])
    x = torch.stack(xs)
    y = torch.tensor(ys)

    base_preds = pipe.enc.classify(x)[1].cpu()
    base_acc = float((base_preds == y).float().mean())
    out = pipe.run_batch(x, labels=y, mu=0.0)
    assert out["accuracy"] >= base_acc - 0.1  # 量化近无损，掉点很小
