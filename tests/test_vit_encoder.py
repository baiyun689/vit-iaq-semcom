"""ViTEncoder 测试。

逻辑用例用随机权重（不联网）；真实 CIFAR-100 权重加载用例打 ``network`` mark，
离线时自动跳过。
"""

import numpy as np
import pytest
import torch

from vit_iaq_semcom.vit_encoder import ViTEncoder, attention_heatmap

NUM_CLASSES = 100


@pytest.fixture(scope="module")
def enc():
    # 随机权重 + cpu：无需联网、结果确定，足以验证结构与注意力提取逻辑。
    return ViTEncoder(weights="random", num_classes=NUM_CLASSES, device="cpu")


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224)


def test_structure(enc):
    # 默认骨干 = DeiT-Tiny（论文设置）：196 patch / 14×14 网格 / 1 前缀 token / 3 头。
    assert enc.num_patches == 196
    assert enc.grid_size == (14, 14)
    assert enc.num_prefix_tokens == 1
    assert enc.num_heads == 3


def test_classify_shape(enc, images):
    logits, preds = enc.classify(images)
    assert logits.shape == (2, NUM_CLASSES)
    assert preds.shape == (2,)
    assert preds.min() >= 0 and preds.max() < NUM_CLASSES


def test_attention_scores_normalized(enc, images):
    a = enc.attention_scores(images)  # (B, num_patches)
    assert a.shape == (2, 196)
    assert (a >= 0).all()
    torch.testing.assert_close(a.sum(dim=1), torch.ones(2), atol=1e-5, rtol=0)


def test_attention_multihead_average(enc, images):
    per_head = enc.attention_scores(images, normalize=False, reduce_heads=False)
    avg = enc.attention_scores(images, normalize=False, reduce_heads=True)
    assert per_head.shape == (2, enc.num_heads, 196)
    # 输出等于各头 class-token→patch 注意力的算术平均
    torch.testing.assert_close(avg, per_head.mean(dim=1))


def test_device_fallback(monkeypatch):
    # CUDA 不可用时，请求 cuda 应回退到 cpu 而非报错。
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ViTEncoder._resolve_device("cuda") == torch.device("cpu")


def test_attention_heatmap_shape(enc, images):
    a = enc.attention_scores(images)[0].cpu().numpy()
    img = np.random.rand(224, 224, 3).astype(np.float32)
    overlay = attention_heatmap(img, a, enc.grid_size)
    assert overlay.shape == (224, 224, 3)
    assert overlay.min() >= 0.0 and overlay.max() <= 1.0


@pytest.mark.network
def test_load_cifar100_weights():
    """真实 fine-tuned 权重加载 smoke（需联网，离线跳过）。"""
    try:
        e = ViTEncoder(
            arch="vit_base_patch16_224",  # 该 hf 权重是 ViT-Base，须显式指定以匹配默认的 DeiT-Tiny 之外的架构
            weights="hf:edadaltocg/vit_base_patch16_224_in21k_ft_cifar100",
            num_classes=NUM_CLASSES,
            device="cpu",
        )
    except Exception as exc:  # noqa: BLE001 - 网络/下载失败则跳过
        pytest.skip(f"无法下载 CIFAR-100 权重（离线？）：{exc}")
    logits, preds = e.classify(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, NUM_CLASSES)
