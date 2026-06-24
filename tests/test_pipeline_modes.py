"""pipeline 多模式管路单测（spec: iaq-pipeline）。

用 mock encoder 验证 A1 / EEP / UEP 端到端管路,不加载真实 ViT(纯 CPU、不下载)。
"""

import numpy as np
import torch
import torch.nn.functional as F

from vit_iaq_semcom.pipeline import IAQPipeline
from vit_iaq_semcom.repair import SemanticRepairNet

# N=16 桶配置(grid 4×4):m_base=1,top-4 到 3bit、次-4 到 2bit、其余到 1bit
BUCKETS = [(4, 3), (4, 2), (8, 1)]


class FakeEncoder:
    """最小 ViTEncoder 替身:确定性注意力 + 平凡分类。grid 4×4,patch 2×2。"""

    def __init__(self, gh=4, gw=4):
        self.grid_size = (gh, gw)
        self.arch = "fake"
        self.device = torch.device("cpu")

    def attention_scores(self, images: torch.Tensor) -> torch.Tensor:
        # 每 2×2 patch 的通道均值幅度 → softmax,行优先(对齐 _to_patches)
        pooled = F.avg_pool2d(images.abs(), kernel_size=2).mean(dim=1)  # (B,gh,gw)
        flat = pooled.reshape(images.shape[0], -1)                       # (B,N)
        return torch.softmax(flat, dim=1)

    def classify(self, images: torch.Tensor):
        # 平凡“分类”:按整图均值符号分两类(确定性,供管路验证)
        preds = (images.reshape(images.shape[0], -1).mean(dim=1) > 0).long()
        logits = torch.zeros(images.shape[0], 2)
        return logits, preds


def _imgs(b=4, seed=0):
    return torch.from_numpy(
        np.random.default_rng(seed).standard_normal((b, 3, 8, 8))).float()


def _pipe(b_target=16):
    # N=16 patch,b_target=16 → A1 基础层 16 + 增强层 0..;m_max 小
    return IAQPipeline(FakeEncoder(), m_max=4, b_target=b_target)


def test_a1_mode_runs_and_reconstructs_at_high_snr():
    pipe = _pipe(b_target=32)
    x = _imgs()
    out = pipe.run_batch(x, channel_type="awgn", ebn0_db=float("inf"),
                         mode="a1", base_fec_r=1, m_base=1)
    assert "preds" in out and out["preds"].shape[0] == 4
    # 无噪声:A1 重建应接近原图(受量化限制),远小于信号尺度
    assert out["recon_mse"] < 0.5


def test_a1_floor_no_crater_at_low_snr():
    """极低 SNR:A1 仍跑通、重建有限(无 NaN/爆炸),体现有底降级。"""
    pipe = _pipe(b_target=32)
    x = _imgs(seed=1)
    out = pipe.run_batch(x, channel_type="awgn", ebn0_db=-10.0,
                         mode="a1", base_fec_r=5, seed=3)
    assert np.isfinite(out["recon_mse"])
    assert out["preds"].shape[0] == 4


def test_eep_uep_lossless_at_inf_snr():
    """无噪声下 EEP/UEP 元信息无损 → 与 IAQ 理想一致。"""
    pipe = _pipe(b_target=48)
    x = _imgs(seed=2)
    ideal = pipe.run_batch(x, channel_type="awgn", ebn0_db=float("inf"),
                           mode="iaq", alloc="incremental")
    eep = pipe.run_batch(x, channel_type="awgn", ebn0_db=float("inf"),
                         mode="eep", meta_fec_r=3, payload_fec_r=3)
    uep = pipe.run_batch(x, channel_type="awgn", ebn0_db=float("inf"),
                         mode="uep", meta_fec_r=5, payload_fec_r=1)
    assert eep["recon_mse"] == ideal["recon_mse"]
    assert uep["recon_mse"] == ideal["recon_mse"]


def test_a1_uses_attn_encoder_when_set():
    """A1 派生分配走 self._attn:设了微调 student 就用它(不同注意力→不同分配→不同重建);
    清空后回退冻结 self.enc。分类裁判始终是 self.enc。"""
    pipe = _pipe(b_target=32)
    x = _imgs(seed=5)
    assert pipe._attn is pipe.enc  # 缺省回退分类网络
    base = pipe.run_batch(x, channel_type="awgn", ebn0_db=float("inf"),
                          mode="a1", m_base=1)

    class FlipEncoder(FakeEncoder):
        def attention_scores(self, images):  # 反向重要性 → 不同派生
            pooled = F.avg_pool2d(images.abs(), kernel_size=2).mean(dim=1)
            return torch.softmax(-pooled.reshape(images.shape[0], -1), dim=1)

    pipe.attn_enc = FlipEncoder()
    assert pipe._attn is pipe.attn_enc
    flipped = pipe.run_batch(x, channel_type="awgn", ebn0_db=float("inf"),
                             mode="a1", m_base=1)
    assert abs(flipped["recon_mse"] - base["recon_mse"]) > 1e-9

    pipe.attn_enc = None
    assert pipe._attn is pipe.enc


def test_iaq_mode_backward_compatible():
    """默认 mode='iaq' 行为不变(回归保护)。"""
    pipe = _pipe(b_target=48)
    x = _imgs(seed=4)
    out = pipe.run_batch(x, channel_type="awgn", ebn0_db=8.0, alloc="incremental")
    assert "preds" in out and np.isfinite(out["recon_mse"])


def _repair_net():
    return SemanticRepairNet(grid=(4, 4), patch_dim=(3, 2, 2), m_max=4,
                             dim=24, depth=2, heads=2)


def test_bucket_mode_runs_and_reconstructs_at_high_snr():
    pipe = _pipe(b_target=28)
    x = _imgs(seed=6)
    out = pipe.run_batch(x, channel_type="awgn", ebn0_db=float("inf"),
                         mode="bucket", m_base=1, buckets=BUCKETS)
    assert out["preds"].shape[0] == 4
    assert out["recon_mse"] < 0.5                    # 无噪:粗桶重建接近原图


def test_bucket_floor_no_crater_at_low_snr():
    """极低 SNR:粗桶仍跑通、重建有限(无 NaN/爆炸),局部损伤不级联崩。"""
    pipe = _pipe(b_target=28)
    x = _imgs(seed=7)
    out = pipe.run_batch(x, channel_type="awgn", ebn0_db=-10.0,
                         mode="bucket", m_base=1, base_fec_r=5, buckets=BUCKETS, seed=3)
    assert np.isfinite(out["recon_mse"]) and out["preds"].shape[0] == 4


def test_bucket_repair_zero_extra_bandwidth():
    """bucket+repair 与 bucket 过信道比特完全相同:差异仅来自收端修补(纯算力)。

    修补网络零初始化(恒等)时两者重建逐元素相同 → 证明信道路径不变、修补是收端加法。
    """
    pipe = _pipe(b_target=28)
    x = _imgs(seed=8)
    base = pipe.run_batch(x, channel_type="awgn", ebn0_db=5.0,
                          mode="bucket", m_base=1, buckets=BUCKETS, seed=2)
    pipe.repair_net = _repair_net()                  # head 零初始化 → 恒等兜底
    rep0 = pipe.run_batch(x, channel_type="awgn", ebn0_db=5.0,
                          mode="bucket+repair", m_base=1, buckets=BUCKETS, seed=2)
    assert rep0["recon_mse"] == base["recon_mse"]    # 同信道比特,修补恒等→重建一致

    # 解除零初始化 → 修补真的改动重建(仍零额外带宽)
    torch.nn.init.normal_(pipe.repair_net.head.weight, std=0.05)
    rep1 = pipe.run_batch(x, channel_type="awgn", ebn0_db=5.0,
                          mode="bucket+repair", m_base=1, buckets=BUCKETS, seed=2)
    assert rep1["recon_mse"] != base["recon_mse"]


def test_bucket_repair_without_net_falls_back_to_bucket():
    """未加载修补网络时 bucket+repair 优雅退化为纯 bucket(不报错)。"""
    pipe = _pipe(b_target=28)
    x = _imgs(seed=9)
    base = pipe.run_batch(x, channel_type="awgn", ebn0_db=5.0,
                          mode="bucket", m_base=1, buckets=BUCKETS, seed=1)
    rep = pipe.run_batch(x, channel_type="awgn", ebn0_db=5.0,
                         mode="bucket+repair", m_base=1, buckets=BUCKETS, seed=1)
    assert rep["recon_mse"] == base["recon_mse"]
