"""端到端 IAQ 流程编排 (Sec. II 图 1)。

device: 图像张量 -> ViT 注意力 a_i -> 增量比特分配 M_i -> patch 均匀量化 -> 载荷比特 + 元信息
channel: (BSC) 载荷恒过；元信息按开关
server : 按收到的 M_i 切分载荷 -> 反量化重建张量 -> ViT 分类 -> Top-1

约定（design D6）：量化作用在 ViT 输入张量上（已归一化、224×224）。一个 patch =
16×16×3 = 768 个标量，整 patch 用同一比特数 M_i。umin/umax 取每图全局 min/max。
本流程纯推理，不训练任何网络。
"""

from __future__ import annotations

import numpy as np
import torch

from .bit_allocation import allocate
from .channel import Packet, apply_channel
from .quantization import (
    bitstream_to_indices,
    quantize_to_bitstream,
    uniform_dequantize,
)
from .vit_encoder import ViTEncoder


def _to_patches(img: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """(C,H,W) -> (N, C*ph*pw)，行优先 patch 顺序（与 timm patch_embed 一致）。"""
    c, h, w = img.shape
    ph, pw = h // gh, w // gw
    p = img.reshape(c, gh, ph, gw, pw)          # (C, gh, ph, gw, pw)
    p = p.transpose(1, 3, 0, 2, 4)              # (gh, gw, C, ph, pw)
    return p.reshape(gh * gw, c * ph * pw)      # (N, C*ph*pw)


def _from_patches(patches: np.ndarray, c: int, gh: int, gw: int, ph: int, pw: int) -> np.ndarray:
    """逆操作：(N, C*ph*pw) -> (C, H, W)。"""
    p = patches.reshape(gh, gw, c, ph, pw)      # (gh, gw, C, ph, pw)
    p = p.transpose(2, 0, 3, 1, 4)              # (C, gh, ph, gw, pw)
    return p.reshape(c, gh * ph, gw * pw)


class IAQPipeline:
    """重要性感知量化语义通信端到端流程。"""

    def __init__(
        self,
        encoder: ViTEncoder,
        m_max: int = 8,
        b_target: int = 980,
        weight: str = "attention",
        strategy: str = "incremental",
    ) -> None:
        self.enc = encoder
        self.m_max = m_max
        self.b_target = b_target
        self.alloc_cfg = {"weight": weight, "strategy": strategy}
        self.gh, self.gw = encoder.grid_size

    @classmethod
    def from_config(cls, cfg: dict) -> "IAQPipeline":
        enc = ViTEncoder.from_config(cfg, role="server")
        q = cfg.get("quantization", {})
        return cls(
            encoder=enc,
            m_max=q.get("m_max", 8),
            b_target=q.get("b_target", 980),
            weight=q.get("weight", "attention"),
            strategy=q.get("strategy", "incremental"),
        )

    # ---------------------------------------------------------------- device
    def encode(self, img: np.ndarray, a_i: np.ndarray) -> Packet:
        """一张图（C,H,W 归一化张量）+ 其 a_i -> Packet。"""
        c, h, w = img.shape
        ph, pw = h // self.gh, w // self.gw
        patches = _to_patches(img, self.gh, self.gw)      # (N, P)
        n, p = patches.shape

        umin, umax = float(img.min()), float(img.max())
        m_map = allocate(a_i, self.b_target, self.m_max, self.alloc_cfg)

        streams = [
            quantize_to_bitstream(patches[i], int(m_map[i]), umin, umax)
            for i in range(n)
        ]
        payload = (
            np.concatenate(streams).astype(np.uint8)
            if streams else np.zeros(0, dtype=np.uint8)
        )
        return Packet(payload, m_map, umin, umax, values_per_patch=p, m_max=self.m_max)

    # ---------------------------------------------------------------- server
    def decode(self, pkt: Packet) -> np.ndarray:
        """Packet -> 重建 patch 张量 (N, P)。按收到的 M_i 顺序切分（失步即崩）。"""
        n = pkt.m_map.shape[0]
        p = pkt.values_per_patch
        out = np.empty((n, p), dtype=np.float64)
        cursor = 0
        for i in range(n):
            m = int(pkt.m_map[i])
            nbits = m * p
            seg = pkt.payload_bits[cursor:cursor + nbits]
            idx = bitstream_to_indices(seg, m, count=p)
            out[i] = uniform_dequantize(idx, m, pkt.umin, pkt.umax)
            cursor += nbits
        return out

    # ------------------------------------------------------------- end-to-end
    @torch.no_grad()
    def run_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        mu: float = 0.0,
        metadata_through_channel: bool = False,
        seed: int = 0,
    ) -> dict:
        """images: (B,3,224,224) 预处理后张量。返回 logits/preds/(accuracy)。"""
        rng = np.random.default_rng(seed)
        a = self.enc.attention_scores(images).cpu().numpy()   # (B, N)
        x = images.cpu().numpy()
        b, c, h, w = x.shape
        ph, pw = h // self.gh, w // self.gw

        recon = np.empty_like(x)
        for k in range(b):
            pkt = self.encode(x[k], a[k])
            pkt = apply_channel(pkt, mu, metadata_through_channel, rng)
            patches = self.decode(pkt)
            recon[k] = _from_patches(patches, c, self.gh, self.gw, ph, pw)

        recon = np.nan_to_num(recon, nan=0.0, posinf=10.0, neginf=-10.0)
        x_hat = torch.from_numpy(recon).to(images.dtype)
        logits, preds = self.enc.classify(x_hat)

        out = {"logits": logits, "preds": preds, "recon_mse": float(np.mean((recon - x) ** 2))}
        if labels is not None:
            labels = labels.to(preds.device)
            out["accuracy"] = float((preds == labels).float().mean())
        return out
