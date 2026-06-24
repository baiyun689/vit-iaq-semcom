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

from .a1 import (
    A1Packet,
    a1_decode,
    a1_decode_base,
    a1_encode,
    a1_through_channel,
    base_reconstruct,
)
from .bit_allocation import allocate
from .channel import Packet, apply_channel, awgn_channel, awgn_channel_fec
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
        # 可选：A1 在压缩图上派生分配的专职注意力提取器(§2.5 微调后的 student)。
        # None 时回退 self.enc。分类裁判恒为冻结 self.enc,不受影响。
        self.attn_enc: ViTEncoder | None = None

    @property
    def _attn(self) -> ViTEncoder:
        """A1 派生分配用的注意力提取器:有微调 student 用它,否则回退分类网络。"""
        return self.attn_enc if self.attn_enc is not None else self.enc

    def load_attn_encoder(self, ckpt_path: str, cfg: dict, role: str = "server") -> ViTEncoder:
        """加载 §2.5 微调后的 student 注意力提取器(.pt = state_dict),供 A1 派生分配用。

        分类裁判仍是冻结 self.enc(不挪尺子);收发两端共用此 attn_enc → metadata-free 不破。
        注意:student 是针对某个 m_base 压缩分布训的,run_batch 的 m_base 须与训练时一致。
        用 weights='random' 建同架构空壳再灌 student 权重,避免重复下载原 hf 权重。
        """
        m = cfg["model"]
        enc = ViTEncoder(
            arch=m.get(f"{role}_arch", m["arch"]),
            weights="random",
            num_classes=m["num_classes"],
            device=cfg.get("device", "cuda"),
        )
        state = torch.load(ckpt_path, map_location=enc.device, weights_only=True)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        enc.model.load_state_dict(state, strict=True)
        enc.model.eval()
        self.attn_enc = enc
        return enc

    @classmethod
    def from_config(cls, cfg: dict) -> "IAQPipeline":
        enc = ViTEncoder.from_config(cfg, role="server")
        q = cfg.get("quantization", {})
        return cls(
            encoder=enc,
            m_max=q.get("m_max", 8),
            b_target=q.get("b_target", 980),
            weight=q.get("weight", "attention"),
            strategy=q.get("alloc", q.get("strategy", "incremental")),
        )

    # ---------------------------------------------------------------- device
    def encode(self, img: np.ndarray, a_i: np.ndarray, alloc_cfg: dict | None = None) -> Packet:
        """一张图（C,H,W 归一化张量）+ 其 a_i -> Packet。alloc_cfg 可覆盖分配策略。"""
        c, h, w = img.shape
        ph, pw = h // self.gh, w // self.gw
        patches = _to_patches(img, self.gh, self.gw)      # (N, P)
        n, p = patches.shape

        umin, umax = float(img.min()), float(img.max())
        m_map = allocate(a_i, self.b_target, self.m_max, alloc_cfg or self.alloc_cfg)

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

    # -------------------------------------------------------------------- A1
    def _run_a1(
        self, x: np.ndarray, ebn0_db: float, m_base: int, base_fec_r: int,
        b_enh: int | None, rng: np.random.Generator, dtype: torch.dtype,
    ) -> np.ndarray:
        """A1 元信息-free 批处理：两端在基础层重建图上各自算注意力派生分配。

        发端在自己的基础层重建上算 a_i 编码；收端解出基础层(经 FEC)后**重新**算 a_i
        派生 ΔM_i，故基础层受损时收发分配会随之一致地变化(诚实建模)。
        """
        b, c, h, w = x.shape
        ph, pw = h // self.gh, w // self.gw
        n = self.gh * self.gw
        if b_enh is None:
            b_enh = self.b_target - n * m_base

        umins = np.array([float(x[k].min()) for k in range(b)])
        umaxs = np.array([float(x[k].max()) for k in range(b)])

        # 发端：基础层重建 → 注意力 → 编码 → 过信道
        x_base_tx = np.stack(
            [base_reconstruct(x[k], m_base, umins[k], umaxs[k]) for k in range(b)])
        a_tx = self._attn.attention_scores(
            torch.from_numpy(x_base_tx).to(dtype)).cpu().numpy()
        pkts: list[A1Packet] = []
        for k in range(b):
            pkt = a1_encode(_to_patches(x[k], self.gh, self.gw), a_tx[k],
                            umins[k], umaxs[k], m_base, b_enh, self.m_max, base_fec_r)
            pkts.append(a1_through_channel(pkt, ebn0_db, rng))

        # 收端：解基础层 → 重新算注意力 → 派生 ΔM_i → 解增强层
        x_base_rx = np.empty_like(x)
        for k in range(b):
            _, base_patches = a1_decode_base(pkts[k])
            x_base_rx[k] = _from_patches(base_patches, c, self.gh, self.gw, ph, pw)
        a_rx = self._attn.attention_scores(
            torch.from_numpy(x_base_rx).to(dtype)).cpu().numpy()
        recon = np.empty_like(x)
        for k in range(b):
            recon[k] = _from_patches(
                a1_decode(pkts[k], a_rx[k]), c, self.gh, self.gw, ph, pw)
        return recon

    # ------------------------------------------------------------- end-to-end
    @torch.no_grad()
    def run_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        channel_type: str = "bsc",
        mu: float = 0.0,
        ebn0_db: float = float("inf"),
        metadata_through_channel: bool = False,
        alloc: str | None = None,
        seed: int = 0,
        mode: str = "iaq",
        meta_fec_r: int = 1,
        payload_fec_r: int = 1,
        base_fec_r: int = 3,
        m_base: int = 1,
        b_enh: int | None = None,
    ) -> dict:
        """images: (B,3,224,224) 预处理后张量。返回 logits/preds/(accuracy)。

        channel_type: 'bsc'(用 mu) | 'awgn'(用 ebn0_db)。
        alloc: 覆盖比特分配策略('incremental'|'uniform'|...)，None 用 pipeline 默认。
        mode: 'iaq'(默认,既有路径) | 'eep' | 'uep' | 'a1'(元信息-free)。
        meta_fec_r/payload_fec_r: EEP/UEP 的元信息/载荷重复码速率分母。
        base_fec_r/m_base/b_enh: A1 的基础层 FEC、基础层比特、增强层预算(缺省 b_target−N·m_base)。
        """
        rng = np.random.default_rng(seed)
        x = images.cpu().numpy()
        b, c, h, w = x.shape
        ph, pw = h // self.gh, w // self.gw

        if mode == "a1":
            recon = self._run_a1(x, ebn0_db, m_base, base_fec_r, b_enh, rng, images.dtype)
        else:
            alloc_cfg = {**self.alloc_cfg, "strategy": alloc} if alloc else None
            a = self.enc.attention_scores(images).cpu().numpy()   # (B, N)
            recon = np.empty_like(x)
            for k in range(b):
                pkt = self.encode(x[k], a[k], alloc_cfg)
                if mode in ("eep", "uep"):
                    pkt = awgn_channel_fec(pkt, ebn0_db, rng, meta_fec_r, payload_fec_r)
                elif channel_type == "awgn":
                    pkt = awgn_channel(pkt, ebn0_db, metadata_through_channel, rng)
                else:
                    pkt = apply_channel(pkt, mu, metadata_through_channel, rng)
                recon[k] = _from_patches(self.decode(pkt), c, self.gh, self.gw, ph, pw)

        recon = np.nan_to_num(recon, nan=0.0, posinf=10.0, neginf=-10.0)
        x_hat = torch.from_numpy(recon).to(images.dtype)
        logits, preds = self.enc.classify(x_hat)

        out = {"logits": logits, "preds": preds, "recon_mse": float(np.mean((recon - x) ** 2))}
        if labels is not None:
            labels = labels.to(preds.device)
            out["accuracy"] = float((preds == labels).float().mean())
        return out
