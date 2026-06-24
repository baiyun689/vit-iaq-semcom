"""收端学习式语义修补网络（spec: iaq-semantic-repair）—— 本 change 的创新内核。

粗桶非级联编码（bucket.py）保证信道误码只造成**局部值错配**；本网络在收端把这些
局部损伤补回，优雅退化、无悬崖。本质是带可靠性先验的 MAE / inpainting。

输入：
  - 受损重建 patch 网格（粗桶解码图，已含基础层 + 错配的增强细节）
  - **比特深度可靠性图**（bucket.bit_depth_map 的免费副产品）：每 patch 实得总比特，
    低比特/边界 patch 更不可信、更该靠上下文修复。重要性在此**二次进场指导修复**。

输出：对重建图的**残差修正**（head 零初始化 → 初始即恒等 → 残差兜底：最坏≈不修≈粗图
地板，结构上保证 AI 不帮倒忙）。

训练（见 scripts/run_repair_train.py）：冻结分类裁判 + patch_embed，只训本网络；损失=
分类交叉熵(主) + 重建 MSE(辅)，跨 SNR。**用任务目标训练**，不重蹈"逼近干净图注意力"
代理目标摊平注意力的覆辙。

纯像素/patch 域，冻结分类器原样复用、可视化友好、好训。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def patchify(img: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """(B,C,H,W) -> (B,N,C*ph*pw)，行优先 patch 顺序（对齐 pipeline._to_patches）。"""
    b, c, h, w = img.shape
    ph, pw = h // gh, w // gw
    p = img.reshape(b, c, gh, ph, gw, pw)        # (B,C,gh,ph,gw,pw)
    p = p.permute(0, 2, 4, 1, 3, 5)              # (B,gh,gw,C,ph,pw)
    return p.reshape(b, gh * gw, c * ph * pw)    # (B,N,C*ph*pw)


def unpatchify(patches: torch.Tensor, c: int, gh: int, gw: int, ph: int, pw: int) -> torch.Tensor:
    """逆操作：(B,N,C*ph*pw) -> (B,C,H,W)。"""
    b = patches.shape[0]
    p = patches.reshape(b, gh, gw, c, ph, pw)    # (B,gh,gw,C,ph,pw)
    p = p.permute(0, 3, 1, 4, 2, 5)              # (B,C,gh,ph,gw,pw)
    return p.reshape(b, c, gh * ph, gw * pw)


class SemanticRepairNet(nn.Module):
    """轻量 patch-Transformer：受损重建网格 + 比特深度可靠性图 → 残差修正。"""

    def __init__(
        self,
        grid: tuple[int, int] = (14, 14),
        patch_dim: tuple[int, int, int] = (3, 16, 16),  # (C, ph, pw)
        m_max: int = 8,
        dim: int = 192,
        depth: int = 4,
        heads: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.gh, self.gw = grid
        self.c, self.ph, self.pw = patch_dim
        self.m_max = m_max
        n = self.gh * self.gw
        pdim = self.c * self.ph * self.pw

        self.patch_embed = nn.Linear(pdim, dim)
        # 比特深度可靠性图：m_max+1 档(含 0)嵌入,作为可靠性通道加到 patch token
        self.depth_embed = nn.Embedding(m_max + 1, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, pdim)
        # 残差兜底：head 零初始化 → 初始残差=0 → 输出==输入(恒等,不帮倒忙)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, img: torch.Tensor, bit_depth: torch.Tensor) -> torch.Tensor:
        """img: (B,C,H,W) 受损重建；bit_depth: (B,N) int 每 patch 总比特。返回修正图 (B,C,H,W)。"""
        x = patchify(img, self.gh, self.gw)                  # (B,N,pdim)
        d = bit_depth.clamp(0, self.m_max).long()            # (B,N)
        tok = self.patch_embed(x) + self.depth_embed(d) + self.pos_embed
        tok = self.blocks(tok)
        residual = self.head(self.norm(tok))                 # (B,N,pdim)
        out = x + residual                                   # 残差兜底
        return unpatchify(out, self.c, self.gh, self.gw, self.ph, self.pw)


@torch.no_grad()
def repair_reconstruction(
    net: SemanticRepairNet,
    recon: np.ndarray,
    bit_depth: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """推理接口（task 3.3）：受损重建 (B,C,H,W) + 比特深度图 (B,N) → 修正重建 (B,C,H,W) numpy。

    纯前向、不训练；网络若未训练(head 零初始化)则输出恒等于输入(兜底)。
    """
    net.eval()
    dev = torch.device(device) if device is not None else next(net.parameters()).device
    x = torch.as_tensor(recon, dtype=dtype, device=dev)
    d = torch.as_tensor(bit_depth, device=dev)
    if x.ndim == 3:                                          # (C,H,W) → (1,C,H,W)
        x = x.unsqueeze(0)
        d = d.unsqueeze(0)
    fixed = net(x, d)
    return fixed.cpu().numpy().astype(recon.dtype)
