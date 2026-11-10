"""Capacity-controlled receiver repair and soft trusted-prefix constraints.

Only explicitly listed received evidence is consumed. Simulation diagnostics,
clean images and true error masks cannot enter the conditioning path.
"""

from __future__ import annotations

import torch
from torch import nn


VARIANTS = ("image", "depth", "llr", "bitwise", "uncertainty", "prefix")


class IdentityRepair(nn.Module):
    def forward(self, image, evidence):
        return image


class NativeRepairNet(nn.Module):
    def __init__(self, cfg: dict, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported repair variant: {variant}")
        self.variant = variant
        self.m_max = int(cfg["transmission"]["m_max"])
        self.patch_size = int(cfg["transmission"]["patch_size"])
        self.native_size = int(cfg["data"]["native_size"])
        spec = cfg["repair"]
        self.architecture = spec.get("architecture", "transformer")
        dim, depth, heads = int(spec["dim"]), int(spec["depth"]), int(spec["heads"])
        # Fixed input channels and weights across all variants: image 3, depth 1,
        # aggregate LLR 6, bit-level LLR/probability/mask 9m, uncertainty 3,
        # prefix lower/upper/confidence 9. Variants only mask feature groups.
        self.condition_channels = 19 + 9 * self.m_max
        if self.architecture == "cnn":
            self.embed = nn.Conv2d(3 + self.condition_channels, dim, 3, padding=1)
            self.blocks = nn.Sequential(*[layer for _ in range(depth)
                                         for layer in (nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())])
            self.head = nn.Conv2d(dim, 3, 3, padding=1)
        elif self.architecture == "transformer":
            if self.native_size % self.patch_size:
                raise ValueError("Repair patch size must divide native image size")
            self.embed = nn.Conv2d(3 + self.condition_channels, dim, self.patch_size, stride=self.patch_size)
            count = (self.native_size // self.patch_size) ** 2
            self.position = nn.Parameter(torch.zeros(1, count, dim))
            nn.init.trunc_normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
            self.head = nn.Linear(dim, 3 * self.patch_size ** 2)
        else:
            raise ValueError(f"Unsupported repair architecture: {self.architecture}")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def conditioning(self, image: torch.Tensor, evidence: dict) -> torch.Tensor:
        b, _, h, w = image.shape
        zeros = lambda channels: image.new_zeros((b, channels, h, w))
        def get(key, channels):
            if key not in evidence:
                raise ValueError(f"Missing received evidence: {key}")
            value = evidence[key].to(device=image.device, dtype=image.dtype)
            if value.shape != (b, channels, h, w) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid received evidence shape/value for {key}")
            return value
        depth, aggregate, bits, uncertainty, prefix = zeros(1), zeros(6), zeros(9 * self.m_max), zeros(3), zeros(9)
        if self.variant != "image":
            depth = get("depth", 1).clamp(0, self.m_max) / self.m_max
        if self.variant in ("llr", "bitwise", "uncertainty", "prefix"):
            mask = get("bit_mask", 3 * self.m_max).clamp(0, 1)
            llr = get("llr", 3 * self.m_max)
            probability = get("bit_prob", 3 * self.m_max).clamp(0, 1)
            llr_masked = torch.tanh(llr / 10.0) * mask
            probability_masked = (probability - 0.5) * mask
            if self.variant == "llr":
                color_mask = mask.reshape(b, 3, self.m_max, h, w)
                count = color_mask.sum(2).clamp_min(1)
                aggregate = torch.cat((llr_masked.reshape(b, 3, self.m_max, h, w).sum(2) / count,
                                       llr_masked.abs().reshape(b, 3, self.m_max, h, w).sum(2) / count), 1)
            else:
                bits = torch.cat((llr_masked, probability_masked, mask), 1)
        if self.variant in ("uncertainty", "prefix"):
            uncertainty = get("uncertainty", 3).clamp(0, 1)
        if self.variant == "prefix":
            confidence = get("prefix_confidence", 3).clamp(0, 1)
            prefix = torch.cat((get("prefix_lower", 3) * confidence,
                                get("prefix_upper", 3) * confidence, confidence), 1)
        return torch.cat((depth, aggregate, bits, uncertainty, prefix), 1)

    def forward(self, image: torch.Tensor, evidence: dict) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Repair expects native B x 3 x H x W RGB images")
        features = self.embed(torch.cat((image, self.conditioning(image, evidence)), 1))
        if self.architecture == "cnn":
            residual = self.head(self.blocks(features))
        else:
            b, _, gh, gw = features.shape
            tokens = features.flatten(2).transpose(1, 2)
            if tokens.shape[1] != self.position.shape[1]:
                raise ValueError("Repair image grid differs from configured native size")
            patches = self.head(self.blocks(tokens + self.position))
            p = self.patch_size
            residual = patches.reshape(b, gh, gw, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape_as(image)
        return (image + residual).clamp(0, 1)


def prefix_penalty(output: torch.Tensor, evidence: dict) -> torch.Tensor:
    """Soft interval violation weighted by received prefix confidence.

    A confidently wrong prefix can still hurt. This is a training regularizer,
    not a no-harm guarantee or an assertion that confidence is calibrated.
    """
    lower, upper, confidence = (evidence[key].to(output) for key in
                               ("prefix_lower", "prefix_upper", "prefix_confidence"))
    if any(value.shape != output.shape for value in (lower, upper, confidence)):
        raise ValueError("Prefix evidence must match output BCHW shape")
    if (not all(torch.isfinite(value).all() for value in (lower, upper, confidence)) or
            (lower > upper).any() or (confidence < 0).any() or (confidence > 1).any()):
        raise ValueError("Invalid prefix interval or confidence")
    violation = (lower - output).clamp_min(0).square() + (output - upper).clamp_min(0).square()
    return (violation * confidence).mean()


def build_repair(cfg: dict, variant: str) -> nn.Module:
    return IdentityRepair() if variant == "none" else NativeRepairNet(cfg, variant)
