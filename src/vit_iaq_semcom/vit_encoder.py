"""Sec. II — ViT encoder 与注意力分数提取。

实现：
- 经 timm 加载预训练 ViT，作为 device 端 / server 端 encoder（论文图 1）。
- 前向得到分类 logits 与 top-1 预测。
- 提取最后一层 class-token 对各 patch 的多头注意力分数，多头求平均得到每个
  patch 的重要性 a_i（论文式 3-4）。
- 将 a_i 重排为网格并叠加到原图，复现 Fig.3 注意力热图。

权重来源（weights 参数）：
- ``"random"``    随机初始化（测试用，不下载）。
- ``"imagenet"``  timm 默认 ImageNet 预训练。
- ``"hf:<repo>"`` 从 HF repo 手动加载 ``pytorch_model.bin``（CIFAR-100 fine-tuned）。
"""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn.functional as F


class ViTEncoder:
    """预训练 ViT 封装：分类 + class-token→patch 注意力分数提取。"""

    def __init__(
        self,
        arch: str = "vit_base_patch16_224",
        weights: str = "imagenet",
        num_classes: int = 100,
        device: str = "cuda",
    ) -> None:
        self.arch = arch
        self.device = self._resolve_device(device)
        self.model = self._build(arch, weights, num_classes).to(self.device).eval()

        self.num_prefix_tokens: int = self.model.num_prefix_tokens
        self.grid_size = tuple(self.model.patch_embed.grid_size)  # (gh, gw)
        self.num_patches: int = self.model.patch_embed.num_patches
        self.num_heads: int = self.model.blocks[-1].attn.num_heads

        # 预处理 transform（resize→224 + 模型归一化常数）
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)

    # ------------------------------------------------------------------ build
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _build(arch: str, weights: str, num_classes: int) -> torch.nn.Module:
        if weights == "random":
            return timm.create_model(arch, pretrained=False, num_classes=num_classes)
        if weights == "imagenet":
            return timm.create_model(arch, pretrained=True, num_classes=num_classes)
        if isinstance(weights, str) and weights.startswith("hf:"):
            # 手动加载：build 基座 + load_state_dict（绕开 timm 对该 repo 架构名的识别失败）。
            from huggingface_hub import hf_hub_download

            repo = weights[len("hf:") :]
            model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
            ckpt = hf_hub_download(repo, filename="pytorch_model.bin")
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state, strict=True)
            return model
        raise ValueError(
            f"未知 weights={weights!r}；应为 'random' | 'imagenet' | 'hf:<repo>'"
        )

    @classmethod
    def from_config(cls, cfg: dict, role: str = "device") -> "ViTEncoder":
        """从 config 的 ``model`` 段构造（role: 'device' | 'server'）。"""
        m = cfg["model"]
        return cls(
            arch=m["arch"],
            weights=m[f"{role}_weights"],
            num_classes=m["num_classes"],
            device=cfg.get("device", "cuda"),
        )

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def classify(self, images: torch.Tensor):
        """前向分类。返回 (logits[B, num_classes], preds[B])。"""
        images = images.to(self.device)
        logits = self.model(images)
        preds = logits.argmax(dim=1)
        return logits, preds

    @torch.no_grad()
    def attention_scores(
        self,
        images: torch.Tensor,
        normalize: bool = True,
        reduce_heads: bool = True,
    ) -> torch.Tensor:
        """提取最后一层 class-token→patch 注意力分数 a_i（式 3-4）。

        返回：
        - reduce_heads=True : (B, num_patches) 多头平均后的 a_i。
        - reduce_heads=False: (B, num_heads, num_patches) 各头分数。
        normalize=True 时在 patch 维度上重归一化，使 a_i 之和为 1（合法重要性分布）。
        """
        images = images.to(self.device)
        attn_module = self.model.blocks[-1].attn

        # 关闭融合注意力，强制走显式 softmax 路径，使 attn_drop 收到注意力矩阵。
        prev_fused = attn_module.fused_attn
        attn_module.fused_attn = False
        store: dict[str, torch.Tensor] = {}

        def _hook(_module, inputs, _output):
            store["attn"] = inputs[0].detach()  # (B, H, N, N) softmax 后权重

        handle = attn_module.attn_drop.register_forward_hook(_hook)
        try:
            self.model(images)
        finally:
            handle.remove()
            attn_module.fused_attn = prev_fused

        attn = store["attn"]  # (B, H, N, N)
        # class-token (query, 行 0) 对各 patch (key, 去掉前缀 token 列)
        cls_to_patch = attn[:, :, 0, self.num_prefix_tokens :]  # (B, H, num_patches)

        if not reduce_heads:
            if normalize:
                cls_to_patch = cls_to_patch / cls_to_patch.sum(dim=-1, keepdim=True)
            return cls_to_patch

        a = cls_to_patch.mean(dim=1)  # (B, num_patches) 多头算术平均
        if normalize:
            a = a / a.sum(dim=-1, keepdim=True)
        return a


def attention_heatmap(
    image: np.ndarray,
    a_i: np.ndarray,
    grid_size: tuple[int, int],
    alpha: float = 0.5,
    cmap: str = "jet",
) -> np.ndarray:
    """将 patch 重要性 a_i 重排为网格、上采样到原图并叠加为热图（复现 Fig.3）。

    参数：
    - image: (H, W, 3) 原图，取值 [0, 1] 或 [0, 255]。
    - a_i: 长度 num_patches 的重要性向量。
    - grid_size: (gh, gw) patch 网格尺寸。
    返回：(H, W, 3) float [0, 1] 的叠加图。
    """
    import matplotlib

    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    gh, gw = grid_size
    heat = np.asarray(a_i, dtype=np.float32).reshape(gh, gw)
    # 上采样到原图尺寸（bilinear）
    heat_t = torch.from_numpy(heat)[None, None]
    heat_up = F.interpolate(
        heat_t, size=(img.shape[0], img.shape[1]), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    # 归一化到 [0, 1] 便于着色
    rng = heat_up.max() - heat_up.min()
    heat_norm = (heat_up - heat_up.min()) / rng if rng > 0 else np.zeros_like(heat_up)

    colored = matplotlib.colormaps[cmap](heat_norm)[..., :3]  # (H, W, 3)
    overlay = (1 - alpha) * img + alpha * colored
    return np.clip(overlay, 0.0, 1.0)
