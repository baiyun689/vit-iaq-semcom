"""Frozen classifiers with one differentiable native-RGB preprocessing path."""

from __future__ import annotations

import hashlib
import json
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _overlap_matrix(source: int, target: int, *, device, dtype):
    """Fraction of each source cell's area falling in each target cell."""
    left = torch.arange(source, device=device, dtype=dtype) / source
    right = (torch.arange(source, device=device, dtype=dtype) + 1) / source
    tleft = torch.arange(target, device=device, dtype=dtype) / target
    tright = (torch.arange(target, device=device, dtype=dtype) + 1) / target
    return (torch.minimum(tright[:, None], right[None, :]) -
            torch.maximum(tleft[:, None], left[None, :])).clamp_min(0) * source


def map_attention_to_grid(mass: torch.Tensor, target_grid: tuple[int, int]) -> torch.Tensor:
    """Conserve attention mass across arbitrary rectangular grid resolutions.

    ``mass`` is B x source_rows x source_columns; output is B x target_patches.
    Resizing heatmap intensities is not equivalent when patch areas differ.
    """
    if mass.ndim != 3 or min(target_grid) < 1 or not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Expected nonnegative finite B x rows x columns attention")
    rows = _overlap_matrix(mass.shape[1], target_grid[0], device=mass.device, dtype=mass.dtype)
    cols = _overlap_matrix(mass.shape[2], target_grid[1], device=mass.device, dtype=mass.dtype)
    mapped = torch.einsum("is,bst,jt->bij", rows, mass, cols)
    flattened = mapped.flatten(1)
    total = flattened.sum(1, keepdim=True)
    return torch.where(total > 0, flattened / total.clamp_min(1e-12),
                       torch.full_like(flattened, 1 / flattened.shape[1]))


class _TinyClassifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(12, 16, 3, padding=1), nn.GELU())
        self.head = nn.Linear(16, classes)

    def forward(self, image):
        return self.head(self.features(image).mean((-2, -1)))


def _load_state(path, expected):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    metadata = {"metadata_status": "external_unknown", "source": str(path),
                "training_exposure_status": "external_unknown", "training_exposure": None}
    own_checkpoint = isinstance(artifact, dict) and (
        artifact.get("format") == "vit-iaq-native-classifier-v1" or "training_fingerprint" in artifact)
    if own_checkpoint:
        for key in ("kind", "arch", "num_classes", "preprocessing"):
            if artifact.get(key) != expected[key]:
                raise ValueError(f"Classifier checkpoint {key} differs from current configuration")
        exposure = artifact.get("training_exposure")
        def valid_ids(record):
            return isinstance(record, dict) and all(
                isinstance(record.get(key), list) and
                all(isinstance(sample_id, str) and sample_id for sample_id in record[key]) and
                len(set(record[key])) == len(record[key])
                for key in ("train_ids", "selection_ids"))
        if not valid_ids(exposure):
            raise ValueError("Framework classifier checkpoint must retain valid training_exposure train_ids and selection_ids")
        ancestors = exposure.get("ancestor_exposure", [])
        if not isinstance(ancestors, list) or any(not valid_ids(record) for record in ancestors):
            raise ValueError("Framework classifier ancestor_exposure must retain valid historical training and selection IDs")
        if any(set(record["train_ids"]) & set(record["selection_ids"]) for record in [exposure, *ancestors]):
            raise ValueError("Framework classifier training_exposure contains overlapping train and selection IDs")
        metadata = {"metadata_status": "verified", "source": str(path),
                    **{key: artifact[key] for key in expected},
                    "training_fingerprint": artifact.get("training_fingerprint"),
                    "split_fingerprint": artifact.get("split_fingerprint"),
                    "training_exposure_status": "recorded_local_fit_prior_pretraining_may_be_unknown",
                    "training_exposure": copy.deepcopy(exposure)}
    state = artifact.get("state_dict", artifact.get("model", artifact)) if isinstance(artifact, dict) else artifact
    if not isinstance(state, dict):
        raise ValueError("Classifier checkpoint must contain a tensor state_dict")
    return {key.removeprefix("module."): value for key, value in state.items()}, metadata


class ClassifierAdapter(nn.Module):
    """Input gradients remain enabled, while classifier parameters stay frozen.

    Tiny/random models are workflow checks. ImageNet with a replacement random
    class head is also a random target classifier and requires explicit consent.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        spec = cfg["classifier"]
        self.kind = spec.get("kind", "tiny")
        self.input_size = int(spec.get("input_size", cfg["data"]["native_size"]))
        classes = int(cfg["data"]["num_classes"])
        weights, checkpoint = spec.get("weights", "random"), spec.get("checkpoint")
        self.checkpoint_metadata = {"metadata_status": "initial_weights", "source": str(weights),
                                    "training_exposure_status": "untrained" if weights == "random" or self.kind == "tiny" else "external_unknown",
                                    "training_exposure": None}
        self.is_random = not checkpoint and (self.kind == "tiny" or weights == "random" or
                                             (weights == "imagenet" and classes != 1000))
        if self.is_random and cfg["data"]["name"] != "synthetic" and not spec.get("allow_random", False):
            raise ValueError("Random target classifiers are not valid real-data experiments; set allow_random only for smoke checks")
        requested_device = cfg.get("device", "cpu")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; explicitly select device=cpu")
        # Weight initialization is independent of repair-training seeds.
        with torch.random.fork_rng(devices=[]):
            # Initialize CPU weights without resetting a caller's CUDA RNG.
            torch.random.default_generator.manual_seed(314159)
            if self.kind == "tiny":
                self.model = _TinyClassifier(classes)
            elif self.kind == "vit":
                import timm
                self.model = timm.create_model(spec["arch"], pretrained=(weights == "imagenet" and not checkpoint),
                                               num_classes=classes, img_size=self.input_size)
                if not checkpoint and isinstance(weights, str) and weights.startswith("hf:"):
                    from huggingface_hub import hf_hub_download
                    checkpoint = hf_hub_download(weights[3:], filename="pytorch_model.bin")
                elif not checkpoint and weights not in ("random", "imagenet"):
                    raise ValueError("classifier.weights must be random, imagenet or hf:<repository>")
            else:
                raise ValueError(f"Unsupported classifier kind: {self.kind}")
            if checkpoint:
                expected = {"kind": self.kind, "arch": spec["arch"] if self.kind == "vit" else "tiny",
                            "num_classes": classes, "preprocessing": {"input_size": self.input_size,
                            "mean": spec.get("mean", [0.5] * 3), "std": spec.get("std", [0.5] * 3)}}
                state, self.checkpoint_metadata = _load_state(checkpoint, expected)
                self.model.load_state_dict(state, strict=True)
                self.is_random = False
        self.register_buffer("mean", torch.tensor(spec.get("mean", [0.5] * 3)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec.get("std", [0.5] * 3)).view(1, 3, 1, 1))
        if self.mean.numel() != 3 or self.std.numel() != 3 or (self.std <= 0).any():
            raise ValueError("Classifier mean/std must have three values with positive std")
        self.eval().requires_grad_(False)
        self.to(torch.device(requested_device))
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
        # Exposure declarations affect what may be treated as held-out data,
        # even when the tensor weights themselves are unchanged.
        digest.update(json.dumps(self.checkpoint_metadata, sort_keys=True).encode())
        for key, value in self.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.fingerprint = digest.hexdigest()

    @property
    def device(self):
        return self.mean.device

    def train(self, mode=True):
        # Repair fitting must never update classifier dropout/batchnorm state.
        return super().train(False)

    def preprocess(self, native_tensor):
        if native_tensor.ndim != 4 or native_tensor.shape[1] != 3:
            raise ValueError("Classifier expects B x 3 x H x W native RGB")
        image = native_tensor.to(device=self.device, dtype=self.mean.dtype)
        image = F.interpolate(image, size=(self.input_size, self.input_size), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        return (image - self.mean) / self.std

    def forward(self, native_tensor):
        return self.model(self.preprocess(native_tensor))

    @torch.no_grad()
    def importance(self, native_tensor, patch_size: int) -> np.ndarray:
        height, width = native_tensor.shape[-2:]
        if height % patch_size or width % patch_size:
            raise ValueError("Communication patch size must divide native image dimensions")
        image = self.preprocess(native_tensor)
        if self.kind == "tiny":
            # Smoke-only proxy; not a ViT-attention baseline.
            mass = self.model.features(image).square().mean(1)
        else:
            attention = self.model.blocks[-1].attn
            previous_fused = getattr(attention, "fused_attn", False)
            attention.fused_attn = False
            captured = {}
            def capture(_module, inputs, _output):
                captured["weights"] = inputs[0].detach()
            handle = attention.attn_drop.register_forward_hook(capture)
            try:
                self.model(image)
            finally:
                handle.remove()
                attention.fused_attn = previous_fused
            prefix_tokens = self.model.num_prefix_tokens
            if prefix_tokens < 1:
                raise ValueError("ViT attention importance requires a class prefix token")
            scores = captured["weights"][:, :, 0, prefix_tokens:].mean(1)
            grid = self.model.patch_embed.grid_size
            mass = scores.reshape(-1, *grid)
        return map_attention_to_grid(mass, (height // patch_size, width // patch_size)).cpu().numpy()


def build_classifier(cfg: dict) -> ClassifierAdapter:
    return ClassifierAdapter(cfg)
