"""Native RGB datasets with deterministic, persisted, disjoint sample splits.

Official test images never participate in repair fitting or checkpoint selection.
This does not assert that an externally supplied classifier has never seen the
training pool: that exposure must be described in the experiment report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SPLITS = ("train", "selection", "calibration", "audit", "test")


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


class _SyntheticPool(Dataset):
    """Small deterministic pattern images, strictly for offline workflow checks."""

    def __init__(self, count: int, size: int, classes: int, seed: int):
        self.count, self.size, self.classes, self.seed = count, size, classes, seed

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        label = int(index % self.classes)
        image = rng.uniform(0.05, 0.35, (3, self.size, self.size)).astype(np.float32)
        image[label % 3, :, label % self.size::max(2, self.classes)] += 0.5
        return torch.from_numpy(image.clip(0, 1)), label


class NativeSubset(Dataset):
    """Subset returning (RGB CHW float32, integer label, stable official ID)."""

    def __init__(self, pool, indices, name, official_split, native_size, fingerprint):
        self.pool = pool
        self.indices = [int(x) for x in indices]
        self.sample_ids = [f"{name}:{official_split}:{i:06d}" for i in self.indices]
        self.native_size = int(native_size)
        self.manifest_fingerprint = fingerprint

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.pool[self.indices[index]]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        image = image.to(dtype=torch.float32)
        if image.shape != (3, self.native_size, self.native_size):
            raise ValueError(f"Native image shape mismatch: {tuple(image.shape)}")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Dataset must supply finite native RGB in [0, 1]")
        return image, int(label), self.sample_ids[index]


def build_datasets(cfg: dict, output_dir: str | Path) -> dict[str, NativeSubset]:
    """Build only explicitly requested data; downloads require data.download=true.

    Sizes are exact limits, not percentages. A different dataset, seed, content,
    or split size cannot silently reuse an existing experiment directory.
    """
    data = cfg["data"]
    name = str(data["name"]).lower()
    size, classes = int(data["native_size"]), int(data["num_classes"])
    seed = int(data.get("split_seed", 42))
    sizes = {split: int(data[f"{split}_size"]) for split in SPLITS}
    if min(sizes.values()) < 0 or classes < 2 or size < 1 or seed < 0:
        raise ValueError("Dataset sizes/seed must be nonnegative; classes >= 2")
    train_count = sum(sizes[split] for split in SPLITS[:-1])
    root = Path(data.get("root", "data")).expanduser().resolve()
    if name == "synthetic":
        train_pool = _SyntheticPool(train_count, size, classes, seed)
        test_pool = _SyntheticPool(sizes["test"], size, classes, seed + 1000003)
        content = {"generator": "native-pattern-v1", "training_count": len(train_pool),
                   "test_count": len(test_pool)}
    elif name in ("cifar100", "stl10"):
        expected_size, expected_classes = (32, 100) if name == "cifar100" else (96, 10)
        if (size, classes) != (expected_size, expected_classes):
            raise ValueError(f"{name} requires native_size={expected_size}, num_classes={expected_classes}")
        from torchvision.datasets import CIFAR100, STL10
        download = bool(data.get("download", False))
        if name == "cifar100":
            train_pool = CIFAR100(str(root), train=True, download=download)
            test_pool = CIFAR100(str(root), train=False, download=download)
        else:
            train_pool = STL10(str(root), split="train", download=download)
            test_pool = STL10(str(root), split="test", download=download)
        content = {}
        for split, pool in (("train", train_pool), ("test", test_pool)):
            labels = np.asarray(pool.targets if hasattr(pool, "targets") else pool.labels, dtype=np.int64)
            content[split] = {"count": len(pool), "images_sha256": _array_digest(pool.data),
                              "labels_sha256": _array_digest(labels)}
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    if train_count > len(train_pool) or sizes["test"] > len(test_pool):
        raise ValueError("Requested disjoint split sizes exceed the official dataset pools")
    rng = np.random.default_rng(seed)
    train_indices, test_indices = rng.permutation(len(train_pool)), rng.permutation(len(test_pool))
    indices, start = {}, 0
    for split in SPLITS[:-1]:
        indices[split] = train_indices[start:start + sizes[split]].tolist()
        start += sizes[split]
    indices["test"] = test_indices[:sizes["test"]].tolist()
    specification = {"version": 1, "dataset": name, "root": str(root), "native_size": size,
                     "num_classes": classes, "split_seed": seed, "sizes": sizes,
                     "content": content, "indices": indices}
    fingerprint = _digest(specification)
    manifest = {**specification, "fingerprint": fingerprint,
                "sample_ids": {split: [f"{name}:{'test' if split == 'test' else 'train'}:{i:06d}"
                                       for i in values] for split, values in indices.items()}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "splits.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Persisted split manifest differs; choose a new output directory")
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    return {split: NativeSubset(test_pool if split == "test" else train_pool,
                               values, name, "test" if split == "test" else "train", size, fingerprint)
            for split, values in indices.items()}
