"""Fit a dataset classifier on train/selection only, with epoch recovery."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .classifier import build_classifier
from .data import build_datasets


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint():
    root = Path(__file__).parent
    return {name: _file_digest(root / name) for name in ("classifier_training.py", "classifier.py", "data.py")}


def _state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _rng_state():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _heads(model):
    value = model.get_classifier() if hasattr(model, "get_classifier") else getattr(model, "head", None)
    modules = value if isinstance(value, (list, tuple)) else [value]
    if not modules or any(not isinstance(module, nn.Module) for module in modules):
        raise ValueError("Classifier has no supported head; select full_model=True")
    return modules


def fit_classifier(cfg: dict, output_path: str | Path, full_model=False, resume=False) -> dict:
    """Train a head by default; full_model explicitly enables backbone updates.

    The supplied config's training epochs/batch_size/lr/grad_accum are reused.
    Seed is experiment.training_seeds[0]. Epoch checkpoints contain optimizer,
    RNG and best state so interrupted writes cannot change the selected model.
    Completed output files are checksum-verified. During an interrupted fit the
    exported best file is a projection, rebuilt from the committed latest state;
    it may have been written before that epoch's recovery checkpoint committed.
    No calibration, audit or test samples are consumed by optimization/selection.
    """
    cfg = copy.deepcopy(cfg)
    root, output = Path(cfg["output_dir"]), Path(output_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_path, latest_path = root / "classifier_fit.json", root / "classifier_latest.pt"
    if metadata_path.exists() and not resume:
        raise ValueError("Classifier training directory exists; use resume or a new output_dir")
    if output.exists() and not metadata_path.exists():
        raise ValueError("Refusing to overwrite an unrelated classifier checkpoint")
    if output == latest_path.resolve():
        raise ValueError("Output checkpoint must differ from classifier_latest.pt")
    datasets = build_datasets(cfg, root)
    if not len(datasets["train"]) or not len(datasets["selection"]):
        raise ValueError("Classifier fitting requires nonempty train and selection splits")
    seed = int(cfg["experiment"]["training_seeds"][0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    init_cfg = copy.deepcopy(cfg)
    init_cfg["classifier"]["allow_random"] = True
    adapter = build_classifier(init_cfg)
    heads = [] if full_model else _heads(adapter.model)
    adapter.model.requires_grad_(bool(full_model))
    for head in heads:
        head.requires_grad_(True)
    parameters = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Classifier has no trainable parameters")
    protocol = {"version": 1, "config": cfg, "full_model": bool(full_model), "output": str(output),
                "initial_classifier_fingerprint": adapter.fingerprint,
                "source_fingerprint": _source_fingerprint(),
                "split_fingerprint": datasets["train"].manifest_fingerprint}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Classifier training configuration, weights or splits differ (including source code)")
        if previous.get("status") == "completed" and (
                not output.exists() or _file_digest(output) != previous.get("output_sha256")):
            raise ValueError("Completed classifier output checkpoint is missing or its checksum differs")
    training = cfg["training"]
    optimizer = torch.optim.AdamW(parameters, lr=float(training["lr"]))
    epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
    accumulation = int(training.get("grad_accum", 1))
    loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0)
    selection = DataLoader(datasets["selection"], batch_size=batch_size, shuffle=False, num_workers=0)
    history, start, best_key, best_state, best_epoch = [], 0, (-1.0, -float("inf")), None, None
    if resume and latest_path.exists():
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        if latest["fingerprint"] != fingerprint:
            raise ValueError("Classifier recovery checkpoint differs from current configuration")
        if (latest.get("state_sha256") != _state_digest(latest["state_dict"]) or
                latest.get("best_state_sha256") != _state_digest(latest["best_state"])):
            raise ValueError("Classifier recovery checkpoint tensor integrity check failed")
        adapter.model.load_state_dict(latest["state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer"])
        history, start = latest["history"], int(latest["epoch"]) + 1
        best_key, best_state, best_epoch = tuple(latest["best_key"]), latest["best_state"], latest["best_epoch"]
        _restore_rng(latest["rng"])
    initial_exposure = adapter.checkpoint_metadata.get("training_exposure")
    ancestors = []
    if initial_exposure is not None:
        # Continuing a local fit cannot erase its earlier optimization or model
        # selection exposure when the new round uses a different partition.
        ancestors = copy.deepcopy(initial_exposure.get("ancestor_exposure", []))
        ancestors.append({"source": adapter.checkpoint_metadata.get("source"),
                          "training_fingerprint": adapter.checkpoint_metadata.get("training_fingerprint"),
                          "train_ids": list(initial_exposure["train_ids"]),
                          "selection_ids": list(initial_exposure["selection_ids"])})
    exposure = {"train_ids": datasets["train"].sample_ids, "selection_ids": datasets["selection"].sample_ids,
                "ancestor_exposure": ancestors,
                "excluded_splits": ["calibration", "audit", "test"],
                "excluded_splits_scope": "current_fit_only; earlier known exposure is recorded in ancestor_exposure",
                "initial_weights": cfg["classifier"].get("checkpoint") or cfg["classifier"].get("weights"),
                "initial_pretraining_exposure": "Known local fit ancestors are recorded; exposure of any external pretraining weights remains unknown",
                "initial_checkpoint_metadata_status": adapter.checkpoint_metadata.get("metadata_status")}
    def checkpoint():
        return {"format": "vit-iaq-native-classifier-v1", "state_dict": best_state,
                "kind": cfg["classifier"]["kind"],
                "arch": cfg["classifier"].get("arch") if cfg["classifier"]["kind"] == "vit" else "tiny",
                "preprocessing": {key: cfg["classifier"][key] for key in ("input_size", "mean", "std")},
                "num_classes": cfg["data"]["num_classes"], "split_fingerprint": protocol["split_fingerprint"],
                "training_exposure": exposure, "training_fingerprint": fingerprint,
                "best_epoch": best_epoch, "selection_accuracy": best_key[0]}
    report = {"fingerprint": fingerprint, **protocol, "status": "running", "history": history,
              "trainable_parameters": sum(parameter.numel() for parameter in parameters)}
    if resume and best_state is not None:
        _atomic_save(output, checkpoint())
        report["output_rebuilt_from_committed_checkpoint"] = True
    _atomic_json(metadata_path, report)
    for epoch in range(start, epochs):
        adapter.model.train(bool(full_model))
        for head in heads:
            head.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch_index, (images, labels, _ids) in enumerate(loader):
            labels = labels.to(adapter.device)
            loss = F.cross_entropy(adapter(images), labels, reduction="sum")
            group_start = (batch_index // accumulation) * accumulation * batch_size
            group_size = min(accumulation * batch_size, len(datasets["train"]) - group_start)
            (loss / group_size).backward()
            total_loss += float(loss.detach())
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        adapter.model.eval()
        correct, selection_loss = 0, 0.0
        with torch.no_grad():
            for images, labels, _ids in selection:
                labels = labels.to(adapter.device)
                logits = adapter(images)
                correct += int((logits.argmax(1) == labels).sum())
                selection_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        accuracy, validation_loss = correct / len(datasets["selection"]), selection_loss / len(datasets["selection"])
        key = (accuracy, -validation_loss)
        if key > best_key:
            best_key, best_state, best_epoch = key, _cpu_state(adapter.model), epoch
            _atomic_save(output, checkpoint())
        history.append({"epoch": epoch, "train_loss": total_loss / len(datasets["train"]),
                        "selection_loss": validation_loss, "selection_accuracy": accuracy})
        current_state = _cpu_state(adapter.model)
        _atomic_save(latest_path, {"fingerprint": fingerprint, "epoch": epoch,
                     "state_dict": current_state, "optimizer": optimizer.state_dict(),
                     "state_sha256": _state_digest(current_state), "best_state_sha256": _state_digest(best_state),
                     "best_state": best_state, "best_epoch": best_epoch, "best_key": best_key,
                     "history": history, "rng": _rng_state()})
        report["history"] = history
        _atomic_json(metadata_path, report)
    if best_state is None:
        raise ValueError("No complete classifier training epoch is available")
    _atomic_save(output, checkpoint())
    report.update(status="completed", best_epoch=best_epoch, selection_accuracy=best_key[0],
                  checkpoint=str(output), output_sha256=_file_digest(output), training_exposure=exposure)
    _atomic_json(metadata_path, report)
    return report
