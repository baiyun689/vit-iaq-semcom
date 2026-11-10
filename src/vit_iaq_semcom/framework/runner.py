"""Single-device experiment lifecycle with atomic, reproducible recovery.

Training checkpoints commit completed epochs. Evaluation batch files are the
authoritative journal; JSONL is a rebuildable projection, never an append-only
resume cursor. Receiver models only receive the explicit evidence whitelist.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as import_metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import PROTOCOL_VERSION
from .config import fingerprint, validate_config


EVIDENCE_KEYS = (
    "hard", "soft", "depth", "uncertainty", "prefix_lower", "prefix_upper",
    "prefix_confidence", "llr", "bit_prob", "bit_mask",
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def source_fingerprint():
    """Include executable package sources and dependency declarations, not logs."""
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    paths = sorted(package.rglob("*.py")) + [root / "pyproject.toml"]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _seed(*parts):
    value = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**32)


def _set_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def _event(path, event, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_jsonable({"time": _now(), "event": event, **fields}), ensure_ascii=False) + "\n")


def _is_smoke(cfg):
    c = cfg["classifier"]
    return cfg["data"]["name"] == "synthetic" or c["kind"] == "tiny" or (
        not c["checkpoint"] and (c["weights"] == "random" or
                                 (c["weights"] == "imagenet" and cfg["data"]["num_classes"] != 1000))
    )


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _ssim(reference, reconstructed):
    """RGB mean SSIM, Gaussian sigma=1.5, population moments, reflect borders."""
    from scipy.ndimage import gaussian_filter
    x, y = np.asarray(reference, dtype=np.float64), np.asarray(reconstructed, dtype=np.float64)
    average = lambda value: gaussian_filter(value, sigma=(0, 1.5, 1.5), truncate=3.5, mode="reflect")
    ux, uy = average(x), average(y)
    vx = np.maximum(average(x * x) - ux * ux, 0)
    vy = np.maximum(average(y * y) - uy * uy, 0)
    covariance = average(x * y) - ux * uy
    result = ((2 * ux * uy + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (ux * ux + uy * uy + 0.01**2) * (vx + vy + 0.03**2)
    )
    return float(result.mean())


def _runtime_environment(device):
    versions = {}
    for distribution in ("torch", "torchvision", "timm", "numpy", "scipy", "PyYAML", "huggingface_hub"):
        try:
            versions[distribution] = import_metadata.version(distribution)
        except import_metadata.PackageNotFoundError:
            versions[distribution] = None
    selected = torch.device(device)
    result = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
              "machine": platform.machine(), "processor": platform.processor(), "packages": versions,
              "torch_loaded_version": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
              "cudnn_version": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
              "selected_device": str(selected), "device_name": platform.processor() or platform.machine(),
              "reproducibility_scope": "Recorded for review, not part of resume rejection. Library or hardware changes require renewed numerical checks; cross-environment bitwise determinism is not guaranteed."}
    if selected.type == "cuda":
        properties = torch.cuda.get_device_properties(selected)
        result.update(device_name=properties.name, device_total_memory_bytes=properties.total_memory,
                      device_compute_capability=[properties.major, properties.minor])
    return result


def plan(cfg):
    """Return a complete serial work matrix without touching data or models."""
    validate_config(cfg)
    jobs = []
    for method in cfg["experiment"]["methods"]:
        for budget in cfg["experiment"]["budgets"]:
            for train_seed in cfg["experiment"]["training_seeds"]:
                for variant in cfg["repair"]["variants"]:
                    relative = f"jobs/{method}/budget_{budget}/seed_{train_seed}/{variant}"
                    jobs.append({
                        "method": method, "budget": budget, "train_seed": train_seed,
                        "variant": variant, "directory": relative,
                        "train": variant != "none",
                    })
    e = cfg["experiment"]
    return {
        "protocol_version": PROTOCOL_VERSION, "name": cfg["name"],
        "output_dir": str(Path(cfg["output_dir"]).resolve()),
        "config_fingerprint": fingerprint(cfg), "source_fingerprint": source_fingerprint(),
        "purpose": "workflow_check" if _is_smoke(cfg) else "research_experiment",
        "jobs": jobs, "training_jobs": sum(j["train"] for j in jobs),
        "evaluation_jobs": len(jobs), "evaluation_split": e["split"],
        "expected_rows": len(jobs) * cfg["data"][e["split"] + "_size"] * len(e["snrs"]) * len(e["channel_seeds"]),
        "notes": [
            "Runs sequentially on one configured device; no runtime estimate is assumed.",
            "No-repair baselines are repeated by training seed for paired comparisons; these are not independent models.",
            "Selection uses only the selection split and a fixed channel seed over the configured SNR grid.",
            "A successful run establishes framework execution, not research novelty or performance gains.",
        ],
    }


def _dependencies():
    from .data import build_datasets
    from .classifier import build_classifier
    from .models import build_repair, prefix_penalty
    from .transport import transmit
    return build_datasets, build_classifier, build_repair, prefix_penalty, transmit


def _audit_classifier_exposure(classifier, datasets):
    """Reject known classifier fit/selection exposure in reserved image pools.

    Repair training and model selection may reuse classifier train/selection
    pools. Calibration, audit and official test must remain unseen by the known
    local classifier fit. External pretraining exposure cannot be inferred.
    """
    metadata = classifier.checkpoint_metadata
    exposure = metadata.get("training_exposure")
    if exposure is None:
        return {"status": metadata.get("training_exposure_status", "external_unknown"),
                "source": metadata.get("source"), "checked_known_local_exposure": False,
                "note": "No recorded local fit IDs are available; external pretraining exposure is not verified."}
    recorded_fits = [exposure, *exposure.get("ancestor_exposure", [])]
    known_train = set().union(*(set(record["train_ids"]) for record in recorded_fits))
    known_selection = set().union(*(set(record["selection_ids"]) for record in recorded_fits))
    exposed = known_train | known_selection
    overlap = {split: sorted(exposed & set(dataset.sample_ids)) for split, dataset in datasets.items()}
    for split in ("calibration", "audit", "test"):
        if overlap[split]:
            raise ValueError(
                f"Classifier training/selection exposure overlaps reserved split {split}: "
                f"{len(overlap[split])} image(s), examples={overlap[split][:3]}. "
                "Preserve the classifier fit partition or use weights fitted without these reserved images."
            )
    return {"status": "recorded_local_fit_no_reserved_overlap", "source": metadata.get("source"),
            "checked_known_local_exposure": True, "known_train_images": len(known_train),
            "known_selection_images": len(known_selection),
            "known_fit_generations": len(recorded_fits),
            "overlap_counts": {split: len(ids) for split, ids in overlap.items()},
            "permitted_overlap_splits": ["train", "selection"],
            "prior_pretraining_exposure": exposure.get("initial_pretraining_exposure", "external_unknown")}


def _context(cfg, output, resume):
    build_datasets, build_classifier, build_repair, penalty, transmit = _dependencies()
    signature = {"protocol_version": PROTOCOL_VERSION, "config_fingerprint": fingerprint(cfg), "source_fingerprint": source_fingerprint()}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        old = _load(manifest_path)
        if not resume:
            raise ValueError("Run directory already exists; use resume or a new output_dir")
        if any(old.get(k) != v for k, v in signature.items()):
            raise ValueError("Run fingerprint mismatch: configuration or executable source changed; use a new output_dir")
    elif any((output / name).exists() for name in ("jobs", "run_state.json")):
        raise ValueError("Output has run artifacts but no manifest; refusing unsafe recovery")
    else:
        old = None
    datasets = build_datasets(cfg, output)
    classifier = build_classifier(cfg)
    exposure_audit = _audit_classifier_exposure(classifier, datasets)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    split_path = output / "splits.json"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest() if split_path.exists() else fingerprint({
        name: [str(dataset[i][2]) for i in range(len(dataset))] for name, dataset in datasets.items()
    })
    signature.update(split_fingerprint=split_hash, classifier_fingerprint=classifier.fingerprint)
    if old and any(old.get(k) != v for k, v in signature.items()):
        raise ValueError("Run fingerprint mismatch: data split or classifier weights changed")
    environment = _runtime_environment(classifier.device)
    manifest = {**signature, "config": cfg, "purpose": "workflow_check" if _is_smoke(cfg) else "research_experiment",
                "classifier_is_random": bool(classifier.is_random),
                "classifier_checkpoint_metadata": classifier.checkpoint_metadata,
                "classifier_exposure_audit": exposure_audit,
                "runtime_environment": environment,
                "runtime_environment_history": [{"observed_at": _now(), "environment": environment}],
                "ssim_definition": "RGB channel/spatial mean; data range=1; Gaussian sigma=1.5, truncate=3.5, reflect border, population moments; K1=.01 K2=.03",
                "timing_definition": "Observed batched wall times, synchronized on CUDA; no warmup or dedicated hardware benchmark. CPU/channel and model costs are reported separately."}
    if old is None:
        _atomic_json(manifest_path, manifest)
    elif not any(entry.get("environment") == environment for entry in old.get("runtime_environment_history", [])):
        old.setdefault("runtime_environment_history", []).append({"observed_at": _now(), "environment": environment})
        _atomic_json(manifest_path, old)
        _event(output / "events.jsonl", "runtime_environment_change_requires_review", environment=environment)
    _atomic_json(output / "plan.json", plan(cfg))
    return datasets, classifier, build_repair, penalty, transmit, fingerprint(signature)


def _batches(dataset, batch_size, order=None):
    order = list(range(len(dataset))) if order is None else order
    for start in range(0, len(order), batch_size):
        examples = [dataset[int(i)] for i in order[start:start + batch_size]]
        images = torch.stack([item[0] for item in examples])
        labels = torch.tensor([int(item[1]) for item in examples], dtype=torch.long)
        ids = [str(item[2]) for item in examples]
        yield images, labels, ids


def _damage(images, ids, classifier, cfg, job, snr, phase, repeat_seed, transmit):
    device = classifier.device
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        scores = classifier.importance(images.to(device), cfg["transmission"]["patch_size"])
    importance_seconds = time.perf_counter() - started
    results = []
    for i, sample_id in enumerate(ids):
        # Method, budget and repair variant are intentionally absent: the same
        # physical image realization (including Rayleigh fade) is shared.
        # Transport owns protocol-specific bit/noise streams; different formats
        # need not suffer identical semantic damage.
        channel_seed = _seed("native-channel-v2", phase, float(snr), repeat_seed, sample_id)
        started = time.perf_counter()
        result = transmit(images[i].detach().cpu().numpy(), np.asarray(scores[i]), job["method"], job["budget"], float(snr), channel_seed, cfg["transmission"])
        result["_timings"] = {"importance_seconds": importance_seconds / len(ids), "transmission_seconds": time.perf_counter() - started}
        if int(result["counts"]["total"]) > job["budget"]:
            raise RuntimeError("Transport exceeded the declared channel-bit budget")
        results.append(result)
    started = time.perf_counter()
    damaged = torch.as_tensor(np.stack([r["image"] for r in results]), device=device, dtype=torch.float32)
    evidence = {
        key: torch.as_tensor(np.stack([r[key] for r in results]), device=device, dtype=torch.float32)
        for key in EVIDENCE_KEYS
    }
    if not torch.isfinite(damaged).all() or any(not torch.isfinite(value).all() for value in evidence.values()):
        raise RuntimeError("Transport produced non-finite receiver inputs")
    _synchronize(device)
    transfer_seconds = (time.perf_counter() - started) / len(ids)
    for result in results:
        result["_timings"]["receiver_tensor_transfer_seconds"] = transfer_seconds
    return damaged, evidence, results


def _selection(model, classifier, dataset, cfg, job, transmit):
    model.eval()
    correct, total, mse = 0, 0, 0.0
    with torch.no_grad():
        for snr in cfg["experiment"]["snrs"]:
            for images, labels, ids in _batches(dataset, cfg["training"]["batch_size"]):
                damaged, evidence, _ = _damage(images, ids, classifier, cfg, job, snr, "selection", 0, transmit)
                output = model(damaged, evidence)
                logits = classifier(output)
                correct += int((logits.argmax(1).cpu() == labels).sum())
                mse += float(((output - images.to(classifier.device)) ** 2).flatten(1).mean(1).sum())
                total += len(ids)
    if not total:
        raise ValueError("Selection split is empty")
    return {"accuracy": correct / total, "mse": mse / total, "samples_with_snr_repeats": total}


def _neutral_evidence(evidence):
    result = {key: torch.zeros_like(value) for key, value in evidence.items()}
    result["prefix_upper"] = torch.ones_like(result["prefix_upper"])
    result["bit_prob"] = torch.full_like(result["bit_prob"], 0.5)
    return result


def _train(cfg, job, job_dir, signature, dataset, selection, classifier, build_repair, penalty, transmit):
    _set_rng(job["train_seed"])
    model = build_repair(cfg, job["variant"]).to(classifier.device)
    if not job["train"]:
        return model.eval(), []
    tr = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"])
    latest_path = job_dir / "latest.pt"
    history, completed, best_score, best_state = [], 0, None, None
    if latest_path.exists():
        # These are locally generated full training states, not untrusted weights.
        state = torch.load(latest_path, map_location=classifier.device, weights_only=False)
        if state.get("signature") != signature:
            raise ValueError("Checkpoint signature mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, completed = state["history"], int(state["completed_epochs"])
        best_score, best_state = state["best_score"], state["best_model"]
        _restore_rng(state["rng"])
    for epoch in range(completed, tr["epochs"]):
        start_time = time.perf_counter()
        model.train()
        order = np.random.default_rng(_seed("order", job["train_seed"], epoch)).permutation(len(dataset))
        n_batches = math.ceil(len(dataset) / tr["batch_size"])
        running = {key: 0.0 for key in ("loss", "task", "mse", "prefix", "identity")}
        count, optimizer_steps = 0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, labels, ids) in enumerate(_batches(dataset, tr["batch_size"], order)):
            snr_rng = np.random.default_rng(_seed("train-snr", job["train_seed"], epoch, batch_index))
            snr = float(snr_rng.uniform(tr["snr_min"], tr["snr_max"]))
            damaged, evidence, _ = _damage(images, ids, classifier, cfg, job, snr, "train", [job["train_seed"], epoch], transmit)
            images, labels = images.to(classifier.device), labels.to(classifier.device)
            output = model(damaged, evidence)
            task = F.cross_entropy(classifier(output), labels) if tr["lambda_task"] else output.sum() * 0.0
            mse = F.mse_loss(output, images)
            prefix = penalty(output, evidence) if job["variant"] == "prefix" else output.sum() * 0.0
            identity = F.mse_loss(model(images, _neutral_evidence(evidence)), images) if tr["lambda_identity"] else output.sum() * 0.0
            loss = tr["lambda_task"] * task + tr["lambda_mse"] * mse + tr["lambda_prefix"] * prefix + tr["lambda_identity"] * identity
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            # Weight each microbatch by its actual example count. The final
            # accumulation group can contain fewer batches and a short batch.
            group_start = (batch_index // tr["grad_accum"]) * tr["grad_accum"]
            group_end = min(group_start + tr["grad_accum"], n_batches)
            group_examples = min(group_end * tr["batch_size"], len(dataset)) - group_start * tr["batch_size"]
            (loss * (len(ids) / group_examples)).backward()
            if (batch_index + 1) % tr["grad_accum"] == 0 or batch_index + 1 == n_batches:
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise RuntimeError("Non-finite training gradient; previous complete epoch remains recoverable")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            for key, value in zip(running, (loss, task, mse, prefix, identity)):
                running[key] += float(value.detach()) * len(ids)
            count += len(ids)
        if count == 0:
            raise ValueError("Training split is empty")
        selected = _selection(model, classifier, selection, cfg, job, transmit)
        score = (selected["accuracy"], -selected["mse"])
        if best_score is None or tuple(score) > tuple(best_score):
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch + 1, **{key: value / count for key, value in running.items()},
                        "selection": selected, "optimizer_steps": optimizer_steps, "seconds": time.perf_counter() - start_time})
        state = {
            "signature": signature, "config": cfg, "job": job, "completed_epochs": epoch + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state(),
            "history": history, "best_score": best_score, "best_model": best_state,
        }
        _atomic_torch(latest_path, state)
        _atomic_json(job_dir / "history.json", history)
        _event(job_dir / "events.jsonl", "epoch_completed", epoch=epoch + 1, selection=selected)
    if best_state is None:
        raise RuntimeError("Training has no completed model checkpoint")
    model.load_state_dict(best_state)
    _atomic_torch(job_dir / "best.pt", {"signature": signature, "config": cfg, "job": job, "model": best_state, "best_score": best_score})
    return model.eval(), history


def _projection(job_dir):
    """Rebuild result rows from committed batch journals, without duplicate IDs."""
    destination = job_dir / "predictions.jsonl"
    temporary = destination.with_name(destination.name + ".tmp")
    seen, count = set(), 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for batch_path in sorted((job_dir / "evaluation_batches").glob("*.json")):
            batch = _load(batch_path)
            if batch.get("rows_fingerprint") != fingerprint(batch["rows"]):
                raise ValueError(f"Evaluation batch contents changed: {batch_path}")
            for row in batch["rows"]:
                key = (row["sample_id"], row["snr"], row["channel_seed"])
                if key in seen:
                    raise ValueError("Duplicate row in evaluation journal")
                seen.add(key)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                count += 1
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return count


def _evaluate(cfg, job, job_dir, signature, dataset, classifier, model, transmit):
    batch_index, skipped = 0, 0
    try:
        for snr in cfg["experiment"]["snrs"]:
            for channel_seed in cfg["experiment"]["channel_seeds"]:
                for images, labels, ids in _batches(dataset, cfg["training"]["batch_size"]):
                    batch_path = job_dir / "evaluation_batches" / f"{batch_index:08d}.json"
                    metadata = {"signature": signature, "snr": float(snr), "channel_seed": channel_seed, "ids": ids}
                    batch_index += 1
                    if batch_path.exists():
                        old = _load(batch_path)
                        if old.get("metadata") != metadata or old.get("rows_fingerprint") != fingerprint(old["rows"]):
                            raise ValueError("Evaluation batch fingerprint mismatch")
                        skipped += len(ids)
                        continue
                    started = time.perf_counter()
                    with torch.no_grad():
                        damaged, evidence, transports = _damage(images, ids, classifier, cfg, job, snr, "evaluation", channel_seed, transmit)
                        stage_started = time.perf_counter()
                        after_image = model(damaged, evidence)
                        if not torch.isfinite(after_image).all():
                            raise RuntimeError("Repair model produced non-finite pixels")
                        _synchronize(classifier.device)
                        repair_seconds = (time.perf_counter() - stage_started) / len(ids)
                        stage_started = time.perf_counter()
                        clean = classifier(images.to(classifier.device)).argmax(1).cpu().tolist()
                        before = classifier(damaged).argmax(1).cpu().tolist()
                        after = classifier(after_image).argmax(1).cpu().tolist()
                        classification_seconds = (time.perf_counter() - stage_started) / len(ids)
                        mse = ((after_image - images.to(classifier.device)) ** 2).flatten(1).mean(1).cpu().tolist()
                        before_mse = ((damaged - images.to(classifier.device)) ** 2).flatten(1).mean(1).cpu().tolist()
                        after_numpy = after_image.cpu().numpy()
                        before_numpy = damaged.cpu().numpy()
                    elapsed = time.perf_counter() - started
                    rows = []
                    for i, sample_id in enumerate(ids):
                        label = int(labels[i])
                        rows.append(_jsonable({
                            "method": job["method"], "variant": job["variant"], "budget": job["budget"],
                            "snr": float(snr), "train_seed": job["train_seed"], "channel_seed": channel_seed,
                            "sample_id": sample_id, "split": cfg["experiment"]["split"], "label": label,
                            "before": before[i], "after": after[i], "clean": clean[i],
                            "correct": after[i] == label, "before_correct": before[i] == label,
                            "clean_correct": clean[i] == label, "mse": mse[i], "before_mse": before_mse[i],
                            "psnr": -10 * math.log10(max(mse[i], 1e-12)), "psnr_cap_db": 120,
                            "ssim": _ssim(images[i].numpy(), after_numpy[i]),
                            "before_ssim": _ssim(images[i].numpy(), before_numpy[i]),
                            "total_bits": int(transports[i]["counts"]["total"]),
                            "counts": transports[i]["counts"], "diagnostics": transports[i].get("diagnostics", {}),
                            "workflow_check": _is_smoke(cfg), "batch_seconds_per_image": elapsed / len(ids),
                            "timings": {**transports[i]["_timings"], "repair_seconds": repair_seconds,
                                        "three_classifications_seconds": classification_seconds},
                        }))
                    _atomic_json(batch_path, {"metadata": metadata, "rows": rows, "rows_fingerprint": fingerprint(rows)})
                    _event(job_dir / "events.jsonl", "evaluation_batch_completed", batch=batch_index, samples=len(ids))
    finally:
        count = _projection(job_dir)
    expected = len(dataset) * len(cfg["experiment"]["snrs"]) * len(cfg["experiment"]["channel_seeds"])
    if count != expected:
        raise RuntimeError(f"Incomplete evaluation: {count} / {expected} rows")
    return {"rows": count, "resumed_rows": skipped}


def run(cfg, resume=True):
    """Execute every configured job; failure leaves a recoverable durable state."""
    validate_config(cfg)
    output = Path(cfg["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists() and _load(output / "run_manifest.json").get("evaluate_only"):
        raise ValueError("This is an evaluation-only directory; use evaluate with the original checkpoints")
    datasets, classifier, build_repair, penalty, transmit, run_signature = _context(cfg, output, resume)
    run_plan = plan(cfg)
    results = []
    _atomic_json(output / "run_state.json", {"status": "running", "updated_at": _now(), "signature": run_signature})
    try:
        for job in run_plan["jobs"]:
            job_dir = output / job["directory"]
            job_dir.mkdir(parents=True, exist_ok=True)
            signature = fingerprint({"run": run_signature, "job": job})
            state_path = job_dir / "state.json"
            old = _load(state_path) if state_path.exists() else None
            if old and old.get("signature") != signature:
                raise ValueError("Job fingerprint mismatch")
            if old and old.get("status") == "completed":
                for name, expected_hash in old.get("artifact_hashes", {}).items():
                    artifact = job_dir / name
                    if not artifact.is_file() or _file_digest(artifact) != expected_hash:
                        raise ValueError(f"Completed model artifact changed or missing: {artifact}")
                rows = _projection(job_dir)
                if rows != old["rows"]:
                    raise ValueError("Completed result rows have changed")
                results.append({**job, "status": "completed", "rows": rows, "skipped_completed": True})
                continue
            _atomic_json(state_path, {"status": "running", "signature": signature, "job": job, "updated_at": _now()})
            _event(output / "events.jsonl", "job_started", **job)
            try:
                job_started = time.perf_counter()
                if torch.device(classifier.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(classifier.device)
                model, history = _train(cfg, job, job_dir, signature, datasets["train"], datasets["selection"], classifier, build_repair, penalty, transmit)
                evaluated = _evaluate(cfg, job, job_dir, signature, datasets[cfg["experiment"]["split"]], classifier, model, transmit)
                result = {**job, **evaluated, "status": "completed", "completed_epochs": len(history), "signature": signature, "updated_at": _now()}
                result["seconds_this_invocation"] = time.perf_counter() - job_started
                result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(classifier.device) if torch.device(classifier.device).type == "cuda" else None
                result["artifact_hashes"] = {
                    name: _file_digest(job_dir / name) for name in ("best.pt", "latest.pt", "history.json")
                    if (job_dir / name).is_file()
                }
                _atomic_json(state_path, result)
                results.append(result)
                _event(output / "events.jsonl", "job_completed", **job, rows=evaluated["rows"])
                del model
            except BaseException as error:
                _atomic_json(state_path, {"status": "failed", "signature": signature, "job": job, "error": str(error), "updated_at": _now()})
                raise
        result = {"status": "completed", "purpose": run_plan["purpose"], "output_dir": str(output), "signature": run_signature,
                  "jobs": results, "rows": sum(item["rows"] for item in results), "updated_at": _now()}
        _atomic_json(output / "run_state.json", result)
        return result
    except BaseException as error:
        _atomic_json(output / "run_state.json", {"status": "failed", "signature": run_signature, "error": str(error), "completed_jobs": len(results), "updated_at": _now()})
        _event(output / "events.jsonl", "run_failed", error=str(error))
        raise


def _evaluation_compatibility(cfg, source_cfg, allow_channel_shift=False):
    """Permit only evaluation settings and a subset of the trained matrix."""
    current, original = copy.deepcopy(cfg), copy.deepcopy(source_cfg)
    for item in (current, original):
        item.pop("name")
        item.pop("output_dir")
        for key in ("split", "snrs", "channel_seeds", "bootstrap_samples"):
            item["experiment"].pop(key)
    for section, key in (("experiment", "methods"), ("experiment", "budgets"),
                         ("experiment", "training_seeds"), ("repair", "variants")):
        if not set(current[section][key]) <= set(original[section][key]):
            raise ValueError(f"Evaluation {section}.{key} must be a subset of the checkpoint run")
        current[section].pop(key)
        original[section].pop(key)
    if allow_channel_shift:
        for key in ("channel", "noise_mismatch_db"):
            current["transmission"].pop(key)
            original["transmission"].pop(key)
    if current != original:
        raise ValueError("Evaluation configuration changes training, data, classifier, model or transport settings; only evaluation settings and matrix subsets may change")


def evaluate(cfg, checkpoint_root, resume=False, allow_channel_shift=False):
    """Evaluate frozen source-run checkpoints on new SNRs or held-out splits.

    No training or model selection occurs. The source run must be complete and
    its source code, data, classifier, model structure and checkpoint hashes must
    still agree. With allow_channel_shift, only the channel family and receiver
    noise-variance mismatch may additionally change. Results use a separate
    recoverable evaluation journal and record that explicit shift.
    """
    validate_config(cfg)
    if type(allow_channel_shift) is not bool:
        raise ValueError("allow_channel_shift must be an explicit boolean")
    source = Path(checkpoint_root).resolve()
    output = Path(cfg["output_dir"]).resolve()
    if source == output:
        raise ValueError("Frozen-checkpoint evaluation requires a separate output_dir")
    if not (source / "run_manifest.json").is_file() or not (source / "run_state.json").is_file():
        raise ValueError("Checkpoint source has no complete run manifest/state")
    source_manifest = _load(source / "run_manifest.json")
    source_state = _load(source / "run_state.json")
    if source_state.get("status") != "completed" or source_manifest.get("evaluate_only"):
        raise ValueError("Checkpoint source must be a completed original training run")
    source_cfg = source_manifest["config"]
    if source_manifest.get("config_fingerprint") != fingerprint(source_cfg):
        raise ValueError("Checkpoint source configuration fingerprint mismatch")
    if source_manifest.get("source_fingerprint") != source_fingerprint() or source_manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Checkpoint source code or protocol differs from the current framework")
    _evaluation_compatibility(cfg, source_cfg, allow_channel_shift)
    source_signature_fields = {key: source_manifest[key] for key in (
        "protocol_version", "config_fingerprint", "source_fingerprint", "split_fingerprint", "classifier_fingerprint"
    )}
    if source_state.get("signature") != fingerprint(source_signature_fields):
        raise ValueError("Checkpoint source run signature mismatch")
    jobs = plan(cfg)["jobs"]
    source_jobs = {}
    checkpoint_hashes = {}
    for job in jobs:
        directory = source / job["directory"]
        if not (directory / "state.json").is_file():
            raise ValueError(f"Source checkpoint job is missing: {job['directory']}")
        state = _load(directory / "state.json")
        expected_signature = fingerprint({"run": source_state["signature"], "job": job})
        if state.get("status") != "completed" or state.get("signature") != expected_signature:
            raise ValueError(f"Source checkpoint job is incomplete or incompatible: {job['directory']}")
        if job["train"]:
            best_path = directory / "best.pt"
            expected_hash = state.get("artifact_hashes", {}).get("best.pt")
            if not expected_hash or not best_path.is_file() or _file_digest(best_path) != expected_hash:
                raise ValueError(f"Source best checkpoint is missing or changed: {best_path}")
            checkpoint_hashes[job["directory"]] = expected_hash
        source_jobs[job["directory"]] = state
    channel_shift = {
        "source": {key: source_cfg["transmission"][key] for key in ("channel", "noise_mismatch_db")},
        "target": {key: cfg["transmission"][key] for key in ("channel", "noise_mismatch_db")},
    }
    source_identity = {"directory": str(source), "signature": source_state["signature"],
                       "config_fingerprint": source_manifest["config_fingerprint"], "best_checkpoint_hashes": checkpoint_hashes,
                       "allow_channel_shift": allow_channel_shift, "channel_shift": channel_shift}
    target_manifest_path = output / "run_manifest.json"
    if target_manifest_path.exists():
        existing = _load(target_manifest_path)
        if not existing.get("evaluate_only") or existing.get("source_run") != source_identity:
            raise ValueError("Evaluation directory belongs to a different source run or lifecycle")
    output.mkdir(parents=True, exist_ok=True)
    datasets, classifier, build_repair, _, transmit, target_signature = _context(cfg, output, resume)
    manifest = _load(target_manifest_path)
    for key in ("split_fingerprint", "classifier_fingerprint"):
        if manifest[key] != source_manifest[key]:
            raise ValueError(f"Evaluation {key} differs from the trained source")
    manifest.update(evaluate_only=True, source_run=source_identity,
                    allow_channel_shift=allow_channel_shift, channel_shift=channel_shift)
    _atomic_json(target_manifest_path, manifest)
    run_signature = fingerprint({"run": target_signature, "source_run": source_identity, "evaluate_only": True})
    evaluation_plan = plan(cfg)
    evaluation_plan.update(evaluate_only=True, source_run=source_identity, training_jobs=0)
    evaluation_plan["jobs"] = [{**job, "train": False} for job in jobs]
    _atomic_json(output / "plan.json", evaluation_plan)
    results = []
    _atomic_json(output / "run_state.json", {"status": "running", "evaluate_only": True, "source_run": source_identity,
                                            "signature": run_signature, "updated_at": _now()})
    try:
        for job in jobs:
            job_dir = output / job["directory"]
            job_dir.mkdir(parents=True, exist_ok=True)
            signature = fingerprint({"run": run_signature, "job": job})
            state_path = job_dir / "state.json"
            old = _load(state_path) if state_path.exists() else None
            if old and old.get("signature") != signature:
                raise ValueError("Frozen evaluation job fingerprint mismatch")
            if old and old.get("status") == "completed":
                count = _projection(job_dir)
                if count != old["rows"]:
                    raise ValueError("Completed frozen evaluation rows changed")
                results.append({**job, "train": False, "status": "completed", "rows": count, "skipped_completed": True})
                continue
            _atomic_json(state_path, {"status": "running", "evaluate_only": True, "signature": signature,
                                      "job": job, "updated_at": _now()})
            _event(output / "events.jsonl", "frozen_evaluation_job_started", **job)
            try:
                model = build_repair(cfg, job["variant"]).to(classifier.device)
                if job["train"]:
                    # Source hashes are checked above before loading these
                    # locally produced frozen inference checkpoint tensors.
                    checkpoint = torch.load(source / job["directory"] / "best.pt", map_location=classifier.device, weights_only=True)
                    if checkpoint.get("signature") != source_jobs[job["directory"]]["signature"]:
                        raise ValueError("Best checkpoint signature differs from its completed source job")
                    model.load_state_dict(checkpoint["model"], strict=True)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                started = time.perf_counter()
                if torch.device(classifier.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(classifier.device)
                evaluated = _evaluate(cfg, job, job_dir, signature, datasets[cfg["experiment"]["split"]], classifier, model, transmit)
                result = {**job, **evaluated, "train": False, "status": "completed", "evaluate_only": True,
                          "signature": signature, "source_job_signature": source_jobs[job["directory"]]["signature"],
                          "seconds_this_invocation": time.perf_counter() - started, "updated_at": _now(),
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(classifier.device) if torch.device(classifier.device).type == "cuda" else None}
                _atomic_json(state_path, result)
                results.append(result)
                _event(output / "events.jsonl", "frozen_evaluation_job_completed", **job, rows=evaluated["rows"])
                del model
            except BaseException as error:
                _atomic_json(state_path, {"status": "failed", "evaluate_only": True, "signature": signature,
                                          "job": job, "error": str(error), "updated_at": _now()})
                raise
        result = {"status": "completed", "evaluate_only": True, "source_run": source_identity,
                  "signature": run_signature, "purpose": evaluation_plan["purpose"], "output_dir": str(output),
                  "jobs": results, "rows": sum(item["rows"] for item in results), "updated_at": _now()}
        _atomic_json(output / "run_state.json", result)
        return result
    except BaseException as error:
        _atomic_json(output / "run_state.json", {"status": "failed", "evaluate_only": True, "source_run": source_identity,
                                                 "signature": run_signature, "error": str(error),
                                                 "completed_jobs": len(results), "updated_at": _now()})
        _event(output / "events.jsonl", "frozen_evaluation_failed", error=str(error))
        raise


def calibrate(cfg):
    """Audit prefix confidence on the calibration split, never tune on test data.

    This reports observed coverage and candidate thresholds. It neither rewrites
    configuration nor treats simulated conditional probabilities as calibrated.
    """
    validate_config(cfg)
    output = Path(cfg["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    datasets, classifier, _, _, transmit, signature = _context(cfg, output, True)
    thresholds = sorted(set([0.9, 0.99, float(cfg["transmission"]["prefix_threshold"])]))
    candidate_configs = {}
    for threshold in thresholds:
        candidate_configs[threshold] = copy.deepcopy(cfg)
        candidate_configs[threshold]["transmission"]["prefix_threshold"] = threshold
    results = []
    for method in cfg["experiment"]["methods"]:
        for budget in cfg["experiment"]["budgets"]:
            for snr in cfg["experiment"]["snrs"]:
                counters = {threshold: {"accepted": 0, "covered": 0, "interval_width_sum": 0.0} for threshold in thresholds}
                total = 0
                bit_count, bit_brier_sum = 0, 0.0
                bin_counts, bin_confidence, bin_correct = (np.zeros(10, dtype=np.float64) for _ in range(3))
                job = {"method": method, "budget": budget}
                for channel_seed in cfg["experiment"]["channel_seeds"]:
                    for images, _, ids in _batches(datasets["calibration"], cfg["training"]["batch_size"]):
                        truth = images.to(classifier.device)
                        evidence = None
                        for threshold in thresholds:
                            # Reconstruct the longest admissible prefix for this
                            # candidate. Filtering intervals formed at another
                            # threshold loses shorter/longer valid prefixes.
                            _, candidate_evidence, _ = _damage(
                                images, ids, classifier, candidate_configs[threshold], job,
                                snr, "calibration", channel_seed, transmit,
                            )
                            if evidence is None:
                                evidence = candidate_evidence  # Bit probabilities do not depend on prefix selection.
                            lower, upper = candidate_evidence["prefix_lower"], candidate_evidence["prefix_upper"]
                            covered = (truth >= lower) & ((truth < upper) | ((truth == 1.0) & (upper == 1.0)))
                            accepted = (candidate_evidence["prefix_confidence"] >= threshold) & ((upper - lower) < 1.0)
                            counters[threshold]["accepted"] += int(accepted.sum())
                            counters[threshold]["covered"] += int((accepted & covered).sum())
                            counters[threshold]["interval_width_sum"] += float((upper - lower)[accepted].sum())
                        total += truth.numel()
                        # Ground truth is used here only for held-out scoring,
                        # never exposed to the repair network as evidence.
                        m = cfg["transmission"]["m_max"]
                        quantized = np.minimum(np.floor(images.numpy() * (2 ** m)), 2 ** m - 1).astype(np.int64)
                        shifts = np.arange(m - 1, -1, -1)[None, None, :, None, None]
                        true_bits = ((quantized[:, :, None] >> shifts) & 1).reshape(len(ids), 3 * m, *images.shape[-2:])
                        present = evidence["bit_mask"].cpu().numpy().astype(bool)
                        bit_probability = evidence["bit_prob"].cpu().numpy()[present]
                        bit_truth = true_bits[present]
                        bit_count += bit_probability.size
                        bit_brier_sum += float(np.square(bit_probability - bit_truth).sum())
                        marginal_confidence = np.maximum(bit_probability, 1.0 - bit_probability)
                        marginal_correct = ((bit_probability >= 0.5) == bit_truth)
                        indexes = np.clip(((marginal_confidence - 0.5) * 20).astype(np.int64), 0, 9)
                        bin_counts += np.bincount(indexes, minlength=10)
                        bin_confidence += np.bincount(indexes, weights=marginal_confidence, minlength=10)
                        bin_correct += np.bincount(indexes, weights=marginal_correct, minlength=10)
                reliability = [{"confidence_lower": 0.5 + i * 0.05, "confidence_upper": 0.55 + i * 0.05,
                                "count": int(bin_counts[i]),
                                "mean_confidence": float(bin_confidence[i] / bin_counts[i]) if bin_counts[i] else None,
                                "marginal_map_accuracy": float(bin_correct[i] / bin_counts[i]) if bin_counts[i] else None}
                               for i in range(10)]
                results.append({"method": method, "budget": budget, "snr": snr, "pixel_channels": total,
                                "received_bit_count": bit_count,
                                "bit_brier_score": bit_brier_sum / bit_count if bit_count else None,
                                "bit_confidence_ece": float(np.abs(bin_confidence - bin_correct).sum() / bit_count) if bit_count else None,
                                "bit_reliability_bins": reliability,
                                "thresholds": [{"threshold": threshold, **values,
                                                "acceptance_fraction": values["accepted"] / total if total else 0,
                                                "empirical_prefix_coverage": values["covered"] / values["accepted"] if values["accepted"] else None,
                                                "mean_accepted_interval_width": values["interval_width_sum"] / values["accepted"] if values["accepted"] else None}
                                               for threshold, values in counters.items()]})
    result = {"signature": signature, "split": "calibration", "purpose": "workflow_check" if _is_smoke(cfg) else "calibration_audit",
              "results": results, "config_unchanged": True, "updated_at": _now(),
              "notes": ["Pixel-channel repeats are correlated; counts are descriptive, not independent trials.",
                        "Every candidate threshold regenerates its longest admissible prefix using the same image/channel seed; candidates are 0.9, 0.99 and the configured threshold (duplicates removed).",
                        "Use only these calibration observations to predeclare thresholds before audit/test evaluation."]}
    _atomic_json(output / "calibration.json", result)
    return result
