"""Strict, serializable configuration shared by every experiment command."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {
    "schema_version": 1, "name": "native-experiment", "output_dir": "outputs/experiments/default", "device": "cpu",
    "data": {"name": "synthetic", "root": "data", "download": False, "native_size": 32,
             "num_classes": 10, "train_size": 16, "selection_size": 4, "calibration_size": 4,
             "audit_size": 4, "test_size": 4, "split_seed": 20261008},
    "classifier": {"kind": "tiny", "arch": "vit_base_patch16_224", "weights": "random",
                   "input_size": 32, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                   "checkpoint": None, "allow_random": False},
    "transmission": {"patch_size": 4, "m_max": 8, "m_base": 1, "header_repetition": 3,
                     "payload_repetition": 1, "base_repetition": 1, "code": "repetition",
                     "channel": "awgn", "noise_mismatch_db": 0.0, "decode": "soft", "prefix_threshold": 0.99},
    "experiment": {"methods": ["uniform", "iaq", "addressed"], "budgets": [6144, 12288],
                   "snrs": [0, 6, 12], "channel_seeds": [0, 1], "training_seeds": [0],
                   "split": "audit", "bootstrap_samples": 1000},
    "repair": {"variants": ["none", "image", "prefix"], "dim": 24, "depth": 1,
               "heads": 3, "architecture": "transformer"},
    "training": {"epochs": 1, "batch_size": 4, "lr": 0.0003, "snr_min": 0.0, "snr_max": 12.0,
                 "lambda_task": 1.0, "lambda_mse": 0.1, "lambda_prefix": 0.1, "lambda_identity": 0.0, "grad_accum": 1},
}


def merge(base, override, prefix=""):
    result = copy.deepcopy(base)
    if not isinstance(override, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a mapping")
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        result[key] = merge(base[key], value, prefix + key + ".") if isinstance(base[key], dict) else value
    return result


def _read(path, seen):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Configuration extends cycle")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping")
    parent = value.pop("extends", None)
    base = _read(path.parent / parent, seen | {path}) if parent else copy.deepcopy(DEFAULTS)
    return merge(base, value)


def _integer(value, key, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")


def validate_config(cfg):
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported schema_version; expected 1")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be nonempty")
    if not isinstance(cfg["device"], str) or not (cfg["device"] == "cpu" or cfg["device"].startswith("cuda")):
        raise ValueError("device must be cpu or cuda[:index]")
    d, c, t, e, r, tr = (cfg[k] for k in ("data", "classifier", "transmission", "experiment", "repair", "training"))
    for key in ("native_size", "num_classes", "train_size", "selection_size", "calibration_size", "audit_size", "test_size"):
        _integer(d[key], "data." + key)
    _integer(d["split_seed"], "data.split_seed", 0)
    if d["name"] not in ("synthetic", "cifar100", "stl10"):
        raise ValueError("Unknown dataset")
    if type(d["download"]) is not bool or type(c["allow_random"]) is not bool:
        raise ValueError("download and allow_random must be booleans")
    expected = {"cifar100": (32, 100, 50000, 10000), "stl10": (96, 10, 5000, 8000)}.get(d["name"])
    if expected:
        if (d["native_size"], d["num_classes"]) != expected[:2]:
            raise ValueError("Dataset native_size/num_classes do not match the official dataset")
        if sum(d[k] for k in ("train_size", "selection_size", "calibration_size", "audit_size")) > expected[2] or d["test_size"] > expected[3]:
            raise ValueError("Requested splits exceed the official dataset")
    if c["kind"] not in ("tiny", "vit"):
        raise ValueError("classifier.kind must be tiny or vit")
    _integer(c["input_size"], "classifier.input_size")
    for key in ("mean", "std"):
        if not isinstance(c[key], list) or len(c[key]) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in c[key]):
            raise ValueError(f"classifier.{key} must contain three finite values")
    if min(c["std"]) <= 0:
        raise ValueError("Normalization std must be positive")
    random_target = not c["checkpoint"] and (c["kind"] == "tiny" or c["weights"] == "random" or (c["weights"] == "imagenet" and d["num_classes"] != 1000))
    if d["name"] != "synthetic" and not c["allow_random"] and random_target:
        raise ValueError("Real-data random classifier requires classifier.allow_random: true (workflow checks only)")
    for key in ("patch_size", "m_max", "m_base", "header_repetition", "payload_repetition", "base_repetition"):
        _integer(t[key], "transmission." + key)
    if d["native_size"] % t["patch_size"] or not 1 <= t["m_base"] <= t["m_max"] <= 8:
        raise ValueError("Native size must divide into patches, and require 1 <= m_base <= m_max <= 8")
    for key in ("header_repetition", "payload_repetition", "base_repetition"):
        if t[key] % 2 != 1:
            raise ValueError(f"{key} must be odd")
    for key, choices in (("code", ("repetition", "hamming74")), ("channel", ("awgn", "rayleigh")), ("decode", ("hard", "soft"))):
        if t[key] not in choices:
            raise ValueError(f"Unknown transmission.{key}")
    if any(type(t[k]) not in (int,float) for k in ("prefix_threshold","noise_mismatch_db")) or not 0.5 < t["prefix_threshold"] <= 1 or not math.isfinite(t["noise_mismatch_db"]):
        raise ValueError("Invalid prefix threshold or noise mismatch")
    for key, choices in (("methods", ("uniform", "iaq", "addressed")),):
        if not isinstance(e[key], list) or not e[key] or len(set(e[key])) != len(e[key]) or not set(e[key]) <= set(choices):
            raise ValueError(f"Invalid experiment.{key}")
    for key in ("budgets", "channel_seeds", "training_seeds"):
        if not isinstance(e[key], list) or not e[key] or len(set(e[key])) != len(e[key]):
            raise ValueError(f"experiment.{key} must be a nonempty unique list")
        for v in e[key]:
            _integer(v, "experiment." + key, 1 if key == "budgets" else 0)
    if not isinstance(e["snrs"], list) or not e["snrs"] or len(set(e["snrs"])) != len(e["snrs"]) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in e["snrs"]):
        raise ValueError("snrs must contain unique finite numbers")
    if e["split"] not in ("audit", "test", "calibration"):
        raise ValueError("Evaluation split must be audit, test or calibration")
    _integer(e["bootstrap_samples"], "bootstrap_samples")
    choices = {"none", "image", "depth", "llr", "bitwise", "uncertainty", "prefix"}
    if not isinstance(r["variants"], list) or not r["variants"] or len(set(r["variants"])) != len(r["variants"]) or not set(r["variants"]) <= choices:
        raise ValueError("Unknown or duplicated repair variant")
    for key in ("dim", "depth", "heads"):
        _integer(r[key], "repair." + key)
    if r["architecture"] not in ("transformer", "cnn") or r["dim"] % r["heads"]:
        raise ValueError("Invalid repair architecture or dim/heads")
    for key in ("epochs", "batch_size", "grad_accum"):
        _integer(tr[key], "training." + key)
    for key in ("lr", "snr_min", "snr_max", "lambda_task", "lambda_mse", "lambda_prefix", "lambda_identity"):
        if not isinstance(tr[key], (int, float)) or not math.isfinite(tr[key]):
            raise ValueError(f"training.{key} must be finite")
    if tr["lr"] <= 0 or tr["snr_min"] > tr["snr_max"] or min(tr[k] for k in ("lambda_task", "lambda_mse", "lambda_prefix", "lambda_identity")) < 0:
        raise ValueError("Invalid training ranges")
    return cfg


def load_config(path):
    cfg = validate_config(_read(path, set()))
    for group, key in ((None, "output_dir"), ("data", "root"), ("classifier", "checkpoint")):
        obj = cfg if group is None else cfg[group]
        if obj[key] is not None:
            obj[key] = str((ROOT / str(obj[key])).resolve())
    return cfg


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
