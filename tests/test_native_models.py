"""Native-domain data/gradient/evidence invariants, requiring no downloads."""

import copy
import json

import numpy as np
import pytest
import torch

from vit_iaq_semcom.framework.classifier import build_classifier, map_attention_to_grid
from vit_iaq_semcom.framework.data import build_datasets
from vit_iaq_semcom.framework.models import VARIANTS, build_repair, prefix_penalty


def config():
    return {"device": "cpu", "data": {"name": "synthetic", "native_size": 8, "num_classes": 3,
             "train_size": 8, "selection_size": 3, "calibration_size": 3, "audit_size": 2,
             "test_size": 4, "split_seed": 41, "download": False},
            "classifier": {"kind": "tiny", "weights": "random", "input_size": 16,
                           "mean": [0.5] * 3, "std": [0.5] * 3, "allow_random": False},
            "transmission": {"patch_size": 2, "m_max": 4},
            "repair": {"architecture": "transformer", "dim": 12, "depth": 1, "heads": 3}}


def received_evidence(b=2, size=8, m=4):
    return {"depth": torch.ones(b, 1, size, size), "llr": torch.ones(b, 3 * m, size, size),
            "bit_prob": torch.full((b, 3 * m, size, size), 0.7),
            "bit_mask": torch.ones(b, 3 * m, size, size),
            "uncertainty": torch.full((b, 3, size, size), 0.3),
            "prefix_lower": torch.full((b, 3, size, size), 0.2),
            "prefix_upper": torch.full((b, 3, size, size), 0.8),
            "prefix_confidence": torch.full((b, 3, size, size), 0.9)}


def test_splits_disjoint_deterministic_persisted(tmp_path):
    datasets = build_datasets(config(), tmp_path)
    ids = [set(dataset.sample_ids) for dataset in datasets.values()]
    assert sum(map(len, ids)) == len(set.union(*ids))
    for split, dataset in datasets.items():
        rebuilt = build_datasets(config(), tmp_path)[split]
        assert dataset.sample_ids == rebuilt.sample_ids
        assert torch.equal(dataset[0][0], rebuilt[0][0])
        image, label, sample_id = dataset[0]
        assert image.shape == (3, 8, 8) and image.dtype == torch.float32
        assert image.min() >= 0 and image.max() <= 1
        assert 0 <= label < 3 and sample_id in dataset.sample_ids
    manifest = json.loads((tmp_path / "splits.json").read_text())
    assert manifest["fingerprint"] == datasets["train"].manifest_fingerprint


def test_existing_split_manifest_cannot_silently_change(tmp_path):
    cfg = config()
    build_datasets(cfg, tmp_path)
    cfg["data"]["split_seed"] += 1
    with pytest.raises(ValueError, match="manifest differs"):
        build_datasets(cfg, tmp_path)


def test_real_dataset_native_domain_is_enforced_before_loading(tmp_path):
    cfg = config()
    cfg["data"]["name"] = "cifar100"
    with pytest.raises(ValueError, match="native_size=32"):
        build_datasets(cfg, tmp_path)


def test_frozen_classifier_keeps_input_gradient_and_deterministic_weights():
    cfg = config()
    first = build_classifier(cfg)
    torch.manual_seed(1001)
    second = build_classifier(cfg)
    assert first.fingerprint == second.fingerprint
    native = torch.rand(2, 3, 8, 8, requires_grad=True)
    logits = first(native)
    logits.square().sum().backward()
    assert native.grad is not None and native.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in first.parameters())
    first.train()
    assert not first.model.training
    scores = first.importance(native, 2)
    assert scores.shape == (2, 16)
    np.testing.assert_allclose(scores.sum(1), 1, atol=1e-6)
    assert np.isfinite(scores).all()


def test_random_real_classifier_rejected():
    cfg = config()
    cfg["data"]["name"] = "cifar100"
    with pytest.raises(ValueError, match="Random target classifiers"):
        build_classifier(cfg)
    cfg["classifier"]["allow_random"] = True
    assert build_classifier(cfg).is_random


def test_local_checkpoint_changes_classifier_fingerprint(tmp_path):
    cfg = config()
    baseline = build_classifier(cfg)
    state = copy.deepcopy(baseline.model.state_dict())
    state["head.bias"] += 1
    path = tmp_path / "trained.pt"
    torch.save({"state_dict": state}, path)
    cfg["classifier"]["checkpoint"] = str(path)
    loaded = build_classifier(cfg)
    assert not loaded.is_random
    assert loaded.fingerprint != baseline.fingerprint
    assert loaded.checkpoint_metadata["metadata_status"] == "external_unknown"


def test_attention_mapping_preserves_mass_and_area_overlap():
    # The left half has all mass; each of its 4 destination cells receives 1/4.
    source = torch.tensor([[[1.0, 0.0]]])
    mapped = map_attention_to_grid(source, (2, 4))
    torch.testing.assert_close(mapped, torch.tensor([[0.25, 0.25, 0., 0., 0.25, 0.25, 0., 0.]]))
    rectangular = map_attention_to_grid(torch.ones(2, 14, 14), (8, 8))
    torch.testing.assert_close(rectangular, torch.full((2, 64), 1 / 64))
    zeros = map_attention_to_grid(torch.zeros(1, 2, 2), (3, 3))
    torch.testing.assert_close(zeros.sum(1), torch.ones(1))


@pytest.mark.parametrize("architecture", ["cnn", "transformer"])
def test_repair_capacity_initialization_identity_and_input_gradient(architecture):
    cfg = config()
    cfg["repair"]["architecture"] = architecture
    image = torch.rand(2, 3, 8, 8, requires_grad=True)
    evidence = received_evidence()
    counts, states = [], []
    for variant in VARIANTS:
        torch.manual_seed(99)
        model = build_repair(cfg, variant)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        states.append(model.state_dict())
        output = model(image, evidence)
        torch.testing.assert_close(output, image)
        output.sum().backward()
        assert model.head.weight.grad.abs().sum() > 0
    assert len(set(counts)) == 1
    for name, value in states[0].items():
        for state in states[1:]:
            torch.testing.assert_close(value, state[name])


def test_absent_bits_do_not_encode_false_zero_certainty_and_truth_is_ignored():
    cfg = config()
    model = build_repair(cfg, "bitwise")
    image, evidence = torch.rand(2, 3, 8, 8), received_evidence()
    evidence["bit_mask"].zero_()
    baseline = model.conditioning(image, evidence)
    evidence["llr"].fill_(-1000)
    evidence["bit_prob"].fill_(0)
    evidence["diagnostics"] = {"true_error_mask": torch.ones_like(image)}
    torch.testing.assert_close(model.conditioning(image, evidence), baseline)
    assert baseline[:, 7:7 + 9 * 4].abs().sum() == 0


def test_condition_variants_do_change_evidence_not_network_size():
    cfg, image, evidence = config(), torch.rand(2, 3, 8, 8), received_evidence()
    image_condition = build_repair(cfg, "image").conditioning(image, evidence)
    assert image_condition.abs().sum() == 0
    bit_condition = build_repair(cfg, "bitwise").conditioning(image, evidence)
    prefix_condition = build_repair(cfg, "prefix").conditioning(image, evidence)
    assert bit_condition.shape == prefix_condition.shape
    assert not torch.equal(bit_condition, prefix_condition)


def test_prefix_penalty_only_constrains_trusted_intervals_and_exposes_risk():
    evidence = received_evidence()
    output = torch.full((2, 3, 8, 8), 0.5, requires_grad=True)
    assert prefix_penalty(output, evidence) == 0
    # An incorrectly trusted interval pulls even a correct estimate downward:
    # the test deliberately documents why this is not a no-harm guarantee.
    evidence["prefix_upper"].fill_(0.3)
    penalty = prefix_penalty(output, evidence)
    assert penalty > 0
    penalty.backward()
    assert output.grad.mean() > 0
    evidence["prefix_confidence"].zero_()
    assert prefix_penalty(output, evidence) == 0


def test_prefix_rejects_inverted_interval():
    evidence = received_evidence()
    evidence["prefix_lower"].fill_(0.9)
    with pytest.raises(ValueError, match="Invalid prefix interval"):
        prefix_penalty(torch.rand(2, 3, 8, 8), evidence)
