import copy

import pytest
import torch

from vit_iaq_semcom.framework.classifier import build_classifier
from vit_iaq_semcom.framework import classifier_training
from vit_iaq_semcom.framework.config import DEFAULTS
from vit_iaq_semcom.framework.data import NativeSubset


def configuration(tmp_path, name="fit"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["output_dir"] = str(tmp_path / name)
    cfg["data"].update(native_size=8, num_classes=3, train_size=7, selection_size=3,
                       calibration_size=2, audit_size=2, test_size=2)
    cfg["classifier"]["input_size"] = 8
    cfg["training"].update(epochs=1, batch_size=3, lr=0.01, grad_accum=2)
    return cfg


def test_head_fit_changes_head_only_and_checkpoint_loads(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    initial = build_classifier(cfg)
    allowed_ids = set()
    original_getitem = NativeSubset.__getitem__
    def guarded_getitem(self, index):
        assert self.sample_ids[index] in allowed_ids
        return original_getitem(self, index)
    original_build = classifier_training.build_datasets
    def guarded_build(config, output):
        data = original_build(config, output)
        allowed_ids.update(data["train"].sample_ids + data["selection"].sample_ids)
        return data
    monkeypatch.setattr(classifier_training, "build_datasets", guarded_build)
    monkeypatch.setattr(NativeSubset, "__getitem__", guarded_getitem)
    output = tmp_path / "classifier.pt"
    report = classifier_training.fit_classifier(cfg, output)
    assert report["status"] == "completed" and len(report["history"]) == 1
    artifact = torch.load(output, weights_only=True)
    assert set(artifact["training_exposure"]["train_ids"]).isdisjoint(artifact["training_exposure"]["selection_ids"])
    cfg["classifier"]["checkpoint"] = str(output)
    trained = build_classifier(cfg)
    assert trained.checkpoint_metadata["metadata_status"] == "verified"
    torch.testing.assert_close(trained.model.features[0].weight, initial.model.features[0].weight)
    assert not torch.equal(trained.model.head.weight, initial.model.head.weight)
    assert len(artifact["training_exposure"]["train_ids"]) == 7
    assert artifact["split_fingerprint"]


def test_classifier_resume_preserves_completed_epochs_and_rejects_mismatch(tmp_path):
    cfg = configuration(tmp_path)
    output = tmp_path / "classifier.pt"
    first = classifier_training.fit_classifier(cfg, output)
    saved = torch.load(output, weights_only=True)["state_dict"]
    resumed = classifier_training.fit_classifier(cfg, output, resume=True)
    assert resumed["history"] == first["history"]
    for key, value in torch.load(output, weights_only=True)["state_dict"].items():
        torch.testing.assert_close(value, saved[key])
    with pytest.raises(ValueError, match="directory exists"):
        classifier_training.fit_classifier(cfg, output)
    cfg["training"]["lr"] = 0.2
    with pytest.raises(ValueError, match="configuration, weights or splits differ"):
        classifier_training.fit_classifier(cfg, output, resume=True)


def test_full_model_updates_backbone(tmp_path):
    cfg = configuration(tmp_path)
    initial = build_classifier(cfg)
    output = tmp_path / "classifier.pt"
    classifier_training.fit_classifier(cfg, output, full_model=True)
    state = torch.load(output, weights_only=True)["state_dict"]
    assert not torch.equal(state["features.0.weight"], initial.model.features[0].weight)


def test_classifier_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    uninterrupted = configuration(tmp_path, "full")
    uninterrupted["training"]["epochs"] = 2
    classifier_training.fit_classifier(uninterrupted, tmp_path / "full.pt", full_model=True)
    resumed = configuration(tmp_path, "resumed")
    resumed["training"]["epochs"] = 2
    original_save = classifier_training._atomic_save
    def interrupt_after_epoch(path, value):
        original_save(path, value)
        if str(path).endswith("classifier_latest.pt") and value["epoch"] == 0:
            raise RuntimeError("simulated power loss")
    monkeypatch.setattr(classifier_training, "_atomic_save", interrupt_after_epoch)
    with pytest.raises(RuntimeError, match="simulated power loss"):
        classifier_training.fit_classifier(resumed, tmp_path / "resumed.pt", full_model=True)
    monkeypatch.setattr(classifier_training, "_atomic_save", original_save)
    classifier_training.fit_classifier(resumed, tmp_path / "resumed.pt", full_model=True, resume=True)
    full = torch.load(tmp_path / "full.pt", weights_only=True)
    recovered = torch.load(tmp_path / "resumed.pt", weights_only=True)
    for key in full["state_dict"]:
        torch.testing.assert_close(full["state_dict"][key], recovered["state_dict"][key], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["kind", "arch", "num_classes", "input_size", "mean", "std"])
def test_own_classifier_checkpoint_metadata_mismatch_is_rejected(tmp_path, field):
    cfg = configuration(tmp_path)
    output = tmp_path / "classifier.pt"
    classifier_training.fit_classifier(cfg, output)
    cfg["classifier"]["checkpoint"] = str(output)
    artifact = torch.load(output, weights_only=True)
    if field in ("kind", "arch", "num_classes"):
        artifact[field] = 100 if field == "num_classes" else "different"
    else:
        artifact["preprocessing"][field] = 16 if field == "input_size" else [0.1, 0.2, 0.3]
    torch.save(artifact, output)
    with pytest.raises(ValueError, match="differs from current configuration"):
        build_classifier(cfg)


def test_resume_rejects_modified_completed_export(tmp_path):
    cfg, output = configuration(tmp_path), tmp_path / "classifier.pt"
    classifier_training.fit_classifier(cfg, output)
    artifact = torch.load(output, weights_only=True)
    artifact["state_dict"]["head.bias"] += 1
    torch.save(artifact, output)
    with pytest.raises(ValueError, match="output checkpoint is missing or its checksum differs"):
        classifier_training.fit_classifier(cfg, output, resume=True)


def test_resume_rejects_changed_training_source(tmp_path, monkeypatch):
    cfg, output = configuration(tmp_path), tmp_path / "classifier.pt"
    classifier_training.fit_classifier(cfg, output)
    monkeypatch.setattr(classifier_training, "_source_fingerprint", lambda: {"classifier_training.py": "changed"})
    with pytest.raises(ValueError, match="including source code"):
        classifier_training.fit_classifier(cfg, output, resume=True)


def test_resume_rejects_modified_recovery_weights(tmp_path):
    cfg, output = configuration(tmp_path), tmp_path / "classifier.pt"
    classifier_training.fit_classifier(cfg, output)
    latest_path = tmp_path / "fit" / "classifier_latest.pt"
    latest = torch.load(latest_path, weights_only=True)
    latest["state_dict"]["head.bias"] += 1
    torch.save(latest, latest_path)
    with pytest.raises(ValueError, match="tensor integrity check failed"):
        classifier_training.fit_classifier(cfg, output, resume=True)
