"""Lifecycle checks for real transport/model integration, without downloads."""
import copy
import json
from pathlib import Path

import pytest
import torch
import numpy as np

from vit_iaq_semcom.framework import runner
from vit_iaq_semcom.framework.config import DEFAULTS


@pytest.fixture
def cfg(tmp_path):
    value = copy.deepcopy(DEFAULTS)
    value["output_dir"] = str(tmp_path / "run")
    value["data"].update(native_size=8, num_classes=3, train_size=3, selection_size=2,
                         calibration_size=2, audit_size=3, test_size=2)
    value["classifier"].update(input_size=8)
    value["transmission"].update(patch_size=4, m_max=3)
    value["experiment"].update(methods=["uniform"], budgets=[384], snrs=[0, 6],
                               channel_seeds=[0], training_seeds=[11], bootstrap_samples=10)
    value["repair"].update(variants=["none"], dim=8, heads=2, architecture="cnn")
    value["training"].update(batch_size=2, epochs=1, grad_accum=2)
    return value


def rows(root):
    return [json.loads(line) for path in Path(root).rglob("predictions.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]


def test_plan_is_read_only_without_loading_models_or_data(cfg, monkeypatch):
    monkeypatch.setattr(runner, "_dependencies", lambda: pytest.fail("plan loaded runtime"))
    cfg["repair"]["variants"] = ["none", "image", "prefix"]
    result = runner.plan(cfg)
    assert result["training_jobs"] == 2
    assert result["expected_rows"] == 18
    assert result["purpose"] == "workflow_check"
    assert not Path(cfg["output_dir"]).exists()


def test_pixel_only_loss_ablation_does_not_use_task_cross_entropy(cfg, monkeypatch):
    cfg["repair"]["variants"] = ["image"]
    cfg["training"].update(lambda_task=0.0, lambda_mse=1.0, lambda_prefix=0.0)
    monkeypatch.setattr(runner.F, "cross_entropy", lambda *a, **k: pytest.fail("pixel-only ablation used task loss"))
    result = runner.run(cfg)
    history_path = Path(cfg["output_dir"]) / result["jobs"][0]["directory"] / "history.json"
    epoch = json.loads(history_path.read_text(encoding="utf8"))[0]
    assert epoch["task"] == 0 and epoch["mse"] > 0
    assert epoch["loss"] == pytest.approx(epoch["mse"])


def test_completed_run_skips_transmission_and_rejects_drift(cfg, monkeypatch):
    first = runner.run(cfg)
    original_rows = rows(cfg["output_dir"])
    assert first["rows"] == 6
    assert all(r["workflow_check"] and r["total_bits"] <= 384 for r in original_rows)
    assert all(r["before"] == r["after"] and r["before_mse"] == r["mse"] for r in original_rows)
    monkeypatch.setattr(runner, "_damage", lambda *args, **kwargs: pytest.fail("completed job recomputed"))
    result = runner.run(cfg)
    assert all(job["skipped_completed"] for job in result["jobs"])
    assert rows(cfg["output_dir"]) == original_rows
    with pytest.raises(ValueError, match="already exists"):
        runner.run(cfg, resume=False)
    cfg["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        runner.run(cfg)


def test_evaluation_interruption_keeps_complete_batches_and_no_duplicate_rows(cfg, monkeypatch):
    damage, calls = runner._damage, []
    def interrupt(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return damage(*args, **kwargs)
    monkeypatch.setattr(runner, "_damage", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run(cfg)
    partial = rows(cfg["output_dir"])
    assert len(partial) == 2
    assert json.loads((Path(cfg["output_dir"]) / "run_state.json").read_text())["status"] == "failed"
    resumed_ids = []
    def record(*args, **kwargs):
        resumed_ids.extend((sample_id, float(args[5])) for sample_id in args[1])
        return damage(*args, **kwargs)
    monkeypatch.setattr(runner, "_damage", record)
    result = runner.run(cfg)
    finished = rows(cfg["output_dir"])
    assert len(finished) == len({(r["sample_id"], r["snr"], r["channel_seed"]) for r in finished}) == 6
    assert finished[:2] == partial
    assert result["jobs"][0]["resumed_rows"] == 2
    assert len(resumed_ids) == 4


def test_variants_share_corruption_and_train_gradient_reaches_repair(cfg):
    cfg["repair"]["variants"] = ["none", "image", "prefix"]
    result = runner.run(cfg)
    assert result["rows"] == 18
    grouped = {}
    for row in rows(cfg["output_dir"]):
        grouped.setdefault((row["sample_id"], row["snr"]), []).append(row)
    for group in grouped.values():
        assert len({r["before"] for r in group}) == 1
        assert len({r["before_mse"] for r in group}) == 1
        assert len({json.dumps(r["counts"], sort_keys=True) for r in group}) == 1
    for path in Path(cfg["output_dir"]).rglob("latest.pt"):
        state = torch.load(path, weights_only=False)
        assert state["completed_epochs"] == 1
        assert state["history"][0]["optimizer_steps"] == 1  # three examples / two microbatches
        assert state["model"]["head.weight"].abs().sum() > 0
        assert state["optimizer"]["state"]
        assert state["rng"]["torch"].numel() > 0


def test_epoch_resume_matches_uninterrupted_training(cfg, monkeypatch, tmp_path):
    cfg["repair"]["variants"] = ["image"]
    cfg["training"]["epochs"] = 2
    selection, calls = runner._selection, 0
    def interrupted_selection(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted epoch two")
        return selection(*args, **kwargs)
    monkeypatch.setattr(runner, "_selection", interrupted_selection)
    with pytest.raises(RuntimeError, match="epoch two"):
        runner.run(cfg)
    checkpoint = next(Path(cfg["output_dir"]).rglob("latest.pt"))
    assert torch.load(checkpoint, weights_only=False)["completed_epochs"] == 1
    monkeypatch.setattr(runner, "_selection", selection)
    runner.run(cfg)
    resumed = torch.load(checkpoint, weights_only=False)
    fresh_cfg = copy.deepcopy(cfg)
    fresh_cfg["output_dir"] = str(tmp_path / "uninterrupted")
    runner.run(fresh_cfg)
    fresh = torch.load(next(Path(fresh_cfg["output_dir"]).rglob("latest.pt")), weights_only=False)
    assert resumed["completed_epochs"] == fresh["completed_epochs"] == 2
    for key in fresh["model"]:
        assert torch.equal(resumed["model"][key], fresh["model"][key]), key
    for key in ("loss", "task", "mse", "prefix", "identity", "selection"):
        assert [h[key] for h in resumed["history"]] == [h[key] for h in fresh["history"]]


def test_calibration_only_reads_held_out_calibration(cfg, monkeypatch):
    damage, ids_seen = runner._damage, []
    def capture(*args, **kwargs):
        assert args[6] == "calibration"
        ids_seen.extend(args[1])
        return damage(*args, **kwargs)
    monkeypatch.setattr(runner, "_damage", capture)
    result = runner.calibrate(cfg)
    manifest = json.loads((Path(cfg["output_dir"]) / "splits.json").read_text())
    assert set(ids_seen) == set(manifest["sample_ids"]["calibration"])
    assert not set(ids_seen) & set(manifest["sample_ids"]["test"])
    assert result["split"] == "calibration" and result["config_unchanged"]
    assert not list(Path(cfg["output_dir"]).rglob("predictions.jsonl"))


def test_corrupted_batch_journal_cannot_be_silently_reused(cfg):
    runner.run(cfg)
    path = next(Path(cfg["output_dir"]).rglob("evaluation_batches/*.json"))
    batch = json.loads(path.read_text())
    batch["rows"][0]["label"] += 1
    path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ValueError, match="contents changed"):
        runner.run(cfg)


def test_short_final_accumulation_group_steps_and_checkpoint_integrity(cfg):
    cfg["data"]["train_size"] = 5
    cfg["repair"]["variants"] = ["image"]
    runner.run(cfg)
    latest = next(Path(cfg["output_dir"]).rglob("latest.pt"))
    checkpoint = torch.load(latest, weights_only=False)
    # Batches of 2,2,1: the short final group still takes a full optimizer step.
    assert checkpoint["history"][0]["optimizer_steps"] == 2
    best = latest.with_name("best.pt")
    best.write_bytes(best.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.run(cfg)


def test_ssim_identity_and_constant_shift():
    image = np.full((3, 8, 8), 0.3, dtype=np.float32)
    assert runner._ssim(image, image) == pytest.approx(1.0)
    assert 0 < runner._ssim(image, image + 0.3) < 0.9


def test_physical_channel_seed_shared_across_protocols_and_budgets(cfg):
    _, build_classifier, _, _, transmit = runner._dependencies()
    classifier = build_classifier(cfg)
    images = torch.full((1, 3, 8, 8), 0.35)
    observed = []
    def capture(image, scores, method, budget, snr, seed, config):
        observed.append(seed)
        return transmit(image, scores, method, budget, snr, seed, config)
    cfg["transmission"]["channel"] = "rayleigh"
    for method in ("uniform", "iaq", "addressed"):
        for budget in (384, 576):
            runner._damage(images, ["shared-image"], classifier, cfg, {"method": method, "budget": budget},
                           6, "evaluation", 17, capture)
    assert len(observed) == 6 and len(set(observed)) == 1
    runner._damage(images, ["shared-image"], classifier, cfg, {"method": "uniform", "budget": 384},
                   6, "evaluation", 18, capture)
    assert observed[-1] != observed[0]


def test_calibration_regenerates_prefix_for_each_candidate(cfg, monkeypatch):
    cfg["transmission"]["prefix_threshold"] = 0.95
    cfg["experiment"]["snrs"] = [0]
    seen = []
    def candidate_damage(images, ids, classifier, candidate, job, snr, phase, repeat_seed, transmit):
        threshold = candidate["transmission"]["prefix_threshold"]
        seen.append((threshold, tuple(ids), phase, repeat_seed))
        # Lower confidence threshold admits a longer prefix, not merely more
        # copies of the shorter interval generated at a higher threshold.
        width = 0.25 if threshold == 0.9 else 0.5
        evidence = {"prefix_lower": torch.zeros_like(images), "prefix_upper": torch.full_like(images, width),
                    "prefix_confidence": torch.full_like(images, (threshold + 1) / 2),
                    "bit_mask": torch.ones(images.shape[0], 3 * candidate["transmission"]["m_max"], 8, 8),
                    "bit_prob": torch.full((images.shape[0], 3 * candidate["transmission"]["m_max"], 8, 8), 0.5)}
        return images, evidence, []
    monkeypatch.setattr(runner, "_damage", candidate_damage)
    result = runner.calibrate(cfg)
    candidates = result["results"][0]["thresholds"]
    assert [row["threshold"] for row in candidates] == [0.9, 0.95, 0.99]
    assert [row["mean_accepted_interval_width"] for row in candidates] == [0.25, 0.5, 0.5]
    assert len({record[1:] for record in seen}) == 1  # same images and physical seed inputs
    assert cfg["transmission"]["prefix_threshold"] == 0.95


def test_calibration_candidate_transmissions_reuse_physical_noise(cfg, monkeypatch):
    dependencies = runner._dependencies()
    transmit, seen = dependencies[-1], []
    def capture(image, scores, method, budget, snr, seed, config):
        result = transmit(image, scores, method, budget, snr, seed, config)
        seen.append((config["prefix_threshold"], seed, result["bit_prob"].copy()))
        return result
    monkeypatch.setattr(runner, "_dependencies", lambda: (*dependencies[:-1], capture))
    cfg["experiment"]["snrs"] = [0]
    runner.calibrate(cfg)
    assert {entry[0] for entry in seen} == {0.9, 0.99}
    by_seed = {}
    for threshold, seed, probability in seen:
        by_seed.setdefault(seed, []).append(probability)
    assert len(by_seed) == cfg["data"]["calibration_size"]
    for probabilities in by_seed.values():
        assert len(probabilities) == 2
        np.testing.assert_array_equal(*probabilities)


def test_frozen_evaluation_uses_test_and_new_snrs_without_training(cfg, monkeypatch, tmp_path):
    cfg["repair"]["variants"] = ["none", "image"]
    cfg["experiment"]["snrs"] = [0]
    runner.run(cfg)
    source = Path(cfg["output_dir"])
    source_best = next(source.rglob("best.pt"))
    original_checkpoint_hash = runner._file_digest(source_best)
    evaluation = copy.deepcopy(cfg)
    evaluation["name"] = "frozen-test"
    evaluation["output_dir"] = str(tmp_path / "frozen-test")
    evaluation["experiment"].update(split="test", snrs=[-2, 14], channel_seeds=[8], bootstrap_samples=20)
    evaluation["repair"]["variants"] = ["image"]
    monkeypatch.setattr(runner, "_train", lambda *args, **kwargs: pytest.fail("frozen evaluation trained a model"))
    monkeypatch.setattr(runner, "_selection", lambda *args, **kwargs: pytest.fail("frozen evaluation selected a model"))
    result = runner.evaluate(evaluation, source)
    assert result["evaluate_only"] and result["rows"] == 4
    saved = rows(evaluation["output_dir"])
    split_manifest = json.loads((source / "splits.json").read_text())
    assert {row["sample_id"] for row in saved} == set(split_manifest["sample_ids"]["test"])
    assert {row["snr"] for row in saved} == {-2, 14}
    assert all(row["split"] == "test" for row in saved)
    assert not list(Path(evaluation["output_dir"]).rglob("*.pt"))
    assert runner._file_digest(source_best) == original_checkpoint_hash
    manifest = json.loads((Path(evaluation["output_dir"]) / "run_manifest.json").read_text())
    assert manifest["evaluate_only"] and manifest["source_run"]["directory"] == str(source.resolve())
    assert json.loads((Path(evaluation["output_dir"]) / "plan.json").read_text())["training_jobs"] == 0
    monkeypatch.setattr(runner, "_damage", lambda *args, **kwargs: pytest.fail("completed frozen evaluation recomputed"))
    resumed = runner.evaluate(evaluation, source, resume=True)
    assert resumed["jobs"][0]["skipped_completed"]
    assert rows(evaluation["output_dir"]) == saved
    with pytest.raises(ValueError, match="evaluation-only directory"):
        runner.run(evaluation)


def test_frozen_evaluation_rejects_incompatible_configuration_and_checkpoint(cfg, tmp_path):
    cfg["repair"]["variants"] = ["image"]
    cfg["experiment"]["snrs"] = [0]
    runner.run(cfg)
    source = Path(cfg["output_dir"])
    evaluation = copy.deepcopy(cfg)
    evaluation["output_dir"] = str(tmp_path / "frozen-mismatch")
    evaluation["experiment"]["split"] = "test"
    changed = copy.deepcopy(evaluation)
    changed["repair"]["dim"] = 16
    with pytest.raises(ValueError, match="changes training, data"):
        runner.evaluate(changed, source)
    changed = copy.deepcopy(evaluation)
    changed["experiment"]["budgets"] = [576]
    with pytest.raises(ValueError, match="must be a subset"):
        runner.evaluate(changed, source)
    assert not Path(evaluation["output_dir"]).exists()
    best = next(source.rglob("best.pt"))
    best.write_bytes(best.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="missing or changed"):
        runner.evaluate(evaluation, source)


def test_frozen_evaluation_recovers_partial_batches(cfg, monkeypatch, tmp_path):
    cfg["experiment"]["snrs"] = [0]
    cfg["training"]["batch_size"] = 1
    runner.run(cfg)
    evaluation = copy.deepcopy(cfg)
    evaluation["output_dir"] = str(tmp_path / "frozen-resume")
    evaluation["experiment"]["split"] = "test"
    damage, calls = runner._damage, 0
    def fail_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("frozen interruption")
        return damage(*args, **kwargs)
    monkeypatch.setattr(runner, "_damage", fail_second_batch)
    with pytest.raises(RuntimeError, match="frozen interruption"):
        runner.evaluate(evaluation, cfg["output_dir"])
    assert len(rows(evaluation["output_dir"])) == 1
    monkeypatch.setattr(runner, "_damage", damage)
    result = runner.evaluate(evaluation, cfg["output_dir"], resume=True)
    assert result["rows"] == 2 and result["jobs"][0]["resumed_rows"] == 1


def test_local_classifier_exposure_retained_and_same_partition_allowed(cfg, tmp_path):
    from vit_iaq_semcom.framework.classifier_training import fit_classifier
    from vit_iaq_semcom.framework.classifier import build_classifier
    fitting = copy.deepcopy(cfg)
    fitting["output_dir"] = str(tmp_path / "classifier-fit")
    path = tmp_path / "classifier.pt"
    fit_classifier(fitting, path)
    cfg["classifier"]["checkpoint"] = str(path)
    loaded = build_classifier(cfg)
    artifact = torch.load(path, weights_only=True)
    assert loaded.checkpoint_metadata["training_exposure"] == artifact["training_exposure"]
    runner.run(cfg)
    manifest = json.loads((Path(cfg["output_dir"]) / "run_manifest.json").read_text())
    audit = manifest["classifier_exposure_audit"]
    assert audit["checked_known_local_exposure"]
    assert audit["overlap_counts"]["train"] == cfg["data"]["train_size"]
    assert audit["overlap_counts"]["selection"] == cfg["data"]["selection_size"]
    assert all(audit["overlap_counts"][split] == 0 for split in ("calibration", "audit", "test"))


def test_runtime_environment_recorded_and_change_logged_without_rejecting_resume(cfg, monkeypatch):
    first = runner.run(cfg)
    path = Path(cfg["output_dir"]) / "run_manifest.json"
    manifest = json.loads(path.read_text())
    environment = manifest["runtime_environment"]
    assert environment["python"] and environment["platform"] and environment["device_name"]
    assert environment["selected_device"] == "cpu"
    assert all(environment["packages"][key] for key in ("torch", "torchvision", "numpy", "scipy"))
    changed = copy.deepcopy(environment)
    changed["device_name"] = "another host device"
    monkeypatch.setattr(runner, "_runtime_environment", lambda device: changed)
    resumed = runner.run(cfg)
    assert resumed["signature"] == first["signature"]
    history = json.loads(path.read_text())["runtime_environment_history"]
    assert len(history) == 2 and history[-1]["environment"] == changed


@pytest.mark.parametrize("operation", ["run", "calibrate"])
def test_changed_partition_rejects_actual_classifier_exposure(cfg, tmp_path, monkeypatch, operation):
    from vit_iaq_semcom.framework.classifier_training import fit_classifier
    from vit_iaq_semcom.framework.data import build_datasets
    fitting = copy.deepcopy(cfg)
    fitting["output_dir"] = str(tmp_path / "classifier-fit")
    path = tmp_path / "classifier.pt"
    fit_classifier(fitting, path)
    artifact = torch.load(path, weights_only=True)
    exposed = set(artifact["training_exposure"]["train_ids"] + artifact["training_exposure"]["selection_ids"])
    cfg["classifier"]["checkpoint"] = str(path)
    # Pick a different seeded partition that demonstrably moves a locally used
    # image into calibration. This exercises real exported-weight loading and
    # the run context, rather than calling the audit helper with fake metadata.
    for index in range(1, 30):
        candidate = copy.deepcopy(cfg)
        candidate["data"]["split_seed"] += index
        datasets = build_datasets(candidate, tmp_path / f"partition-probe-{index}")
        if exposed & set(datasets["calibration"].sample_ids):
            cfg = candidate
            break
    else:
        pytest.fail("No overlapping deterministic partition found")
    monkeypatch.setattr(runner, "_damage", lambda *args, **kwargs: pytest.fail("leaking experiment reached transmission"))
    with pytest.raises(ValueError, match="exposure overlaps reserved split calibration"):
        getattr(runner, operation)(cfg)
    assert not (Path(cfg["output_dir"]) / "run_manifest.json").exists()


def test_external_classifier_exposure_remains_unknown_and_metadata_is_fingerprinted(cfg, tmp_path):
    from vit_iaq_semcom.framework.classifier import build_classifier
    from vit_iaq_semcom.framework.classifier_training import fit_classifier
    initial = build_classifier(cfg)
    external = tmp_path / "external.pt"
    torch.save({"state_dict": initial.model.state_dict()}, external)
    cfg["classifier"]["checkpoint"] = str(external)
    runner.run(cfg)
    manifest = json.loads((Path(cfg["output_dir"]) / "run_manifest.json").read_text())
    assert manifest["classifier_exposure_audit"]["status"] == "external_unknown"
    assert not manifest["classifier_exposure_audit"]["checked_known_local_exposure"]
    fitting = copy.deepcopy(cfg)
    fitting["output_dir"] = str(tmp_path / "classifier-fit")
    own = tmp_path / "own.pt"
    fit_classifier(fitting, own)
    cfg["classifier"]["checkpoint"] = str(own)
    first = build_classifier(cfg).fingerprint
    artifact = torch.load(own, weights_only=True)
    artifact["training_exposure"]["initial_pretraining_exposure"] = "changed exposure declaration"
    torch.save(artifact, own)
    assert build_classifier(cfg).fingerprint != first
    artifact.pop("training_exposure")
    torch.save(artifact, own)
    with pytest.raises(ValueError, match="must retain valid training_exposure"):
        build_classifier(cfg)


def test_chained_classifier_fit_retains_ancestor_exposure_and_blocks_leak(cfg, tmp_path):
    from vit_iaq_semcom.framework.classifier_training import fit_classifier
    from vit_iaq_semcom.framework.classifier import build_classifier
    from vit_iaq_semcom.framework.data import build_datasets
    first_cfg = copy.deepcopy(cfg)
    first_cfg["output_dir"] = str(tmp_path / "first-fit")
    first_path = tmp_path / "first-classifier.pt"
    fit_classifier(first_cfg, first_path)
    first = torch.load(first_path, weights_only=True)
    first_ids = set(first["training_exposure"]["train_ids"] + first["training_exposure"]["selection_ids"])
    second_cfg = copy.deepcopy(cfg)
    second_cfg["output_dir"] = str(tmp_path / "second-fit")
    second_cfg["classifier"]["checkpoint"] = str(first_path)
    for index in range(1, 30):
        second_cfg["data"]["split_seed"] = cfg["data"]["split_seed"] + index
        data = build_datasets(second_cfg, tmp_path / f"chain-probe-{index}")
        if first_ids & set(data["calibration"].sample_ids):
            break
    else:
        pytest.fail("No ancestor-overlapping partition found")
    second_path = tmp_path / "second-classifier.pt"
    fit_classifier(second_cfg, second_path)
    exported = torch.load(second_path, weights_only=True)["training_exposure"]
    assert len(exported["ancestor_exposure"]) == 1
    ancestor = exported["ancestor_exposure"][0]
    assert set(ancestor["train_ids"] + ancestor["selection_ids"]) == first_ids
    assert ancestor["source"] == str(first_path)
    assert "external pretraining" in exported["initial_pretraining_exposure"]
    # The current fit alone is disjoint from calibration; rejection must use
    # historical exposure, even after the original weight file is unavailable.
    assert not set(exported["train_ids"] + exported["selection_ids"]) & set(data["calibration"].sample_ids)
    first_path.unlink()
    evaluation = copy.deepcopy(second_cfg)
    evaluation["output_dir"] = str(tmp_path / "chained-evaluation")
    evaluation["classifier"]["checkpoint"] = str(second_path)
    assert build_classifier(evaluation).checkpoint_metadata["training_exposure"]["ancestor_exposure"]
    with pytest.raises(ValueError, match="exposure overlaps reserved split calibration"):
        runner.run(evaluation)


def test_frozen_channel_and_noise_shift_require_explicit_opt_in_and_never_train(cfg, tmp_path, monkeypatch):
    cfg["repair"]["variants"] = ["image"]
    cfg["experiment"]["snrs"] = [6]
    runner.run(cfg)
    target = copy.deepcopy(cfg)
    target["output_dir"] = str(tmp_path / "shifted-test")
    target["experiment"]["split"] = "test"
    target["transmission"].update(channel="rayleigh", noise_mismatch_db=3.0)
    with pytest.raises(ValueError, match="changes training, data"):
        runner.evaluate(target, cfg["output_dir"])
    forbidden = copy.deepcopy(target)
    forbidden["transmission"]["header_repetition"] = 5
    with pytest.raises(ValueError, match="changes training, data"):
        runner.evaluate(forbidden, cfg["output_dir"], allow_channel_shift=True)
    monkeypatch.setattr(runner, "_train", lambda *args, **kwargs: pytest.fail("shifted evaluation trained"))
    monkeypatch.setattr(runner, "_selection", lambda *args, **kwargs: pytest.fail("shifted evaluation selected"))
    result = runner.evaluate(target, cfg["output_dir"], allow_channel_shift=True)
    assert result["status"] == "completed" and result["rows"] == 2
    manifest = json.loads((Path(target["output_dir"]) / "run_manifest.json").read_text())
    assert manifest["allow_channel_shift"] is True
    assert manifest["channel_shift"]["source"] == {"channel": "awgn", "noise_mismatch_db": 0.0}
    assert manifest["channel_shift"]["target"] == {"channel": "rayleigh", "noise_mismatch_db": 3.0}
    assert result["source_run"]["channel_shift"] == manifest["channel_shift"]
    resumed = runner.evaluate(target, cfg["output_dir"], resume=True, allow_channel_shift=True)
    assert resumed["jobs"][0]["skipped_completed"]


def test_frozen_resume_cannot_mix_channel_shift_opt_in(cfg, tmp_path):
    cfg["experiment"]["snrs"] = [0]
    runner.run(cfg)
    target = copy.deepcopy(cfg)
    target["output_dir"] = str(tmp_path / "declared-shift")
    target["experiment"]["split"] = "test"
    # Even if the channel values currently happen to match, the recorded
    # lifecycle declaration cannot silently change on a resumed evaluation.
    runner.evaluate(target, cfg["output_dir"], allow_channel_shift=True)
    with pytest.raises(ValueError, match="different source run or lifecycle"):
        runner.evaluate(target, cfg["output_dir"], resume=True, allow_channel_shift=False)
