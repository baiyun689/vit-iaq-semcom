import copy
import json
from pathlib import Path

import numpy as np

import pytest

from vit_iaq_semcom.framework.config import DEFAULTS, ROOT, load_config, validate_config
from vit_iaq_semcom.framework.metrics import summarize


def test_examples_validate():
    for path in (ROOT / "configs/experiments").glob("*.yaml"):
        cfg=load_config(path)
        assert cfg["schema_version"] == 1


@pytest.mark.parametrize("section,key,value", [
    ("data","native_size",31), ("transmission","header_repetition",2),
    ("transmission","prefix_threshold",.2), ("repair","variants",["magic"]),
    ("experiment","training_seeds",[0,0]), ("training","grad_accum",0),
    ("training","lr",float("nan")),
])
def test_invalid_scientific_configs_rejected(section,key,value):
    cfg=copy.deepcopy(DEFAULTS);cfg[section][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_unknown_and_cyclic_configs_rejected(tmp_path):
    path=tmp_path/"bad.yaml";path.write_text("training:\n  epchos: 1\n")
    with pytest.raises(ValueError,match="Unknown"):load_config(path)
    path.write_text("extends: bad.yaml\n")
    with pytest.raises(ValueError,match="cycle"):load_config(path)


def test_summary_clusters_repeats_and_checks_duplicate_ids(tmp_path,monkeypatch):
    monkeypatch.setattr("vit_iaq_semcom.framework.metrics._plots",lambda *a:None)
    (tmp_path/"run_state.json").write_text(json.dumps({"status":"completed"}))
    (tmp_path/"run_manifest.json").write_text(json.dumps({"config":DEFAULTS}))
    (tmp_path/"plan.json").write_text(json.dumps({"jobs":[{"directory":"jobs/test"}],"expected_rows":6}))
    rows=[]
    for image_id in ("a","b"):
        for channel_seed in (0,1,2):
            rows.append(dict(method="uniform",variant="image",budget=100,snr=6,train_seed=0,
                channel_seed=channel_seed,sample_id=image_id,label=0,before=1,after=0 if image_id=="a" else 1,
                clean=0,correct=image_id=="a",before_correct=False,mse=.01,before_mse=.02,psnr=20,total_bits=96))
    path=tmp_path/"jobs/test/predictions.jsonl";path.parent.mkdir(parents=True);path.write_text("\n".join(json.dumps(r) for r in rows))
    result=summarize(tmp_path,200);group=result["groups"][0]
    assert group["images"]==2 and group["observations"]==6
    assert group["accuracy"]==group["gain"]==.5
    assert group["training_accuracy_sd"] is None
    assert result["complete"] and result["purpose"]=="workflow_check"
    path.write_text(path.read_text()+"\n"+json.dumps(rows[0]))
    with pytest.raises(ValueError,match="Duplicate"):summarize(tmp_path,200)


def test_summary_rejects_parent_or_nested_run(tmp_path):
    with pytest.raises(ValueError,match="one experiment"):summarize(tmp_path)
    (tmp_path/"run_manifest.json").write_text("{}")
    (tmp_path/"plan.json").write_text('{"jobs": []}')
    nested=tmp_path/"nested";nested.mkdir();(nested/"run_manifest.json").write_text("{}")
    with pytest.raises(ValueError,match="Nested"):summarize(tmp_path)


def test_summary_streams_repeated_observations_and_preserves_seed_statistics(tmp_path, monkeypatch):
    """Many repeated rows must be iterated, then averaged by image and seed."""
    monkeypatch.setattr("vit_iaq_semcom.framework.metrics._plots", lambda *a: None)
    image_ids = [f"image-{i:03d}" for i in range(32)]
    repeats = 64
    jobs = [dict(directory=f"jobs/seed-{seed}", method="iaq", variant="prefix", budget=100, train_seed=seed)
            for seed in (0, 1)]
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config": DEFAULTS}))
    (tmp_path / "run_state.json").write_text(json.dumps({"status": "completed"}))
    (tmp_path / "splits.json").write_text(json.dumps({"sample_ids": {"audit": image_ids}}))
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": jobs, "evaluation_split": "audit",
                                                       "expected_rows": 2 * len(image_ids) * repeats}))
    for job in jobs:
        path = tmp_path / job["directory"] / "predictions.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w") as stream:
            for index, image_id in enumerate(image_ids):
                for channel_seed in range(repeats):
                    before_correct = index % 2 == 1
                    correct = index % 2 == 0 or job["train_seed"] == 1
                    row = dict(method="iaq", variant="prefix", budget=100, snr=6,
                               train_seed=job["train_seed"], channel_seed=channel_seed, sample_id=image_id,
                               label=0, before=0 if before_correct else 1, after=0 if correct else 1,
                               clean=0, correct=correct, before_correct=before_correct,
                               mse=.01 if channel_seed % 2 else None, before_mse=.02, psnr=20, ssim=.8,
                               total_bits=96, diagnostics={"header_failures": job["train_seed"],
                                                          "transmitted_depth": [4] * 128})
                    stream.write(json.dumps(row) + "\n")
    original_read_text = Path.read_text
    def reject_prediction_bulk_read(path, *args, **kwargs):
        if path.name == "predictions.jsonl":
            raise AssertionError("Prediction files must be consumed line by line")
        return original_read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", reject_prediction_bulk_read)
    result = summarize(tmp_path, 200)
    group = result["groups"][0]
    assert result["observations"] == 4096
    assert group["images"] == 32 and group["evaluation_seed_repeats"] == 2
    assert group["accuracy"] == .75 and group["gain"] == .25
    assert group["training_accuracy_sd"] == pytest.approx(np.std([.5, 1], ddof=1))
    assert group["rescued_rate"] == .5 and group["harmful_repair_rate"] == .25
    assert group["mse"] == pytest.approx(.01) and group["ssim"] == pytest.approx(.8)
    assert group["diagnostics_mean"] == {"header_failures": .5}
    assert group["total_bits_max"] == group["total_bits_mean"] == 96


def test_summary_rejects_same_job_claimed_by_two_files(tmp_path, monkeypatch):
    monkeypatch.setattr("vit_iaq_semcom.framework.metrics._plots", lambda *a: None)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config": DEFAULTS}))
    jobs = [{"directory": "jobs/one"}, {"directory": "jobs/two"}]
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": jobs, "expected_rows": 2}))
    for number, job in enumerate(jobs):
        path = tmp_path / job["directory"] / "predictions.jsonl"
        path.parent.mkdir(parents=True)
        # Even disjoint sample IDs cannot split one logical job across files.
        row = dict(method="uniform", variant="none", budget=100, snr=6, train_seed=0,
                   channel_seed=0, sample_id=f"sample-{number}", label=0, before=0, after=0,
                   clean=0, mse=.01, psnr=20, total_bits=96)
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Duplicate job identity across"):
        summarize(tmp_path, 10)


def test_summary_rejects_row_job_mismatch_and_unknown_split_id(tmp_path, monkeypatch):
    monkeypatch.setattr("vit_iaq_semcom.framework.metrics._plots", lambda *a: None)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config": DEFAULTS}))
    job = dict(directory="jobs/one", method="uniform", variant="none", budget=100, train_seed=0)
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": [job], "expected_rows": 1}))
    path = tmp_path / job["directory"] / "predictions.jsonl"
    path.parent.mkdir(parents=True)
    row = dict(method="uniform", variant="none", budget=100, snr=6, train_seed=1,
               channel_seed=0, sample_id="x", label=0, before=0, after=0, clean=0, total_bits=96)
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="job identity differs"):
        summarize(tmp_path, 10)
    row["train_seed"] = 0
    path.write_text(json.dumps(row))
    (tmp_path / "splits.json").write_text(json.dumps({"sample_ids": {"audit": ["y"]}}))
    with pytest.raises(ValueError, match="outside the persisted"):
        summarize(tmp_path, 10)
