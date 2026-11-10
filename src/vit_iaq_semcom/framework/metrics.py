"""Image-clustered paired statistics and publication-source tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _add_mean(accumulators, name, value):
    if value is None:
        return
    number = float(value)
    if np.isfinite(number):
        aggregate = accumulators.setdefault(name, [0.0, 0])
        aggregate[0] += number
        aggregate[1] += 1


def _mean(accumulators, name):
    total, count = accumulators.get(name, (0.0, 0))
    return total / count if count else None


class _Group:
    """Sufficient statistics; memory scales with images, never observations."""

    def __init__(self, image_index):
        self.image_index = image_index  # Shared across all groups.
        self.images = np.zeros((len(image_index), 3), dtype=np.int64) if image_index is not None else {}
        self.seeds = {}
        self.means = {}
        self.diagnostics = {}
        self.observations = 0
        self.max_bits = 0

    def add(self, row, image_key):
        before = bool(row.get("before_correct", row["before"] == row["label"]))
        after = bool(row.get("correct", row["after"] == row["label"]))
        if self.image_index is None:
            values = self.images.setdefault(image_key, [0, 0, 0])
        else:
            values = self.images[image_key]
        values[0] += int(after)
        values[1] += int(after) - int(before)
        values[2] += 1
        seed = self.seeds.setdefault(row["train_seed"], [0, 0])
        seed[0] += int(after)
        seed[1] += 1
        self.observations += 1
        self.max_bits = max(self.max_bits, row["total_bits"])
        label_before = row["before"] == row["label"]
        label_after = row["after"] == row["label"]
        for key, value in (
            ("clean_accuracy", row["clean"] == row["label"]),
            ("before_accuracy", label_before),
            ("rescued_rate", not label_before and label_after),
            ("harmful_repair_rate", label_before and not label_after),
            ("total_bits_mean", row["total_bits"]),
        ):
            _add_mean(self.means, key, value)
        for key in ("mse", "before_mse", "psnr", "ssim"):
            _add_mean(self.means, key, row.get(key))
        for name, value in row.get("diagnostics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.diagnostics.setdefault(name, [0.0, 0])
                _add_mean(self.diagnostics, name, value)

    def image_means(self):
        if self.image_index is None:
            values = np.asarray([value for _, value in sorted(self.images.items())], dtype=np.float64)
        else:
            values = self.images[self.images[:, 2] > 0]
        return values[:, :2] / values[:, 2:3]


def _image_index(root, manifest, plan):
    """One compact ID lookup from the persisted evaluated split, if present."""
    path = root / "splits.json"
    if not path.is_file():
        return None  # Small hand-authored / legacy fixtures remain supported.
    split = plan.get("evaluation_split", manifest.get("config", {}).get("experiment", {}).get("split"))
    ids = json.loads(path.read_text(encoding="utf8")).get("sample_ids", {}).get(split)
    if ids is None:
        raise ValueError("Split manifest has no sample IDs for the evaluated split")
    if len(set(ids)) != len(ids):
        raise ValueError("Split manifest contains duplicate sample IDs")
    return {sample_id: index for index, sample_id in enumerate(sorted(ids))}


def _interval(values, repeats, rng):
    values = np.asarray(values, dtype=float)
    samples = []
    for start in range(0, repeats, 100):
        draws = rng.integers(len(values), size=(min(100, repeats-start), len(values)))
        samples.extend(values[draws].mean(axis=1).tolist())
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def summarize(run_dir, bootstrap_samples=1000):
    root = Path(run_dir).resolve()
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf8")) if state_path.exists() else {}
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file() or not (root/"plan.json").is_file():
        raise ValueError("Select one experiment root containing run_manifest.json and plan.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if any(path.parent != root for path in root.rglob("run_manifest.json")):
        raise ValueError("Nested experiment manifests found; do not mix separate runs")
    plan = json.loads((root/"plan.json").read_text(encoding="utf8"))
    image_index = _image_index(root, manifest, plan)
    grouped, job_owners, paths_seen = {}, {}, set()
    observations = 0
    job_fields = ("method", "variant", "budget", "train_seed")
    for job in plan["jobs"]:
        path=(root/job["directory"]/"predictions.jsonl").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Job output escapes experiment directory")
        if path in paths_seen:
            raise ValueError(f"Duplicate job output path in plan: {path}")
        paths_seen.add(path)
        if not path.is_file():
            continue
        expected_job = tuple(job[k] for k in job_fields) if all(k in job for k in job_fields) else None
        # Only this job's identities are retained. Distinct files cannot claim
        # the same job, so no global set of millions of observations is needed.
        seen_in_job = set()
        with path.open("r", encoding="utf8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_job = tuple(row[k] for k in job_fields)
                if expected_job is None:
                    expected_job = row_job
                if row_job != expected_job:
                    raise ValueError(f"Prediction job identity differs from its plan/file: {path}")
                owner = job_owners.setdefault(row_job, path)
                if owner != path:
                    raise ValueError(f"Duplicate job identity across prediction files: {row_job}")
                sample_id = row["sample_id"]
                if image_index is not None:
                    if sample_id not in image_index:
                        raise ValueError(f"Prediction sample is outside the persisted evaluated split: {sample_id}")
                    image_key = image_index[sample_id]
                else:
                    image_key = sample_id
                identity = (row["snr"], row["channel_seed"], image_key)
                if identity in seen_in_job:
                    raise ValueError(f"Duplicate prediction identity in {path}: {identity}")
                seen_in_job.add(identity)
                key = tuple(row[k] for k in ("method", "variant", "budget", "snr"))
                if row["total_bits"] > row["budget"]:
                    raise ValueError("Saved run violates its channel budget")
                if key not in grouped:
                    grouped[key] = _Group(image_index)
                grouped[key].add(row, image_key)
                observations += 1
        # Release a large single-job duplicate set before opening another job.
        del seen_in_job
    if not grouped:
        raise ValueError("No evaluated predictions found; run the experiment first")
    if state.get("status")=="completed" and observations!=plan["expected_rows"]:
        raise ValueError("Completed run has missing or unexpected predictions")
    summaries = []
    rng = np.random.default_rng(20261008)
    for key, group in sorted(grouped.items()):
        image_means = group.image_means()
        training_means = [total / count for _, (total, count) in sorted(group.seeds.items())]
        saved = dict(zip(("method", "variant", "budget", "snr"), key))
        saved.update(
            images=len(image_means), observations=group.observations, training_seeds=len(group.seeds) if key[1]!="none" else 0,
            evaluation_seed_repeats=len(group.seeds),
            accuracy=float(image_means[:, 0].mean()), gain=float(image_means[:, 1].mean()),
            accuracy_ci95=_interval(image_means[:, 0], bootstrap_samples, rng),
            gain_ci95=_interval(image_means[:, 1], bootstrap_samples, rng),
            training_accuracy_sd=float(np.std(training_means, ddof=1)) if len(training_means)>1 and key[1]!="none" else None,
            clean_accuracy=_mean(group.means, "clean_accuracy"),
            before_accuracy=_mean(group.means, "before_accuracy"),
            rescued_rate=_mean(group.means, "rescued_rate"),
            harmful_repair_rate=_mean(group.means, "harmful_repair_rate"),
            mse=_mean(group.means, "mse"), before_mse=_mean(group.means, "before_mse"),
            psnr=_mean(group.means, "psnr"), ssim=_mean(group.means, "ssim"),
            total_bits_mean=_mean(group.means, "total_bits_mean"), total_bits_max=group.max_bits,
        )
        if saved["total_bits_max"] > saved["budget"]:
            raise ValueError("Saved run violates its channel budget")
        saved["diagnostics_mean"] = {name: _mean(group.diagnostics, name) for name in group.diagnostics}
        summaries.append(saved)
    cfg = manifest.get("config", {})
    c = cfg.get("classifier", {})
    random_target = not c.get("checkpoint") and (c.get("kind") == "tiny" or c.get("weights") == "random" or (c.get("weights") == "imagenet" and cfg.get("data",{}).get("num_classes") != 1000))
    smoke = cfg.get("data", {}).get("name") == "synthetic" or manifest.get("classifier_is_random",random_target)
    result = {"schema_version": 1, "complete": state.get("status") == "completed", "purpose": "workflow_check" if smoke else "configured_evaluation",
              "observations": observations, "groups": summaries,
              "statistics": "Image-clustered paired percentile bootstrap after averaging channel and fitted-model repeats per image. Intervals are conditional on fitted models; training-seed sample SD is separate. PSNR is the mean of finite per-image dB values.",
              "bootstrap_samples": bootstrap_samples}
    folder = root / "summary"
    folder.mkdir(exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),encoding="utf8")
    columns = [k for k in summaries[0] if k != "diagnostics_mean"]
    with (folder / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream, fieldnames=columns);writer.writeheader()
        writer.writerows({k: json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in columns} for r in summaries)
    lines = ["# 实验结果汇总", "", f"用途：{'流程验证，不作为论文性能证据' if smoke else '按已保存配置评价，结果不自动证明创新性'}。",
             f"运行完整：{result['complete']}；预测记录：{observations}。", "",
             "区间以图像为抽样单位，先合并同一图像的信道和已训练模型重复。它不包含完整训练不确定性；训练种子标准差单独给出。准确率差值单位为百分点。", "",
             "| 方法 | 修补 | 预算 bit | SNR dB | 图像数 | 准确率 | 净增益及95%区间 | 实际bit均值 |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for row in summaries:
        lo,hi=row["gain_ci95"]
        lines.append(f"| {row['method']} | {row['variant']} | {row['budget']} | {row['snr']} | {row['images']} | {100*row['accuracy']:.2f}% | {100*row['gain']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {row['total_bits_mean']:.1f} |")
    (folder / "report.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    _plots(summaries,folder)
    return result


def _plots(rows, folder):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    for budget in sorted({r["budget"] for r in rows}):
        subset=[r for r in rows if r["budget"]==budget]
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for key in sorted({(r["method"],r["variant"]) for r in subset}):
            values=sorted([r for r in subset if (r["method"],r["variant"])==key],key=lambda r:r["snr"])
            x=[r["snr"] for r in values];label="/".join(key)
            axes[0].plot(x,[100*r["accuracy"] for r in values],marker="o",label=label)
            axes[1].plot(x,[100*r["gain"] for r in values],marker="o",label=label)
        for ax,ylabel in zip(axes,["Accuracy (%)","Paired gain (percentage points)"]):
            ax.set(xlabel="Coded-bit Eb/N0 (dB)",ylabel=ylabel);ax.grid(alpha=.25)
        axes[0].legend(fontsize=7);fig.suptitle(f"Native RGB transmission - budget {budget} bits")
        fig.tight_layout();fig.savefig(folder/f"curves_{budget}.png",dpi=160);plt.close(fig)
