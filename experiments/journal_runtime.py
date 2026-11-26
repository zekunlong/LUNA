"""Logging, provenance, and the real-DTD sanity gate for journal training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = Path("E:/LUNA/results_reliability_analysis")
MODES = ("legacy", "equal", "static_mean", "reliability_adaptive", "instance_only")
METRICS = ("r_ins", "r_batch", "r_cls", "lambda_ins", "lambda_batch", "lambda_cls",
           "L_ins", "L_batch", "L_cls", "L_adapt")
WEIGHTS = ("lambda_ins", "lambda_batch", "lambda_cls")
DATASETS = {"FGVCAircraft": "FGVCAircraft", "DescribableTextures": "DTD", "EuroSAT": "EuroSAT"}
SANITY_CHECKS = (
    "ten_real_batches", "no_nan_inf", "reliability_in_unit_interval", "weights_in_unit_interval",
    "weights_sum_to_one", "weights_change_across_batches", "student_prompt_gradients",
    "teacher_frozen_without_gradients", "independent_weighted_scalars", "relative_contributions_change",
    "legacy_is_mathematically_different", "same_sample_two_views", "no_test_checkpoint_selection",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    paths = ["train.py", "trainers/luna.py", "trainers/luna_journal.py",
             "experiments/journal_losses.py", "experiments/journal_runtime.py",
             "experiments/journal_preflight.py", "experiments/run_journal_sanity.py"]
    for directory in ("datasets", "configs", "clip", "Dassl.pytorch/dassl"):
        paths.extend(str(p.relative_to(REPO)).replace("\\", "/")
                     for p in (REPO / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml"))
    return {name: sha256_file(REPO / name) for name in sorted(set(paths))}


def plain_config(value):
    if hasattr(value, "items"):
        return {str(k): plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(v) for v in value]
    return value


def protocol_fingerprint(cfg):
    config = plain_config(cfg)
    for key in ("OUTPUT_DIR", "RESUME"):
        config.pop(key, None)
    config["DATASET"].pop("FEWSHOT_CACHE_ROOT", None)
    journal = config["JOURNAL"]
    for key in ("MODE", "STATIC_WEIGHTS", "STATIC_WEIGHTS_FILE", "OUTPUT_ROOT",
                "SANITY_BATCHES", "SANITY_REPORT", "LOG_BATCH_SAMPLES"):
        journal.pop(key, None)
    encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_state_hash(model):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def dataset_fingerprint(dataset):
    digest = hashlib.sha256()
    for split in ("train_x", "val", "test"):
        digest.update(split.encode())
        for item in getattr(dataset, split):
            digest.update(json.dumps([item.impath, item.label, item.classname], ensure_ascii=False).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def configure_journal(cfg, args):
    """Apply explicit new-mode flags after config merge; legacy is unchanged."""
    mode = getattr(args, "distill_weighting", None) or cfg.JOURNAL.MODE
    if mode not in MODES:
        raise ValueError(f"Unsupported distill weighting: {mode}")
    cfg.JOURNAL.MODE = mode
    if mode == "legacy":
        if getattr(args, "selection_protocol", None) not in (None, "legacy"):
            raise ValueError("Use a journal weighting mode with final_epoch; legacy keeps its original protocol")
        return
    if cfg.TRAINER.NAME != "LUNA" or cfg.TRAINER.MODAL != "base2novel":
        raise ValueError("Journal comparisons require the existing LUNA base2novel pipeline")
    if cfg.DATASET.NAME not in DATASETS:
        raise ValueError("Journal comparisons are restricted to FGVCAircraft, DTD, and EuroSAT")
    requested_selection = getattr(args, "selection_protocol", None) or "final_epoch"
    if requested_selection != "final_epoch":
        raise ValueError("Journal mode requires --selection-protocol final_epoch")
    cfg.JOURNAL.SELECTION_PROTOCOL = "final_epoch"
    cfg.TEST.FINAL_MODEL = "last_step"
    cfg.TEST.NO_TEST = True
    mappings = {
        "teacher_checkpoint": "TEACHER_CHECKPOINT", "teacher_total_epochs": "TEACHER_TOTAL_EPOCHS",
        "teacher_quality": "TEACHER_QUALITY", "static_weights": "STATIC_WEIGHTS_FILE",
        "analysis_output_dir": "OUTPUT_ROOT", "sanity_batches": "SANITY_BATCHES",
        "sanity_report": "SANITY_REPORT",
    }
    for argument, field in mappings.items():
        value = getattr(args, argument, None)
        if value is not None:
            setattr(cfg.JOURNAL, field, value)
    if not cfg.DATASET.ROOT:
        cfg.DATASET.ROOT = os.environ.get("LUNA_DATA_ROOT", "")
    if not cfg.TRAINER.LUNA.TEACHER_CLIP_PATH:
        cfg.TRAINER.LUNA.TEACHER_CLIP_PATH = os.environ.get("LUNA_TEACHER_CLIP_PATH", "")
    if not cfg.JOURNAL.TEACHER_CHECKPOINT:
        cfg.JOURNAL.TEACHER_CHECKPOINT = os.environ.get("LUNA_TEACHER_CHECKPOINT", "")
    if not cfg.JOURNAL.TEACHER_TOTAL_EPOCHS:
        cfg.JOURNAL.TEACHER_TOTAL_EPOCHS = int(os.environ.get("LUNA_TEACHER_TOTAL_EPOCHS", "0"))
    requested_seed = getattr(args, "seed", None)
    if requested_seed is not None and requested_seed >= 0:
        cfg.SEED = requested_seed
    if cfg.SEED < 0:
        raise ValueError("Journal mode requires the fixed main-run seed (existing queue: 1)")
    if cfg.RESUME:
        raise ValueError("Journal runs start from their recorded initialization; automatic resume is disabled")
    if not getattr(args, "output_dir", ""):
        raise ValueError("Journal mode requires a separate explicit --output-dir")
    output = Path(cfg.OUTPUT_DIR).expanduser().resolve()
    legacy_output = (REPO / "output").resolve()
    if output == legacy_output or legacy_output in output.parents:
        raise ValueError("Journal output must be separate from the repository's legacy output directory")
    if (output / "journal_manifest.json").exists() or (output / "VLPromptLearner").exists():
        raise ValueError(f"Journal run already has a manifest/checkpoint; use a fresh output directory: {output}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Journal output must be fresh; existing run files will not be overwritten: {output}")
    cfg.DATASET.FEWSHOT_CACHE_ROOT = str(output / "fewshot_cache")
    if mode == "static_mean":
        provenance = validate_static_means(cfg)
        cfg.JOURNAL.STATIC_WEIGHTS = tuple(provenance["lambda_mean"][key] for key in WEIGHTS)


def validate_static_means(cfg):
    path = Path(cfg.JOURNAL.STATIC_WEIGHTS_FILE)
    if not cfg.JOURNAL.STATIC_WEIGHTS_FILE or not path.is_file():
        raise ValueError("static_mean requires --static-weights from a completed adaptive training run")
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "COMPLETED_TRAINING", "source_mode": "reliability_adaptive",
        "data_usage": "training_only", "selection_protocol": "final_epoch",
        "dataset": DATASETS[cfg.DATASET.NAME], "seed": cfg.SEED,
        "protocol_fingerprint": protocol_fingerprint(cfg),
        "teacher_checkpoint_sha256": sha256_file(cfg.JOURNAL.TEACHER_CHECKPOINT),
        "teacher_clip_sha256": sha256_file(cfg.TRAINER.LUNA.TEACHER_CLIP_PATH),
        "source_hashes": source_hashes(),
    }
    for name, expected in required.items():
        if record.get(name) != expected:
            raise ValueError(f"Static-mean provenance mismatch: {name}")
    values = [record.get("lambda_mean", {}).get(key) for key in WEIGHTS]
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Static means must be three finite normalized weights")
    if abs(sum(values) - 1) >= 1e-5 or record.get("n_observations", 0) <= 0:
        raise ValueError("Static means have invalid normalization or no training observations")
    return record


def require_sanity_pass(path):
    if not path or not Path(path).is_file():
        raise ValueError("Full journal training is gated on a passed real DTD sanity_result.json")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("status") != "PASS_REAL_DTD_SANITY" or record.get("dataset") != "DTD":
        raise ValueError("Synthetic tests, blocked reports, and other datasets cannot unlock journal experiments")
    checks = record.get("checks", {})
    if set(checks) != set(SANITY_CHECKS) or not all(value is True for value in checks.values()):
        raise ValueError("The real DTD sanity check did not pass all required checks on ten batches")
    records = record.get("records", [])
    if len(records) != record.get("batch_count") or len(records) < 10:
        raise ValueError("A passed sanity report must contain at least ten actual batch records")
    identities = {(row.get("epoch"), row.get("batch_index")) for row in records}
    if len(identities) != len(records) or any(row.get("run_kind") != "sanity_training_batches" for row in records):
        raise ValueError("Sanity records must identify distinct real training batches")
    preflight = record.get("preflight", {})
    if preflight.get("status") != "READY" or preflight.get("environment", {}).get("checked") is not True:
        raise ValueError("Actual DTD assets and the training environment must have passed preflight")
    if record.get("source_hashes") != source_hashes():
        raise ValueError("Training code changed since the DTD sanity check; rerun it before full experiments")
    return record


class JournalRecorder:
    """Stream scalar/sample records; epoch and global statistics are sample-weighted."""

    def __init__(self, trainer, cfg):
        self.cfg = cfg
        self.output = Path(cfg.OUTPUT_DIR)
        self.output.mkdir(parents=True, exist_ok=True)
        self.dataset = DATASETS[cfg.DATASET.NAME]
        self.groups = {}
        self.global_group = self._group()
        self.expected_batches_per_epoch = len(trainer.train_loader_x)
        self.handles = []
        self.manifest = {
            "status": "RUNNING", "dataset": self.dataset, "seed": cfg.SEED,
            "variant": cfg.JOURNAL.MODE, "teacher_quality": cfg.JOURNAL.TEACHER_QUALITY,
            "teacher_checkpoint": str(Path(cfg.JOURNAL.TEACHER_CHECKPOINT).resolve()),
            "teacher_checkpoint_sha256": sha256_file(cfg.JOURNAL.TEACHER_CHECKPOINT),
            "teacher_clip_sha256": sha256_file(cfg.TRAINER.LUNA.TEACHER_CLIP_PATH),
            "dataset_fingerprint": dataset_fingerprint(trainer.dm.dataset),
            "selection_protocol": "final_epoch", "test_used_for_selection": False,
            "configuration": plain_config(cfg), "protocol_fingerprint": protocol_fingerprint(cfg),
            "source_hashes": source_hashes(), "initialization_sha256": model_state_hash(trainer.model),
            "python": platform.python_version(), "reliability_granularity": "batch",
            "command_argv": list(sys.argv),
            "reliability_temperature": 1.0, "loss_temperatures": [1, 3, 5, 2, 6],
            "paired_views": True, "loss_views": "sum of weak and augmented",
            "Base": None, "Novel": None, "HM": None,
        }
        if cfg.JOURNAL.MODE == "static_mean":
            origin = validate_static_means(cfg)
            for key in ("initialization_sha256", "dataset_fingerprint"):
                if origin.get(key) != self.manifest[key]:
                    raise ValueError(f"StaticMean and Adaptive provenance mismatch: {key}")
        manifest_path = self.output / "journal_manifest.json"
        if manifest_path.exists():
            raise ValueError(f"Refusing to overwrite an existing journal run: {manifest_path}")
        write_json(manifest_path, self.manifest)
        self.batch_fields = ("dataset", "seed", "epoch", "batch_index", "n_samples",
                             "teacher_checkpoint", "variant", *METRICS, "legacy_loss")
        self.batch_writer = self._csv(self.output / "journal_batches.csv", self.batch_fields)
        run_id = hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()[:12]
        raw_dir = (Path(cfg.JOURNAL.OUTPUT_ROOT) / "raw_reliability" / self.dataset
                   / cfg.JOURNAL.MODE / f"seed_{cfg.SEED}_{run_id}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.raw_writer = self._csv(raw_dir / "batches.csv", self.batch_fields)
        self.sample_writer = None
        if cfg.JOURNAL.LOG_BATCH_SAMPLES:
            self.sample_writer = self._csv(raw_dir / "samples.csv", (
                "dataset", "seed", "epoch", "batch_index", "sample_index", "impath",
                "teacher_entropy", "teacher_entropy_aug", "teacher_max_probability",
                "teacher_prediction", "ground_truth", "teacher_correct"))
        self.manifest["raw_reliability_directory"] = str(raw_dir)
        trainer.journal_batch_callback = self.on_batch
        trainer.journal_epoch_callback = self.on_epoch

    def on_epoch(self, record):
        epoch = int(record["epoch"])
        statistics = self._statistics(self.groups[epoch])
        print(f"Journal epoch {epoch} means and lambda standard deviations:", json.dumps(statistics, sort_keys=True))

    @staticmethod
    def _group():
        return {"weight": 0, "batches": 0, "sum": {k: 0.0 for k in METRICS},
                "sum_sq": {k: 0.0 for k in WEIGHTS}}

    def _csv(self, path, fields):
        handle = Path(path).open("x", encoding="utf-8", newline="")
        self.handles.append(handle)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        return writer

    def on_batch(self, record, sample_records):
        row = {**record, "dataset": self.dataset, "seed": self.cfg.SEED,
               "teacher_checkpoint": self.manifest["teacher_checkpoint"], "variant": self.cfg.JOURNAL.MODE}
        for key in METRICS:
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Non-finite journal metric: {key}")
        weight = int(row["n_samples"])
        if weight <= 0:
            raise ValueError("A journal batch must contain samples")
        for group in (self.global_group, self.groups.setdefault(int(row["epoch"]), self._group())):
            group["weight"] += weight
            group["batches"] += 1
            for key in METRICS:
                group["sum"][key] += weight * float(row[key])
            for key in WEIGHTS:
                group["sum_sq"][key] += weight * float(row[key]) ** 2
        self.batch_writer.writerow(row)
        self.raw_writer.writerow(row)
        if self.sample_writer is not None:
            for sample in sample_records:
                self.sample_writer.writerow({**sample, "dataset": self.dataset, "seed": self.cfg.SEED,
                                             "epoch": row["epoch"], "batch_index": row["batch_index"]})
        for handle in self.handles:
            handle.flush()

    @staticmethod
    def _statistics(group):
        if not group["weight"]:
            return {}
        result = {"n_observations": group["weight"], "batch_count": group["batches"]}
        for key in METRICS:
            result["mean_" + key] = group["sum"][key] / group["weight"]
        for key in WEIGHTS:
            variance = group["sum_sq"][key] / group["weight"] - result["mean_" + key] ** 2
            result["std_" + key] = math.sqrt(max(0.0, variance))
        return result

    def finish(self, status, *, metrics=None, error=None):
        if status == "COMPLETED_TRAINING":
            if self.cfg.JOURNAL.SANITY_BATCHES > 0:
                raise ValueError("Sanity batches cannot be exported as completed training")
            if set(self.groups) != set(range(1, self.cfg.OPTIM.MAX_EPOCH + 1)):
                raise ValueError("Completed training requires the full contiguous epoch trace")
            if any(group["batches"] != self.expected_batches_per_epoch for group in self.groups.values()):
                raise ValueError("Completed training requires every batch of every epoch")
        for handle in self.handles:
            handle.close()
        epochs = [{"epoch": epoch, **self._statistics(group)} for epoch, group in sorted(self.groups.items())]
        statistics = self._statistics(self.global_group)
        self.manifest.update(status=status, statistics=statistics, epochs=epochs, error=error)
        if metrics:
            self.manifest.update(metrics)
            self.manifest["dataset"] = self.dataset
        write_json(self.output / "journal_manifest.json", self.manifest)
        write_json(self.output / "journal_epochs.json", epochs)
        if epochs:
            with (self.output / "journal_epochs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(epochs[0]))
                writer.writeheader()
                writer.writerows(epochs)
        lines = [f"Status: {status}", f"Dataset: {self.dataset}; seed: {self.cfg.SEED}",
                 "Selection: final_epoch; test accuracy never selects a checkpoint.",
                 "Statistics: sample-weighted population means/std; raw scores are batch-level."]
        for epoch in epochs:
            lines.append(json.dumps(epoch, sort_keys=True))
            print("Journal epoch statistics:", json.dumps(epoch, sort_keys=True))
        if error:
            lines.append(f"Error: {error}")
        (self.output / "journal_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if status == "COMPLETED_TRAINING" and self.cfg.JOURNAL.MODE == "reliability_adaptive":
            means = [statistics["mean_" + key] for key in WEIGHTS]
            total = sum(means)
            provenance = {
                "status": status, "source_mode": "reliability_adaptive", "data_usage": "training_only",
                "dataset": self.dataset, "seed": self.cfg.SEED, "selection_protocol": "final_epoch",
                "source_training_run": str(self.output.resolve()),
                "n_observations": statistics["n_observations"],
                "lambda_mean": dict(zip(WEIGHTS, [value / total for value in means])),
                **{key: self.manifest[key] for key in ("protocol_fingerprint", "teacher_checkpoint_sha256",
                                                     "teacher_clip_sha256", "dataset_fingerprint",
                                                     "initialization_sha256", "source_hashes")},
            }
            write_json(self.output / "training_mean_weights.json", provenance)
        return self.manifest
