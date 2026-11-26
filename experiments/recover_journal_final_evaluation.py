"""Recover final evaluation of a completed training trace; never train/resume.

Core journal sources remain unchanged. Original failure artifacts are copied to
a new read-only evidence directory, and training statistics are reconstructed
from the untouched batch CSV before the existing recorder finalizes the run.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import stat
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    options = parser.parse_args()
    from experiments.journal_runtime import (
        DATASETS, JournalRecorder, METRICS, WEIGHTS, dataset_fingerprint,
        model_state_hash, protocol_fingerprint, require_sanity_pass,
        sha256_file, source_hashes, write_json,
    )

    run_dir = options.run_dir.resolve()
    manifest_path = run_dir / "journal_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(original.get("status") == "FAILED_TRAINING", "Recovery requires an explicitly failed original run")
    require(original.get("source_hashes") == source_hashes(), "Original core source hashes changed; cannot recover this run")
    config = original["configuration"]
    require(config["JOURNAL"]["SANITY_BATCHES"] == 0, "Sanity traces cannot be recovered as full training")
    require(Path(config["OUTPUT_DIR"]).resolve() == run_dir, "Run/configuration output paths disagree")
    require(original["protocol_fingerprint"] == protocol_fingerprint(config), "Original protocol fingerprint mismatch")
    require(original["selection_protocol"] == "final_epoch" and original["test_used_for_selection"] is False,
            "Only final-epoch runs without performance selection may recover")
    require_sanity_pass(config["JOURNAL"]["SANITY_REPORT"])
    epochs = int(config["OPTIM"]["MAX_EPOCH"])
    checkpoint = run_dir / "VLPromptLearner" / f"model.pth.tar-{epochs}"
    require(checkpoint.is_file(), "Configured final checkpoint is missing")
    checkpoint_before = sha256_file(checkpoint)
    teacher = Path(original["teacher_checkpoint"])
    require(sha256_file(teacher) == original["teacher_checkpoint_sha256"], "Teacher checkpoint changed")
    require(sha256_file(config["TRAINER"]["LUNA"]["TEACHER_CLIP_PATH"]) == original["teacher_clip_sha256"],
            "Teacher CLIP foundation changed")

    trace = run_dir / "journal_batches.csv"
    with trace.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, "Training batch trace is empty")
    groups, global_group, indices = {}, JournalRecorder._group(), {}
    for row in rows:
        epoch, index, count = int(row["epoch"]), int(row["batch_index"]), int(row["n_samples"])
        require(row["dataset"] == original["dataset"] and row["variant"] == original["variant"]
                and int(row["seed"]) == original["seed"], "Batch identity differs from original run")
        require(Path(row["teacher_checkpoint"]).resolve() == teacher.resolve(), "Batch teacher differs from original run")
        require(count == config["DATALOADER"]["TRAIN_X"]["BATCH_SIZE"], "Unexpected partial/empty training batch")
        require(index not in indices.setdefault(epoch, set()), "Duplicated batch identity")
        indices[epoch].add(index)
        values = {key: float(row[key]) for key in METRICS}
        require(all(math.isfinite(value) for value in values.values()), "Nonfinite training trace")
        require(all(0 <= values[key] <= 1 for key in (*WEIGHTS, "r_ins", "r_batch", "r_cls")),
                "Invalid training reliability/weight")
        require(abs(sum(values[key] for key in WEIGHTS) - 1) < 1e-5, "Invalid recorded lambda normalization")
        for group in (global_group, groups.setdefault(epoch, JournalRecorder._group())):
            group["weight"] += count
            group["batches"] += 1
            for key in METRICS:
                group["sum"][key] += count * values[key]
            for key in WEIGHTS:
                group["sum_sq"][key] += count * values[key] ** 2
    require(set(groups) == set(range(1, epochs + 1)), "Training did not finish all configured epochs")
    batches_per_epoch = groups[1]["batches"]
    require(all(group["batches"] == batches_per_epoch for group in groups.values()), "Unequal epoch coverage")
    require(all(value == set(range(batches_per_epoch)) for value in indices.values()), "Noncontiguous training batches")
    reconstructed = JournalRecorder._statistics(global_group)
    for key, value in reconstructed.items():
        require(math.isclose(value, original["statistics"][key], rel_tol=1e-10, abs_tol=1e-12),
                "CSV/original overall statistics disagree: " + key)
    saved_epochs = {int(row["epoch"]): row for row in original["epochs"]}
    require(set(saved_epochs) == set(groups), "Original epoch summary coverage mismatch")
    for epoch, group in groups.items():
        for key, value in JournalRecorder._statistics(group).items():
            require(math.isclose(value, saved_epochs[epoch][key], rel_tol=1e-10, abs_tol=1e-12),
                    f"CSV/original epoch statistics disagree: {epoch}/{key}")

    protected = [checkpoint, trace]
    raw_dir = Path(original["raw_reliability_directory"])
    protected.extend(path for path in (raw_dir / "batches.csv", raw_dir / "samples.csv") if path.is_file())
    protected_hashes = {str(path): sha256_file(path) for path in protected}
    recovery_dir = run_dir / "recovery" / datetime.now(timezone.utc).strftime("final_eval_%Y%m%dT%H%M%S_%fZ")
    evidence_dir = recovery_dir / "original_failure"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    originals = {}
    for name in ("journal_manifest.json", "journal_epochs.json", "journal_epochs.csv",
                 "journal_log.txt", "journal_batches.csv", "log.txt"):
        source = run_dir / name
        if source.is_file():
            destination = evidence_dir / name
            shutil.copy2(source, destination)
            originals[str(destination)] = sha256_file(destination)
            require(originals[str(destination)] == sha256_file(source), "Failure evidence copy differs")
            destination.chmod(stat.S_IREAD)
    evidence_inventory = evidence_dir / "SHA256.json"
    write_json(evidence_inventory, originals)
    evidence_inventory.chmod(stat.S_IREAD)
    recovery = {
        "status": "RUNNING_FINAL_EVALUATION_RECOVERY", "mode": "recovered_eval_only",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "original_status": original["status"], "original_error": original.get("error"),
        "original_failure_artifacts_sha256": originals,
        "original_manifest_sha256": sha256_file(evidence_dir / "journal_manifest.json"),
        "evidence_directory": str(evidence_dir), "recovery_command_argv": list(sys.argv),
        "helper_path": str(Path(__file__).resolve()), "helper_sha256": sha256_file(__file__),
        "core_source_hashes": original["source_hashes"],
        "training_batch_count": len(rows), "training_observations": global_group["weight"],
        "training_restarted": False, "additional_optimizer_steps": 0,
        "student_checkpoint": str(checkpoint), "checkpoint_sha256_before": checkpoint_before,
        "protected_artifacts_sha256_before": protected_hashes,
        "statistics_source": "Original complete training CSV; unchanged batch sequence/sample weighting",
    }
    recovery_path = recovery_dir / "final_evaluation_recovery.json"
    write_json(recovery_path, recovery)

    # Imports and exact configuration reconstruction do not call the train entry.
    # Disable only WILDS' optional version-notification network hook.
    import outdated
    outdated.check_outdated = None
    import train
    import torch
    from yacs.config import CfgNode
    from dassl.config import get_cfg_default
    from dassl.utils import set_random_seed
    from trainers.luna_journal import LUNAJournal

    cfg = get_cfg_default()
    train.extend_cfg(cfg)
    cfg.merge_from_other_cfg(CfgNode(config))
    cfg.freeze()
    require(protocol_fingerprint(cfg) == original["protocol_fingerprint"], "Reconstructed configuration differs")
    # This is our own checkpoint from the verified complete run. Dassl stores
    # live scheduler metadata, which requires full loading rather than weights_only.
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    require(payload.get("epoch") == epochs and payload.get("val_result") is None,
            "Saved checkpoint epoch/performance-selection metadata is invalid")
    require(isinstance(payload.get("state_dict"), dict) and payload["state_dict"], "Missing saved model state")
    require(all(not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())
                for value in payload["state_dict"].values()), "Saved model contains nonfinite state")
    del payload
    set_random_seed(cfg.SEED)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = None
    started = time.perf_counter()
    try:
        trainer = LUNAJournal(cfg)
        require(model_state_hash(trainer.model) == original["initialization_sha256"], "Rebuilt student initialization differs")
        require(dataset_fingerprint(trainer.dm.dataset) == original["dataset_fingerprint"], "Actual dataset/splits differ")
        require(len(trainer.train_loader_x) == batches_per_epoch, "Actual full-loader batch count differs")
        trainer.epoch = epochs - 1
        trainer.load_model(str(run_dir), epoch=epochs)
        model_before = model_state_hash(trainer.model)
        trainer.init_writer(str(recovery_dir / "tensorboard"))
        base = float(trainer.test(split="val"))
        novel = float(trainer.test(split="test"))
        require(all(math.isfinite(value) and 0 <= value <= 100 for value in (base, novel)), "Invalid recovered evaluation")
        metrics = {
            "dataset": cfg.DATASET.NAME, "seed": int(cfg.SEED), "epoch": epochs,
            "variant": cfg.JOURNAL.MODE, "Base": base, "Novel": novel,
            "HM": 0.0 if base + novel == 0 else 2 * base * novel / (base + novel),
            "selection_protocol": "final_epoch", "student_checkpoint": str(checkpoint),
            "teacher_checkpoint": original["teacher_checkpoint"],
        }
        model_after = model_state_hash(trainer.model)
        require(model_before == model_after and trainer._journal_optimizer_steps == 0,
                "Recovery must not change model weights or execute optimizer steps")
        after_hashes = {path: sha256_file(path) for path in protected_hashes}
        require(after_hashes == protected_hashes, "An original checkpoint/raw training trace changed during recovery")
        recovery.update(status="COMPLETED_FINAL_EVALUATION_RECOVERY", elapsed_seconds=time.perf_counter() - started,
                        checkpoint_sha256_after=after_hashes[str(checkpoint)],
                        model_state_sha256_before=model_before, model_state_sha256_after=model_after,
                        protected_artifacts_sha256_after=after_hashes, metrics=metrics,
                        initialization_verified=True, dataset_fingerprint_verified=True,
                        protocol_fingerprint_verified=True, training_csv_verified=True)
        write_json(recovery_path, recovery)

        recorder = JournalRecorder.__new__(JournalRecorder)
        recorder.cfg, recorder.output, recorder.dataset = cfg, run_dir, DATASETS[cfg.DATASET.NAME]
        recorder.groups, recorder.global_group = groups, global_group
        recorder.expected_batches_per_epoch, recorder.handles = batches_per_epoch, []
        recorder.manifest = copy.deepcopy(original)
        recorder.manifest["final_evaluation_recovery"] = {**recovery, "report_path": str(recovery_path)}
        recorder.finish("COMPLETED_TRAINING", metrics=metrics)
        require(all(sha256_file(path) == digest for path, digest in protected_hashes.items()),
                "Finalization must preserve original checkpoint and raw training CSVs")
        print("COMPLETED_FINAL_EVALUATION_RECOVERY " + json.dumps(metrics), flush=True)
    except Exception as exc:
        recovery.update(status="FAILED_FINAL_EVALUATION_RECOVERY", error=f"{type(exc).__name__}: {exc}")
        write_json(recovery_path, recovery)
        raise
    finally:
        if trainer is not None:
            trainer.close_writer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
