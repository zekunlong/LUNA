"""Complete the two independently predeclared Exp1 StaticMean controls.

Run only after all original21 training conditions finish. The matching completed
StrongAdaptive training_mean_weights.json is fixed throughout training. Existing
StrongEqual results are checked and reused; this helper never trains Equal.
--validate-only does CPU/file checks only and reports pending dependencies honestly.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.journal_runtime import (
    WEIGHTS, protocol_fingerprint, require_sanity_pass, sha256_file,
    source_hashes, validate_static_means,
)
from experiments.run_first_real_dtd import run
from experiments.run_journal_exp234_training import ROOT, DATA_ROOT, DATASETS, COHORT, destination, teacher_directory

PLAN = ROOT / "logs/exp1_static_mean_supplemental_predeclared_plan_20260906.json"
PLAN_SHA256 = "adb8f5d9568fbe95de6fcb8ee6abdde04fc6175a24fea59c22cb0be08a28f062"
TARGETS = ("FGVCAircraft", "EuroSAT")
PAIRED_KEYS = ("seed", "protocol_fingerprint", "dataset_fingerprint", "initialization_sha256",
               "teacher_checkpoint_sha256", "teacher_clip_sha256", "source_hashes")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def event(path, kind, **fields):
    record = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print("STATIC_MEAN_EVENT " + json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def load_plan():
    require(sha256_file(PLAN) == PLAN_SHA256, "Supplemental predeclaration changed")
    plan = read(PLAN)
    require(sha256_file(plan["original_plan"]) == plan["original_plan_sha256"], "Original21 plan changed")
    original = read(plan["original_plan"])
    current = source_hashes()
    require(current == plan["core_source_sha256"] == original["core_source_sha256"], "Predeclared core source changed")
    require_sanity_pass(ROOT / "sanity_result.json")
    require(plan["execution_order"] == list(TARGETS), "Supplemental execution order changed")
    require(len(plan["conditions"]) == 2, "Exactly two supplemental conditions are required")
    for dataset, item in zip(TARGETS, plan["conditions"]):
        cohort = ROOT / "exp1_weighting" / dataset / COHORT
        expected = {"dataset": dataset, "quality": "Strong", "variant": "StaticMean", "teacher_epoch": 20,
                    "run_manifest": str(cohort / "StaticMean/journal_manifest.json"),
                    "adaptive_source_manifest": str(cohort / "AdaptiveLUNA/journal_manifest.json"),
                    "static_weights_file": str(cohort / "AdaptiveLUNA/training_mean_weights.json"),
                    "equal_reused_manifest": str(cohort / "Equal/journal_manifest.json"),
                    "teacher_checkpoint": str(teacher_directory(dataset) / "VLPromptLearner/model.pth.tar-20")}
        require(all(item.get(k) == v for k, v in expected.items()), "Supplemental paths/identity differ from paired protocol")
    return plan, original, current


def completed_manifest(path, dataset, mode, current):
    record = read(path)
    required = {"status": "COMPLETED_TRAINING", "dataset": dataset, "seed": 1, "variant": mode,
                "teacher_quality": "strong", "epoch": 5, "selection_protocol": "final_epoch",
                "test_used_for_selection": False, "source_hashes": current}
    require(all(record.get(k) == v for k, v in required.items()), "Incomplete/incompatible paired run: " + str(path))
    require(Path(record["student_checkpoint"]).is_file(), "Missing final student checkpoint")
    cfg = record["configuration"]
    require(cfg["OPTIM"]["MAX_EPOCH"] == 5 and cfg["DATALOADER"]["TRAIN_X"]["BATCH_SIZE"] == 8,
            "Paired run must use final5/batch8")
    require(cfg["DATASET"]["NUM_SHOTS"] == 0 and cfg["SEED"] == 1, "Paired run seed/shots changed")
    require(Path(cfg["DATASET"]["ROOT"]).resolve() == DATA_ROOT.resolve(), "Paired dataset root changed")
    require(protocol_fingerprint(cfg) == record["protocol_fingerprint"], "Paired configuration fingerprint is inconsistent")
    return record


def original_pending(original, current):
    pending = []
    for item in original["conditions"]:
        path = Path(item["run_manifest"])
        record = read(path) if path.is_file() else {}
        if record.get("status") != "COMPLETED_TRAINING":
            pending.append({"dataset": item["dataset"], "quality": item["quality"], "variant": item["variant"],
                            "status": record.get("status", "NOT_STARTED")})
            continue
        require(record["source_hashes"] == current and record["seed"] == 1 and record["epoch"] == 5,
                "Original21 completed condition identity changed")
        require(Path(record["student_checkpoint"]).is_file(), "Original21 final checkpoint missing")
    return pending


def command_for(item, paired_root):
    dataset = item["dataset"]
    return [sys.executable, "-X", "utf8", "-u", "train.py", "--root", paired_root, "--seed", "1",
            "--trainer", "LUNA", "--dataset-config-file", f"configs/datasets/{DATASETS[dataset]}.yaml",
            "--config-file", "configs/trainers/LUNA/vit_b16_c2_ep20_batch8_4+4ctx.yaml",
            "--output-dir", str(Path(item["run_manifest"]).parent), "--distill-weighting", "static_mean",
            "--selection-protocol", "final_epoch", "--teacher-checkpoint", item["teacher_checkpoint"],
            "--teacher-total-epochs", "20", "--teacher-quality", "strong", "--analysis-output-dir", str(ROOT),
            "--sanity-report", str(ROOT / "sanity_result.json"), "--static-weights", item["static_weights_file"],
            "DATASET.NUM_SHOTS", "0", "TRAINER.MODAL", "base2novel", "TRAINER.LUNA.TEMPERATURE", "1.0",
            "TRAINER.LUNA.KD_WEIGHT", "1000.0", "TRAINER.LUNA.TEACHER_CLIP_PATH",
            str(Path.home() / ".cache/clip/ViT-L-14.pt"), "TEST.SPLIT", "val"]


def paired_arguments(arguments):
    """Only output and the explicitly declared weighting treatment may differ."""
    ignored = {"--output-dir", "--distill-weighting", "--static-weights"}
    result = []
    index = 0
    while index < len(arguments):
        if arguments[index] in ignored:
            require(index + 1 < len(arguments), "Incomplete paired command argument")
            index += 2
        else:
            result.append(arguments[index])
            index += 1
    return result


def prepare(item, current):
    dataset = item["dataset"]
    adaptive_path = Path(item["adaptive_source_manifest"])
    result = {"dataset": dataset, "gpu_launched": False, "paired_protocol_validated": False,
              "command": command_for(item, DATA_ROOT.as_posix()), "static_weights_file": item["static_weights_file"]}
    if not adaptive_path.is_file() or read(adaptive_path).get("status") != "COMPLETED_TRAINING":
        result.update(status="PREPARED_PENDING_COMPLETED_STRONG_ADAPTIVE", fixed_weight_values=None)
        return result
    adaptive = completed_manifest(adaptive_path, dataset, "reliability_adaptive", current)
    command = command_for(item, adaptive["configuration"]["DATASET"]["ROOT"])
    require(paired_arguments(command[command.index("train.py") + 1:]) == paired_arguments(adaptive["command_argv"][1:]),
            "StaticMean CLI differs from actual paired Adaptive beyond output/weighting")
    means_path = Path(item["static_weights_file"])
    require(means_path.is_file(), "Completed adaptive run lacks training_mean_weights.json")
    teacher_receipt = read(teacher_directory(dataset) / "teacher_provenance.json")
    require(teacher_receipt["status"] == "COMPLETED_REAL_TEACHER_TRAINING", "Teacher is not complete")
    checkpoint_hash = sha256_file(item["teacher_checkpoint"])
    require(checkpoint_hash == teacher_receipt["checkpoint_sha256"][item["teacher_checkpoint"]]
            == adaptive["teacher_checkpoint_sha256"], "Paired final teacher checkpoint changed")
    from yacs.config import CfgNode
    config = copy.deepcopy(adaptive["configuration"])
    config["OUTPUT_DIR"] = str(Path(item["run_manifest"]).parent)
    config["DATASET"]["FEWSHOT_CACHE_ROOT"] = str(Path(item["run_manifest"]).parent / "fewshot_cache")
    config["JOURNAL"].update(MODE="static_mean", STATIC_WEIGHTS_FILE=str(means_path))
    cfg = CfgNode(config)
    means = validate_static_means(cfg)
    require(Path(means["source_training_run"]).resolve() == adaptive_path.parent.resolve(), "Static weights came from a different adaptive run")
    require(all(means[k] == adaptive[k] for k in PAIRED_KEYS), "Static weight provenance differs from matching adaptive manifest")
    # Recompute normalized sample-weighted means from the complete original CSV.
    csv_path = adaptive_path.parent / "journal_batches.csv"
    with csv_path.open(encoding="utf-8") as stream:
        batches = list(csv.DictReader(stream))
    require(len(batches) == adaptive["statistics"]["batch_count"], "Adaptive batch trace is incomplete")
    require({int(row["epoch"]) for row in batches} == set(range(1, 6)), "Adaptive means must include all5epochs")
    require(all(int(row["n_samples"]) == 8 and row["variant"] == "reliability_adaptive" for row in batches), "Adaptive mean trace batch protocol changed")
    observations = sum(int(row["n_samples"]) for row in batches)
    require(observations == means["n_observations"] == adaptive["statistics"]["n_observations"], "Adaptive observation count differs")
    raw = {key: sum(float(row[key]) * int(row["n_samples"]) for row in batches) / observations for key in WEIGHTS}
    total = sum(raw.values())
    require(all(math.isclose(raw[k] / total, means["lambda_mean"][k], rel_tol=1e-10, abs_tol=1e-12) for k in WEIGHTS),
            "Fixed means differ from complete training-only adaptive trace")
    require(protocol_fingerprint(cfg) == adaptive["protocol_fingerprint"], "StaticMean paired protocol differs")
    result.update(status="PAIRED_STATIC_PROTOCOL_VALIDATED_PENDING_EQUAL", paired_protocol_validated=True,
                  command=command, fixed_weight_values=means["lambda_mean"],
                  actual_paired_command_arguments_match=True,
                  protocol_fingerprint=adaptive["protocol_fingerprint"], adaptive_manifest_sha256=sha256_file(adaptive_path),
                  adaptive_checkpoint_sha256=sha256_file(adaptive["student_checkpoint"]),
                  fixed_weights_sha256=sha256_file(means_path), adaptive_training_csv_sha256=sha256_file(csv_path),
                  teacher_checkpoint_sha256=checkpoint_hash, n_observations=observations)
    equal_path = Path(item["equal_reused_manifest"])
    if not equal_path.is_file() or read(equal_path).get("status") != "COMPLETED_TRAINING":
        return result
    equal = completed_manifest(equal_path, dataset, "equal", current)
    require(all(equal[k] == adaptive[k] for k in PAIRED_KEYS), "Existing StrongEqual differs from paired adaptive protocol")
    result.update(status="PASS_READY_PAIRED_STATIC_MEAN", equal_manifest_sha256=sha256_file(equal_path),
                  equal_checkpoint_sha256=sha256_file(equal["student_checkpoint"]), equal_retrained=False)
    output = Path(item["run_manifest"])
    if output.is_file():
        completed = completed_manifest(output, dataset, "static_mean", current)
        require(all(completed[k] == adaptive[k] for k in PAIRED_KEYS), "Existing StaticMean is incompatible")
        require(Path(completed["configuration"]["JOURNAL"]["STATIC_WEIGHTS_FILE"]).resolve() == means_path.resolve(),
                "Existing StaticMean used another weight source")
        used = completed["configuration"]["JOURNAL"]["STATIC_WEIGHTS"]
        require(len(used) == 3 and all(math.isclose(used[i], means["lambda_mean"][key], rel_tol=1e-12, abs_tol=1e-12)
                                     for i, key in enumerate(WEIGHTS)), "Existing StaticMean fixed values differ")
        result["status"] = "PASS_REUSABLE_COMPLETED_STATIC_MEAN"
    elif output.parent.exists():
        require(not any(output.parent.iterdir()), "Preserve incomplete/nonempty StaticMean output; automatic restart disabled")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan, original, current = load_plan()
    pending = original_pending(original, current)
    preparations = [prepare(item, current) for item in plan["conditions"]]
    ready = not pending and all(item["status"] in ("PASS_READY_PAIRED_STATIC_MEAN", "PASS_REUSABLE_COMPLETED_STATIC_MEAN") for item in preparations)
    if args.validate_only:
        print(json.dumps({"status": "PASS_READY_SUPPLEMENTAL_STATIC_MEAN" if ready else "PREPARED_PENDING_DEPENDENCIES",
                          "gpu_launched": False, "plan": str(PLAN), "plan_sha256": PLAN_SHA256,
                          "original21_pending": pending, "preparations": preparations}, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    require(ready, "Supplement starts only after all original21 complete and both paired adaptive/equal protocols validate; use --validate-only for pending state")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    logs = ROOT / "logs/static_mean_supplemental_execution"
    logs.mkdir(parents=True, exist_ok=True)
    events = logs / f"supplement_{stamp}.jsonl"
    event(events, "START_SUPPLEMENT", plan_sha256=PLAN_SHA256, helper_sha256=sha256_file(__file__),
          declared_order=list(TARGETS), original21_all_completed=True, existing_equal_policy="reuse_only_never_retrain")
    try:
        for item, prepared in zip(plan["conditions"], preparations):
            dataset = item["dataset"]
            event(events, "REUSE_EXISTING_EQUAL", dataset=dataset, manifest=item["equal_reused_manifest"],
                  manifest_sha256=prepared["equal_manifest_sha256"], checkpoint_sha256=prepared["equal_checkpoint_sha256"])
            if prepared["status"] == "PASS_REUSABLE_COMPLETED_STATIC_MEAN":
                event(events, "REUSE_EXISTING_STATIC_MEAN", dataset=dataset, manifest=item["run_manifest"],
                      manifest_sha256=sha256_file(item["run_manifest"]))
                continue
            event(events, "START_STATIC_MEAN", dataset=dataset, prepared=prepared)
            run(prepared["command"], logs / f"{dataset}_StaticMean_{stamp}.stdout.txt")
            after = prepare(item, current)
            require(after["status"] == "PASS_REUSABLE_COMPLETED_STATIC_MEAN", "Supplemental training did not complete")
            for key in ("fixed_weights_sha256", "adaptive_manifest_sha256", "adaptive_checkpoint_sha256",
                        "adaptive_training_csv_sha256", "equal_manifest_sha256", "equal_checkpoint_sha256", "teacher_checkpoint_sha256"):
                require(after[key] == prepared[key], "Protected paired evidence changed during StaticMean: " + key)
            measured = read(item["run_manifest"])
            event(events, "COMPLETED_STATIC_MEAN", dataset=dataset, manifest=item["run_manifest"],
                  manifest_sha256=sha256_file(item["run_manifest"]),
                  checkpoint_sha256=sha256_file(measured["student_checkpoint"]),
                  Base=measured["Base"], Novel=measured["Novel"], HM=measured["HM"], ResultSource="REAL_EXPERIMENT")
        event(events, "COMPLETED_SUPPLEMENT", datasets=list(TARGETS), equal_retrained=False, ResultSource="REAL_EXPERIMENT")
    except BaseException as exc:
        event(events, "STOPPED_ERROR", error=repr(exc), traceback=traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
