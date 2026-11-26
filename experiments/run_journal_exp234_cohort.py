"""Complete the predeclared three-dataset cohort, reusing authenticated results.

GPU tasks run sequentially. Failures stop the queue and preserve all artifacts.
The append-only event log records progress independently of console sessions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.run_first_real_dtd import run
from experiments.run_journal_exp234_training import (
    ROOT, COHORT, DATASETS, STAGES, destination, teacher_directory, train_condition,
)
from experiments.run_journal_posthoc import evaluate_condition
from experiments.journal_runtime import sha256_file, source_hashes


def now():
    return datetime.now(timezone.utc).isoformat()


def append_event(path, kind, **values):
    event = {"utc": now(), "event": kind, **values}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print("COHORT_EVENT " + json.dumps(event, ensure_ascii=False), flush=True)


def ensure_teacher(dataset, event_path, stamp):
    receipt_path = teacher_directory(dataset) / "teacher_provenance.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "COMPLETED_REAL_TEACHER_TRAINING":
            raise ValueError("Preserved incomplete teacher requires inspection: " + str(receipt_path))
        for epoch in STAGES.values():
            checkpoint = teacher_directory(dataset) / "VLPromptLearner" / f"model.pth.tar-{epoch}"
            if sha256_file(checkpoint) != receipt["checkpoint_sha256"][str(checkpoint)]:
                raise ValueError("Teacher checkpoint hash changed: " + str(checkpoint))
        append_event(event_path, "REUSED_TEACHER", dataset=dataset, receipt=str(receipt_path))
        return
    if dataset == "DTD":
        raise ValueError("This continuation requires the existing completed DTD teacher")
    adapter = Path("E:/LUNA/external/PromptKD/run_dataset_promptsrc_final.py")
    append_event(event_path, "START_TEACHER", dataset=dataset, adapter_sha256=sha256_file(adapter))
    run([sys.executable, "-X", "utf8", "-B", "-u", str(adapter), "--dataset", dataset],
        ROOT / "logs/exp234_execution" / dataset / f"teacher_{stamp}.stdout.txt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETED_REAL_TEACHER_TRAINING":
        raise ValueError("Teacher did not complete")
    append_event(event_path, "COMPLETED_TEACHER", dataset=dataset, receipt=str(receipt_path))


def publish(dataset, event_path, stamp):
    for script in ("export_measured_exp234_results.py", "summarize_measured_dtd.py"):
        run([sys.executable, "-X", "utf8", "-u", "experiments/" + script],
            ROOT / "logs/exp234_execution" / dataset / f"{script}_{stamp}.stdout.txt")
    append_event(event_path, "PUBLISHED_DATASET", dataset=dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate dataset")
    plan_path = ROOT / "logs/exp234_predeclared_plan_20260906.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["core_source_sha256"] != source_hashes():
        raise ValueError("Predeclared core source changed")
    declared = {(item["dataset"], item["quality"], item["variant"]): item for item in plan["conditions"]}
    order = [(quality, variant) for quality in ("Strong", "Weak", "Medium")
             for variant in ("AdaptiveLUNA", "Equal")] + [("Strong", "InstanceOnly")]
    for dataset in args.datasets:
        for quality, variant in order:
            item = declared[(dataset, quality, variant)]
            if (Path(item["run_manifest"]).resolve() != (destination(dataset, quality, variant) / "journal_manifest.json").resolve()
                    or item["teacher_epoch"] != STAGES[quality]):
                raise ValueError("Continuation differs from predeclared condition")
    if args.validate_only:
        print(json.dumps({"status": "PASS_PREDECLARED_QUEUE", "datasets": args.datasets,
                          "conditions": len(args.datasets) * len(order), "gpu_launched": False}))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    events = ROOT / "logs/exp234_execution" / f"cohort_{stamp}.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    append_event(events, "START_COHORT", datasets=args.datasets, cohort=COHORT,
                 plan_sha256=sha256_file(plan_path), helper_sha256=sha256_file(__file__))
    try:
        for dataset in args.datasets:
            ensure_teacher(dataset, events, stamp)
            for quality, variant in order:
                append_event(events, "START_CONDITION", dataset=dataset, quality=quality, variant=variant)
                measured = train_condition(dataset, quality, variant)
                append_event(events, "COMPLETED_CONDITION", dataset=dataset, quality=quality, variant=variant,
                             Base=measured["Base"], Novel=measured["Novel"], HM=measured["HM"])
                if quality == "Strong" or variant == "AdaptiveLUNA":
                    record = evaluate_condition(dataset, quality, variant)
                    append_event(events, "COMPLETED_EVALUATION", dataset=dataset, quality=quality, variant=variant,
                                 teacher_accuracy=record["teacher_accuracy_micro_percent"],
                                 relation_metrics=record["relation_metrics"])
            publish(dataset, events, stamp)
        append_event(events, "COMPLETED_COHORT", datasets=args.datasets, ResultSource="REAL_EXPERIMENT")
    except BaseException as exc:
        append_event(events, "STOPPED_ERROR", error=repr(exc), traceback=traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
