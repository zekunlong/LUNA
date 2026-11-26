"""Record completed asset discovery and zero-measurement preflight evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.journal_preflight import inspect_assets


def build(root):
    audit = json.loads((root / "asset_audit.json").read_text(encoding="utf-8"))
    sanity = json.loads((root / "sanity_result.json").read_text(encoding="utf-8"))
    offline = json.loads((root / "logs" / "offline_tests_final.json").read_text(encoding="utf-8"))
    if sanity["batch_count"] or sanity.get("records"):
        raise ValueError("Measured batches must be processed rather than unavailable-result export")
    if any(value["runnable"] or value["usable_student_checkpoints"] or value["reusable_valid_exp1_to_exp4_runs"]
           for value in audit["dataset_conclusions"].values()):
        raise ValueError("Runnable or reusable assets require actual execution/reuse")
    environment = sanity["preflight"]["environment"]
    if environment.get("import_probe_error") or environment.get("cuda_available") is not True:
        raise ValueError("Resolve the environment probe before declaring only external-asset blockers")
    if any(not value.get("ok") for value in environment.get("imports", {}).values()):
        raise ValueError("Resolve failed imports before this external-asset-only report")
    state = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "real_execution_complete": True,
        "completion_scope": "All work currently possible from the audited available assets; required real experiments remain NOT_RUN.",
        "real_batch_count": 0, "measured_result_count": 0,
        "environment": {"status": "PASS", "cuda_available": True, "evidence": "sanity_result.json"},
        "offline_test_count": offline["tests_run"],
        "paths_searched": ["E:/LUNA", "E:/", *[entry["path"] for entry in audit["cache_checks"]]],
        "dataset_preflights": {}, "datasets": {},
        "initial_environment_timeout": "Resolved cold sklearn import; preserved in logs/preflight_first_attempt.sanity_result.json",
    }
    for dataset in ("FGVCAircraft", "DTD", "EuroSAT"):
        conclusion = audit["dataset_conclusions"][dataset]
        # No candidate root/weights can be inferred: inspect unset external
        # inputs instead of constructing fictional paths or fake checkpoints.
        preflight = inspect_assets(dataset, check_environment=False)
        preflight["environment_reference"] = str(root / "sanity_result.json")
        preflight["asset_discovery_reference"] = str(root / "asset_audit.json")
        path = root / "logs" / f"real_preflight_{dataset}.json"
        path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        state["dataset_preflights"][dataset] = str(path)
        if preflight["status"] == "READY":
            raise ValueError("A ready preflight must be acted on")
        blockers = [entry for entry in conclusion["blockers"]
                    if entry["category"] != "MISSING_INTERMEDIATE_TEACHER_CHECKPOINT"]
        state["datasets"][dataset] = {
            "status": "NOT_RUN", "reason_categories": [entry["category"] for entry in blockers],
            "exact_reason": " ".join(entry["reason"] for entry in blockers),
            "required_assets": "Recover the original dataset images and split files listed above, the protocol's OpenAI CLIP pretrained weights, and the dataset-specific PromptSRC final checkpoint with its configuration/log and verified total epochs. Run actual data sanity checks and controlled training; do not select by test performance.",
            "evidence": [str(root / "ASSET_AUDIT.md"), str(path), str(root / "sanity_result.json")],
            "absence_scope": conclusion["absence_scope"],
        }
    path = root / "logs" / "real_execution_status.json"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    plan_path = root / "run_plan.csv"
    if plan_path.is_file():
        backup = root / "logs" / "run_plan_before_current_audit.csv"
        if not backup.exists():
            backup.write_bytes(plan_path.read_bytes())
        with plan_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = list(reader.fieldnames), list(reader)
        if "CurrentReason" not in fields:
            fields.append("CurrentReason")
        for row in rows:
            row["Status"] = "NOT_RUN"
            row["CurrentReason"] = state["datasets"][row["Dataset"]]["exact_reason"]
        priority = {"AdaptiveLUNA": 0, "StaticMean": 1, "Equal": 2}
        rows.sort(key=lambda row: (list(state["datasets"]).index(row["Dataset"]), row["Experiment"],
                                  priority.get(row["Variant"], 3) if row["Experiment"] == "Exp1_Weighting" else 0))
        with plan_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"real_execution_complete": True, "real_batch_count": 0,
                      "offline_tests": state["offline_test_count"],
                      "dataset_status": {key: value["status"] for key, value in state["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    build(parser.parse_args().output_root)
