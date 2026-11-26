"""Incremental, provenance-checked REAL Exp1-4 export; no training or simulation.

Only predeclared completed training conditions and authenticated fixed-checkpoint
evaluations are imported. Partial measured comparisons retain their available
cells and mark missing cells explicitly; an earlier measurement cannot disappear.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import export_measured_real_results as base

SOURCE = base.SOURCE
MODES = {**base.VARIANTS, "instance_only": "InstanceOnly"}
STAGES = {"Weak": 5, "Medium": 12, "Strong": 20}
COMMON = ["Seed", "MeasurementEvidence", "EvaluationManifest", "EvaluationManifestSHA256",
          "AnalyzedSampleHash", "EvaluationProtocol", "N_samples", "TeacherAccuracyScope",
          "UndefinedReason"]
for sheet in ("Exp2_Reliability", "Exp3_TeacherQuality", "Exp4_RelationPreservation"):
    base.EXTENDED_COLUMNS[sheet] = list(COMMON)
base.EXTENDED_COLUMNS["Exp3_TeacherQuality"] += ["EqualRunManifest", "AdaptiveRunManifest", "TeacherAccuracyEvaluationManifest"]
base.EXTENDED_COLUMNS["Exp4_RelationPreservation"] += ["RunManifest", "RunManifestSHA256", "StudentCheckpoint", "StudentCheckpointSHA256"]
METRIC_COLUMNS = {
    "Exp1_Weighting": ["Base", "Novel", "HM"],
    "Exp2_Reliability": ["rho", "p_value", "N_batches", "teacher_batch_accuracy"],
    "Exp3_TeacherQuality": ["TeacherAccuracy", "Equal_Base", "Equal_Novel", "Equal_HM", "Adaptive_Base", "Adaptive_Novel", "Adaptive_HM", "HM_Gain", "Drop_equal", "Drop_adaptive"],
    "Exp4_RelationPreservation": ["D_ins", "D_batch", "D_cls", "HM"],
}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def key_for(row):
    return row["Dataset"], row["Variant"]


def normalized_student_config(manifest):
    config = copy.deepcopy(manifest["configuration"])
    for key in ("OUTPUT_DIR", "RESUME"):
        config.pop(key, None)
    config["DATASET"].pop("FEWSHOT_CACHE_ROOT", None)
    config["DATASET"]["ROOT"] = str(Path(config["DATASET"]["ROOT"]).resolve()).casefold()
    journal = config["JOURNAL"]
    for key in ("MODE", "STATIC_WEIGHTS", "STATIC_WEIGHTS_FILE", "OUTPUT_ROOT", "SANITY_BATCHES",
                "SANITY_REPORT", "LOG_BATCH_SAMPLES", "TEACHER_CHECKPOINT", "TEACHER_QUALITY"):
        journal.pop(key, None)
    return config


def training_evidence(path, verified):
    manifest, manifest_hash, checkpoint_hash = verified
    recovery = manifest.get("final_evaluation_recovery")
    recovery_summary = ({key: recovery[key] for key in ("status", "report_path", "training_restarted", "additional_optimizer_steps", "original_error", "checkpoint_sha256_before", "checkpoint_sha256_after")}
                        if recovery else None)
    return {"kind": "COMPLETED_TRAINING", "path": str(path), "sha256": manifest_hash,
            "student_checkpoint": manifest["student_checkpoint"], "student_checkpoint_sha256": checkpoint_hash,
            "teacher_checkpoint": manifest["teacher_checkpoint"], "teacher_checkpoint_sha256": manifest["teacher_checkpoint_sha256"],
            "teacher_quality": manifest["teacher_quality"], "seed": manifest["seed"],
            "protocol_fingerprint": manifest["protocol_fingerprint"], "dataset_fingerprint": manifest["dataset_fingerprint"],
            "final_evaluation_recovery": recovery_summary}


def supplemental_conditions(root, verifier, primary_path, primary):
    """Read the fixed optional Exp1 completion plan without changing the 21-run plan."""
    path = root / "logs/exp1_static_mean_supplemental_predeclared_plan_20260906.json"
    if not path.is_file():
        return []
    plan = base.read_json(path)
    expected = {"status": "SUPPLEMENTAL_PREDECLARED_NOT_EXECUTED", "seed": 1, "student_epochs": 5,
                "student_batch_size": 8, "teacher_quality": "Strong", "teacher_epoch": 20, "teacher_total_epochs": 20,
                "selection_protocol": "final_epoch", "existing_equal_policy": "reuse_only_never_retrain",
                "selection_by_target_results": False, "heldout_used_for_weights_or_selection": False,
                "original_plan_modified": False, "core_source_sha256": verifier.current_hashes}
    for key, value in expected.items():
        base.require(plan.get(key) == value, f"Supplemental StaticMean declaration mismatch: {key}")
    base.require(Path(plan["original_plan"]).resolve() == Path(primary_path).resolve()
                 and plan["original_plan_sha256"] == verifier.file_hash(primary_path), "Supplemental plan does not reference the unchanged original plan")
    base.require(plan["cohort"] == primary["cohort"] and plan["execution_order"] == ["FGVCAircraft", "EuroSAT"], "Supplemental cohort/order differs")
    conditions = plan["conditions"]
    base.require(len(conditions) == 2 and {(row["dataset"], row["quality"], row["variant"]) for row in conditions}
                 == {("FGVCAircraft", "Strong", "StaticMean"), ("EuroSAT", "Strong", "StaticMean")}, "Supplemental plan must contain exactly the two declared StaticMean identities")
    original = {(row["dataset"], row["quality"], row["variant"]): Path(row["run_manifest"]).resolve() for row in primary["conditions"]}
    result = []
    for condition in conditions:
        dataset = condition["dataset"]
        adaptive, equal = original[dataset, "Strong", "AdaptiveLUNA"], original[dataset, "Strong", "Equal"]
        expected_run = adaptive.parent.parent / "StaticMean/journal_manifest.json"
        base.require(Path(condition["run_manifest"]).resolve() == expected_run, "Supplemental target run path differs from the fixed Exp1 cohort")
        base.require(Path(condition["adaptive_source_manifest"]).resolve() == adaptive and Path(condition["equal_reused_manifest"]).resolve() == equal, "Supplemental Adaptive/Equal references differ from the original declaration")
        base.require(Path(condition["static_weights_file"]).resolve() == adaptive.parent / "training_mean_weights.json", "Supplemental static weights do not name the matching Adaptive training mean")
        base.require(condition["teacher_epoch"] == 20 and condition["target_student_outcomes_present_at_declaration"] is False, "Supplemental teacher stage or pre-outcome declaration differs")
        result.append({**condition, "plan_scope": "SUPPLEMENTAL_EXP1_STATICMEAN", "declaration_path": str(path)})
    verifier.file_hash(path)
    verifier.checks.append("Optional supplemental plan verified: exactly two predeclared Strong StaticMean identities, seed1/epoch5/batch8, unchanged original plan/core hashes and matching Adaptive/Equal paths")
    return result


def collect_training(root, verifier, plan_path, old_tables):
    plan = base.read_json(plan_path)
    base.require(plan.get("seed") == 1 and plan.get("student_epochs") == 5 and plan.get("student_batch_size") == 8, "Unexpected predeclared seed/student schedule")
    base.require(plan.get("teacher_stages") == STAGES, "Unexpected teacher stage schedule")
    base.require(plan.get("core_source_sha256") == verifier.current_hashes, "Predeclared core source revision changed")
    verifier.file_hash(plan_path)
    by_path, selected, pending = {}, {}, []
    declared_conditions = [{**row, "plan_scope": "ORIGINAL_EXP234"} for row in plan["conditions"]]
    declared_conditions += supplemental_conditions(root, verifier, plan_path, plan)
    for condition in declared_conditions:
        path = Path(condition["run_manifest"]).resolve()
        identity = condition["dataset"], condition["quality"], condition["variant"]
        base.require(identity not in selected and str(path) not in by_path, "Duplicated predeclared condition")
        by_path[str(path)] = condition
        if not path.is_file():
            pending.append({**condition, "current_status": "QUEUED"})
            continue
        try:
            raw = base.read_json(path)
        except json.JSONDecodeError:
            pending.append({**condition, "current_status": "UPDATING_MANIFEST"})
            continue
        if raw.get("status") != "COMPLETED_TRAINING":
            pending.append({**condition, "current_status": raw.get("status")})
            continue
        verified = verifier.manifest(path)
        m = verified[0]
        base.require((m["dataset"], m["teacher_quality"].title(), MODES[m["variant"]]) == identity, "Completed run differs from predeclared condition")
        base.require(m["seed"] == plan["seed"] and m["epoch"] == plan["student_epochs"], "Run seed/epoch differs from plan")
        base.require(int(Path(m["teacher_checkpoint"]).name.rsplit("-", 1)[-1]) == condition["teacher_epoch"] == STAGES[condition["quality"]], "Teacher stage differs from plan")
        base.require(m["configuration"]["DATALOADER"]["TRAIN_X"]["BATCH_SIZE"] == plan["student_batch_size"], "Student batch size differs from plan")
        if condition["plan_scope"] == "SUPPLEMENTAL_EXP1_STATICMEAN":
            base.require(all((row["dataset"], row["quality"], row["variant"]) in selected for row in plan["conditions"]), "Supplemental result precedes completion of the original 21 conditions")
            base.require(Path(m["configuration"]["JOURNAL"]["STATIC_WEIGHTS_FILE"]).resolve() == Path(condition["static_weights_file"]).resolve(), "Measured StaticMean used a different weight source")
            base.require(Path(m["teacher_checkpoint"]).resolve() == Path(condition["teacher_checkpoint"]).resolve(), "Supplemental StaticMean teacher checkpoint differs from declaration")
            weights = base.read_json(condition["static_weights_file"])
            base.require(Path(weights["source_training_run"]).resolve() == Path(condition["adaptive_source_manifest"]).parent.resolve(), "Supplemental weights came from another Adaptive run")
        selected[identity] = (path, verified)
    # Older DTD StaticMean is not an Exp2-4 condition but must remain verified.
    exp1_paths = {entry[0] for identity, entry in selected.items() if identity[1] == "Strong" and identity[2] in base.VARIANTS.values()}
    for row in old_tables["Exp1_Weighting"]:
        if row.get("ResultSource") == SOURCE:
            if row["Variant"] == "StaticMean" and row["Dataset"] != "DTD":
                base.require(str(Path(row["RunManifest"]).resolve()) in by_path, "Previously measured supplemental StaticMean has lost its predeclaration")
            exp1_paths.add(Path(row["RunManifest"]).resolve())
    for dataset in {identity[0] for identity in selected}:
        cohort = [entry[1][0] for identity, entry in selected.items() if identity[0] == dataset]
        first = cohort[0]
        for other in cohort[1:]:
            for key in ("seed", "dataset_fingerprint", "initialization_sha256", "teacher_clip_sha256", "source_hashes"):
                base.require(first[key] == other[key], f"Across-stage controlled identity mismatch: {dataset} {key}")
            base.require(normalized_student_config(first) == normalized_student_config(other), f"Across-stage student configuration mismatch: {dataset}")
        for quality in STAGES:
            pair = [entry[1][0] for identity, entry in selected.items() if identity[0] == dataset and identity[1] == quality]
            for other in pair[1:]:
                for key in base.COHORT_KEYS:
                    base.require(pair[0][key] == other[key], f"Within-stage cohort mismatch: {dataset} {quality} {key}")
    return selected, exp1_paths, pending, by_path


def publish_row(row, updates, evidence, required):
    row.update(updates)
    merged = json.loads(row.get("MeasurementEvidence") or "{}")
    merged.update(evidence)
    row["MeasurementEvidence"] = encoded(merged)
    missing = [key for key in required if row.get(key) is None]
    row.update(ResultSource=SOURCE, Status="PARTIAL_REAL" if missing else "REAL",
               ReasonCategories="PENDING_MEASUREMENTS" if missing else None,
               ExactReason=("Pending authenticated measurements: " + ", ".join(missing)) if missing else None,
               Evidence="; ".join(sorted({value["path"] for value in merged.values()})))


def update_quality(tables, selected, evaluations):
    for row in tables["Exp3_TeacherQuality"]:
        dataset, quality = row["Dataset"], row["TeacherQuality"]
        updates, evidence = {}, {}
        for variant, prefix in (("Equal", "Equal"), ("AdaptiveLUNA", "Adaptive")):
            entry = selected.get((dataset, quality, variant))
            if entry:
                path, verified = entry
                updates.update({prefix + "_" + metric: verified[0][metric] for metric in ("Base", "Novel", "HM")})
                updates[prefix + "RunManifest"] = str(path)
                updates.update(TeacherEpoch=STAGES[quality], Seed=verified[0]["seed"])
                evidence[prefix] = training_evidence(path, verified)
        stage_evals = [entry for identity, entry in evaluations.items() if identity[:2] == (dataset, quality)]
        if stage_evals:
            evaluation = stage_evals[0]
            for other in stage_evals[1:]:
                base.require(evaluation["manifest"]["analyzed_sample_hash"] == other["manifest"]["analyzed_sample_hash"], "Teacher accuracy evaluated on differing stage subsets")
                base.require(base.close(evaluation["manifest"]["teacher_accuracy_micro_percent"], other["manifest"]["teacher_accuracy_micro_percent"]), "Same teacher produces conflicting measured accuracies")
            updates.update(TeacherAccuracy=evaluation["manifest"]["teacher_accuracy_micro_percent"],
                           TeacherAccuracyEvaluationManifest=str(evaluation["path"]),
                           TeacherAccuracyScope=evaluation["accuracy_scope"],
                           AnalyzedSampleHash=evaluation["manifest"]["analyzed_sample_hash"], N_samples=evaluation["manifest"]["N_samples"])
            evidence["TeacherAccuracy"] = evaluation["evidence"]
        if not evidence:
            continue
        available = {**row, **updates}
        if available.get("Equal_HM") is not None and available.get("Adaptive_HM") is not None:
            updates["HM_Gain"] = float(available["Adaptive_HM"]) - float(available["Equal_HM"])
        for variant, column in (("Equal", "Drop_equal"), ("AdaptiveLUNA", "Drop_adaptive")):
            strong, weak = selected.get((dataset, "Strong", variant)), selected.get((dataset, "Weak", variant))
            if strong and weak:
                updates[column] = strong[1][0]["HM"] - weak[1][0]["HM"]
                evidence[column + "_Strong"] = training_evidence(*strong)
                evidence[column + "_Weak"] = training_evidence(*weak)
        if evidence:
            publish_row(row, updates, evidence, METRIC_COLUMNS["Exp3_TeacherQuality"])


def resolve_failures(tables, failures):
    failure_map = {row["FailureID"]: row for row in failures}
    for sheet, rows in tables.items():
        for row in rows:
            if row.get("ResultSource") != SOURCE or not row.get("FailureID"):
                continue
            failure = failure_map[row["FailureID"]]
            if row["Status"] in ("REAL", "UNDEFINED_REAL"):
                failure.update(Status="RESOLVED", ReasonCategory=None,
                               ExactReason="Previously unavailable cell now has authenticated measured evidence; undefined statistics are explicitly identified.",
                               Evidence=row["Evidence"], ResolutionEvidence=row["Evidence"], MissingFields=None,
                               RequiredForRealResult="Resolved; retain original audit and measured provenance.")
                row["FailureID"] = None
            else:
                missing = [key for key in METRIC_COLUMNS.get(sheet, []) if row.get(key) is None]
                failure.update(Status="PARTIAL_REAL", ReasonCategory="PENDING_MEASUREMENTS", ExactReason=row["ExactReason"],
                               Evidence=row["Evidence"], MissingFields=", ".join(missing),
                               RequiredForRealResult="Complete the remaining predeclared runs/evaluations; retain already measured cells.")


def assert_preserved(old, new):
    for sheet, rows in old.items():
        current = {key_for(row): row for row in new[sheet]}
        for row in rows:
            if row.get("ResultSource") != SOURCE:
                continue
            target = current[key_for(row)]
            base.require(target.get("ResultSource") == SOURCE, "Earlier measured source was removed")
            for key in METRIC_COLUMNS.get(sheet, []):
                if row.get(key) is not None:
                    same = (target.get(key) is not None and
                            (base.math.isclose(float(row[key]), float(target[key]), rel_tol=1e-9, abs_tol=0)
                             if key == "p_value" else base.close(row[key], target[key])))
                    base.require(same, f"Earlier measurement changed/disappeared: {sheet} {key_for(row)} {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--evaluation-root", type=Path, action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    verifier = base.InputVerifier(root / "sanity_result.json", (key.lower() for key in STAGES), MODES)
    old, _ = base.load_tables(root)
    selected, exp1_paths, pending, declared = collect_training(root, verifier, args.plan or root / "logs/exp234_predeclared_plan_20260906.json", old)
    tables, failures, _ = base.refresh_tables(root, verifier, exp1_paths)
    from experiments.verify_measured_checkpoint_evaluation import collect_evaluations, update_evaluation_tables
    evaluations = collect_evaluations(root, verifier, selected, old, args.evaluation_root)
    update_quality(tables, selected, evaluations)
    update_evaluation_tables(tables, selected, evaluations)
    resolve_failures(tables, failures)
    assert_preserved(old, tables)
    summary = {"status": "PASS", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "real_sanity_batches": verifier.sanity["batch_count"],
               "measured_rows": {sheet: sum(row["Status"] in ("REAL", "UNDEFINED_REAL") for row in rows) for sheet, rows in tables.items()},
               "partial_measured_rows": {sheet: sum(row["Status"] == "PARTIAL_REAL" for row in rows) for sheet, rows in tables.items()},
               "resolved_failure_entries": sum(row["Status"] == "RESOLVED" for row in failures),
               "completed_training_conditions": len(selected), "completed_evaluations": len(evaluations), "pending_conditions": pending,
               "completed_original_training_conditions": sum(identity[2] != "StaticMean" for identity in selected),
               "completed_supplemental_training_conditions": sum(identity[2] == "StaticMean" for identity in selected),
               "checks": verifier.checks, "input_sha256": verifier.files,
               "verified_training_manifests": {path: training_evidence(Path(path), value) for path, value in verifier.manifests.items()},
               "final_evaluation_recovery": verifier.recovery_evidence, "recovered_run_count": len(verifier.recovery_evidence),
               "hypothetical_artifacts_read_or_written": 0, "previous_measured_cells_preserved": True,
               "selection": "Predeclared schedule and final checkpoints; signed effects retained without monotonicity constraints"}
    if args.validate_only:
        print(encoded({**summary, "mode": "VALIDATION_ONLY_NO_REPORT_WRITES"}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = root / "logs/real_export_snapshots" / stamp
    stage = root / "logs/real_export_staging" / stamp
    snapshot.mkdir(parents=True)
    old_paths = list((root / "tables").glob("REAL_*")) + [root / "Journal_LUNA_REAL_Results.xlsx", root / "FAILURE_ANALYSIS.md", root / "logs/real_export_validation.json"]
    snapshot_rows = []
    for path in old_paths:
        if path.is_file():
            target = snapshot / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            snapshot_rows.append({"original": str(path), "snapshot": str(target), "sha256": base.sha256(target)})
    (snapshot / "snapshot_manifest.json").write_text(encoded(snapshot_rows), encoding="utf-8")
    paths = base.output_artifacts(stage, tables, failures, snapshot)
    summary.update(snapshot=str(snapshot), staging=str(stage), workbook_sheets=[*base.SCHEMAS, "Failure_Log"], serialized_outputs_validated=True)
    for path in paths:
        target = root / path.relative_to(stage)
        shutil.copy2(path, target)
    summary["output_sha256"] = {str(root / path.relative_to(stage)): base.sha256(path) for path in paths}
    (root / "logs/real_export_validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(encoded({key: summary[key] for key in ("status", "measured_rows", "partial_measured_rows", "completed_training_conditions", "completed_evaluations", "previous_measured_cells_preserved", "serialized_outputs_validated")}))


if __name__ == "__main__":
    main()
