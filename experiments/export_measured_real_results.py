"""Export verified real sanity and completed Exp1 runs; never import simulation.

Missing Exp2-4 measurements remain blank. Every overwritten report is copied to
a timestamped audit snapshot first. --validate-only reads and verifies inputs
without changing reports. Run only when actual sanity/training artifacts exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.export_unavailable_real_results import META, NAMES, SCHEMAS, worksheet, write_csv
from experiments.journal_runtime import METRICS, WEIGHTS, protocol_fingerprint, require_sanity_pass, source_hashes
from experiments.run_journal_sanity import assess_sanity

SOURCE = "REAL_EXPERIMENT"
VARIANTS = {"reliability_adaptive": "AdaptiveLUNA", "static_mean": "StaticMean", "equal": "Equal"}
EXTRA = ["Seed", "RunManifest", "RunManifestSHA256", "StudentCheckpoint", "StudentCheckpointSHA256", "ProvenanceStatus"]
HISTORY = ["HistoricalStatus", "HistoricalReasonCategory", "HistoricalExactReason", "HistoricalEvidence", "ResolutionEvidence"]
COHORT_KEYS = ("seed", "protocol_fingerprint", "dataset_fingerprint", "initialization_sha256", "teacher_checkpoint_sha256", "teacher_clip_sha256", "source_hashes")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [{key: None if value == "" else value for key, value in row.items()} for row in csv.DictReader(handle)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-10)


def hm(base, novel):
    return 0.0 if base + novel == 0 else 2 * base * novel / (base + novel)


class InputVerifier:
    def __init__(self, sanity_path, allowed_stages=("strong",), allowed_variants=None):
        self.allowed_stages = set(allowed_stages)
        self.allowed_variants = set(allowed_variants or VARIANTS)
        self.sanity_path = Path(sanity_path).resolve()
        self.current_hashes = source_hashes()
        self.files = {}
        self.manifests = {}
        self.teacher_proofs = set()
        self.recovery_evidence = {}
        self.checks = []
        self.sanity = require_sanity_pass(str(self.sanity_path))
        recomputed, deviations, means = assess_sanity(self.sanity["records"])
        require(recomputed == self.sanity["checks"] and all(recomputed.values()), "Recomputed real sanity checks do not match the passed report")
        for key in METRICS:
            require(close(means[key], self.sanity["means"][key]), f"Sanity mean mismatch: {key}")
        for key in WEIGHTS:
            require(close(deviations[key], self.sanity["lambda_std"][key]), f"Sanity std mismatch: {key}")
        self.sanity_means, self.sanity_std = means, deviations
        self.file_hash(self.sanity_path)
        self.checks.append("Real DTD sanity gate, all batch diagnostics and unweighted population means/std revalidated")
        if self.sanity.get("run_dir"):
            run_manifest = Path(self.sanity["run_dir"]) / "journal_manifest.json"
            evidence = read_json(run_manifest)
            require(evidence.get("status") == "COMPLETED_SANITY", "Sanity run manifest is not COMPLETED_SANITY")
            require(evidence.get("source_hashes") == self.sanity["source_hashes"], "Sanity report/run source hashes differ")
            for key in ("Base", "Novel", "HM"):
                require(evidence.get(key) is None, "Sanity must not supply experiment accuracies")
            self.verify_teacher(evidence)
            self.file_hash(run_manifest)
            self.checks.append("Sanity completed-run manifest and teacher asset fingerprints verified")

    def file_hash(self, path):
        path = Path(path).expanduser().resolve()
        require(path.is_file(), f"Provenance file is missing: {path}")
        if str(path) not in self.files:
            self.files[str(path)] = sha256(path)
        return self.files[str(path)]

    def verify_teacher(self, manifest):
        config = manifest["configuration"]
        teacher = Path(manifest["teacher_checkpoint"]).resolve()
        require("best" not in teacher.name.lower(), "Best-selected teacher is not allowed")
        require(self.file_hash(teacher) == manifest["teacher_checkpoint_sha256"], "Teacher checkpoint fingerprint mismatch")
        foundation = config["TRAINER"]["LUNA"]["TEACHER_CLIP_PATH"]
        require(self.file_hash(foundation) == manifest["teacher_clip_sha256"], "Teacher CLIP fingerprint mismatch")
        require(manifest.get("teacher_quality") in self.allowed_stages, "Teacher quality is outside the declared export scope")
        # A locally trained official teacher has an additional, stronger receipt.
        # Check it when present; external released teachers are already checked by
        # the actual LUNAJournal loader recorded by the passed sanity/run source.
        proof_path = teacher.parent.parent / "teacher_provenance.json"
        proof_key = (str(proof_path), str(teacher))
        if proof_path.is_file() and proof_key not in self.teacher_proofs:
            import yaml

            proof = read_json(proof_path)
            require(proof.get("status") == "COMPLETED_REAL_TEACHER_TRAINING", "Local teacher provenance does not record completed real training")
            teacher_config = yaml.safe_load(proof["config_yaml"])
            total = teacher_config["OPTIM"]["MAX_EPOCH"]
            require(total == config["JOURNAL"]["TEACHER_TOTAL_EPOCHS"], "Teacher total epochs disagree with the actual training configuration")
            require(teacher_config["TEST"]["NO_TEST"] is True and proof.get("accuracy_status") == "not_evaluated_for_selection", "Teacher provenance lacks no-evaluation-selection evidence")
            require(teacher_config["SEED"] == manifest["seed"], "Teacher/student seed differs from the declared matched protocol")
            receipt_hashes = {str(Path(path).resolve()): digest for path, digest in proof["checkpoint_sha256"].items()}
            require(receipt_hashes.get(str(teacher)) == manifest["teacher_checkpoint_sha256"], "Teacher-training receipt and selected checkpoint fingerprints differ")
            stages = sorted(int(Path(path).name.rsplit("-", 1)[-1]) for path in receipt_hashes if Path(path).name.startswith("model.pth.tar-"))
            target = {"weak": .25 * total, "medium": .60 * total, "strong": total}[manifest["teacher_quality"]]
            selected_epoch = min(stages, key=lambda epoch: (abs(epoch - target), epoch))
            require(teacher.name == f"model.pth.tar-{selected_epoch}", "Teacher was not selected by the prespecified nearest-epoch schedule")
            require(manifest["teacher_quality"] != "strong" or selected_epoch == total, "Strong teacher must be the configured final epoch")
            require(proof["clip_sha256"] == manifest["teacher_clip_sha256"], "Teacher-training CLIP differs from the selected foundation")
            if proof.get("dataset_split"):
                require(self.file_hash(proof["dataset_split"]) == proof["split_sha256"], "Teacher training split fingerprint changed")
            else:
                require(proof.get("dataset_split_files") and proof.get("metadata_sha256"), "Multi-file teacher split provenance is missing")
                for split_path in proof["dataset_split_files"]:
                    require(self.file_hash(split_path) == proof["metadata_sha256"][split_path], "Teacher training split metadata changed")
            require(proof["total_training_batches"] == total * proof["batches_per_epoch"], "Teacher training batch/epoch receipt is inconsistent")
            self.file_hash(proof_path)
            self.teacher_proofs.add(proof_key)
            self.checks.append("Official locally trained teacher receipt: schedule-defined stage, matching epoch/seed/checkpoint/CLIP/split and no evaluation selection")

    def training_trace(self, path, manifest):
        trace = path.parent / "journal_batches.csv"
        rows = read_csv(trace)
        total_epochs = int(manifest["configuration"]["OPTIM"]["MAX_EPOCH"])
        require(rows, f"No real training batch trace: {trace}")
        groups = {}
        seen = set()
        for row in rows:
            epoch, batch = int(row["epoch"]), int(row["batch_index"])
            require((epoch, batch) not in seen, "Duplicate epoch/batch identity in training trace")
            seen.add((epoch, batch))
            require(row["dataset"] == manifest["dataset"] and row["variant"] == manifest["variant"] and int(row["seed"]) == manifest["seed"], "Training trace identity disagrees with manifest")
            count = int(row["n_samples"])
            require(count > 0, "Training trace contains an empty batch")
            for key in METRICS:
                require(math.isfinite(float(row[key])), f"Nonfinite training trace metric: {key}")
            for key in ("r_ins", "r_batch", "r_cls", *WEIGHTS):
                require(0 <= float(row[key]) <= 1, f"Out-of-range reliability/weight: {key}")
            require(abs(sum(float(row[key]) for key in WEIGHTS) - 1) < 1e-5, "Training lambda normalization failed")
            groups.setdefault(epoch, []).append(row)
        require(set(groups) == set(range(1, total_epochs + 1)), "Full contiguous epoch trace is required")
        epoch_summaries = {int(row["epoch"]): row for row in manifest["epochs"]}
        require(set(epoch_summaries) == set(groups), "Manifest epoch summary coverage differs from batch trace")
        batch_counts = {len(group) for group in groups.values()}
        require(len(batch_counts) == 1, "Completed training must include the same full batch count for every epoch")
        for epoch, group in groups.items():
            summary = epoch_summaries[epoch]
            require(summary["batch_count"] == len(group), "Manifest epoch batch count mismatch")
            require(summary["n_observations"] == sum(int(row["n_samples"]) for row in group), "Manifest epoch sample count mismatch")
        stats = manifest["statistics"]
        count = sum(int(row["n_samples"]) for row in rows)
        require(stats["batch_count"] == len(rows) and stats["n_observations"] == count, "Manifest overall counts mismatch")
        for key in METRICS:
            mean = sum(int(row["n_samples"]) * float(row[key]) for row in rows) / count
            require(close(mean, stats["mean_" + key]), f"Manifest trace mean mismatch: {key}")
        self.file_hash(trace)
        self.checks.append(f"{manifest['dataset']} {manifest['variant']}: full {total_epochs}-epoch, {len(rows)}-batch trace verified")

    def manifest(self, path):
        import torch

        path = Path(path).resolve()
        if str(path) in self.manifests:
            return self.manifests[str(path)]
        manifest = read_json(path)
        require(manifest.get("status") == "COMPLETED_TRAINING", f"Run is not COMPLETED_TRAINING: {path}")
        require(manifest.get("variant") in self.allowed_variants, f"Unsupported variant for this export: {path}")
        require(manifest.get("dataset") in ("DTD", "FGVCAircraft", "EuroSAT"), "Unknown dataset")
        require(manifest.get("selection_protocol") == "final_epoch" and manifest.get("test_used_for_selection") is False, "Missing final-epoch/no-test-selection provenance")
        require(manifest.get("source_hashes") == self.current_hashes, "Run source hashes differ from the current verified journal implementation")
        config = manifest["configuration"]
        require(config["JOURNAL"]["SANITY_BATCHES"] == 0, "Sanity training cannot provide Exp1 results")
        require(manifest["protocol_fingerprint"] == protocol_fingerprint(config), "Manifest protocol fingerprint mismatch")
        require(manifest.get("paired_views") is True and manifest.get("reliability_temperature") == 1.0 and manifest.get("loss_temperatures") == [1, 3, 5, 2, 6], "Journal fixed formulation protocol mismatch")
        require(Path(config["OUTPUT_DIR"]).resolve() == path.parent, "Run directory disagrees with configuration")
        gate_path = Path(config["JOURNAL"]["SANITY_REPORT"]).resolve()
        gate = require_sanity_pass(str(gate_path))
        require(gate["source_hashes"] == self.sanity["source_hashes"], "Training sanity gate used a different source revision")
        self.file_hash(gate_path)
        for metric in ("Base", "Novel", "HM"):
            require(finite(manifest.get(metric)) and 0 <= manifest[metric] <= 100, f"Missing/invalid measured {metric}: {path}")
        require(close(manifest["HM"], hm(manifest["Base"], manifest["Novel"])), "Measured Base/Novel/HM are inconsistent")
        final_epoch = int(config["OPTIM"]["MAX_EPOCH"])
        require(final_epoch > 0 and manifest.get("epoch") == final_epoch, "Completed manifest does not identify the configured final epoch")
        checkpoint = Path(manifest["student_checkpoint"]).resolve()
        require(checkpoint == path.parent / "VLPromptLearner" / f"model.pth.tar-{final_epoch}", "Student checkpoint is not the explicit final run checkpoint")
        require(checkpoint.is_relative_to(self.sanity_path.parent), "Full checkpoint deserialization is restricted to this locally produced analysis tree")
        student_hash = self.file_hash(checkpoint)
        self.verify_teacher(manifest)
        self.training_trace(path, manifest)
        # Dassl stores the live successor scheduler/optimizer inside its warmup
        # scheduler state, which torch 2.0 weights_only=True cannot reconstruct.
        # These are our newly produced local checkpoints: source revision,
        # configuration, complete trace and fixed final path are verified above.
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        require(payload.get("epoch") == final_epoch and isinstance(payload.get("state_dict"), dict) and payload["state_dict"], "Final student checkpoint epoch/state_dict mismatch")
        require(all(not isinstance(value, torch.Tensor) or torch.isfinite(value).all().item() for value in payload["state_dict"].values()), "Student checkpoint contains nonfinite tensors")
        del payload
        if manifest.get("final_evaluation_recovery"):
            recovery = manifest["final_evaluation_recovery"]
            require(recovery.get("status") == "COMPLETED_FINAL_EVALUATION_RECOVERY" and recovery.get("mode") == "recovered_eval_only", "Incomplete or unsupported final-evaluation recovery")
            require(recovery.get("training_restarted") is False and recovery.get("additional_optimizer_steps") == 0, "Recovery must not retrain the student")
            require(recovery["checkpoint_sha256_before"] == recovery["checkpoint_sha256_after"] == student_hash, "Recovery changed the final checkpoint")
            require(recovery["model_state_sha256_before"] == recovery["model_state_sha256_after"], "Recovery evaluation changed model state")
            require(recovery["protected_artifacts_sha256_before"] == recovery["protected_artifacts_sha256_after"], "Recovery modified protected training artifacts")
            require(recovery["core_source_hashes"] == manifest["source_hashes"], "Recovery changed source-hashed training code")
            require(recovery["training_batch_count"] == manifest["statistics"]["batch_count"], "Recovery training-batch count disagrees with the original trace")
            for metric in ("Base", "Novel", "HM"):
                require(close(recovery["metrics"][metric], manifest[metric]), "Recovered metrics differ from final manifest")
            report = Path(recovery["report_path"]).resolve()
            self.file_hash(report)
            external_receipt = read_json(report)
            require(all(external_receipt.get(key) == value for key, value in recovery.items() if key != "report_path"), "Recovery report differs from manifest provenance")
            for original, expected_hash in recovery["original_failure_artifacts_sha256"].items():
                require(self.file_hash(original) == expected_hash, "Preserved original failure evidence changed")
            self.recovery_evidence[str(path)] = recovery
            self.checks.append("Eval-only recovery verified: preserved failure evidence, zero extra optimizer steps, unchanged checkpoint/model/training artifacts/source, matching final metrics")
        if manifest["variant"] == "static_mean":
            weights_path = Path(config["JOURNAL"]["STATIC_WEIGHTS_FILE"]).resolve()
            weights = read_json(weights_path)
            require(weights.get("status") == "COMPLETED_TRAINING" and weights.get("source_mode") == "reliability_adaptive" and weights.get("data_usage") == "training_only", "StaticMean weights lack completed training-only adaptive provenance")
            adaptive_path = Path(weights["source_training_run"]) / "journal_manifest.json"
            adaptive, _, _ = self.manifest(adaptive_path)
            for key in COHORT_KEYS:
                require(weights.get(key) == manifest.get(key) == adaptive.get(key), f"StaticMean/adaptive identity differs: {key}")
            expected = [adaptive["statistics"]["mean_" + key] for key in WEIGHTS]
            normalized = [value / sum(expected) for value in expected]
            for key, value, applied in zip(WEIGHTS, normalized, config["JOURNAL"]["STATIC_WEIGHTS"]):
                require(close(weights["lambda_mean"][key], value) and close(applied, value), "StaticMean did not use the adaptive training mean")
            self.file_hash(weights_path)
            self.checks.append("StaticMean coefficients match the complete adaptive training-only means")
        manifest_hash = self.file_hash(path)
        result = (manifest, manifest_hash, student_hash)
        self.manifests[str(path)] = result
        self.checks.append(f"{manifest['dataset']} {manifest['variant']}: completed final checkpoint, finite metrics, teacher/source/protocol provenance verified")
        self.checks.append("Owned local Dassl checkpoint loaded with full scheduler metadata only after manifest/source/path/training-trace verification")
        return result


EXTENDED_COLUMNS = {}


def columns_for(sheet):
    return list(dict.fromkeys(SCHEMAS[sheet] + META + (EXTRA if sheet == "Exp1_Weighting" else []) + EXTENDED_COLUMNS.get(sheet, [])))


def load_tables(root):
    tables = {sheet: read_csv(root / "tables" / f"REAL_{NAMES.get(sheet, 'sanity')}.csv") for sheet in SCHEMAS}
    failures = read_csv(root / "tables" / "REAL_failure_log.csv")
    require(all(row.get("ResultSource") in (None, SOURCE) for rows in tables.values() for row in rows), "REAL input tables contain an invalid source label")
    return tables, failures


def refresh_tables(root, verifier, manifest_paths):
    tables, failures = load_tables(root)
    # A previous measured row is kept only if its original manifest passes again.
    for row in tables["Exp1_Weighting"]:
        if row.get("ResultSource") == SOURCE:
            require(row.get("RunManifest"), "Previously measured row has no manifest provenance")
            manifest_paths.add(Path(row["RunManifest"]).resolve())
    selected = {}
    for path in sorted(manifest_paths):
        manifest, manifest_hash, student_hash = verifier.manifest(path)
        key = manifest["dataset"], VARIANTS[manifest["variant"]]
        require(key not in selected or selected[key][0] == path, f"Multiple completed runs for {key}; choose an explicit protocol/seed set, not by accuracy")
        selected[key] = (path, manifest, manifest_hash, student_hash)
    for dataset in {key[0] for key in selected}:
        cohort = [entry[1] for key, entry in selected.items() if key[0] == dataset]
        for other in cohort[1:]:
            for key in COHORT_KEYS:
                require(other[key] == cohort[0][key], f"Exp1 controlled cohort mismatch for {dataset}: {key}")
    resolved = {}
    sanity_row = tables["Sanity"][0]
    resolved[sanity_row["FailureID"]] = str(verifier.sanity_path)
    sanity_row.update(Status="REAL", ResultSource=SOURCE, BatchCount=verifier.sanity["batch_count"], FailureID=None,
                      ReasonCategories=None, ExactReason=None, Evidence=f"{verifier.sanity_path}; sha256={verifier.file_hash(verifier.sanity_path)}; statistics=unweighted batch population mean/std")
    for key in ("r_ins", "r_batch", "r_cls"):
        sanity_row[key] = verifier.sanity_means[key]
    for key in WEIGHTS:
        sanity_row["mean_" + key] = verifier.sanity_means[key]
        sanity_row["std_" + key] = verifier.sanity_std[key]
    mapping = {"lambda_sum_approximately_one": "weights_sum_to_one", "student_prompt_gradients_positive": "student_prompt_gradients",
               "teacher_gradients_zero": "teacher_frozen_without_gradients", "adaptive_not_scalar_rescaled_legacy": "legacy_is_mathematically_different"}
    sanity_row.update({column: verifier.sanity["checks"][check] for column, check in mapping.items()})
    for row in tables["Exp1_Weighting"]:
        key = row["Dataset"], row["Variant"]
        if key not in selected:
            continue
        path, manifest, manifest_hash, student_hash = selected[key]
        if row.get("FailureID"):
            resolved[row["FailureID"]] = str(path)
        row.update({metric: manifest[metric] for metric in ("Base", "Novel", "HM")})
        row.update(Status="REAL", ResultSource=SOURCE, FailureID=None, ReasonCategories=None, ExactReason=None,
                   Evidence=str(path), Seed=manifest["seed"], RunManifest=str(path), RunManifestSHA256=manifest_hash,
                   StudentCheckpoint=manifest["student_checkpoint"], StudentCheckpointSHA256=student_hash,
                   ProvenanceStatus="VERIFIED_COMPLETED_FINAL_EPOCH_TRAINING")
    failure_by_id = {row["FailureID"]: row for row in failures}
    for failure in failures:
        for old, new in (("Status", "HistoricalStatus"), ("ReasonCategory", "HistoricalReasonCategory"), ("ExactReason", "HistoricalExactReason"), ("Evidence", "HistoricalEvidence")):
            if not failure.get(new):
                failure[new] = failure.get(old)
        if failure["FailureID"] in resolved:
            failure.update(Status="RESOLVED", ReasonCategory=None, ExactReason="Previously missing result is now covered by verified REAL experimental evidence.",
                           Evidence=resolved[failure["FailureID"]], ResolutionEvidence=resolved[failure["FailureID"]], MissingFields=None,
                           RequiredForRealResult="Resolved; retain provenance and the historical assessment below.")
    for sheet, rows in tables.items():
        for row in rows:
            if row["Status"] in ("REAL", "PARTIAL_REAL", "UNDEFINED_REAL") and row.get("ResultSource") == SOURCE:
                continue
            require(row.get("ResultSource") is None, "An unsupported measured row cannot be silently discarded")
            failure = failure_by_id[row["FailureID"]]
            if sheet == "Exp1_Weighting":
                reason = f"No validated COMPLETED_TRAINING journal manifest with final Base/Novel/HM was supplied for {row['Dataset']} {row['Variant']}."
                required = "Complete and evaluate the controlled final-epoch run, then export its journal_manifest.json. StaticMean additionally needs a validated completed AdaptiveLUNA training mean."
            elif sheet == "Exp2_Reliability":
                reason = "No deterministic checkpoint evaluation batch records and correctness correlations/quintiles were produced for this cell. Training/sanity reliabilities do not substitute for the requested held-out analysis."
                required = "Evaluate the actual AdaptiveLUNA checkpoint with fixed split/order/batching/seed; export genuine teacher batch correctness, rho, p and quantiles."
            elif sheet == "Exp3_TeacherQuality":
                reason = "The requested teacher-quality experiment has not been completed and evaluated. Exp1 Strong accuracies alone do not supply teacher accuracy or Weak/Medium runs; this exporter does not claim partial Exp3 completion."
                required = "Run and evaluate the schedule-defined Weak/Medium/Strong teacher comparisons; reuse the matching Strong Exp1 run and measure teacher accuracy."
            elif sheet == "Exp4_RelationPreservation":
                reason = "No matched final-checkpoint teacher/student probability evaluation was produced for the requested relation-preservation metrics. Exp1 HM alone does not supply D_ins/D_batch/D_cls."
                required = "Evaluate matched InstanceOnly/Equal/Adaptive checkpoints and compute the prescribed KL and normalized Frobenius distances on the same fixed split."
            else:
                continue
            row.update(Status="NOT_RUN", ResultSource=None, ReasonCategories="OTHER", ExactReason=reason)
            failure.update(Status="NOT_RUN", ReasonCategory="OTHER", ExactReason=reason, RequiredForRealResult=required)
    return tables, failures, selected


def escape_tex(value):
    translation = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(translation.get(char, char) for char in str(value))


def write_tex(path, sheet, rows):
    columns = SCHEMAS[sheet] + ["Status", "ResultSource", "FailureID"]
    lines = ["% REAL experimental results only. -- means no measurement; no imputation.", r"% Requires \usepackage{graphicx}.", r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
             r"\caption{REAL " + escape_tex(sheet) + ": verified measurements and explicit missing cells.}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline", " & ".join(map(escape_tex, columns)) + r" \\", r"\hline"]
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key)
            values.append("--" if value is None else escape_tex(f"{value:.8g}" if isinstance(value, float) else value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", "}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def failure_markdown(failures, snapshot):
    lines = ["# Failure analysis for REAL experimental results", "", "Current status is listed separately from the preserved historical asset assessment. **RESOLVED** entries are covered by authenticated measurements. **NOT_RUN** entries have blank metrics and no measured source label.", "",
             f"Reports immediately before this update are preserved at `{snapshot}`. Historical missing-asset statements describe the earlier search, not current asset availability.", ""]
    for fail in failures:
        lines.extend([f"## {fail['FailureID']}", "", f"Experiment: {fail['Experiment']}  ", f"Dataset: {fail['Dataset']}  ", f"Variant: {fail['Variant']}  ", f"Status: {fail['Status']}", "",
                      f"Reason category: {fail.get('ReasonCategory') or 'RESOLVED'}", "", f"Exact reason: {fail['ExactReason']}", "", f"Evidence: {fail['Evidence']}", "",
                      f"Paths searched: {fail['PathsSearched']}", "", f"What would be required to obtain the real result: {fail['RequiredForRealResult']}", "", f"Missing fields: {fail.get('MissingFields') or 'None; resolved'}", "",
                      f"Historical status: {fail.get('HistoricalStatus')}", "", f"Historical reason category: {fail.get('HistoricalReasonCategory')}", "", f"Historical exact reason: {fail.get('HistoricalExactReason')}", "", f"Historical evidence: {fail.get('HistoricalEvidence')}", ""])
    return "\n".join(lines)


def output_artifacts(stage, tables, failures, snapshot):
    from openpyxl import Workbook, load_workbook

    (stage / "tables").mkdir(parents=True, exist_ok=True)
    paths = []
    for sheet, rows in tables.items():
        stem = NAMES.get(sheet, "sanity")
        csv_path = stage / "tables" / f"REAL_{stem}.csv"
        write_csv(csv_path, rows, columns_for(sheet))
        paths.append(csv_path)
        if sheet != "Sanity":
            tex_path = csv_path.with_suffix(".tex")
            write_tex(tex_path, sheet, rows)
            paths.append(tex_path)
    failure_path = stage / "tables" / "REAL_failure_log.csv"
    fail_columns = list(dict.fromkeys([*failures[0], *HISTORY]))
    write_csv(failure_path, failures, fail_columns)
    paths.append(failure_path)
    analysis_path = stage / "FAILURE_ANALYSIS.md"
    analysis_path.write_text(failure_markdown(failures, snapshot), encoding="utf-8")
    paths.append(analysis_path)
    workbook = Workbook()
    workbook.remove(workbook.active)
    for sheet, rows in tables.items():
        ws = worksheet(workbook, sheet, f"REAL {sheet} - verified measured results / explicit unavailable cells", rows, columns_for(sheet))
        ws.cell(2, 1, "REAL_EXPERIMENT labels authenticated measurements only. Blank cells are pending or explicitly undefined; PARTIAL_REAL rows identify incomplete comparisons. Historical failures remain in Failure_Log.")
    worksheet(workbook, "Failure_Log", "REAL current execution status and preserved historical failure evidence", failures, fail_columns)
    workbook_path = stage / "Journal_LUNA_REAL_Results.xlsx"
    workbook.save(workbook_path)
    paths.append(workbook_path)
    restored = load_workbook(workbook_path, read_only=True, data_only=True)
    require(restored.sheetnames == [*SCHEMAS, "Failure_Log"], "REAL workbook sheet schema differs")
    for sheet, expected_rows in tables.items():
        stored_rows = list(restored[sheet].iter_rows(min_row=4, values_only=True))
        require(len(stored_rows) == len(expected_rows), "REAL workbook row count differs")
        for cells, expected in zip(stored_rows, expected_rows):
            for key, actual in zip(columns_for(sheet), cells):
                target = expected.get(key)
                require(close(actual, target) if finite(target) else actual == target, f"Workbook serialization mismatch: {sheet} {key}")
        csv_rows = read_csv(stage / "tables" / f"REAL_{NAMES.get(sheet, 'sanity')}.csv")
        require(len(csv_rows) == len(expected_rows), "REAL CSV row count differs")
        for actual, expected in zip(csv_rows, expected_rows):
            for key in columns_for(sheet):
                target = expected.get(key)
                require(actual[key] == (None if target is None else str(target)), f"CSV precision/value mismatch: {sheet} {key}")
    restored.close()
    return paths


def run(args):
    root = args.output_root.resolve()
    verifier = InputVerifier(args.sanity_report or root / "sanity_result.json")
    manifests = {path.resolve() for path in args.manifest}
    skipped = []
    for run_root in args.run_root:
        for path in run_root.resolve().rglob("journal_manifest.json"):
            try:
                record = read_json(path)
            except json.JSONDecodeError:
                skipped.append({"path": str(path), "status": "UNREADABLE_DURING_SCAN", "reason": "Manifest may be in the middle of an update; no result imported"})
                continue
            status = record.get("status")
            declared_output = record.get("configuration", {}).get("OUTPUT_DIR")
            if declared_output and Path(declared_output).resolve() != path.parent.resolve():
                skipped.append({"path": str(path), "status": status, "reason": "Preserved provenance copy outside its declared run directory; not an independent run"})
                continue
            if status == "COMPLETED_TRAINING":
                manifests.add(path.resolve())
            else:
                skipped.append({"path": str(path), "status": status, "reason": "No completed training result"})
    tables, failures, selected = refresh_tables(root, verifier, manifests)
    measured_rows = {sheet: sum(row["Status"] == "REAL" for row in rows) for sheet, rows in tables.items()}
    for sheet in ("Exp2_Reliability", "Exp3_TeacherQuality", "Exp4_RelationPreservation"):
        require(measured_rows[sheet] == 0, "This exporter must not fabricate Exp2-4 measurements")
    summary = {"status": "PASS", "generated_utc": datetime.now(timezone.utc).isoformat(), "real_sanity_batches": verifier.sanity["batch_count"],
               "measured_rows": measured_rows, "resolved_failure_entries": sum(row["Status"] == "RESOLVED" for row in failures),
               "checks": verifier.checks, "input_sha256": verifier.files, "skipped_incomplete_manifests": skipped,
               "final_evaluation_recovery": verifier.recovery_evidence,
               "recovered_run_count": len(verifier.recovery_evidence),
               "hypothetical_artifacts_read_or_written": 0, "selection": "Explicit final epochs; no accuracy-based selection"}
    if args.validate_only:
        summary["mode"] = "VALIDATION_ONLY_NO_REPORT_WRITES"
        print(json.dumps(summary, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = root / "logs" / "real_export_snapshots" / stamp
    stage = root / "logs" / "real_export_staging" / stamp
    require(snapshot.resolve().is_relative_to(root) and stage.resolve().is_relative_to(root), "Audit output must remain inside the analysis root")
    snapshot.mkdir(parents=True)
    old_files = list((root / "tables").glob("REAL_*")) + [root / "Journal_LUNA_REAL_Results.xlsx", root / "FAILURE_ANALYSIS.md", root / "logs" / "real_export_validation.json"]
    snapshot_manifest = []
    for path in old_files:
        if path.is_file():
            target = snapshot / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            snapshot_manifest.append({"original": str(path), "snapshot": str(target), "sha256": sha256(target)})
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(snapshot_manifest, indent=2), encoding="utf-8")
    paths = output_artifacts(stage, tables, failures, snapshot)
    summary.update(snapshot=str(snapshot), staging=str(stage), workbook_sheets=[*SCHEMAS, "Failure_Log"], serialized_outputs_validated=True)
    for path in paths:
        target = root / path.relative_to(stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    summary["output_sha256"] = {str(root / path.relative_to(stage)): sha256(path) for path in paths}
    (root / "logs" / "real_export_validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("input_sha256", "output_sha256", "final_evaluation_recovery")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    parser.add_argument("--sanity-report", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--run-root", type=Path, action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
