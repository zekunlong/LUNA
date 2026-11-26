"""Read-only CPU validation of actual held-out checkpoint evaluation artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments import export_measured_real_results as base

KEYS = ("r_ins", "r_batch", "r_cls")
DISTANCES = ("D_ins", "D_batch", "D_cls")


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_summary(manifest):
    """Keep workbook cells compact; full sample lists stay in the hashed receipt."""
    protocol = manifest["protocol"]
    keys = ("protocol_id", "split", "prediction_space", "sample_order", "batch_size", "shuffle", "drop_last",
            "evaluation_seed", "N_original_test", "N_analyzed", "N_discarded", "temperature", "distance_formulas",
            "distance_units", "distance_aggregation", "spearman", "quintiles", "inferential_limitations")
    return json.dumps({"protocol_fingerprint": manifest["protocol_fingerprint"], **{key: protocol[key] for key in keys}}, sort_keys=True)


def numeric_close(left, right):
    # Independent float64 CPU reference compared with measured float32 GPU ops.
    return math.isclose(float(left), float(right), rel_tol=5e-5, abs_tol=3e-6)


def independent_metrics(log_t, log_s, log_a):
    import numpy as np

    pt, ps, pa = np.exp(log_t.astype(np.float64)), np.exp(log_s.astype(np.float64)), np.exp(log_a.astype(np.float64))
    entropy_t = -(pt * np.log(pt + 1e-8)).sum(1).mean()
    entropy_a = -(pa * np.log(pa + 1e-8)).sum(1).mean()
    tb, sb, ab = pt @ pt.T, ps @ ps.T, pa @ pa.T
    tc, sc, ac = pt.T @ pt, ps.T @ ps, pa.T @ pa
    cosine = lambda x, y: float(np.clip(np.dot(x.ravel(), y.ravel()) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8), 0, 1))
    return {"r_ins": float(np.clip(1 - .5 * (entropy_t + entropy_a) / math.log(pt.shape[1]), 0, 1)),
            "r_batch": cosine(tb, ab), "r_cls": cosine(tc, ac),
            "D_ins": float((pt * (log_t.astype(np.float64) - log_s.astype(np.float64))).sum(1).mean()),
            "D_batch": float(np.linalg.norm(tb - sb) / (np.linalg.norm(tb) + 1e-8)),
            "D_cls": float(np.linalg.norm(tc - sc) / (np.linalg.norm(tc) + 1e-8))}


def validate_evaluation(path, verifier, selected):
    import numpy as np
    from scipy.stats import spearmanr
    from experiments.evaluate_journal_reliability_relations import original_test_records

    path = Path(path).resolve()
    manifest = base.read_json(path)
    base.require(manifest.get("schema_version") == 1 and manifest.get("status") == "COMPLETED_REAL_EVALUATION" and manifest.get("ResultSource") == base.SOURCE, "Incomplete or unsupported real evaluation manifest")
    source = Path(manifest["source_run_manifest"]).resolve()
    sources = {str(value[0]): identity for identity, value in selected.items()}
    base.require(str(source) in sources, "Evaluation source is not a completed predeclared training condition")
    identity = sources[str(source)]
    verified = selected[identity][1]
    train, train_hash, checkpoint_hash = verified
    for key in ("dataset", "variant", "seed", "teacher_quality", "teacher_checkpoint", "teacher_checkpoint_sha256", "source_hashes"):
        base.require(manifest[key] == train[key], f"Evaluation/source identity mismatch: {key}")
    base.require(manifest["source_run_manifest_sha256"] == train_hash and manifest["source_protocol_fingerprint"] == train["protocol_fingerprint"], "Evaluation references a different source training state")
    base.require(manifest["student_epoch"] == train["epoch"] and manifest["student_checkpoint"] == train["student_checkpoint"], "Evaluation checkpoint is not the source run's final checkpoint")
    base.require(manifest["checkpoint_sha256_before"] == manifest["checkpoint_sha256_after"] == checkpoint_hash, "Evaluation changed or substituted the final checkpoint")
    base.require(manifest["model_state_sha256_before"] == manifest["model_state_sha256_after"] and manifest["teacher_model_state_sha256_before"] == manifest["teacher_model_state_sha256_after"], "Evaluation changed model parameters")
    base.require(manifest.get("training_restarted") is False and manifest.get("additional_optimizer_steps") == 0, "Checkpoint evaluation executed training")
    base.require(verifier.file_hash(manifest["helper_path"]) == manifest["helper_sha256"], "Evaluation helper differs from its recorded implementation")
    base.require(verifier.file_hash(manifest["training_trace"]) == manifest["training_trace_sha256"], "Source training trace changed after evaluation")
    base.require(manifest["teacher_epoch"] == int(Path(train["teacher_checkpoint"]).name.rsplit("-", 1)[-1]), "Evaluation teacher epoch differs from schedule")
    protocol = manifest["protocol"]
    fixed = {"protocol_id": "original_test_order_full8_paired_seeded_T1_v1", "split": "original_test", "prediction_space": "all_classes",
             "batch_size": 8, "shuffle": False, "drop_last": True, "num_workers": 0, "temperature": 1.0,
             "training_restarted": False, "additional_optimizer_steps": 0}
    for key, expected in fixed.items():
        base.require(protocol.get(key) == expected, f"Evaluation fixed protocol mismatch: {key}")
    base.require(protocol["evaluation_seed"] == manifest["evaluation_seed"] == 1, "Unexpected evaluation seed")
    base.require(object_hash(protocol) == manifest["protocol_fingerprint"], "Evaluation protocol fingerprint mismatch")
    records, _ = original_test_records(train["configuration"])
    analyzed = records[:len(records) // 8 * 8]
    expected_samples = [{key: row[key] for key in ("sample_index", "sample_id", "ground_truth", "classname")} for row in analyzed]
    base.require(protocol["analyzed_samples"] == expected_samples and protocol["discarded_samples"] == records[len(analyzed):], "Evaluation subset/order/tail differs from the unmodified original test split")
    base.require(manifest["analyzed_sample_hash"] == protocol["analyzed_sample_hash"] == object_hash(expected_samples), "Analyzed sample identity hash differs")
    base.require(manifest["N_samples"] == len(analyzed) and manifest["N_batches"] * 8 == len(analyzed), "Evaluation sample/batch count mismatch")
    base.require(protocol["N_original_test"] == len(records) and protocol["N_analyzed"] == len(analyzed) and protocol["N_discarded"] == len(records) - len(analyzed), "Protocol tail counts mismatch")
    for entry in protocol["split_sources"]:
        base.require(verifier.file_hash(entry["path"]) == entry["sha256"], "Original evaluation split metadata changed")
    artifacts = {}
    for name, entry in manifest["artifacts"].items():
        artifact = Path(entry["path"]).resolve()
        base.require(artifact.name == name and artifact.is_relative_to(verifier.sanity_path.parent), "Evaluation artifact is outside the local result tree")
        base.require(verifier.file_hash(artifact) == entry["sha256"] and artifact.stat().st_size == entry["size_bytes"], "Evaluation artifact checksum/size mismatch")
        artifacts[name] = artifact
    required = {"batches.csv", "samples.csv", "exp2_correlations.csv", "exp2_quintiles.csv", "exp4_metrics.json", "probabilities.npz"}
    base.require(required <= set(artifacts), "Required raw evaluation artifacts are absent")
    batches, samples = base.read_csv(artifacts["batches.csv"]), base.read_csv(artifacts["samples.csv"])
    base.require(len(batches) == manifest["N_batches"] and len(samples) == manifest["N_samples"], "Raw CSV evaluation count mismatch")
    with np.load(artifacts["probabilities.npz"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("teacher_log_probability", "student_log_probability", "teacher_aug_log_probability")}
    expected_shape = len(samples), len(protocol["classnames"])
    for value in arrays.values():
        base.require(value.shape == expected_shape and value.dtype == np.float32 and np.isfinite(value).all(), "Invalid stored all-class log probabilities")
        base.require(np.allclose(np.exp(value.astype(np.float64)).sum(1), 1, rtol=0, atol=1e-6), "Stored class probabilities do not normalize")
    correct = 0
    for index, row in enumerate(samples):
        expected = analyzed[index]
        base.require(int(row["sample_index"]) == index and row["sample_id"] == expected["sample_id"] and row["impath"] == expected["impath"] and int(row["ground_truth"]) == expected["ground_truth"] and row["classname"] == expected["classname"], "Sample CSV identity/order/label mismatch")
        base.require(int(row["batch_index"]) == index // 8 and int(row["index_in_batch"]) == index % 8, "Sample batch assignment mismatch")
        base.require(verifier.file_hash(row["impath"]) == row["image_sha256"], "Analyzed image bytes changed")
        seed = int.from_bytes(hashlib.sha256(f"1:{train['dataset']}:{index}".encode()).digest()[:4], "big")
        base.require(int(row["augmentation_seed"]) == seed, "Augmented sample seed mismatch")
        teacher_pred, student_pred = int(arrays["teacher_log_probability"][index].argmax()), int(arrays["student_log_probability"][index].argmax())
        is_correct = int(teacher_pred == expected["ground_truth"])
        base.require(int(row["teacher_prediction"]) == teacher_pred and int(row["student_prediction"]) == student_pred and int(row["teacher_correct"]) == is_correct, "Predictions/correctness disagree with stored probabilities")
        correct += is_correct
    numeric_batches = []
    for index, row in enumerate(batches):
        start, stop = index * 8, (index + 1) * 8
        base.require(int(row["batch_index"]) == index and int(row["n_samples"]) == 8, "Noncontiguous or unequal fixed evaluation batch")
        base.require(json.loads(row["sample_indices"]) == list(range(start, stop)) and json.loads(row["sample_ids"]) == [entry["sample_id"] for entry in analyzed[start:stop]], "Raw batch sample list mismatch")
        batch_correct = sum(int(entry["teacher_correct"]) for entry in samples[start:stop])
        base.require(int(row["teacher_correct_count"]) == batch_correct and base.close(row["teacher_batch_accuracy_percent"], 100 * batch_correct / 8), "Teacher batch accuracy/count mismatch")
        recomputed = independent_metrics(*[arrays[key][start:stop] for key in ("teacher_log_probability", "student_log_probability", "teacher_aug_log_probability")])
        for key, value in recomputed.items():
            base.require(numeric_close(row[key], value), f"Stored probability independent recomputation differs: batch {index} {key}")
        numeric_batches.append({key: float(row[key]) for key in (*KEYS, *DISTANCES, "teacher_batch_accuracy_percent")})
    base.require(correct == manifest["teacher_correct_count"] and base.close(100 * correct / len(samples), manifest["teacher_accuracy_micro_percent"]), "Teacher measured micro accuracy mismatch")
    metrics = base.read_json(artifacts["exp4_metrics.json"])
    base.require(metrics["ResultSource"] == base.SOURCE and metrics["N_samples"] == len(samples) and metrics["N_batches"] == len(batches), "Relation metric scope/source mismatch")
    for key in DISTANCES:
        value = sum(row[key] for row in numeric_batches) / len(batches)
        base.require(base.close(value, manifest["relation_metrics"][key]) and base.close(value, metrics["relation_metrics"][key]), "Sample-weighted relation aggregation differs")
    eligible = train["teacher_quality"] == "strong" and train["variant"] == "reliability_adaptive"
    base.require(manifest["exp2_eligible"] is eligible, "Exp2 eligibility differs from declared Strong Adaptive scope")
    correlations = base.read_csv(artifacts["exp2_correlations.csv"])
    quintiles = base.read_csv(artifacts["exp2_quintiles.csv"])
    if eligible:
        base.require(len(correlations) == 3 and {row["reliability"] for row in correlations} == set(KEYS) and len(quintiles) == 15, "Exp2 correlation/quintile coverage differs")
        accuracy = np.asarray([row["teacher_batch_accuracy_percent"] for row in numeric_batches])
        for key in KEYS:
            scores = np.asarray([row[key] for row in numeric_batches])
            correlation = next(row for row in correlations if row["reliability"] == key)
            base.require(int(correlation["N_batches"]) == len(batches), "Spearman batch count differs")
            reason = "constant_reliability" if np.ptp(scores) == 0 else "constant_accuracy" if np.ptp(accuracy) == 0 else None
            if reason:
                base.require(correlation["status"] == "undefined" and correlation["rho"] is None and correlation["p_value"] is None and correlation["reason"] == reason, "Undefined correlation was not explicitly represented")
            else:
                rho, pvalue = spearmanr(scores, accuracy, alternative="two-sided")
                base.require(correlation["status"] == "defined" and base.close(correlation["rho"], rho)
                             and math.isclose(float(correlation["p_value"]), float(pvalue), rel_tol=1e-9, abs_tol=0), "Spearman rho/p-value recomputation differs")
            order = sorted(range(len(batches)), key=lambda index: (scores[index], index))
            for q, indices in enumerate(np.array_split(order, 5), 1):
                group = [row for row in quintiles if row["reliability"] == key and int(row["quintile"]) == q]
                base.require(len(group) == 1, "Duplicate/missing reliability quintile")
                row = group[0]
                base.require(json.loads(row["batch_indices"]) == indices.tolist() and int(row["N_batches"]) == len(indices) and int(row["N_samples"]) == len(indices) * 8, "Stable quintile membership/count mismatch")
                for column, value in (("teacher_accuracy_percent", accuracy[indices].mean()), ("reliability_min", scores[indices].min()), ("reliability_max", scores[indices].max()), ("reliability_mean", scores[indices].mean())):
                    base.require(base.close(row[column], value), "Reliability quintile statistic differs")
    else:
        base.require(not correlations and not quintiles, "Non-Strong-Adaptive evaluation supplied an out-of-scope Exp2 analysis")
    manifest_hash = verifier.file_hash(path)
    verifier.checks.append(f"{identity}: immutable final checkpoint, exact held-out order/tail/labels/image bytes, all stored probabilities, CPU reliability/distance formulas, correctness, rank correlations and stable quintiles independently verified")
    return identity, {"path": path, "manifest": manifest, "correlations": correlations, "quintiles": quintiles,
                      "accuracy_scope": manifest["teacher_accuracy_scope"],
                      "evidence": {"kind": "COMPLETED_REAL_EVALUATION", "path": str(path), "sha256": manifest_hash,
                                   "source_run_manifest": str(source), "source_run_manifest_sha256": train_hash,
                                   "student_checkpoint_sha256": checkpoint_hash, "analyzed_sample_hash": manifest["analyzed_sample_hash"],
                                   "protocol_fingerprint": manifest["protocol_fingerprint"]}}


def collect_evaluations(root, verifier, selected, old_tables, evaluation_roots):
    paths = set()
    for rows in old_tables.values():
        for row in rows:
            if row.get("ResultSource") != base.SOURCE or not row.get("MeasurementEvidence"):
                continue
            for entry in json.loads(row["MeasurementEvidence"]).values():
                base.require(verifier.file_hash(entry["path"]) == entry["sha256"], "Previously exported measurement evidence changed")
                if entry["kind"] == "COMPLETED_REAL_EVALUATION":
                    paths.add(Path(entry["path"]).resolve())
                else:
                    verifier.manifest(entry["path"])
    for search_root in evaluation_roots or [root / name for name in ("exp2_reliability", "exp3_teacher_quality", "exp4_relation_preservation")]:
        for path in Path(search_root).rglob("evaluation_manifest.json"):
            try:
                record = base.read_json(path)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "COMPLETED_REAL_EVALUATION":
                paths.add(path.resolve())
    result = {}
    for path in sorted(paths):
        identity, entry = validate_evaluation(path, verifier, selected)
        if identity in result:
            # A requested Exp2 mirror is the same immutable evaluation receipt.
            base.require(entry["manifest"] == result[identity]["manifest"], "Competing evaluations for one condition; do not select by outcome")
            continue
        result[identity] = entry
    for dataset in {identity[0] for identity in result}:
        entries = [entry for identity, entry in result.items() if identity[0] == dataset]
        for other in entries[1:]:
            base.require(other["manifest"]["protocol"] == entries[0]["manifest"]["protocol"], "Across-arm evaluation subset/batching/transform/class protocol differs")
    return result


def update_evaluation_tables(tables, selected, evaluations):
    from experiments.export_measured_exp234_results import publish_row, training_evidence, encoded

    for row in tables["Exp2_Reliability"]:
        entry = evaluations.get((row["Dataset"], "Strong", "AdaptiveLUNA"))
        if not entry:
            continue
        manifest = entry["manifest"]
        updates = {"Seed": manifest["seed"], "EvaluationManifest": str(entry["path"]), "EvaluationManifestSHA256": entry["evidence"]["sha256"],
                   "AnalyzedSampleHash": manifest["analyzed_sample_hash"], "EvaluationProtocol": protocol_summary(manifest),
                   "TeacherAccuracyScope": entry["accuracy_scope"]}
        if row["Statistic"] == "Spearman":
            raw = next(item for item in entry["correlations"] if item["reliability"] == row["Reliability"])
            updates.update(rho=None if raw["rho"] is None else float(raw["rho"]), p_value=None if raw["p_value"] is None else float(raw["p_value"]),
                           N_batches=int(raw["N_batches"]), N_samples=manifest["N_samples"], UndefinedReason=raw["reason"])
            publish_row(row, updates, {"heldout_evaluation": entry["evidence"]}, ["N_batches"])
            if raw["status"] == "undefined":
                row.update(Status="UNDEFINED_REAL", ExactReason=raw["reason"])
        else:
            raw = next(item for item in entry["quintiles"] if item["reliability"] == row["Reliability"] and f"Q{item['quintile']}" == row["Quantile"])
            updates.update(teacher_batch_accuracy=float(raw["teacher_accuracy_percent"]), N_batches=int(raw["N_batches"]), N_samples=int(raw["N_samples"]))
            publish_row(row, updates, {"heldout_evaluation": entry["evidence"]}, ["teacher_batch_accuracy", "N_batches"])
    for row in tables["Exp4_RelationPreservation"]:
        variant = "Equal" if row["Variant"] == "StaticHierarchical" else row["Variant"]
        identity = row["Dataset"], "Strong", variant
        entry = evaluations.get(identity)
        if not entry:
            continue
        manifest = entry["manifest"]
        path, verified = selected[identity]
        train, train_hash, checkpoint_hash = verified
        updates = {**manifest["relation_metrics"], "HM": train["HM"], "Seed": train["seed"],
                   "EvaluationManifest": str(entry["path"]), "EvaluationManifestSHA256": entry["evidence"]["sha256"],
                   "AnalyzedSampleHash": manifest["analyzed_sample_hash"], "EvaluationProtocol": protocol_summary(manifest),
                   "N_samples": manifest["N_samples"], "RunManifest": str(path), "RunManifestSHA256": train_hash,
                   "StudentCheckpoint": train["student_checkpoint"], "StudentCheckpointSHA256": checkpoint_hash}
        publish_row(row, updates, {"heldout_distances": entry["evidence"], "same_run_HM": training_evidence(path, verified)}, [*DISTANCES, "HM"])
