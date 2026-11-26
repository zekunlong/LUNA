"""Synthetic runtime boundary fixtures; never generate a real sanity artifact."""

from __future__ import annotations

import copy
import ast
import csv
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from experiments import journal_runtime as runtime
from experiments.run_journal_sanity import assess_sanity


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def synthetic_sanity_rows():
    """Schema fixtures, not outputs from DTD, CLIP, or actual training."""
    rows = []
    for index in range(10):
        lambdas = (0.10 + 0.01 * index, 0.30, 0.60 - 0.01 * index)
        losses = (2.0, 4.0, 8.0)
        terms = dict(zip(("L_ins", "L_batch", "L_cls"), [w * loss for w, loss in zip(lambdas, losses)]))
        rows.append({
            "epoch": 1, "batch_index": index, "n_samples": 2,
            "r_ins": 0.2 + 0.01 * index, "r_batch": 0.7, "r_cls": 0.9,
            **dict(zip(runtime.WEIGHTS, lambdas)),
            **dict(zip(("L_ins", "L_batch", "L_cls"), losses)),
            "L_adapt": sum(terms.values()), "weighted_terms": terms,
            "legacy_loss": 12.0, "legacy_numeric_status": "finite",
            "legacy_components": {"L_ins": 2.0, "L_batch": 4.0, "L_cls": 6.0},
            "run_kind": "sanity_training_batches", "selection_protocol": "final_epoch",
            "student_prompt_grad_present": True, "student_prompt_grad_finite": True,
            "student_prompt_grad_nonzero": True, "student_prompt_grad_norm": 0.5,
            "teacher_requires_grad_false": True, "teacher_grad_none": True,
            "teacher_weights_detached": True, "paired_same_sample": True,
            "paired_view_count": 2, "Base": None, "Novel": None,
            "alternate_weight_diagnostic": {
                "used_for_optimization": False,
                "weighted_terms": dict(zip(("L_ins", "L_batch", "L_cls"), (1.2, 1.2, 0.8))),
            },
        })
    return rows


class JournalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="luna_synthetic_runtime_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.teacher = self.root / "teacher_fixture.pth.tar"
        self.teacher.write_bytes(b"unit-test path/hash fixture, never deserialized")
        self.teacher_clip = self.root / "teacher_clip_fixture.pt"
        self.teacher_clip.write_bytes(b"unit-test frozen model asset fixture, never deserialized")
        self.dataset = SimpleNamespace(**{
            split: [SimpleNamespace(impath=f"{split}/fixture_{index}.jpg", label=index, classname=f"class_{index}")
                    for index in range(2)]
            for split in ("train_x", "val", "test")
        })
        self.hashes = {"fixture_only.py": "unit-test-source-hash"}
        self.hash_patch = patch.object(runtime, "source_hashes", return_value=self.hashes)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.model = torch.nn.Linear(2, 3)

    def config(self, name, *, mode="reliability_adaptive", max_epoch=2, sanity_batches=0):
        output = self.root / name
        return AttrDict(
            OUTPUT_DIR=str(output), RESUME="", SEED=1,
            DATASET=AttrDict(NAME="DescribableTextures", ROOT=str(self.root / "data"),
                             FEWSHOT_CACHE_ROOT=str(output / "fewshot_cache"), NUM_SHOTS=0),
            TRAINER=AttrDict(NAME="LUNA", MODAL="base2novel", LUNA=AttrDict(
                TEMPERATURE=1.0, TEACHER_CLIP_PATH=str(self.teacher_clip))),
            OPTIM=AttrDict(MAX_EPOCH=max_epoch, LR=0.002),
            TEST=AttrDict(FINAL_MODEL="best_val", NO_TEST=False),
            JOURNAL=AttrDict(
                MODE=mode, SELECTION_PROTOCOL="final_epoch", TEACHER_QUALITY="strong",
                TEACHER_CHECKPOINT=str(self.teacher), TEACHER_TOTAL_EPOCHS=20,
                STATIC_WEIGHTS=(), STATIC_WEIGHTS_FILE="", OUTPUT_ROOT=str(self.root / "analysis"),
                SANITY_BATCHES=sanity_batches, SANITY_REPORT="", LOG_BATCH_SAMPLES=True,
            ),
        )

    def recorder(self, cfg, model=None, dataset=None):
        trainer = SimpleNamespace(model=self.model if model is None else model, train_loader_x=[None, None],
                                  dm=SimpleNamespace(dataset=self.dataset if dataset is None else dataset))
        recorder = runtime.JournalRecorder(trainer, cfg)
        self.addCleanup(lambda: [handle.close() for handle in recorder.handles])
        self.assertEqual(trainer.journal_batch_callback, recorder.on_batch)
        return recorder

    def batch(self, epoch, index, count, weights=(0.1, 0.3, 0.6), *, kind="journal_training"):
        losses = (2.0, 4.0, 8.0)
        return {
            "epoch": epoch, "batch_index": index, "n_samples": count,
            "run_kind": kind, "r_ins": 0.2, "r_batch": 0.7, "r_cls": 0.9,
            **dict(zip(runtime.WEIGHTS, weights)),
            **dict(zip(("L_ins", "L_batch", "L_cls"), losses)),
            "L_adapt": sum(w * loss for w, loss in zip(weights, losses)), "legacy_loss": 12.0,
        }

    def finish(self, recorder, status):
        with redirect_stdout(io.StringIO()):
            return recorder.finish(status)

    def full_adaptive_mean(self, name="adaptive"):
        cfg = self.config(name)
        recorder = self.recorder(cfg)
        for epoch in (1, 2):
            recorder.on_batch(self.batch(epoch, 0, 1, (0.1, 0.3, 0.6)), [])
            recorder.on_batch(self.batch(epoch, 1, 3, (0.5, 0.2, 0.3)), [])
        self.finish(recorder, "COMPLETED_TRAINING")
        return cfg, Path(cfg.OUTPUT_DIR) / "training_mean_weights.json"

    def gate_fixture(self):
        rows = synthetic_sanity_rows()
        checks, deviations, means = assess_sanity(rows)
        return {
            "_fixture_only": "Synthetic unit test; never a real DTD result",
            "status": "PASS_REAL_DTD_SANITY", "dataset": "DTD", "batch_count": 10,
            "checks": checks, "source_hashes": self.hashes, "records": rows,
            "lambda_std": deviations, "means": means,
            "preflight": {"status": "READY", "environment": {"checked": True}},
        }

    def check_gate(self, report):
        # The fixture is kept only inside TemporaryDirectory, outside analysis.
        path = self.root / "synthetic_gate_fixture.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return runtime.require_sanity_pass(path)

    def test_gate_accepts_complete_real_marker_and_matching_source_fixture(self):
        fixture = self.gate_fixture()
        self.assertTrue(all(fixture["checks"].values()))
        self.assertEqual(self.check_gate(fixture)["status"], "PASS_REAL_DTD_SANITY")

    def test_gate_rejects_missing_synthetic_blocked_failed_other_dataset_and_stale_reports(self):
        with self.assertRaises(ValueError):
            runtime.require_sanity_pass(self.root / "absent.json")
        cases = [
            ("status", "PASS_SYNTHETIC_TESTS"), ("status", "BLOCKED_BEFORE_REAL_DTD_SANITY"),
            ("status", "FAIL_REAL_DTD_SANITY"), ("dataset", "EuroSAT"),
            ("batch_count", 9), ("source_hashes", {"fixture_only.py": "stale-hash"}),
            ("checks", {}),
            ("records", []), ("preflight", {"status": "BLOCKED", "environment": {"checked": True}}),
            ("preflight", {"status": "READY", "environment": {"checked": False}}),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                fixture = self.gate_fixture()
                fixture[key] = value
                with self.assertRaises(ValueError):
                    self.check_gate(fixture)

    def test_gate_rejects_incomplete_check_names_and_truthy_nonbooleans(self):
        for checks in ({"unrelated_check": True}, {"ten_real_batches": True}):
            fixture = self.gate_fixture()
            fixture["checks"] = checks
            with self.subTest(checks=checks), self.assertRaises(ValueError):
                self.check_gate(fixture)
        fixture = self.gate_fixture()
        fixture["checks"]["teacher_frozen_without_gradients"] = "false"
        with self.assertRaises(ValueError):
            self.check_gate(fixture)
        fixture = self.gate_fixture()
        fixture["checks"]["student_prompt_gradients"] = False
        with self.assertRaises(ValueError):
            self.check_gate(fixture)

    def test_scalar_logs_use_sample_weighted_means_and_population_std(self):
        cfg = self.config("weighted_log")
        recorder = self.recorder(cfg)
        rows = [self.batch(1, 0, 1, (0.1, 0.3, 0.6)), self.batch(1, 1, 3, (0.5, 0.2, 0.3))]
        sample = {
            "sample_index": 17, "impath": "synthetic_fixture.jpg", "teacher_entropy": 0.2,
            "teacher_entropy_aug": 0.3, "teacher_max_probability": 0.9,
            "teacher_prediction": 2, "ground_truth": 2, "teacher_correct": True,
        }
        recorder.on_batch(rows[0], [sample])
        recorder.on_batch(rows[1], [])
        manifest = self.finish(recorder, "COMPLETED_SANITY")
        metrics = manifest["statistics"]
        self.assertEqual(metrics["n_observations"], 4)
        self.assertEqual(metrics["batch_count"], 2)
        for key in runtime.METRICS:
            expected = sum(row[key] * row["n_samples"] for row in rows) / 4
            self.assertAlmostEqual(metrics["mean_" + key], expected, places=13)
        for key in runtime.WEIGHTS:
            expanded = [row[key] for row in rows for _ in range(row["n_samples"])]
            self.assertAlmostEqual(metrics["std_" + key], statistics.pstdev(expanded), places=13)
        with (Path(cfg.OUTPUT_DIR) / "journal_batches.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(csv_rows[0]["dataset"], "DTD")
        self.assertEqual(float(csv_rows[1]["lambda_ins"]), 0.5)
        sample_path = Path(manifest["raw_reliability_directory"]) / "samples.csv"
        with sample_path.open(newline="", encoding="utf-8") as handle:
            sample_rows = list(csv.DictReader(handle))
        self.assertEqual(sample_rows[0]["sample_index"], "17")
        self.assertEqual(sample_rows[0]["teacher_entropy_aug"], "0.3")
        self.assertEqual(sample_rows[0]["ground_truth"], "2")
        self.assertFalse(manifest["test_used_for_selection"])

    def test_sanity_and_failed_runs_never_export_training_static_means(self):
        for status in ("COMPLETED_SANITY", "FAILED_SANITY", "FAILED_TRAINING"):
            cfg = self.config(status)
            recorder = self.recorder(cfg)
            recorder.on_batch(self.batch(1, 0, 2, kind="sanity_training_batches"), [])
            self.finish(recorder, status)
            self.assertFalse((Path(cfg.OUTPUT_DIR) / "training_mean_weights.json").exists())

    def test_incomplete_training_cannot_export_static_mean(self):
        cfg = self.config("incomplete", max_epoch=2)
        recorder = self.recorder(cfg)
        recorder.on_batch(self.batch(1, 0, 2), [])
        with self.assertRaises(ValueError):
            self.finish(recorder, "COMPLETED_TRAINING")
        self.assertFalse((Path(cfg.OUTPUT_DIR) / "training_mean_weights.json").exists())
        manifest = json.loads((Path(cfg.OUTPUT_DIR) / "journal_manifest.json").read_text(encoding="utf-8"))
        self.assertNotEqual(manifest["status"], "COMPLETED_TRAINING")

    def test_truncated_batches_or_noncontiguous_epochs_cannot_export_static_mean(self):
        for name, epochs in (("truncated_batches", (1, 2)), ("noncontiguous_epochs", (1, 3))):
            cfg = self.config(name, max_epoch=2)
            recorder = self.recorder(cfg)
            for epoch in epochs:
                recorder.on_batch(self.batch(epoch, 0, 2), [])
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.finish(recorder, "COMPLETED_TRAINING")
            self.assertFalse((Path(cfg.OUTPUT_DIR) / "training_mean_weights.json").exists())

    def test_debug_sanity_cannot_be_relabelled_completed_training(self):
        cfg = self.config("sanity_relabel", max_epoch=1, sanity_batches=10)
        recorder = self.recorder(cfg)
        recorder.on_batch(self.batch(1, 0, 2, kind="sanity_training_batches"), [])
        with self.assertRaises(ValueError):
            self.finish(recorder, "COMPLETED_TRAINING")
        self.assertFalse((Path(cfg.OUTPUT_DIR) / "training_mean_weights.json").exists())

    def test_full_training_exports_train_only_means_and_static_reuses_same_initialization(self):
        _, path = self.full_adaptive_mean()
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["data_usage"], "training_only")
        self.assertEqual(record["n_observations"], 8)
        self.assertAlmostEqual(record["lambda_mean"]["lambda_ins"], 0.4, places=14)
        self.assertAlmostEqual(sum(record["lambda_mean"].values()), 1, places=14)
        static_cfg = self.config("static", mode="static_mean")
        static_cfg.JOURNAL.STATIC_WEIGHTS_FILE = str(path)
        self.assertEqual(runtime.validate_static_means(static_cfg), record)
        static_recorder = self.recorder(static_cfg)
        self.assertEqual(static_recorder.manifest["initialization_sha256"], record["initialization_sha256"])
        self.finish(static_recorder, "FAILED_TRAINING")

    def test_static_means_reject_mismatched_provenance(self):
        _, path = self.full_adaptive_mean()
        original = json.loads(path.read_text(encoding="utf-8"))
        static_cfg = self.config("static_validation", mode="static_mean")
        static_cfg.JOURNAL.STATIC_WEIGHTS_FILE = str(path)
        changes = {
            "status": "COMPLETED_SANITY", "source_mode": "equal", "data_usage": "test",
            "dataset": "EuroSAT", "seed": 2, "selection_protocol": "best_val",
            "protocol_fingerprint": "mismatched-protocol", "teacher_checkpoint_sha256": "other-teacher",
            "teacher_clip_sha256": "other-frozen-teacher-clip",
            "source_hashes": {"fixture_only.py": "old-source"}, "n_observations": 0,
            "lambda_mean": {"lambda_ins": 0.5, "lambda_batch": 0.5, "lambda_cls": 0.5},
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                modified = {**original, key: value}
                path.write_text(json.dumps(modified), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runtime.validate_static_means(static_cfg)
        path.write_text(json.dumps(original), encoding="utf-8")
        static_cfg.OPTIM.LR = 0.004
        with self.assertRaisesRegex(ValueError, "protocol_fingerprint"):
            runtime.validate_static_means(static_cfg)

    def test_static_means_reject_different_student_initialization_before_manifest(self):
        _, path = self.full_adaptive_mean()
        cfg = self.config("changed_initialization", mode="static_mean")
        cfg.JOURNAL.STATIC_WEIGHTS_FILE = str(path)
        different_model = copy.deepcopy(self.model)
        with torch.no_grad():
            different_model.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "initialization_sha256"):
            self.recorder(cfg, model=different_model)
        self.assertFalse((Path(cfg.OUTPUT_DIR) / "journal_manifest.json").exists())

    def test_static_means_reject_changed_split_metadata(self):
        _, path = self.full_adaptive_mean()
        cfg = self.config("changed_dataset", mode="static_mean")
        cfg.JOURNAL.STATIC_WEIGHTS_FILE = str(path)
        changed_dataset = copy.deepcopy(self.dataset)
        changed_dataset.train_x[0].label = 1
        with self.assertRaisesRegex(ValueError, "dataset_fingerprint"):
            self.recorder(cfg, dataset=changed_dataset)
        self.assertFalse((Path(cfg.OUTPUT_DIR) / "journal_manifest.json").exists())

    def test_protocol_ignores_variant_output_and_logging_but_retains_training_changes(self):
        adaptive = self.config("protocol_adaptive")
        static = self.config("protocol_static", mode="static_mean", sanity_batches=10)
        static.JOURNAL.STATIC_WEIGHTS = (0.1, 0.3, 0.6)
        static.JOURNAL.STATIC_WEIGHTS_FILE = "weights.json"
        static.JOURNAL.LOG_BATCH_SAMPLES = False
        self.assertEqual(runtime.protocol_fingerprint(adaptive), runtime.protocol_fingerprint(static))
        static.SEED = 2
        self.assertNotEqual(runtime.protocol_fingerprint(adaptive), runtime.protocol_fingerprint(static))

    def test_invalid_scalar_record_rejected_before_aggregation(self):
        recorder = self.recorder(self.config("invalid_scalars"))
        for key, value in (("L_adapt", float("nan")), ("r_ins", float("inf")), ("n_samples", 0)):
            row = self.batch(1, 0, 2)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                recorder.on_batch(row, [])
        self.assertEqual(recorder.global_group["weight"], 0)

    def test_assessment_synthetic_fixture_reports_statistics_without_publishing(self):
        checks, deviations, means = assess_sanity(synthetic_sanity_rows())
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(means["lambda_ins"], 0.145, places=14)
        self.assertGreater(deviations["lambda_ins"], 0)
        self.assertEqual(deviations["lambda_batch"], 0)
        self.assertEqual(list(self.root.glob("*sanity*")), [])

    def test_assessment_rejects_repeated_batch_identity(self):
        rows = synthetic_sanity_rows()
        for row in rows:
            row["batch_index"] = 0
        checks, _, _ = assess_sanity(rows)
        self.assertFalse(checks["ten_real_batches"])

    def test_assessment_rejects_scalar_rescaled_legacy_components(self):
        rows = synthetic_sanity_rows()
        for row in rows:
            row["legacy_components"] = {key: value / 2 for key, value in row["weighted_terms"].items()}
            row["legacy_loss"] = row["L_adapt"] / 2
        checks, _, _ = assess_sanity(rows)
        self.assertTrue(checks["independent_weighted_scalars"])
        self.assertTrue(checks["relative_contributions_change"])
        self.assertFalse(checks["legacy_is_mathematically_different"])

    def test_assessment_requires_component_evidence_for_legacy_comparison(self):
        rows = synthetic_sanity_rows()
        for row in rows:
            row.pop("legacy_components")
        checks, _, _ = assess_sanity(rows)
        self.assertFalse(checks["legacy_is_mathematically_different"])

    def test_assessment_detects_gradient_view_selection_and_loss_failures(self):
        cases = [
            ("teacher_grad_none", False, "teacher_frozen_without_gradients"),
            ("student_prompt_grad_finite", False, "student_prompt_gradients"),
            ("paired_same_sample", False, "same_sample_two_views"),
            ("Base", 50.0, "no_test_checkpoint_selection"),
            ("L_adapt", 100.0, "independent_weighted_scalars"),
            ("r_ins", float("nan"), "no_nan_inf"),
        ]
        for key, value, expected_check in cases:
            rows = synthetic_sanity_rows()
            rows[0][key] = value
            with self.subTest(key=key):
                checks, _, _ = assess_sanity(rows)
                self.assertFalse(checks[expected_check])
        rows = synthetic_sanity_rows()
        rows[0]["alternate_weight_diagnostic"]["used_for_optimization"] = True
        with self.assertRaises(ValueError):
            assess_sanity(rows)

    def arguments(self, cfg, **changes):
        args = dict(distill_weighting=None, selection_protocol=None, output_dir=cfg.OUTPUT_DIR,
                    seed=None, teacher_checkpoint=None, teacher_total_epochs=None,
                    teacher_quality=None, static_weights=None, analysis_output_dir=None,
                    sanity_batches=None, sanity_report=None)
        args.update(changes)
        return SimpleNamespace(**args)

    def test_legacy_configuration_retains_original_selection_and_every_existing_value(self):
        cfg = self.config("legacy_cfg", mode="legacy")
        cfg.JOURNAL.SELECTION_PROTOCOL = "legacy"
        cfg.SEED = -1
        cfg.RESUME = "legacy_resume_directory"
        before = runtime.plain_config(cfg)
        args = self.arguments(cfg, output_dir="", selection_protocol="legacy", teacher_checkpoint="ignored-new-mode-path")
        runtime.configure_journal(cfg, args)
        self.assertEqual(runtime.plain_config(cfg), before)
        self.assertEqual(cfg.TEST.FINAL_MODEL, "best_val")
        self.assertFalse(cfg.TEST.NO_TEST)

    def test_new_mode_forces_final_selection_and_supports_seed_zero(self):
        for seed in (0, 1, 7):
            cfg = self.config(f"journal_cfg_{seed}", mode="legacy")
            cfg.SEED = -1
            args = self.arguments(cfg, distill_weighting="reliability_adaptive", seed=seed)
            runtime.configure_journal(cfg, args)
            self.assertEqual(cfg.JOURNAL.MODE, "reliability_adaptive")
            self.assertEqual(cfg.JOURNAL.SELECTION_PROTOCOL, "final_epoch")
            self.assertEqual(cfg.TEST.FINAL_MODEL, "last_step")
            self.assertTrue(cfg.TEST.NO_TEST)
            self.assertEqual(cfg.SEED, seed)
            self.assertEqual(cfg.DATASET.FEWSHOT_CACHE_ROOT, str(Path(cfg.OUTPUT_DIR).resolve() / "fewshot_cache"))
        cfg = self.config("unspecified_seed")
        runtime.configure_journal(cfg, self.arguments(cfg, seed=None))
        self.assertEqual(cfg.SEED, 1)
        cfg = self.config("negative_seed")
        cfg.SEED = -1
        with self.assertRaisesRegex(ValueError, "fixed main-run seed"):
            runtime.configure_journal(cfg, self.arguments(cfg))

    def test_new_mode_rejects_test_selection_and_legacy_rejects_journal_selection(self):
        for selection in ("legacy", "best_val"):
            cfg = self.config("invalid_selection")
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "final_epoch"):
                runtime.configure_journal(cfg, self.arguments(cfg, selection_protocol=selection))
        cfg = self.config("legacy_selection", mode="legacy")
        with self.assertRaises(ValueError):
            runtime.configure_journal(cfg, self.arguments(cfg, selection_protocol="final_epoch"))

    def test_new_mode_requires_separate_fresh_output_and_no_resume(self):
        cfg = self.config("missing_output_flag")
        with self.assertRaisesRegex(ValueError, "explicit --output-dir"):
            runtime.configure_journal(cfg, self.arguments(cfg, output_dir=""))
        for output in (runtime.REPO / "output", runtime.REPO / "output" / "old_seed_1"):
            cfg = self.config("invalid_output")
            cfg.OUTPUT_DIR = str(output)
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "legacy output"):
                runtime.configure_journal(cfg, self.arguments(cfg))
        cfg = self.config("resume")
        cfg.RESUME = "previous_journal_run"
        with self.assertRaisesRegex(ValueError, "resume is disabled"):
            runtime.configure_journal(cfg, self.arguments(cfg))
        for artifact in ("journal_manifest.json", "VLPromptLearner"):
            cfg = self.config("existing_" + artifact)
            output = Path(cfg.OUTPUT_DIR)
            output.mkdir()
            if artifact.endswith(".json"):
                (output / artifact).write_text("{}", encoding="utf-8")
            else:
                (output / artifact).mkdir()
            with self.subTest(artifact=artifact), self.assertRaisesRegex(ValueError, "fresh output directory"):
                runtime.configure_journal(cfg, self.arguments(cfg))
        cfg = self.config("old_log_only")
        output = Path(cfg.OUTPUT_DIR)
        output.mkdir()
        previous_log = output / "log.txt"
        previous_log.write_text("previous run log", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing run files will not be overwritten"):
            runtime.configure_journal(cfg, self.arguments(cfg))
        self.assertEqual(previous_log.read_text(encoding="utf-8"), "previous run log")

    def test_explicit_mode_and_assets_win_over_config_or_environment(self):
        cfg = self.config("explicit_assets", mode="legacy")
        explicit_root = cfg.DATASET.ROOT
        explicit_clip = cfg.TRAINER.LUNA.TEACHER_CLIP_PATH
        args = self.arguments(cfg, distill_weighting="equal", teacher_checkpoint="explicit_cli_teacher.tar",
                              teacher_total_epochs=30, teacher_quality="strong", sanity_batches=10,
                              analysis_output_dir=str(self.root / "explicit_analysis"),
                              sanity_report="explicit_sanity.json")
        environment = {"LUNA_DATA_ROOT": "environment_data", "LUNA_TEACHER_CLIP_PATH": "environment_clip",
                       "LUNA_TEACHER_CHECKPOINT": "environment_teacher", "LUNA_TEACHER_TOTAL_EPOCHS": "99"}
        with patch.dict(os.environ, environment):
            runtime.configure_journal(cfg, args)
        self.assertEqual(cfg.JOURNAL.MODE, "equal")
        self.assertEqual(cfg.DATASET.ROOT, explicit_root)
        self.assertEqual(cfg.TRAINER.LUNA.TEACHER_CLIP_PATH, explicit_clip)
        self.assertEqual(cfg.JOURNAL.TEACHER_CHECKPOINT, "explicit_cli_teacher.tar")
        self.assertEqual(cfg.JOURNAL.TEACHER_TOTAL_EPOCHS, 30)
        self.assertEqual(cfg.JOURNAL.SANITY_BATCHES, 10)
        self.assertEqual(cfg.JOURNAL.OUTPUT_ROOT, args.analysis_output_dir)
        self.assertEqual(cfg.JOURNAL.SANITY_REPORT, "explicit_sanity.json")

    def test_environment_assets_fill_only_unspecified_values(self):
        cfg = self.config("environment_assets")
        cfg.DATASET.ROOT = ""
        cfg.TRAINER.LUNA.TEACHER_CLIP_PATH = ""
        cfg.JOURNAL.TEACHER_CHECKPOINT = ""
        cfg.JOURNAL.TEACHER_TOTAL_EPOCHS = 0
        environment = {"LUNA_DATA_ROOT": "environment_data", "LUNA_TEACHER_CLIP_PATH": "environment_clip",
                       "LUNA_TEACHER_CHECKPOINT": "environment_teacher", "LUNA_TEACHER_TOTAL_EPOCHS": "20"}
        with patch.dict(os.environ, environment):
            runtime.configure_journal(cfg, self.arguments(cfg))
        self.assertEqual(cfg.DATASET.ROOT, "environment_data")
        self.assertEqual(cfg.TRAINER.LUNA.TEACHER_CLIP_PATH, "environment_clip")
        self.assertEqual(cfg.JOURNAL.TEACHER_CHECKPOINT, "environment_teacher")
        self.assertEqual(cfg.JOURNAL.TEACHER_TOTAL_EPOCHS, 20)

    def test_config_bridge_rejects_unsupported_trainer_modal_dataset_and_mode(self):
        cases = (("trainer", "OtherTrainer"), ("modal", "cross"), ("dataset", "ImageNet"), ("mode", "learned_gate"))
        for field, value in cases:
            cfg = self.config("invalid_" + field)
            args = self.arguments(cfg)
            if field == "trainer":
                cfg.TRAINER.NAME = value
            elif field == "modal":
                cfg.TRAINER.MODAL = value
            elif field == "dataset":
                cfg.DATASET.NAME = value
            else:
                args.distill_weighting = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime.configure_journal(cfg, args)

    def test_train_entry_routes_only_explicit_new_mode_into_journal_lifecycle(self):
        # Execute only train.py's main function; do not import Dassl or datasets.
        source = runtime.REPO / "train.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        code = compile(ast.fix_missing_locations(ast.Module(body=[main_node], type_ignores=[])), str(source), "exec")
        from experiments import run_journal_sanity
        for mode in ("legacy", "equal", "reliability_adaptive"):
            cfg = self.config("entry_" + mode, mode=mode)
            cfg.USE_CUDA = False
            trainer = Mock()
            namespace = {
                "setup_cfg": Mock(return_value=cfg), "set_random_seed": Mock(), "setup_logger": Mock(),
                "print_args": Mock(), "collect_env_info": Mock(return_value="synthetic test fixture"),
                "build_trainer": Mock(return_value=trainer),
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            }
            exec(code, namespace)
            args = SimpleNamespace(eval_only=False, no_train=False)
            with patch.object(run_journal_sanity, "run_journal_mode", return_value=23) as journal_mode:
                with redirect_stdout(io.StringIO()):
                    result = namespace["main"](args)
                if mode == "legacy":
                    journal_mode.assert_not_called()
                    namespace["build_trainer"].assert_called_once_with(cfg)
                    trainer.train.assert_called_once_with()
                    self.assertIsNone(result)
                else:
                    journal_mode.assert_called_once_with(args, cfg)
                    namespace["build_trainer"].assert_not_called()
                    namespace["setup_logger"].assert_not_called()
                    trainer.train.assert_not_called()
                    self.assertEqual(result, 23)


if __name__ == "__main__":
    unittest.main()
