"""Synthetic structural tests; these do not satisfy the required real DTD check.

The trainer class is AST-loaded with a stub base and small CPU models. No Dassl
loader, CLIP checkpoint, dataset image, or real experimental result is used.
"""

import ast
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]


class StubBase:
    def get_current_lr(self):
        return self.optim.param_groups[0]["lr"]

    def set_model_mode(self, mode):
        self.model.train(mode == "train")

    def update_lr(self):
        self.scheduler_calls += 1

    def train(self):
        self.full_train_calls += 1

    def save_model(self, *args, **kwargs):
        self.save_calls.append((args, kwargs))

    def load_model(self, directory, epoch=None):
        self.load_calls.append((directory, epoch))

    def get_model_names(self):
        return ["VLPromptLearner"]

    def test(self, split=None):
        self.test_calls.append(split)
        return {"val": 80.0, "test": 60.0}[split]

    def init_writer(self, directory):
        self.writer_directory = directory

    def close_writer(self):
        self.writer_closed = True


def load_structural_class():
    spec = importlib.util.spec_from_file_location("journal_math_test_fixture", ROOT / "experiments/journal_losses.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = {
        "LUNA": StubBase, "torch": torch, "np": np, "F": F,
        "Path": Path, "os": os, "math": math, "time": time, "hashlib": hashlib,
        "EPS": module.EPS, "aggregate_losses": module.aggregate_losses,
        "compute_reliability": module.compute_reliability,
    }
    original = ast.parse((ROOT / "trainers/luna.py").read_text(encoding="utf-8-sig"))
    nodes = [
        node for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in {"kd_loss", "bc_loss", "cc_loss"}
    ]
    journal = ast.parse((ROOT / "trainers/luna_journal.py").read_text(encoding="utf-8"))
    nodes.extend(node for node in journal.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "synthetic_journal_trainer_fixture", "exec"), namespace)
    return namespace["LUNAJournal"], namespace


class TinyStudent(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.prompt_learner = torch.nn.Module()
        self.prompt_learner.ctx = torch.nn.Parameter(torch.randn(5, 7, generator=generator))

    def forward(self, image, label):
        if label is not None:
            raise AssertionError("Training forward must not consume ground-truth labels")
        return torch.tensor(1.0), F.normalize(image, dim=-1), F.normalize(self.prompt_learner.ctx, dim=-1)


class TinyTeacher(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.prototype = torch.nn.Parameter(torch.randn(5, 7, generator=generator), requires_grad=False)
        self.seen_inputs = []

    def forward(self, image):
        self.seen_inputs.append(image.detach().clone())
        return F.normalize(image, dim=-1) @ F.normalize(self.prototype, dim=-1).T


class JournalTrainerStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_class, cls.namespace = load_structural_class()

    def fixture(self, mode="reliability_adaptive", batches=10):
        generator = torch.Generator(device="cpu").manual_seed(8021)
        trainer = self.trainer_class.__new__(self.trainer_class)
        trainer.cfg = NS(
            DATASET=NS(NAME="synthetic_unit_fixture"), SEED=8021,
            TRAIN=NS(CHECKPOINT_FREQ=2),
            JOURNAL=NS(
                MODE=mode, STATIC_WEIGHTS=(0.2, 0.3, 0.5), LOG_BATCH_SAMPLES=True,
                VALIDATE_GRADIENTS=True, TEACHER_QUALITY="strong", SANITY_BATCHES=10,
                SANITY_REPORT="not-a-real-sanity-report.json",
            ),
        )
        trainer.device = torch.device("cpu")
        trainer.model = TinyStudent(generator)
        trainer.model_teacher = TinyTeacher(generator)
        trainer.optim = torch.optim.SGD(trainer.model.parameters(), lr=0.005)
        trainer.temperature = trainer.kd_loss_weight = 1.0
        trainer.journal_temperatures = (1.0, 3.0, 5.0, 2.0, 6.0)
        trainer.journal_teacher_checkpoint = "synthetic-unit-fixture-only"
        trainer.journal_teacher_sha256 = "not-a-real-checkpoint"
        trainer.journal_teacher_epoch = 1
        trainer.journal_initial_trainable_sha256 = "synthetic-fixture"
        trainer._journal_optimizer_steps = 0
        trainer._journal_sanity = False
        trainer._journal_epoch_sums = {}
        trainer._journal_epoch_samples = 0
        trainer.journal_records = []
        trainer.journal_last_record = None
        trainer.journal_batch_callback = None
        trainer.journal_epoch_callback = None
        trainer.scheduler_calls = trainer.full_train_calls = 0
        trainer.save_calls, trainer.load_calls, trainer.test_calls = [], [], []
        trainer.epoch, trainer.batch_idx, trainer.max_epoch = 0, 0, 3
        trainer.num_batches = batches
        trainer.output_dir = "synthetic-output-do-not-create"
        trainer.train_loader_x = [
            {
                "img": torch.randn(8, 7, generator=generator),
                "img2": torch.randn(8, 7, generator=generator),
                "index": torch.arange(8), "impath": ["synthetic_fixture"] * 8,
                "label": torch.arange(8) % 5,
            }
            for _ in range(batches)
        ]
        return trainer

    def test_all_modes_preserve_original_two_view_scalar_losses_and_gradients(self):
        for mode in ("equal", "static_mean", "reliability_adaptive", "instance_only"):
            with self.subTest(mode=mode):
                trainer = self.fixture(mode)
                batch = trainer.train_loader_x[0]
                initial = trainer.model.prompt_learner.ctx.detach().clone()
                with torch.no_grad():
                    logits = []
                    teacher_logits = []
                    for image in (batch["img"], batch["img2"]):
                        scale, image_features, text = trainer.model(image, None)
                        logits.append(scale * image_features @ text.T)
                        teacher_logits.append(trainer.model_teacher(image))
                    expected_components = [
                        sum(function(logits[0], teacher_logits[0], temperature) for temperature in trainer.journal_temperatures)
                        + sum(function(logits[1], teacher_logits[1], temperature) for temperature in trainer.journal_temperatures)
                        for function in (self.namespace["kd_loss"], self.namespace["bc_loss"], self.namespace["cc_loss"])
                    ]
                captured_probabilities = []
                original_reliability = self.namespace["compute_reliability"]

                def capture(original, augmented):
                    captured_probabilities.append((original.detach().clone(), augmented.detach().clone()))
                    return original_reliability(original, augmented)

                samples = []
                trainer.journal_batch_callback = lambda record, batch_samples: samples.extend(batch_samples)
                with patch.dict(self.namespace, {"compute_reliability": capture}):
                    trainer.forward_backward(batch)
                record = trainer.journal_last_record
                for key, expected in zip(("L_ins", "L_batch", "L_cls"), expected_components):
                    self.assertEqual(record[key], expected.item())
                for captured, teacher_logits_view in zip(captured_probabilities[0], teacher_logits):
                    self.assertTrue(torch.equal(captured, F.softmax(teacher_logits_view / trainer.temperature, dim=1)))
                self.assertTrue(record["student_prompt_grad_nonzero"])
                self.assertTrue(record["student_prompt_grad_finite"])
                self.assertTrue(record["teacher_grad_none"])
                self.assertTrue(record["teacher_requires_grad_false"])
                self.assertFalse(torch.equal(initial, trainer.model.prompt_learner.ctx.detach()))
                self.assertEqual(record["L_adapt"], record["loss"])
                self.assertEqual(len(samples), 8)
                self.assertTrue({"sample_index", "teacher_entropy_aug", "teacher_max_probability", "teacher_prediction", "ground_truth"} <= samples[0].keys())
                self.assertTrue(torch.equal(trainer.model_teacher.seen_inputs[-2], batch["img"]))
                self.assertTrue(torch.equal(trainer.model_teacher.seen_inputs[-1], batch["img2"]))
                self.assertAlmostEqual(sum(record[key] for key in ("lambda_ins", "lambda_batch", "lambda_cls")), 1, places=6)
                if mode == "instance_only":
                    self.assertEqual(record["weighted_terms"]["L_batch"], 0)
                    self.assertEqual(record["weighted_terms"]["L_cls"], 0)

    def test_structural_sanity_loop_never_saves_evaluates_or_advances_scheduler(self):
        trainer = self.fixture()
        # The name only exercises the policy guard. The fixture above remains
        # tiny synthetic CPU tensors, not actual DTD data or a CLIP model.
        trainer.cfg.DATASET.NAME = "DescribableTextures"
        records = trainer.run_sanity(10)
        self.assertEqual(len(records), 10)
        self.assertEqual(trainer.scheduler_calls, 0)
        self.assertFalse(trainer.save_calls or trainer.test_calls)
        self.assertFalse(trainer._journal_sanity)
        self.assertTrue(all(record["Base"] is None and record["Novel"] is None for record in records))
        self.assertFalse(records[7]["alternate_weight_diagnostic"]["used_for_optimization"])
        self.assertNotEqual(
            records[7]["lambda_ins"] / records[7]["lambda_batch"],
            records[7]["alternate_weight_diagnostic"]["weights"]["lambda_ins"]
            / records[7]["alternate_weight_diagnostic"]["weights"]["lambda_batch"],
        )
        with self.assertRaises(RuntimeError):
            trainer.before_train()

    def test_sanity_policy_rejects_other_datasets_modes_and_short_checks(self):
        with self.assertRaises(ValueError):
            self.fixture().run_sanity(10)
        trainer = self.fixture("equal")
        trainer.cfg.DATASET.NAME = "DescribableTextures"
        with self.assertRaises(ValueError):
            trainer.run_sanity(10)
        trainer.cfg.JOURNAL.MODE = "reliability_adaptive"
        with self.assertRaises(ValueError):
            trainer.run_sanity(9)

    def test_full_training_requires_external_real_sanity_gate(self):
        trainer = self.fixture()
        with self.assertRaises(ValueError):
            trainer.train()
        trainer.cfg.JOURNAL.SANITY_BATCHES = 0
        runtime = ModuleType("experiments.journal_runtime")
        calls = []

        def rejected(path):
            calls.append(path)
            raise ValueError("No genuine DTD pass")

        runtime.require_sanity_pass = rejected
        with patch.dict(sys.modules, {"experiments.journal_runtime": runtime}):
            with self.assertRaises(ValueError):
                trainer.train()
        self.assertEqual(calls, [trainer.cfg.JOURNAL.SANITY_REPORT])
        self.assertEqual(trainer.full_train_calls, 0)

    def test_epoch_hooks_do_not_select_on_base_or_novel_results(self):
        trainer = self.fixture()
        for trainer.epoch in range(trainer.max_epoch):
            trainer.after_epoch()
        self.assertEqual([call[0][0] for call in trainer.save_calls], [1, 2])
        self.assertFalse(trainer.test_calls)
        with self.assertRaises(ValueError):
            trainer.save_model(0, "unused", is_best=True)
        with self.assertRaises(ValueError):
            trainer.load_model("unused")
        with self.assertRaises(ValueError):
            trainer.load_model("unused", epoch=1)
        with patch.dict(self.namespace, {"load_checkpoint": lambda path: {"epoch": 3}}):
            trainer.after_train()
        self.assertEqual(trainer.load_calls, [(trainer.output_dir, 3)])
        self.assertEqual(trainer.test_calls, ["val", "test"])
        self.assertAlmostEqual(trainer.journal_final_result["HM"], 2 * 80 * 60 / 140)

    def test_explicit_teacher_stage_rejects_best_selected_filename(self):
        trainer = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "model-best.pth.tar"
            selected.write_text("synthetic fixture; no weights", encoding="utf-8")
            trainer.cfg.JOURNAL.TEACHER_CHECKPOINT = str(selected)
            with self.assertRaises(ValueError):
                trainer._resolve_teacher_checkpoint()
            stage = Path(directory) / "model.pth.tar-20"
            stage.write_text("synthetic fixture; no weights", encoding="utf-8")
            trainer.cfg.JOURNAL.TEACHER_CHECKPOINT = str(stage)
            self.assertEqual(trainer._resolve_teacher_checkpoint(), str(stage.resolve()))


if __name__ == "__main__":
    unittest.main()
