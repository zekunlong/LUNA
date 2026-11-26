"""Archive-backed legacy regression checks without importing trainer dependencies.

The extracted code_luna.zip is an optional provenance fixture, not a training
asset. If absent, these archive comparisons explicitly skip rather than claim
that legacy compatibility was established.
"""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO.parents[1] / "code_luna.zip"
SOURCE_PATH = REPO / "trainers" / "luna.py"


def ast_equivalent(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def top_level(tree, name):
    return next(node for node in tree.body if getattr(node, "name", None) == name)


def method(class_node, name):
    return next(node for node in class_node.body if getattr(node, "name", None) == name)


def original_checkpoint_block(build_node):
    matches = [node for node in build_node.body if isinstance(node, ast.If)
               and ast.unparse(node.test) == "cfg.TRAINER.MODAL == 'base2novel'"]
    if len(matches) != 1:
        raise AssertionError("Archive must contain one original teacher checkpoint resolution block")
    return copy.deepcopy(matches[0])


def extracted_resolver(tree, *, baseline):
    """Execute only the inspected filesystem helpers and path resolution code."""
    definitions = [copy.deepcopy(top_level(tree, name))
                   for name in ("_required_file", "_required_directory")]
    luna = top_level(tree, "LUNA")
    if baseline:
        resolver = ast.parse("def _resolve_teacher_checkpoint(self):\n    cfg = self.cfg\n").body[0]
        resolver.body.extend((original_checkpoint_block(method(luna, "build_model")),
                              ast.Return(value=ast.Name(id="model_path", ctx=ast.Load()))))
    else:
        resolver = copy.deepcopy(method(luna, "_resolve_teacher_checkpoint"))
        resolver.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=definitions + [resolver], type_ignores=[]))
    namespace = {"os": os, "osp": os.path}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_resolve_teacher_checkpoint"]


def config(modal="base2novel", root="", cross="", dataset="DTD"):
    return SimpleNamespace(cfg=SimpleNamespace(
        DATASET=SimpleNamespace(NAME=dataset),
        TRAINER=SimpleNamespace(MODAL=modal, LUNA=SimpleNamespace(
            TEACHER_CHECKPOINT_ROOT=root, CROSS_DATASET_TEACHER_CHECKPOINT=cross,
        )),
    ))


class LegacyArchiveCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ARCHIVE.is_file():
            raise unittest.SkipTest(f"Optional original source archive unavailable: {ARCHIVE}")
        with zipfile.ZipFile(ARCHIVE) as archive:
            cls.baseline_files = {
                info.filename.removeprefix("LUNA/"): archive.read(info)
                for info in archive.infolist()
                if info.filename.startswith("LUNA/") and not info.is_dir()
            }
        cls.baseline_tree = ast.parse(cls.baseline_files["trainers/luna.py"].decode("utf-8-sig"))
        cls.current_tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8-sig"))
        cls.baseline_luna = top_level(cls.baseline_tree, "LUNA")
        cls.current_luna = top_level(cls.current_tree, "LUNA")
        cls.original_resolver = staticmethod(extracted_resolver(cls.baseline_tree, baseline=True))
        cls.current_resolver = staticmethod(extracted_resolver(cls.current_tree, baseline=False))

    def test_original_loss_helpers_and_nontrainer_classes_are_ast_identical(self):
        # Includes kd_loss, bc_loss, cc_loss, CLIP forward methods, prompt
        # learners, teacher loading, and the original path validation helpers.
        checked = []
        for original in self.baseline_tree.body:
            if isinstance(original, (ast.FunctionDef, ast.ClassDef)) and original.name != "LUNA":
                with self.subTest(definition=original.name):
                    current = top_level(self.current_tree, original.name)
                    self.assertEqual(ast_equivalent(original), ast_equivalent(current))
                    checked.append(original.name)
        for required in ("kd_loss", "bc_loss", "cc_loss", "CustomCLIP", "CustomCLIP_teacher"):
            self.assertIn(required, checked)

    def test_original_trainer_methods_including_forward_backward_are_ast_identical(self):
        checked = []
        for original in self.baseline_luna.body:
            if isinstance(original, ast.FunctionDef) and original.name != "build_model":
                with self.subTest(method=original.name):
                    current = method(self.current_luna, original.name)
                    self.assertEqual(ast_equivalent(original), ast_equivalent(current))
                    checked.append(original.name)
        self.assertIn("forward_backward", checked)

    def test_build_model_changes_only_checkpoint_resolution_call(self):
        original = method(self.baseline_luna, "build_model")
        current = copy.deepcopy(method(self.current_luna, "build_model"))
        seam_indices = [index for index, node in enumerate(current.body)
                        if isinstance(node, ast.Assign)
                        and ast.unparse(node) == "model_path = self._resolve_teacher_checkpoint()"]
        self.assertEqual(len(seam_indices), 1, "Expected exactly one explicit checkpoint override seam")
        current.body[seam_indices[0]] = original_checkpoint_block(original)
        self.assertEqual(ast_equivalent(current), ast_equivalent(original))

    def test_original_data_loaders_datasets_configs_and_scripts_are_byte_identical(self):
        prefixes = ("datasets/", "Dassl.pytorch/dassl/data/", "scripts/", "configs/")
        checked = []
        for relative, original_bytes in self.baseline_files.items():
            if relative.startswith(prefixes):
                with self.subTest(file=relative):
                    self.assertEqual((REPO / relative).read_bytes(), original_bytes)
                    checked.append(relative)
        self.assertTrue(any(path.startswith("datasets/") for path in checked))
        self.assertTrue(any(path.startswith("Dassl.pytorch/dassl/data/") for path in checked))
        self.assertTrue(any(path.startswith("scripts/") for path in checked))
        self.assertTrue(any(path.startswith("configs/") for path in checked))

    def test_base_to_new_resolution_preserves_original_best_checkpoint_path(self):
        with tempfile.TemporaryDirectory(prefix="luna_legacy_path_") as temporary:
            root = Path(temporary)
            expected = root / "DTD" / "VLPromptLearner" / "model-best.pth.tar"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"path-only fixture; never loaded as a checkpoint")
            request = config(root=str(root))
            self.assertEqual(self.current_resolver(request), os.path.realpath(expected))
            self.assertEqual(self.current_resolver(request), self.original_resolver(request))

    def test_cross_dataset_resolution_preserves_external_file_and_expansion(self):
        with tempfile.TemporaryDirectory(prefix="luna_legacy_path_") as temporary:
            expected = Path(temporary) / "teacher prompt final.pth.tar"
            expected.write_bytes(b"path-only fixture; never loaded as a checkpoint")
            with patch.dict(os.environ, {"LUNA_TEST_CROSS_CHECKPOINT": str(expected)}):
                request = config(modal="cross", cross="$LUNA_TEST_CROSS_CHECKPOINT")
                self.assertEqual(self.current_resolver(request), os.path.realpath(expected))
                self.assertEqual(self.current_resolver(request), self.original_resolver(request))

    def test_original_missing_asset_and_unsupported_modal_errors_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix="luna_legacy_path_") as temporary:
            root = Path(temporary)
            existing_file = root / "file-not-directory"
            existing_file.write_bytes(b"fixture")
            cases = [
                (config(root=""), ValueError, "TRAINER.LUNA.TEACHER_CHECKPOINT_ROOT is not set"),
                (config(root=str(root / "missing")), FileNotFoundError, "does not point to a directory"),
                (config(root=str(existing_file)), FileNotFoundError, "does not point to a directory"),
                (config(root=str(root)), FileNotFoundError, "LUNA base-to-new teacher checkpoint"),
                (config(modal="cross", cross=""), ValueError, "CROSS_DATASET_TEACHER_CHECKPOINT is not set"),
                (config(modal="cross", cross=str(root / "missing.pth.tar")), FileNotFoundError, "does not point to a file"),
                (config(modal="cross", cross=str(root)), FileNotFoundError, "does not point to a file"),
                (config(modal="invalid"), ValueError, 'Unsupported TRAINER.MODAL="invalid"'),
            ]
            for request, exception_type, fragment in cases:
                with self.subTest(case=fragment, modal=request.cfg.TRAINER.MODAL):
                    with self.assertRaises(exception_type) as original:
                        self.original_resolver(request)
                    with self.assertRaises(exception_type) as current:
                        self.current_resolver(request)
                    self.assertEqual(str(current.exception), str(original.exception))
                    self.assertIn(fragment, str(current.exception))


if __name__ == "__main__":
    unittest.main()
