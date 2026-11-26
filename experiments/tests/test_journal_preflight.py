"""Fixture tests for read-only journal preflight; no real model/data inference."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.journal_preflight import inspect_assets


class JournalPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clip = self.root / "teacher_clip.pt"
        self.checkpoint = self.root / "model.pth.tar-20"
        self.clip.write_bytes(b"fixture-not-a-real-model")
        self.checkpoint.write_bytes(b"fixture-not-a-real-checkpoint")
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def _fixture(self, dataset="dtd"):
        directory = self.root / dataset
        images = directory / ("2750" if dataset == "eurosat" else "images")
        images.mkdir(parents=True)
        if dataset == "fgvc_aircraft":
            (directory / "variants.txt").write_text("Aircraft One\n", encoding="utf-8")
            for index, split in enumerate(("train", "val", "test"), 1):
                name = str(index).zfill(7)
                (images / (name + ".jpg")).write_bytes(b"image-fixture")
                (directory / ("images_variant_" + split + ".txt")).write_text(name + " Aircraft One\n", encoding="utf-8")
            return directory
        records = {}
        for split in ("train", "val", "test"):
            (images / (split + ".jpg")).write_bytes(b"image-fixture")
            records[split] = [[split + ".jpg", 0, "class"]]
        name = "split_zhou_EuroSAT.json" if dataset == "eurosat" else "split_zhou_DescribableTextures.json"
        (directory / name).write_text(json.dumps(records), encoding="utf-8")
        return directory

    def _inspect(self, dataset="dtd", **overrides):
        args = dict(dataset=dataset, data_root=self.root, teacher_clip_path=self.clip,
                    teacher_checkpoint=self.checkpoint, teacher_total_epochs=20,
                    check_environment=False)
        args.update(overrides)
        return inspect_assets(**args)

    def _rewrite_split(self, edit):
        path = self.root / "dtd" / "split_zhou_DescribableTextures.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        edit(records)
        path.write_text(json.dumps(records), encoding="utf-8")

    def test_existing_dtd_ready_without_loading_fixture_checkpoints(self):
        self._fixture()
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self._inspect("DescribableTextures")
        self.assertEqual(result["status"], "READY")
        self.assertFalse(result["environment"]["checked"])
        self.assertEqual(result["assets"]["teacher_checkpoint"]["metadata_status"], "unvalidated_not_loaded")
        self.assertEqual(result["assets"]["dataset"]["split_counts"], {"train": 1, "val": 1, "test": 1})
        after = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.root / "dtd" / "split_fewshot").exists())

    def test_supported_dataset_aliases(self):
        self._fixture("fgvc_aircraft")
        self._fixture("eurosat")
        for alias in ("Aircraft", "FGVCAircraft", "fgvc_aircraft", "EuroSAT"):
            with self.subTest(alias=alias):
                self.assertEqual(self._inspect(alias)["status"], "READY")

    def test_missing_root_is_not_created(self):
        missing = self.root / "absent"
        result = self._inspect(data_root=missing)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(missing.exists())
        self.assertTrue(any("Dataset root" in item for item in result["blockers"]))

    def test_missing_split_is_not_regenerated(self):
        directory = self._fixture()
        split = directory / "split_zhou_DescribableTextures.json"
        split.unlink()
        result = self._inspect()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(split.exists())

    def test_missing_image_is_reported(self):
        directory = self._fixture()
        (directory / "images" / "val.jpg").unlink()
        result = self._inspect()
        self.assertEqual(result["assets"]["dataset"]["missing_image_count"], 1)
        self.assertEqual(result["status"], "BLOCKED")

    def test_invalid_json_is_reported(self):
        directory = self._fixture()
        (directory / "split_zhou_DescribableTextures.json").write_text("not json", encoding="utf-8")
        self.assertTrue(any("Invalid existing split JSON" in item for item in self._inspect()["blockers"]))

    def test_invalid_record_labels_are_rejected(self):
        self._fixture()
        self._rewrite_split(lambda data: data["train"][0].__setitem__(1, True))
        self.assertEqual(self._inspect()["status"], "BLOCKED")

    def test_path_traversal_and_absolute_references_are_rejected(self):
        self._fixture()
        for reference in ("../outside.jpg", "/outside.jpg", "C:\\outside.jpg"):
            with self.subTest(reference=reference):
                self._rewrite_split(lambda data: data["train"][0].__setitem__(0, reference))
                result = self._inspect()
                self.assertTrue(any("inside the dataset image directory" in item for item in result["blockers"]))

    def test_repeated_image_across_splits_is_rejected(self):
        self._fixture()
        self._rewrite_split(lambda data: data.__setitem__("val", data["train"]))
        result = self._inspect()
        self.assertEqual(result["assets"]["dataset"]["duplicate_reference_count"], 1)
        self.assertEqual(result["status"], "BLOCKED")

    def test_best_teacher_checkpoint_is_forbidden(self):
        self._fixture()
        for filename in ("model-best.pth.tar", "teacher_best.pth"):
            checkpoint = self.root / filename
            checkpoint.write_bytes(b"checkpoint-fixture")
            result = self._inspect(teacher_checkpoint=checkpoint)
            self.assertTrue(any("best-selected" in item for item in result["blockers"]))

    def test_teacher_filename_epoch_must_match_declared_final_epoch(self):
        self._fixture()
        result = self._inspect(teacher_total_epochs=10)
        self.assertTrue(any("filename epoch disagrees" in item for item in result["blockers"]))

    def test_preselected_intermediate_teacher_epochs_are_accepted(self):
        self._fixture()
        for quality, epoch in (("weak", 5), ("medium", 12)):
            checkpoint = self.root / f"model.pth.tar-{epoch}"
            checkpoint.write_bytes(b"intermediate-checkpoint-fixture")
            with self.subTest(quality=quality):
                result = self._inspect(teacher_checkpoint=checkpoint, teacher_quality=quality)
                self.assertEqual(result["status"], "READY", result["blockers"])
                self.assertEqual(result["assets"]["teacher_checkpoint"]["filename_epoch"], epoch)
                self.assertEqual(self._inspect(teacher_checkpoint=checkpoint)["status"], "BLOCKED")

    def test_intermediate_teachers_must_precede_final_epoch(self):
        self._fixture()
        for epoch in (0, 20, 21):
            checkpoint = self.root / f"model.pth.tar-{epoch}"
            checkpoint.write_bytes(b"invalid-intermediate-checkpoint-fixture")
            for quality in ("weak", "medium"):
                with self.subTest(epoch=epoch, quality=quality):
                    result = self._inspect(teacher_checkpoint=checkpoint, teacher_quality=quality)
                    self.assertTrue(any("must precede" in item for item in result["blockers"]))
        self.assertEqual(self._inspect(teacher_quality="selected_by_accuracy")["status"], "BLOCKED")

    def test_invalid_total_epochs(self):
        self._fixture()
        for epochs in (None, 0, -1, True, 1.5, "five", "1.5"):
            with self.subTest(epochs=epochs):
                result = self._inspect(teacher_total_epochs=epochs)
                self.assertTrue(any("positive integer" in item for item in result["blockers"]))

    def test_missing_and_empty_checkpoint_assets(self):
        self._fixture()
        self.clip.write_bytes(b"")
        self.checkpoint.unlink()
        result = self._inspect()
        self.assertTrue(any("base weights is empty" in item for item in result["blockers"]))
        self.assertTrue(any("final teacher checkpoint does not exist" in item for item in result["blockers"]))

    def test_environment_fallback_and_argument_precedence(self):
        self._fixture()
        values = {"LUNA_DATA_ROOT": str(self.root), "LUNA_TEACHER_CLIP_PATH": str(self.clip),
                  "LUNA_TEACHER_CHECKPOINT": str(self.checkpoint), "LUNA_TEACHER_TOTAL_EPOCHS": "20"}
        with patch.dict(os.environ, values):
            result = inspect_assets("dtd", check_environment=False)
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["assets"]["teacher_checkpoint"]["source"], "environment:LUNA_TEACHER_CHECKPOINT")
            explicit_empty = inspect_assets("dtd", data_root="", check_environment=False)
            self.assertEqual(explicit_empty["assets"]["data_root"]["source"], "argument")
            self.assertEqual(explicit_empty["status"], "BLOCKED")

    def test_no_assets_or_invalid_dataset(self):
        result = inspect_assets("dtd", check_environment=False)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertGreaterEqual(len(result["blockers"]), 4)
        for name in ("ImageNet", "", None):
            with self.subTest(dataset=name):
                self.assertEqual(inspect_assets(name, check_environment=False)["status"], "BLOCKED")

    def test_aircraft_unknown_variant_and_missing_metadata(self):
        directory = self._fixture("fgvc_aircraft")
        (directory / "images_variant_train.txt").write_text("0000001 Unknown Plane\n", encoding="utf-8")
        (directory / "images_variant_val.txt").unlink()
        result = self._inspect("aircraft")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("Invalid Aircraft train" in item for item in result["blockers"]))
        self.assertTrue(any("images_variant_val.txt" in item for item in result["blockers"]))


if __name__ == "__main__":
    unittest.main()
