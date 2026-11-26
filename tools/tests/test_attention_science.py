"""CPU-only mathematical/contract tests; synthetic arrays are test fixtures.

No synthetic attribution or image is exported into the production figures.
Run: python tools/tests/test_attention_science.py
"""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import attention_figure as figures
import attention_localization as localization
import audit_attention_localization as audit
import evaluate_attention_comparison as evaluation
import run_attention_queue as queue
import visualize_attention_comparison as inference

torch.set_num_threads(4)


class MathTests(unittest.TestCase):
    def test_class_bootstrap_keeps_pairing_and_equal_class_weights(self):
        result = audit.paired_cluster_bootstrap([.4, .6, .2, .1, np.nan], [.1, .3, .3, .2, 0.], [0, 0, 1, 2, 2])
        self.assertEqual(result["valid_images"], 4)
        self.assertEqual(result["valid_classes"], 3)
        self.assertAlmostEqual(result["estimate"], (.3-.1-.1)/3)
        self.assertNotAlmostEqual(result["estimate"], .1)  # pooled-image mean would be wrong
        self.assertAlmostEqual(result["paired_luna_mean"]-result["paired_rival_mean"], result["estimate"])
        np.testing.assert_allclose(result["ci95"], [-.1, .3], atol=1e-12)
        self.assertEqual(result["valid_images_by_class"], {"0": 2, "1": 1, "2": 1})

    def test_bootstrap_missing_and_single_cluster_are_not_false_certainty(self):
        self.assertEqual(audit.paired_cluster_bootstrap([np.nan], [1.], [0])["interval_status"], "NO_VALID_PAIRS")
        one = audit.paired_cluster_bootstrap([.3, .5], [.1, .1], [0, 0])
        self.assertAlmostEqual(one["estimate"], .3)
        self.assertIsNone(one["ci95"])
        self.assertEqual(one["interval_status"], "INSUFFICIENT_CLASS_CLUSTERS")
        same = audit.paired_cluster_bootstrap([.25, .25], [0., 0.], [0, 1])
        self.assertEqual(same["interval_status"], "DEGENERATE_IDENTICAL_CLASS_DIFFERENCES")
        self.assertEqual(same["ci95"], [.25, .25])

    def test_audit_validity_ties_preserve_mass_but_do_not_invent_pointing(self):
        uniform = audit.checked_box_metrics(np.ones((2, 2)), [0, 0, 112, 224], 224)
        self.assertAlmostEqual(uniform["bbox_mass"], .5)
        self.assertAlmostEqual(uniform["bbox_enrichment"], 1.)
        self.assertIsNone(uniform["pointing_inside_bbox"])
        self.assertTrue(uniform["pointing_tied"])
        for raw, reason in [(np.zeros((2, 2)), "ZERO_RELEVANCE"), (np.full((2, 2), np.nan), "INVALID_RELEVANCE"), (np.asarray([[2., 2.], [-1., 2.]]), "INVALID_RELEVANCE")]:
            result = audit.checked_box_metrics(raw, [0, 0, 224, 224], 224)
            self.assertEqual(result["validity"], reason)
            self.assertTrue(all(result[key] is None for key in audit.METRICS))

    def test_gradcam_excludes_cls_and_uses_relu(self):
        # Channel averages are [2,-1]. Spatial weighted sums are [2,-1,1,-1].
        activation = torch.tensor([[[999., 999.]], [[1., 0.]], [[0., 1.]], [[1., 1.]], [[-1., -1.]]])
        gradient = torch.tensor([[[9999., 9999.]], [[2., -1.]], [[2., -1.]], [[2., -1.]], [[2., -1.]]])
        result = inference.gradcam_relevance(activation, gradient, 4)
        torch.testing.assert_close(result, torch.tensor([[2/3, 0., 1/3, 0.]]))
        activation[0] = -99999
        gradient[0] = -99999
        torch.testing.assert_close(result, inference.gradcam_relevance(activation, gradient, 4))
        self.assertEqual(float(inference.gradcam_relevance(activation, gradient*0, 4).sum()), 0.)

    def test_hook_observes_original_forward_and_is_removed(self):
        layer = torch.nn.Linear(2, 3)
        x = torch.tensor([[1., 2.]], requires_grad=True)
        expected = layer(x)
        with inference.LayerActivationCapture(layer) as captured:
            actual = layer(x)
            self.assertIs(actual, captured.activation)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(layer._forward_hooks)

    def test_entropy_and_spatial_mass_have_known_endpoints(self):
        uniform = figures.spatial_metrics(np.ones((10, 10)))
        point = np.zeros((10, 10)); point[2, 3] = 1
        concentrated = figures.spatial_metrics(point)
        self.assertAlmostEqual(uniform["normalized_entropy"], 1.)
        self.assertAlmostEqual(uniform["top20_mass"], .2)
        self.assertAlmostEqual(concentrated["normalized_entropy"], 0.)
        self.assertAlmostEqual(concentrated["top20_mass"], 1.)
        self.assertTrue(figures.spatial_metrics(np.zeros((2, 2)))["zero_map"])

    def test_renderer_scale_is_shared_and_invariant_to_raw_magnitude(self):
        raw = np.asarray([[[4., 0.], [0., 0.]], [[1., 1.], [1., 1.]]])
        display, bounds = figures.shared_row_maps(raw, (2, 2))
        np.testing.assert_allclose(display[0], [[1, 0], [0, 0]])
        np.testing.assert_allclose(display[1], np.full((2, 2), .25))
        scaled = raw*np.asarray([100., .1])[:, None, None]
        np.testing.assert_allclose(figures.shared_row_maps(scaled, (2, 2))[0], display)
        np.testing.assert_allclose(figures.shared_row_maps(raw[::-1], (2, 2))[0], display[::-1])

    def test_bbox_overlap_integrates_partial_patches(self):
        uniform = np.ones((2, 2))
        # A quarter-width box overlaps half of each left-column patch.
        box = [0, 0, 56, 224]
        result = localization.box_metrics(uniform, box)
        self.assertAlmostEqual(result["bbox_area_fraction"], .25)
        self.assertAlmostEqual(result["bbox_mass"], .25)
        self.assertAlmostEqual(result["bbox_enrichment"], 1.)
        point = np.asarray([[1., 0.], [0., 0.]])
        result = localization.box_metrics(point, box)
        self.assertAlmostEqual(result["bbox_mass"], .5)
        self.assertAlmostEqual(result["background_mass"], .5)
        self.assertAlmostEqual(result["bbox_enrichment"], 2.)
        self.assertFalse(result["pointing_inside_bbox"])  # center lies on excluded x2 edge
        full = localization.box_metrics(point, [0, 0, 224, 224])
        self.assertEqual(full["bbox_mass"], 1.)
        self.assertIsNone(localization.box_metrics(point*0, box)["bbox_mass"])

    def test_bbox_crop_matches_actual_torchvision_resize_center_crop(self):
        transform = transforms.Compose([transforms.Resize(224, interpolation=InterpolationMode.NEAREST), transforms.CenterCrop(224)])
        for size, box in [((333, 501), (90, 100, 260, 440)), ((501, 333), (100, 90, 440, 260)), ((225, 224), (30, 20, 180, 200)), ((224, 227), (20, 30, 200, 180)), ((64, 64), (10, 10, 50, 50))]:
            with self.subTest(size=size):
                mask = np.zeros((size[1], size[0]), dtype=np.uint8)
                x1, y1, x2, y2 = box; mask[y1:y2, x1:x2] = 255
                cropped = np.asarray(transform(Image.fromarray(mask)))
                ys, xs = np.nonzero(cropped)
                observed = np.asarray([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
                expected = localization.crop_box(box, size)
                np.testing.assert_allclose(observed, expected, atol=1.01, rtol=0)
                np.testing.assert_allclose(localization.crop_box([0, 0, *size], size), [0, 0, 224, 224])


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="attention_science_")
        self.root = Path(self.tmp.name).resolve()
        self.assertTrue(self.root.name.startswith("attention_science_"))

    def tearDown(self):
        self.tmp.cleanup()

    def bundle(self, methods=("CLIP", "LUNA"), **extra):
        path = self.root / "fixture.npz"
        values = dict(images=np.zeros((1, 2, 2, 3), np.float32), maps=np.ones((1, len(methods), 2, 2)), methods=np.asarray(methods), classnames=np.asarray(["test-fixture"]), metadata=np.asarray(json.dumps({"status": "complete"})))
        values.update(extra)
        np.savez(path, **values)
        return path

    def test_missing_methods_never_pass_full_render_even_claimed_complete(self):
        path = self.bundle()
        with self.assertRaisesRegex(ValueError, "Incomplete comparison"):
            figures.load_bundle(path)
        partial = figures.load_bundle(path, allow_partial=True)
        self.assertEqual(partial["missing"], ["CoOp", "KgCoOp", "DeKg"])

    def test_invalid_maps_and_pickled_arrays_fail_closed(self):
        path = self.bundle(methods=figures.METHODS, maps=-np.ones((1, 5, 2, 2)))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            figures.load_bundle(path)
        path = self.bundle(methods=figures.METHODS, metadata=np.asarray({"status": "complete"}, dtype=object))
        with self.assertRaises(ValueError):
            figures.load_bundle(path)
        maps = np.ones((1, 5, 2, 2)); maps[:, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "KgCoOp"):
            figures.load_bundle(self.bundle(methods=figures.METHODS, maps=maps))

    def test_preflight_requires_explicit_partial_and_missing_assets_remain_errors(self):
        asset = self.root / "fixture.bin"; asset.write_bytes(b"TEST ONLY, not a model")
        config = {"_config_path": "test-only", "clip_path": str(asset), "classnames": ["x"], "samples": [{"classname": "x", "impath": str(asset)}], "methods": [{"name": "CLIP"}]}
        self.assertEqual(inference.preflight(config)["status"], "BLOCKED")
        self.assertEqual(inference.preflight(config, True)["status"], "READY_PARTIAL")
        config["samples"][0]["impath"] = str(self.root/"missing.jpg")
        self.assertEqual(inference.preflight(config, True)["status"], "BLOCKED")

    def test_queue_requires_declared_outputs_and_completion_receipt(self):
        artifact = self.root/"checkpoint"; artifact.write_bytes(b"TEST ONLY")
        receipt = self.root/"receipt.json"
        stage = {"outputs": [str(artifact)], "completion_json": {"path": str(receipt), "value": "COMPLETED"}}
        self.assertFalse(queue.complete(stage))
        receipt.write_text('{"status":"RUNNING"}')
        self.assertFalse(queue.complete(stage))
        receipt.write_text('{"status":"COMPLETED"}')
        self.assertTrue(queue.complete(stage))
        artifact.write_bytes(b"")
        self.assertFalse(queue.complete(stage))

    def test_selected_examples_are_declared_and_controls_ignore_predictions(self):
        names = ["base0", "base1", "base2", "novel3", "novel4", "novel5"]
        labels = [3, 3, 4, 4, 5, 5]
        samples = [{"impath": "sample{}.jpg".format(i), "label": label, "classname": names[label]} for i, label in enumerate(labels)]
        correct = np.asarray([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=bool)
        margins = np.full((6, 5), .1); margins[:, 4] = .2; margins[1, 4] = .4
        config = {"dataset": "synthetic_test_only", "title": "Proper Title", "clip_path": "unused", "methods": [{"name": name} for name in figures.METHODS]}
        report = evaluation.make_selections(samples, names, correct, margins, config, self.root/"first")
        self.assertEqual(report["selected_success"]["selected_images"], ["sample1.jpg", "sample2.jpg"])
        self.assertEqual(report["selected_success"]["shortfall"], 1)
        second = evaluation.make_selections(samples, names, ~correct, -margins, config, self.root/"second")
        self.assertEqual(report["random_control"]["selected_images"], second["random_control"]["selected_images"])
        for name in ("selected_success", "random_control", "luna_failures"):
            path = self.root/"first"/(name+".json")
            self.assertTrue(path.exists())  # These exact filenames are consumed by the production queue.
            selected = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(selected["display_name"], "Proper Title")
            self.assertEqual(selected["classnames"], names[3:])
            self.assertTrue(selected["figure_note"])
            self.assertEqual(len({row["label"] for row in selected["samples"]}), len(selected["samples"]))

    def test_evaluator_base_novel_probability_indices_with_toy_classifier(self):
        # Test-only fixed embeddings deliberately prefer an outside-group class
        # for some images. Correct evaluation must restrict candidates by group.
        class ToyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_scale = torch.nn.Parameter(torch.tensor(0.))
                self.visual = SimpleNamespace(input_resolution=224)
                self.values = torch.tensor([[1., 0., 10., 0.], [1., 0., 0., 0.], [10., 0., 0., 1.], [0., 0., 1., 2.]])
                self.cursor = 0
                self.calls = 0

            def encode_image(self, tensor):
                start = self.cursor; self.cursor += len(tensor); self.calls += 1
                return self.values[start:self.cursor]

        samples = []
        names = ["base0", "base1", "novel2", "novel3"]
        for index, name in enumerate(names):
            path = self.root/(str(index)+".png")
            Image.new("RGB", (8, 8), color=(index*40, 0, 0)).save(path)
            samples.append({"impath": str(path), "label": index, "classname": name})
        manifest = self.root/"dataset.json"
        manifest.write_text(json.dumps({"dataset": "synthetic_test_only", "classnames": names, "train": [], "val": [], "test": samples}))
        config = self.root/"models.json"
        config.write_text(json.dumps({"dataset": "synthetic_test_only", "clip_path": str(self.root/"unused.pt"), "classnames": names, "methods": [{"name": name} for name in figures.METHODS]}))
        encoder = ToyEncoder()
        loaded = SimpleNamespace(clip_model=encoder, text_features={name: torch.eye(4) for name in figures.METHODS}, metadata={"synthetic_test_only": True})
        args = SimpleNamespace(dataset_manifest=manifest, models_config=config, output_dir=self.root/"evaluation", device="cpu", batch_size=2, cpu_threads=4, selection_seed=1, max_samples=4)
        with patch("attention_models.load_attention_models", return_value=loaded), contextlib.redirect_stdout(io.StringIO()):
            evaluation.evaluate(args)
        self.assertEqual(encoder.calls, 2)  # shared across all five methods
        with np.load(args.output_dir/"evaluation_arrays.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["predictions"], np.asarray([[0]*5, [0]*5, [3]*5, [3]*5]))
            np.testing.assert_array_equal(arrays["correct"], np.asarray([[1]*5, [0]*5, [0]*5, [1]*5], dtype=bool))
            self.assertAlmostEqual(float(arrays["target_probabilities"][1, 0]), 1/(1+np.e), places=6)
            self.assertAlmostEqual(float(arrays["target_probabilities"][2, 0]), 1/(1+np.exp(1/np.sqrt(101))), places=6)
            self.assertAlmostEqual(float(arrays["target_margins"][2, 0]), -1/np.sqrt(101), places=6)
        summary = json.loads((args.output_dir/"evaluation_summary.json").read_text())
        self.assertEqual(summary["status"], "SMOKE_SUBSET_NOT_FULL_EVALUATION")
        for method in figures.METHODS:
            self.assertEqual(summary["methods"][method]["base"]["accuracy_percent"], 50.)
            self.assertEqual(summary["methods"][method]["novel"]["accuracy_percent"], 50.)
            self.assertEqual(summary["methods"][method]["harmonic_mean_percent"], 50.)


class RealAnnotationTests(unittest.TestCase):
    def test_cars_mat_loader_exactly_matches_all_original_bbox_records(self):
        root = Path("E:/LUNA/data")
        source = root/"stanford_cars/bounding_boxes.json"
        if not source.is_file():
            self.skipTest("Cars acquired annotation fixture is unavailable")
        boxes, _ = localization.annotations("stanford_cars", root)
        expected = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(len(boxes), 16185)
        self.assertEqual(set(boxes), set(expected))
        for name, row in expected.items():
            x1, y1, x2, y2 = row["bbox_xyxy"]
            np.testing.assert_array_equal(boxes[name], [x1-1, y1-1, x2, y2])


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
