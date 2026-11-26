"""Synthetic equation checks only; these are NOT the required DTD sanity run."""

from __future__ import annotations

import math
import unittest

import torch

from experiments.journal_losses import (
    EPS,
    RELIABILITY_KEYS,
    WEIGHT_KEYS,
    aggregate_losses,
    compute_reliability,
    reliability_weights,
    select_weights,
)


def reference_reliability(original, augmented):
    """Plain Python reference, independent of torch matrix/reduction code."""
    batch_size, classes = len(original), len(original[0])
    entropy = sum(
        -sum(p * math.log(p + 1e-8) for p in row)
        for matrix in (original, augmented) for row in matrix
    ) / (2 * batch_size)

    def sample_relation(matrix):
        return [sum(matrix[i][c] * matrix[j][c] for c in range(classes))
                for i in range(batch_size) for j in range(batch_size)]

    def class_relation(matrix):
        return [sum(matrix[i][c] * matrix[i][d] for i in range(batch_size))
                for c in range(classes) for d in range(classes)]

    def cosine(left, right):
        return sum(x * y for x, y in zip(left, right)) / (
            math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right)) + 1e-8
        )

    return [max(0, min(1, 1 - entropy / math.log(classes))),
            max(0, min(1, cosine(sample_relation(original), sample_relation(augmented)))),
            max(0, min(1, cosine(class_relation(original), class_relation(augmented))))]


class JournalReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.original = torch.tensor([[0.8, 0.1, 0.1], [0.05, 0.35, 0.6]], dtype=torch.float64)
        self.augmented = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2]], dtype=torch.float64)

    def test_exact_entropy_and_both_relation_equations(self):
        actual = compute_reliability(self.original, self.augmented)
        reference = reference_reliability(self.original.tolist(), self.augmented.tolist())
        self.assertEqual(tuple(actual), RELIABILITY_KEYS)
        for name, expected in zip(RELIABILITY_KEYS, reference):
            self.assertEqual(actual[name].ndim, 0)
            self.assertEqual(actual[name].dtype, torch.float64)
            self.assertAlmostEqual(actual[name].item(), expected, places=14)

    def test_entropy_averages_both_views(self):
        confident = torch.eye(3, dtype=torch.float64)
        uncertain = torch.full((3, 3), 1 / 3, dtype=torch.float64)
        r = compute_reliability(confident, uncertain)
        self.assertAlmostEqual(r["r_ins"].item(), 0.5, places=7)
        reversed_r = compute_reliability(uncertain, confident)
        for key in RELIABILITY_KEYS:
            torch.testing.assert_close(r[key], reversed_r[key])

    def test_probability_inputs_detached_and_teacher_receives_no_gradient(self):
        teacher_logits = torch.randn(4, 3, requires_grad=True)
        teacher_aug_logits = torch.randn(4, 3, requires_grad=True)
        r = compute_reliability(teacher_logits.softmax(-1), teacher_aug_logits.softmax(-1))
        weights = reliability_weights(r)
        for value in (*r.values(), *weights.values()):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        student_prompt = torch.tensor([0.4, -0.3, 0.6], requires_grad=True)
        losses = ((student_prompt[0] - 1).square(), (student_prompt[1] + 1).square(),
                  (student_prompt[2] - 2).square())
        total, _ = aggregate_losses(*losses, reliabilities=r)
        total.backward()
        self.assertGreater(student_prompt.grad.norm().item(), 0)
        self.assertIsNone(teacher_logits.grad)
        self.assertIsNone(teacher_aug_logits.grad)

    def test_supplied_reliability_and_static_weights_are_detached(self):
        r = {name: torch.tensor(value, requires_grad=True)
             for name, value in zip(RELIABILITY_KEYS, (0.1, 0.3, 0.6))}
        static = torch.tensor([0.1, 0.3, 0.6], requires_grad=True)
        for weights in (reliability_weights(r), select_weights("static_mean", static_weights=static)):
            self.assertTrue(all(not value.requires_grad for value in weights.values()))

    def test_exact_normalization_epsilon_bounds_and_zero_case(self):
        for values in ((0.2, 0.3, 0.7), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)):
            r = {name: torch.tensor(value, dtype=torch.float64)
                 for name, value in zip(RELIABILITY_KEYS, values)}
            weights = reliability_weights(r)
            self.assertEqual(tuple(weights), WEIGHT_KEYS)
            for key, value in zip(WEIGHT_KEYS, values):
                self.assertAlmostEqual(weights[key].item(), (value + 1e-8) / (sum(values) + 3e-8), places=15)
                self.assertGreaterEqual(weights[key], 0)
                self.assertLessEqual(weights[key], 1)
            self.assertLess(abs(sum(weights.values()).item() - 1), 1e-5)
        self.assertEqual(EPS, 1e-8)

    def test_fp16_and_bfloat16_avoid_epsilon_underflow(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                p = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
                original_probability_values = p.clone()
                r = compute_reliability(p, p)
                self.assertEqual(p.dtype, dtype)
                torch.testing.assert_close(p, original_probability_values, rtol=0, atol=0)
                zeros = {name: torch.tensor(0, dtype=dtype) for name in RELIABILITY_KEYS}
                weights = reliability_weights(zeros)
                for value in (*r.values(), *weights.values()):
                    self.assertTrue(torch.isfinite(value))
                    self.assertEqual(value.dtype, torch.float32)
                for weight in weights.values():
                    self.assertAlmostEqual(weight.item(), 1 / 3, places=7)

    def test_autocast_does_not_downcast_relation_calculation(self):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = compute_reliability(self.original.float(), self.augmented.float())
        expected = compute_reliability(self.original.float(), self.augmented.float())
        for key in RELIABILITY_KEYS:
            self.assertEqual(actual[key].dtype, torch.float32)
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_uniform_tied_and_single_sample_probabilities_are_finite(self):
        for p in (torch.full((4, 3), 1 / 3, dtype=torch.float64),
                  torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64),
                  torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)):
            r = compute_reliability(p, p)
            for value in r.values():
                self.assertTrue(torch.isfinite(value))
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
        # log(p + eps) means exact uniform r_ins is tiny positive, not forcibly zero.
        uniform = torch.full((4, 3), 1 / 3, dtype=torch.float64)
        r = compute_reliability(uniform, uniform)["r_ins"]
        self.assertGreater(r.item(), 0)
        self.assertLess(r.item(), 1e-7)

    def test_invalid_probability_inputs_fail_explicitly(self):
        invalid = [torch.ones(2, 1), torch.empty(0, 3), torch.ones(2, 3, 1),
                   torch.tensor([[float("nan"), 0.5, 0.5]]),
                   torch.tensor([[float("inf"), 0.5, 0.5]]),
                   torch.tensor([[-0.1, 0.5, 0.6]]), torch.tensor([[0.1, 0.1, 0.1]]),
                   torch.tensor([[1, 0, 0]])]
        for p in invalid:
            with self.subTest(p=p), self.assertRaises(ValueError):
                compute_reliability(p, p)
        with self.assertRaises(ValueError):
            compute_reliability(self.original, self.augmented[:1])

    def test_independent_scalar_gradients_prove_relative_weighting(self):
        gradient_vectors = []
        for values in ((0.1, 0.3, 0.6), (0.6, 0.3, 0.1)):
            losses = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (2.0, 3.0, 7.0)]
            r = {name: torch.tensor(value, dtype=torch.float64)
                 for name, value in zip(RELIABILITY_KEYS, values)}
            total, weights = aggregate_losses(*losses, reliabilities=r)
            expected = sum(loss.item() * weights[key].item() for loss, key in zip(losses, WEIGHT_KEYS))
            self.assertAlmostEqual(total.item(), expected, places=14)
            total.backward()
            gradients = torch.stack([loss.grad for loss in losses])
            torch.testing.assert_close(gradients, torch.stack(list(weights.values())))
            gradient_vectors.append(gradients)
        # Any global scalar applied after sum gives identical derivatives of all
        # three losses. These derivatives differ, and their ratios change.
        self.assertFalse(torch.allclose(gradient_vectors[0], gradient_vectors[0][0].expand(3)))
        self.assertFalse(torch.allclose(gradient_vectors[0] / gradient_vectors[1],
                                        (gradient_vectors[0][0] / gradient_vectors[1][0]).expand(3)))

    def test_controlled_variants_and_static_validation(self):
        losses = [torch.tensor(x, requires_grad=True) for x in (2.0, 3.0, 7.0)]
        equal, _ = aggregate_losses(*losses, mode="equal")
        instance, weights = aggregate_losses(*losses, mode="instance_only")
        self.assertAlmostEqual(equal.item(), 4.0, places=6)
        self.assertEqual(instance.item(), 2.0)
        self.assertEqual([value.item() for value in weights.values()], [1.0, 0.0, 0.0])
        for static in ([0.1, 0.3, 0.6], dict(zip(WEIGHT_KEYS, (0.1, 0.3, 0.6))),
                       torch.tensor([0.1, 0.3, 0.6])):
            total, _ = aggregate_losses(*losses, mode="static_mean", static_weights=static)
            self.assertAlmostEqual(total.item(), 5.3, places=6)
        for static in (None, [1, 1, 1], [0.1, 0.2], [-0.1, 0.5, 0.6],
                       [float("nan"), 0, 1], {"ins": 0.1, "batch": 0.3, "cls": 0.6}, "0.1,0.3,0.6"):
            with self.subTest(static=static), self.assertRaises(ValueError):
                select_weights("static_mean", static_weights=static)

    def test_reliability_and_loss_contracts(self):
        for r in (None, {}, {"r_ins": 1.0}, dict(zip(RELIABILITY_KEYS, (1.1, 0.2, 0.3))),
                  dict(zip(RELIABILITY_KEYS, (float("inf"), 0.2, 0.3))),
                  dict(zip(RELIABILITY_KEYS, (torch.ones(1), 0.2, 0.3)))):
            with self.subTest(r=r), self.assertRaises(ValueError):
                reliability_weights(r)
        for bad in (torch.ones(1), torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor(1), 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                aggregate_losses(bad, torch.tensor(2.0), torch.tensor(3.0), mode="equal")
        with self.assertRaises(ValueError):
            select_weights("legacy")

    def test_ten_synthetic_batches_produce_dynamic_weights(self):
        # This verifies the equation wiring, never an actual DTD run or result.
        generator = torch.Generator().manual_seed(711)
        records = []
        for index in range(10):
            teacher = ((0.2 + index / 3) * torch.randn(8, 5, generator=generator)).softmax(-1)
            augmented = ((0.1 + index / 5) * torch.randn(8, 5, generator=generator)).softmax(-1)
            r = compute_reliability(teacher, augmented)
            weights = reliability_weights(r)
            row = torch.stack([weights[key] for key in WEIGHT_KEYS])
            self.assertTrue(torch.isfinite(row).all())
            self.assertTrue(((row >= 0) & (row <= 1)).all())
            self.assertLess(abs(row.sum().item() - 1), 1e-5)
            records.append(row)
        std = torch.stack(records).std(dim=0, correction=0)
        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue((std > 0).any())


if __name__ == "__main__":
    unittest.main()
