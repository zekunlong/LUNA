"""Fixed teacher-only reliability equations for the separate journal mode.

Inputs are probabilities from LUNA's existing probability computation, using
its existing temperature. This module intentionally neither changes that
temperature nor imports or modifies the legacy trainer or its loss functions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Union

import torch


EPS = 1e-8
RELIABILITY_KEYS = ("r_ins", "r_batch", "r_cls")
WEIGHT_KEYS = ("lambda_ins", "lambda_batch", "lambda_cls")
Scalar = Union[torch.Tensor, float]


def _calculation_dtype(tensors: Sequence[torch.Tensor]) -> torch.dtype:
    # In particular, 1e-8 must not become zero under fp16 or autocast.
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _validate_probabilities(prob: torch.Tensor, name: str) -> None:
    if not isinstance(prob, torch.Tensor) or not prob.is_floating_point():
        raise ValueError(f"{name} must be a floating-point probability tensor")
    if prob.ndim != 2 or prob.shape[0] == 0 or prob.shape[1] < 2:
        raise ValueError(f"{name} must have shape (B, C) with B > 0 and C > 1")
    if not bool(torch.isfinite(prob).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    if bool(((prob < 0) | (prob > 1)).any()):
        raise ValueError(f"{name} contains values outside [0, 1]")
    # Stored half-precision probabilities have small rounding errors. Check the
    # probability contract without renormalizing or changing the supplied data.
    tolerance = 8 * torch.finfo(prob.dtype).eps
    row_sums = prob.detach().to(_calculation_dtype((prob,))).sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=tolerance, rtol=0):
        raise ValueError(f"Rows of {name} must sum to one")


@torch.no_grad()
def compute_reliability(prob_t: torch.Tensor, prob_aug: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the three detached scalar reliabilities from aligned teacher views.

    Entropy uses log(p + 1e-8), and both relation similarities use
    dot(v, w) / (norm(v) * norm(w) + 1e-8), exactly as specified.
    """
    _validate_probabilities(prob_t, "prob_t")
    _validate_probabilities(prob_aug, "prob_aug")
    if prob_t.shape != prob_aug.shape:
        raise ValueError("Teacher views must have identical (B, C) shapes")
    if prob_t.device != prob_aug.device:
        raise ValueError("Teacher views must be on the same device")
    dtype = _calculation_dtype((prob_t, prob_aug))
    with torch.autocast(device_type=prob_t.device.type, enabled=False):
        teacher = prob_t.detach().to(dtype=dtype)
        augmented = prob_aug.detach().to(dtype=dtype)
        entropy_t = -(teacher * torch.log(teacher + EPS)).sum(dim=1).mean()
        entropy_aug = -(augmented * torch.log(augmented + EPS)).sum(dim=1).mean()
        r_ins = (1 - 0.5 * (entropy_t + entropy_aug) / math.log(teacher.shape[1])).clamp(0, 1)

        def relation_cosine(original: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            original = original.flatten()
            other = other.flatten()
            return (
                torch.dot(original, other)
                / (torch.linalg.vector_norm(original) * torch.linalg.vector_norm(other) + EPS)
            ).clamp(0, 1)

        r_batch = relation_cosine(teacher @ teacher.T, augmented @ augmented.T)
        r_cls = relation_cosine(teacher.T @ teacher, augmented.T @ augmented)
        result = dict(zip(RELIABILITY_KEYS, (r_ins, r_batch, r_cls)))
        if not all(bool(torch.isfinite(value)) for value in result.values()):
            raise ValueError("Reliability calculation produced NaN or Inf")
        return result


def _scalar_vector(values: Sequence[Scalar], name: str) -> torch.Tensor:
    tensors = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or not value.is_floating_point():
                raise ValueError(f"{name} values must be floating-point scalars")
            tensors.append(value.detach())
        elif not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"{name} values must be real scalars")
    if tensors and any(t.device != tensors[0].device for t in tensors):
        raise ValueError(f"{name} tensors must share a device")
    device = tensors[0].device if tensors else torch.device("cpu")
    dtype = _calculation_dtype(tensors)
    result = torch.stack([torch.as_tensor(value, device=device, dtype=dtype).detach() for value in values])
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    if bool(((result < 0) | (result > 1)).any()):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return result


def _weight_dict(weights: torch.Tensor) -> dict[str, torch.Tensor]:
    if weights.shape != (3,) or not bool(torch.isfinite(weights).all()):
        raise ValueError("Weights must contain three finite scalars")
    if bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("Weights must lie in [0, 1]")
    # Executed on every batch, including under optimized Python (no assert).
    if not bool(torch.abs(weights.sum() - 1) < 1e-5):
        raise ValueError("Hierarchical weights must sum to one within 1e-5")
    return dict(zip(WEIGHT_KEYS, weights.unbind()))


def reliability_weights(reliabilities: Mapping[str, Scalar]) -> dict[str, torch.Tensor]:
    """Normalize teacher reliability directly, preserving the fixed epsilon."""
    if not isinstance(reliabilities, Mapping) or set(reliabilities) != set(RELIABILITY_KEYS):
        raise ValueError(f"Reliabilities must contain exactly {RELIABILITY_KEYS}")
    values = _scalar_vector([reliabilities[key] for key in RELIABILITY_KEYS], "Reliabilities")
    return _weight_dict((values + EPS) / (values.sum() + 3 * EPS))


def select_weights(
    mode: str,
    *,
    reliabilities: Mapping[str, Scalar] | None = None,
    static_weights: Mapping[str, Scalar] | Sequence[Scalar] | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Select a controlled variant; static weights must already sum to one.

    Loading a training-only mean from disk is the caller's responsibility. This
    function never recomputes or silently renormalizes a supplied static mean.
    """
    if mode == "reliability_adaptive":
        return reliability_weights(reliabilities)
    if mode == "equal":
        return _weight_dict(torch.full((3,), 1 / 3, dtype=torch.float32))
    if mode == "instance_only":
        return _weight_dict(torch.tensor([1.0, 0.0, 0.0]))
    if mode == "static_mean":
        if isinstance(static_weights, Mapping):
            if set(static_weights) != set(WEIGHT_KEYS):
                raise ValueError(f"Static weights must contain exactly {WEIGHT_KEYS}")
            values = [static_weights[key] for key in WEIGHT_KEYS]
        elif isinstance(static_weights, torch.Tensor):
            if static_weights.shape != (3,):
                raise ValueError("Static weights must contain exactly three values")
            values = list(static_weights.unbind())
        elif isinstance(static_weights, Sequence) and not isinstance(static_weights, (str, bytes)):
            values = list(static_weights)
        else:
            raise ValueError("static_mean requires fixed training-set mean weights")
        if len(values) != 3:
            raise ValueError("Static weights must contain exactly three values")
        return _weight_dict(_scalar_vector(values, "Static weights"))
    raise ValueError(f"Unknown journal weighting mode {mode!r}; legacy belongs to the original trainer")


def aggregate_losses(
    loss_ins: torch.Tensor,
    loss_batch: torch.Tensor,
    loss_cls: torch.Tensor,
    *,
    mode: str = "reliability_adaptive",
    reliabilities: Mapping[str, Scalar] | None = None,
    static_weights: Mapping[str, Scalar] | Sequence[Scalar] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weight three independently computed scalar losses, then sum.

    Student loss tensors retain their autograd graphs. Teacher-derived weights
    are detached. No confidence mask or multiplier of the total loss is used.
    """
    losses = (loss_ins, loss_batch, loss_cls)
    for name, loss in zip(("L_ins", "L_batch", "L_cls"), losses):
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.is_floating_point():
            raise ValueError(f"{name} must be an independently computed floating-point scalar tensor")
        if not bool(torch.isfinite(loss.detach())):
            raise ValueError(f"{name} contains NaN or Inf")
    if any(loss.device != loss_ins.device for loss in losses):
        raise ValueError("All three scalar loss terms must be on the same device")
    selected = select_weights(mode, reliabilities=reliabilities, static_weights=static_weights)
    dtype = _calculation_dtype((*losses, *selected.values()))
    weights = torch.stack([selected[key] for key in WEIGHT_KEYS]).to(device=loss_ins.device, dtype=dtype)
    checked_weights = _weight_dict(weights)
    # The three autograd derivatives dL/dL_k are the individual lambda_k.
    total = (torch.stack(losses).to(dtype=dtype) * weights).sum()
    if not bool(torch.isfinite(total.detach())):
        raise ValueError("Weighted loss contains NaN or Inf")
    return total, checked_weights
