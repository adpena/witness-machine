"""Matched-effort policy comparison for the public STAC teaching scene.

This is a transparent toy extension, not a JPEG experiment and not a result
from Xiao et al.  It holds the declared precision proxy ``sum(1 / q_i)``
constant and compares three quantization-step fields under STAC's displayed
first-order bound ``0.5 * sum(|g_i| q_i)``.  The construction exists to make
the paper's boundary-only counterexample explorable without promoting it to an
empirical codec claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PolicyRow:
    """One matched-effort quantization policy."""

    name: str
    first_order_bound: float
    precision_proxy: float
    quant_steps: Array


@dataclass(frozen=True)
class MatchedPolicyComparison:
    """Three policies evaluated on one declared sensitivity field."""

    precision_budget: float
    boundary_focus_ratio: float
    uniform: PolicyRow
    boundary_only: PolicyRow
    sensitivity_projected: PolicyRow
    authority: str = "TOY · matched precision proxy · not JPEG · not an evaluated score"
    exact_score: bool = False


def _matched_precision_steps(base_steps: Array, precision_budget: float) -> Array:
    base = np.asarray(base_steps, dtype=np.float64)
    if base.size == 0 or not np.all(np.isfinite(base)) or np.any(base <= 0.0):
        raise ValueError("base_steps must be finite and strictly positive")
    scale = float(np.sum(1.0 / base, dtype=np.float64)) / precision_budget
    return base * scale


def _row(name: str, sensitivity: Array, quant_steps: Array) -> PolicyRow:
    return PolicyRow(
        name=name,
        first_order_bound=float(
            0.5 * np.sum(sensitivity * quant_steps, dtype=np.float64)
        ),
        precision_proxy=float(np.sum(1.0 / quant_steps, dtype=np.float64)),
        quant_steps=quant_steps,
    )


def compare_matched_precision_policies(
    sensitivity: Array,
    boundary_mask: Array,
    *,
    precision_budget: float,
    boundary_focus_ratio: float = 4.0,
    sensitivity_floor: float = 1e-6,
) -> MatchedPolicyComparison:
    """Compare uniform, outline-only, and sensitivity-projected policies.

    ``boundary_focus_ratio`` is the ratio between the off-boundary and
    boundary quantization steps before the common matched-effort rescaling.
    The sensitivity policy uses the exact minimizer of the displayed linear
    bound under the declared reciprocal-step precision proxy:
    ``q_i proportional to 1 / sqrt(|g_i|)``.
    """

    gradient = np.asarray(sensitivity, dtype=np.float64)
    boundary = np.asarray(boundary_mask, dtype=bool)
    if gradient.ndim != 2 or gradient.size == 0:
        raise ValueError("sensitivity must be a non-empty two-dimensional field")
    if boundary.shape != gradient.shape:
        raise ValueError("boundary_mask must match sensitivity")
    if not np.all(np.isfinite(gradient)) or np.any(gradient < 0.0):
        raise ValueError("sensitivity must be finite and nonnegative")
    if not isfinite(precision_budget) or precision_budget <= 0.0:
        raise ValueError("precision_budget must be finite and positive")
    if not isfinite(boundary_focus_ratio) or boundary_focus_ratio < 1.0:
        raise ValueError("boundary_focus_ratio must be finite and at least one")
    if not isfinite(sensitivity_floor) or sensitivity_floor <= 0.0:
        raise ValueError("sensitivity_floor must be finite and positive")

    uniform_steps = _matched_precision_steps(
        np.ones_like(gradient), float(precision_budget)
    )
    boundary_steps = _matched_precision_steps(
        np.where(boundary, 1.0, float(boundary_focus_ratio)),
        float(precision_budget),
    )
    sensitivity_steps = _matched_precision_steps(
        1.0 / np.sqrt(np.maximum(gradient, float(sensitivity_floor))),
        float(precision_budget),
    )
    return MatchedPolicyComparison(
        precision_budget=float(precision_budget),
        boundary_focus_ratio=float(boundary_focus_ratio),
        uniform=_row("uniform", gradient, uniform_steps),
        boundary_only=_row("boundary_only", gradient, boundary_steps),
        sensitivity_projected=_row(
            "sensitivity_projected", gradient, sensitivity_steps
        ),
    )


__all__ = [
    "MatchedPolicyComparison",
    "PolicyRow",
    "compare_matched_precision_policies",
]
