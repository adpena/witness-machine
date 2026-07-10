"""Transparent toy carrier valuation for the V12 public notebook.

This module does not predict comma.ai evaluator behavior.  It evaluates a
declared table of hypothetical debt reductions with the local derivatives of
the public score law.  The result is a first-order shadow-price construction,
not a trajectory costate, learned controller, archive score, or performance
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable

from .score_law import ORIGINAL_BYTES, RATE_WEIGHT, SEG_WEIGHT


@dataclass(frozen=True)
class CarrierProposal:
    """One declared hypothetical receiver-visible intervention."""

    name: str
    seg_debt_reduction: float
    pose_debt_reduction: float
    added_bytes: int


@dataclass(frozen=True)
class CarrierValuation:
    """First-order value of a :class:`CarrierProposal` at one operating point."""

    proposal: CarrierProposal
    lambda_seg: float
    lambda_pose: float
    lambda_byte: float
    task_drop_proxy: float
    rate_cost: float
    score_drop_proxy: float
    value_per_byte: float
    admitted: bool


def local_score_shadow_prices(
    d_pose: float,
    *,
    source_bytes: int = ORIGINAL_BYTES,
) -> tuple[float, float, float]:
    """Return exact local score derivatives for ``d_pose > 0``.

    The pose derivative is undefined at zero and diverges from the right, so
    this public construction fails closed rather than silently clamping it.
    """

    pose = float(d_pose)
    if not isfinite(pose) or pose <= 0.0:
        raise ValueError("d_pose must be finite and strictly positive")
    if isinstance(source_bytes, bool) or int(source_bytes) != source_bytes or source_bytes <= 0:
        raise ValueError("source_bytes must be a positive integer")
    return SEG_WEIGHT, 5.0 / sqrt(10.0 * pose), RATE_WEIGHT / int(source_bytes)


def value_carrier_proposals(
    proposals: Iterable[CarrierProposal],
    *,
    d_pose: float,
    source_bytes: int = ORIGINAL_BYTES,
) -> tuple[CarrierValuation, ...]:
    """Apply the displayed first-order score rule to an explicit proposal table.

    For reductions ``(r_seg, r_pose)`` and byte increment ``b``, the displayed
    quantity is

    ``Delta S_drop ~= lambda_seg*r_seg + lambda_pose*r_pose - lambda_byte*b``.

    Positive values are locally favorable under the declared hypothetical
    reductions.  They still owe a real carrier, receiver round trip, and exact
    evaluation before they can support an empirical or score claim.
    """

    lambda_seg, lambda_pose, lambda_byte = local_score_shadow_prices(
        d_pose,
        source_bytes=source_bytes,
    )
    rows: list[CarrierValuation] = []
    for proposal in proposals:
        if not proposal.name.strip():
            raise ValueError("proposal name must be non-empty")
        seg = float(proposal.seg_debt_reduction)
        pose = float(proposal.pose_debt_reduction)
        added = proposal.added_bytes
        if not isfinite(seg) or not isfinite(pose) or seg < 0.0 or pose < 0.0:
            raise ValueError("declared debt reductions must be finite and nonnegative")
        if isinstance(added, bool) or int(added) != added or added <= 0:
            raise ValueError("added_bytes must be a positive integer")
        added = int(added)
        task_drop = lambda_seg * seg + lambda_pose * pose
        rate_cost = lambda_byte * added
        score_drop = task_drop - rate_cost
        rows.append(
            CarrierValuation(
                proposal=proposal,
                lambda_seg=lambda_seg,
                lambda_pose=lambda_pose,
                lambda_byte=lambda_byte,
                task_drop_proxy=task_drop,
                rate_cost=rate_cost,
                score_drop_proxy=score_drop,
                value_per_byte=score_drop / added,
                admitted=score_drop > 0.0,
            )
        )
    if not rows:
        raise ValueError("at least one proposal is required")
    return tuple(rows)


def notebook_toy_proposals(
    *,
    boundary_sweep_fraction: float,
    texture_amplitude: float,
    refresh_pressure: float,
    normalized_budget: float,
) -> tuple[CarrierProposal, ...]:
    """Construct the notebook's four fully declared hypothetical rows.

    The controls change the table so readers can inspect causal arithmetic.
    These coefficients are pedagogical constants, not fitted observations.
    """

    values = tuple(
        float(value)
        for value in (
            boundary_sweep_fraction,
            texture_amplitude,
            refresh_pressure,
            normalized_budget,
        )
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("toy proposal controls must be finite")
    boundary, texture, refresh, budget = values
    if boundary < 0.0 or not 0.0 <= texture <= 1.0 or not 0.0 <= refresh <= 1.0 or not 0.0 <= budget <= 1.0:
        raise ValueError("toy proposal controls are outside their declared ranges")

    return (
        CarrierProposal("boundary", 0.0012 + 0.0100 * boundary, 0.00004, 28_000),
        CarrierProposal("motion", 0.00025, 0.00045 + 0.00035 * (1.0 - texture), 38_000),
        CarrierProposal("generator", 0.0011 + 0.0004 * budget, 0.00030, 64_000),
        CarrierProposal("correction", 0.00055 + 0.00035 * refresh, 0.00015, 22_000),
    )


__all__ = [
    "CarrierProposal",
    "CarrierValuation",
    "local_score_shadow_prices",
    "notebook_toy_proposals",
    "value_carrier_proposals",
]
