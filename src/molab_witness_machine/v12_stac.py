"""A small, auditable implementation of STAC's spatial controller.

The primary source is Xiao et al., *DNN-Driven Compressive Offloading for
Edge-Assisted Semantic Video Segmentation* (INFOCOM 2022, arXiv:2203.14481).

This module implements the paper's first-order sensitivity allocation and its
regional choice from a finite bank of offline quantization tables.  It does not
implement JPEG entropy coding, optical flow, a segmentation network, or the
paper's reported experiment.  Its outputs are therefore controller
constructions, never comma.ai score rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RegionalStrategy:
    """A finite-table approximation to the ideal sensitivity allocation."""

    levels: Array
    quant_steps: Array
    regional_bounds: Array
    total_bound: float
    target_per_region: float
    table_count: int
    strategy_bits: int
    requested_total_budget: float
    budget_slack: float
    feasible: bool
    infeasible_regions: int


def _finite_nonnegative(values: Array | Sequence[float], *, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def worst_case_first_order_bound(abs_gradient: Array, quant_steps: Array) -> float:
    """Return ``sum_i |g_i| q_i / 2`` from STAC's first-order bound."""

    gradient = _finite_nonnegative(abs_gradient, name="abs_gradient")
    steps = _finite_nonnegative(quant_steps, name="quant_steps")
    if gradient.shape != steps.shape:
        raise ValueError("abs_gradient and quant_steps must have identical shapes")
    return float(np.sum(gradient * steps * 0.5, dtype=np.float64))


def ideal_quant_steps(
    abs_gradient: Array | Sequence[float],
    budget: float,
    *,
    gradient_floor: float = 1e-9,
    maximum_step: float | None = None,
) -> Array:
    """Compute the paper's ideal ``q_i = 2B / (M |g_i|)`` allocation.

    ``gradient_floor`` makes zero-sensitivity coordinates finite.  An optional
    ``maximum_step`` models a codec's largest supported quantization step; that
    clipping makes the realized first-order bound no larger than ``budget``.
    """

    gradient = _finite_nonnegative(abs_gradient, name="abs_gradient")
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be finite and positive")
    if not np.isfinite(gradient_floor) or gradient_floor <= 0:
        raise ValueError("gradient_floor must be finite and positive")
    if maximum_step is not None and (not np.isfinite(maximum_step) or maximum_step <= 0):
        raise ValueError("maximum_step must be finite and positive")

    count = gradient.size
    steps = (2.0 * float(budget)) / (count * np.maximum(gradient, gradient_floor))
    if maximum_step is not None:
        steps = np.minimum(steps, float(maximum_step))
    return steps


def build_offline_table_bank(
    mean_abs_gradient_by_frequency: Array | Sequence[float],
    budgets: Array | Sequence[float],
    *,
    coefficient_count: int,
    gradient_floor: float = 1e-9,
    minimum_step: float = 1.0,
    maximum_step: float = 255.0,
) -> Array:
    """Build a bank of DNN-specific frequency quantization tables.

    Each row applies STAC's inverse-gradient law to one offline loss budget.
    Rounding models an integer JPEG-like table while clipping records the finite
    actuator range.  The bank is ordered from the finest to coarsest table.
    """

    frequency_gradient = _finite_nonnegative(
        mean_abs_gradient_by_frequency,
        name="mean_abs_gradient_by_frequency",
    ).reshape(-1)
    budget_values = _finite_nonnegative(budgets, name="budgets").reshape(-1)
    if np.any(budget_values <= 0):
        raise ValueError("budgets must be positive")
    if coefficient_count <= 0:
        raise ValueError("coefficient_count must be positive")
    if minimum_step <= 0 or maximum_step < minimum_step:
        raise ValueError("quantization step bounds are invalid")

    rows = []
    safe_gradient = np.maximum(frequency_gradient, gradient_floor)
    for budget in np.sort(budget_values):
        raw = (2.0 * float(budget)) / (coefficient_count * safe_gradient)
        rows.append(np.clip(np.rint(raw), minimum_step, maximum_step))
    return np.asarray(rows, dtype=np.float64)


def _region_slices(height: int, width: int, region_shape: tuple[int, int]):
    region_height, region_width = region_shape
    if region_height <= 0 or region_width <= 0:
        raise ValueError("region_shape values must be positive")
    for y0 in range(0, height, region_height):
        for x0 in range(0, width, region_width):
            yield slice(y0, min(height, y0 + region_height)), slice(x0, min(width, x0 + region_width))


def select_regional_strategy(
    abs_gradient: Array,
    table_bank: Array,
    total_budget: float,
    *,
    region_shape: tuple[int, int] = (3, 3),
) -> RegionalStrategy:
    """Select the coarsest table that does not exceed ``B / r_max``.

    ``abs_gradient`` has shape ``(height, width, frequencies)``. ``table_bank``
    has shape ``(levels, frequencies)``.  The selected level map is sufficient
    to rematerialize the full per-coefficient quantization map at a receiver
    that already holds the offline table bank. If no table can satisfy a
    region's equal budget share, the finest realized candidate is returned but
    ``feasible`` is false; callers must display or reject that state rather
    than silently overspending the declared bound.
    """

    gradient = _finite_nonnegative(abs_gradient, name="abs_gradient")
    tables = _finite_nonnegative(table_bank, name="table_bank")
    if gradient.ndim != 3:
        raise ValueError("abs_gradient must have shape (height, width, frequencies)")
    if tables.ndim != 2 or tables.shape[0] < 1:
        raise ValueError("table_bank must have shape (levels, frequencies)")
    if gradient.shape[-1] != tables.shape[-1]:
        raise ValueError("gradient and table bank frequency dimensions must match")
    if not np.isfinite(total_budget) or total_budget <= 0:
        raise ValueError("total_budget must be finite and positive")

    height, width, _ = gradient.shape
    slices = tuple(_region_slices(height, width, region_shape))
    target = float(total_budget) / len(slices)
    level_grid_height = ceil(height / region_shape[0])
    level_grid_width = ceil(width / region_shape[1])
    levels = np.empty((level_grid_height, level_grid_width), dtype=np.int16)
    regional_bounds = np.empty_like(levels, dtype=np.float64)
    infeasible_regions = 0

    for index, (ys, xs) in enumerate(slices):
        region = gradient[ys, xs, :]
        candidate_bounds = np.sum(
            region[None, :, :, :] * tables[:, None, None, :] * 0.5,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(target)) * 16.0
        admissible = np.flatnonzero(candidate_bounds <= target + tolerance)
        if admissible.size:
            selected = int(admissible[np.argmax(candidate_bounds[admissible])])
        else:
            selected = int(np.argmin(candidate_bounds))
            infeasible_regions += 1
        row, column = divmod(index, level_grid_width)
        levels[row, column] = selected
        regional_bounds[row, column] = candidate_bounds[selected]

    quant_steps = rematerialize_quant_map(
        levels,
        tables,
        spatial_shape=(height, width),
        region_shape=region_shape,
    )
    bits_per_level = 0 if tables.shape[0] == 1 else ceil(log2(tables.shape[0]))
    total_bound = float(regional_bounds.sum(dtype=np.float64))
    total_tolerance = np.finfo(np.float64).eps * max(1.0, abs(total_budget)) * 32.0
    feasible = infeasible_regions == 0 and total_bound <= float(total_budget) + total_tolerance
    return RegionalStrategy(
        levels=levels,
        quant_steps=quant_steps,
        regional_bounds=regional_bounds,
        total_bound=total_bound,
        target_per_region=target,
        table_count=int(tables.shape[0]),
        strategy_bits=int(levels.size * bits_per_level),
        requested_total_budget=float(total_budget),
        budget_slack=float(total_budget) - total_bound,
        feasible=feasible,
        infeasible_regions=infeasible_regions,
    )


def rematerialize_quant_map(
    levels: Array,
    table_bank: Array,
    *,
    spatial_shape: tuple[int, int],
    region_shape: tuple[int, int],
) -> Array:
    """Expand a compact regional level map into a dense quantization map."""

    selected_levels = np.asarray(levels)
    tables = _finite_nonnegative(table_bank, name="table_bank")
    if selected_levels.ndim != 2 or not np.issubdtype(selected_levels.dtype, np.integer):
        raise ValueError("levels must be a two-dimensional integer array")
    if np.any(selected_levels < 0) or np.any(selected_levels >= tables.shape[0]):
        raise ValueError("levels contains a table index outside table_bank")
    height, width = spatial_shape
    if height <= 0 or width <= 0:
        raise ValueError("spatial_shape values must be positive")
    expected = (ceil(height / region_shape[0]), ceil(width / region_shape[1]))
    if selected_levels.shape != expected:
        raise ValueError(f"levels shape must be {expected} for the requested spatial and region shapes")

    output = np.empty((height, width, tables.shape[1]), dtype=np.float64)
    for row, (y0, y1) in enumerate(
        (range_start, min(height, range_start + region_shape[0]))
        for range_start in range(0, height, region_shape[0])
    ):
        for column, (x0, x1) in enumerate(
            (range_start, min(width, range_start + region_shape[1]))
            for range_start in range(0, width, region_shape[1])
        ):
            output[y0:y1, x0:x1, :] = tables[int(selected_levels[row, column])]
    return output


__all__ = [
    "RegionalStrategy",
    "build_offline_table_bank",
    "ideal_quant_steps",
    "rematerialize_quant_map",
    "select_regional_strategy",
    "worst_case_first_order_bound",
]
