"""Optional accelerator search for the V12 geometric controller.

This module answers one narrow scientific question: across a large batch of
boundary, texture, residual, and refresh choices, which construction minimizes
a declared *design energy*?  The energy is a transparent proxy used to explore
the representation.  It is never the comma.ai challenge score.

Torch is imported lazily so the complete notebook remains usable with NumPy
alone.  CUDA and MPS are reported as separate advisory devices; neither is
promoted to exact evaluator authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CandidateSearchReceipt:
    backend: str
    device: str
    authority: str
    batch_size: int
    grid_shape: tuple[int, int]
    elapsed_ms: float
    parity_max_abs: float | None
    best_index: int
    best_parameters: tuple[float, float, float, float, float]
    best_metrics: tuple[float, float, float, float, float]
    exact_score: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProxyRobustnessReceipt:
    """Custody for a batched epsilon sweep on one locked real evidence field."""

    backend: str
    device: str
    authority: str
    evidence_sha256: str
    pixel_count: int
    epsilon_count: int
    epsilon_range: tuple[float, float]
    elapsed_ms: float
    parity_max_abs: float | None
    boundary_mass_range: tuple[float, float]
    effective_sample_size_range: tuple[float, float]
    top_pixel_mass_range: tuple[float, float]
    exact_score: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_candidates(batch_size: int) -> Array:
    """Generate a deterministic, well-spread five-parameter candidate batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    index = np.arange(batch_size, dtype=np.float64) + 0.5
    irrational = np.array(
        [0.6180339887498949, 0.4142135623730950, 0.7320508075688772, 0.2360679774997897, 0.6457513110645906],
        dtype=np.float64,
    )
    unit = np.mod(index[:, None] * irrational[None, :], 1.0)
    return np.column_stack(
        [
            -0.24 + 0.48 * unit[:, 0],  # shared boundary displacement
            -0.18 + 0.36 * unit[:, 1],  # curvature correction
            0.05 + 0.95 * unit[:, 2],  # temporal texture aperture
            unit[:, 3],  # sparse residual fraction
            unit[:, 4],  # refresh frequency
        ]
    ).astype(np.float32)


def _validate_parameters(parameters: Array) -> Array:
    values = np.asarray(parameters, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("parameters must have shape (batch, 5)")
    if values.shape[0] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("parameters must be a non-empty finite batch")
    return values


def evaluate_candidates_numpy(parameters: Array, *, height: int = 28, width: int = 42) -> Array:
    """Evaluate the declared five-term design energy with NumPy.

    Returned columns are ``boundary_debt``, ``motion_debt``, ``topology_debt``,
    ``rate_proxy``, and their weighted ``design_energy``.
    """

    values = _validate_parameters(parameters)
    if height < 12 or width < 18:
        raise ValueError("grid is too small for the geometric probe")
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, None, :]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, :, None]
    reference_curve = 0.18 * np.sin(np.pi * x) + 0.08 * x
    reference = y >= reference_curve

    displacement = values[:, 0, None, None]
    curvature = values[:, 1, None, None]
    candidate_curve = (
        0.18 * np.sin(np.pi * x)
        + (0.08 + curvature) * x
        + displacement
    )
    candidate = y >= candidate_curve
    boundary_debt = np.mean(reference != candidate, axis=(1, 2), dtype=np.float64)

    texture = values[:, 2].astype(np.float64)
    residual = values[:, 3].astype(np.float64)
    refresh = values[:, 4].astype(np.float64)
    motion_debt = (1.0 - texture) * (0.72 + 0.28 * (1.0 - refresh))
    topology_debt = (
        np.maximum(np.abs(values[:, 0]).astype(np.float64) - 0.12, 0.0)
        * (1.0 - residual)
        * 2.5
    )
    rate_proxy = 0.16 + 0.24 * texture + 0.31 * residual + 0.17 * refresh
    design_energy = 2.4 * boundary_debt + 0.75 * motion_debt + topology_debt + 0.42 * rate_proxy
    return np.column_stack(
        [boundary_debt, motion_debt, topology_debt, rate_proxy, design_energy]
    ).astype(np.float64)


def _evaluate_candidates_torch(parameters, torch, *, height: int, width: int):
    values = parameters
    x = torch.linspace(-1.0, 1.0, width, device=values.device, dtype=values.dtype)[None, None, :]
    y = torch.linspace(-1.0, 1.0, height, device=values.device, dtype=values.dtype)[None, :, None]
    reference_curve = 0.18 * torch.sin(torch.pi * x) + 0.08 * x
    reference = y >= reference_curve
    displacement = values[:, 0, None, None]
    curvature = values[:, 1, None, None]
    candidate_curve = 0.18 * torch.sin(torch.pi * x) + (0.08 + curvature) * x + displacement
    candidate = y >= candidate_curve
    boundary_debt = (reference != candidate).to(values.dtype).mean(dim=(1, 2))
    texture = values[:, 2]
    residual = values[:, 3]
    refresh = values[:, 4]
    motion_debt = (1.0 - texture) * (0.72 + 0.28 * (1.0 - refresh))
    topology_debt = torch.clamp(torch.abs(values[:, 0]) - 0.12, min=0.0) * (1.0 - residual) * 2.5
    rate_proxy = 0.16 + 0.24 * texture + 0.31 * residual + 0.17 * refresh
    design_energy = 2.4 * boundary_debt + 0.75 * motion_debt + topology_debt + 0.42 * rate_proxy
    return torch.stack(
        [boundary_debt, motion_debt, topology_debt, rate_proxy, design_energy], dim=1
    )


def evaluate_proxy_robustness_numpy(
    grad_energy: Array,
    margin: Array,
    boundary_mask: Array,
    epsilons: Array,
    *,
    chunk_size: int = 32,
) -> Array:
    """Return boundary mass, ESS, and top-pixel share for each epsilon."""

    energy = np.asarray(grad_energy, dtype=np.float32)
    local_margin = np.asarray(margin, dtype=np.float32)
    boundary = np.asarray(boundary_mask, dtype=bool)
    regularizers = np.asarray(epsilons, dtype=np.float32).reshape(-1)
    if energy.ndim != 2 or energy.size == 0 or local_margin.shape != energy.shape:
        raise ValueError("grad_energy and margin must be nonempty matching 2D arrays")
    if boundary.shape != energy.shape:
        raise ValueError("boundary_mask must match grad_energy")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("grad_energy must be finite and nonnegative")
    if not np.all(np.isfinite(local_margin)) or np.any(local_margin < 0.0):
        raise ValueError("margin must be finite and nonnegative")
    if regularizers.size == 0 or not np.all(np.isfinite(regularizers)) or np.any(regularizers <= 0.0):
        raise ValueError("epsilons must be finite and strictly positive")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    flat_energy = energy.reshape(-1).astype(np.float64)
    flat_margin_sq = np.square(local_margin.reshape(-1).astype(np.float64))
    flat_boundary = boundary.reshape(-1)
    rows: list[Array] = []
    for start in range(0, regularizers.size, chunk_size):
        epsilon_chunk = regularizers[start : start + chunk_size].astype(np.float64)
        proxy = flat_energy[None, :] / (
            flat_margin_sq[None, :] + epsilon_chunk[:, None]
        )
        total = proxy.sum(axis=1, dtype=np.float64)
        if np.any(total <= 0.0):
            raise ValueError("proxy mass must be positive")
        boundary_mass = proxy[:, flat_boundary].sum(axis=1, dtype=np.float64) / total
        sum_squares = np.square(proxy).sum(axis=1, dtype=np.float64)
        effective_sample_size = np.square(total) / sum_squares
        top_pixel_mass = proxy.max(axis=1) / total
        rows.append(
            np.column_stack(
                [boundary_mass, effective_sample_size, top_pixel_mass]
            )
        )
    return np.concatenate(rows, axis=0)


def _evaluate_proxy_robustness_torch(
    grad_energy,
    margin,
    boundary_mask,
    epsilons,
    torch,
    *,
    chunk_size: int,
):
    energy = grad_energy.reshape(-1)
    margin_sq = torch.square(margin.reshape(-1))
    boundary = boundary_mask.reshape(-1)
    accumulator_dtype = (
        torch.float32 if energy.device.type == "mps" else torch.float64
    )
    rows = []
    for start in range(0, int(epsilons.numel()), chunk_size):
        epsilon_chunk = epsilons[start : start + chunk_size]
        proxy = energy[None, :] / (
            margin_sq[None, :] + epsilon_chunk[:, None]
        )
        total = proxy.sum(dim=1, dtype=accumulator_dtype)
        boundary_mass = (
            proxy[:, boundary].sum(dim=1, dtype=accumulator_dtype) / total
        )
        sum_squares = torch.square(proxy).sum(
            dim=1, dtype=accumulator_dtype
        )
        effective_sample_size = torch.square(total) / sum_squares
        top_pixel_mass = proxy.max(dim=1).values.to(accumulator_dtype) / total
        rows.append(
            torch.stack(
                [boundary_mass, effective_sample_size, top_pixel_mass], dim=1
            )
        )
    return torch.cat(rows, dim=0)


def run_proxy_robustness_sweep(
    artifact_path,
    manifest_path,
    *,
    epsilon_count: int = 256,
    use_accelerator: bool = True,
    chunk_size: int = 32,
) -> ProxyRobustnessReceipt:
    """Sweep the declared margin regularizer on locked real one-pair evidence.

    This is an EMPIRICAL-input robustness analysis of an ADVISORY proxy.  It is
    not STAC's DCT gradient, not a codec round trip, and never a challenge score.
    """

    if not isinstance(epsilon_count, int) or epsilon_count < 8:
        raise ValueError("epsilon_count must be an integer of at least eight")
    from .v12_real_evidence import load_locked_evidence

    locked = load_locked_evidence(artifact_path, manifest_path)
    energy = np.asarray(locked.arrays["grad_energy_f32"], dtype=np.float32)
    margin = np.asarray(locked.arrays["margin_f32"], dtype=np.float32)
    boundary = np.asarray(locked.arrays["boundary_u8"], dtype=bool)
    epsilons = np.logspace(-8.0, -1.0, epsilon_count, dtype=np.float32)
    device_name = "cpu"
    backend = "numpy"
    parity: float | None = None

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        if use_accelerator and torch.cuda.is_available():
            device_name = "cuda"
        elif use_accelerator and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device_name = "mps"
        backend = "torch"
        energy_tensor = torch.as_tensor(energy, device=device_name, dtype=torch.float32)
        margin_tensor = torch.as_tensor(margin, device=device_name, dtype=torch.float32)
        boundary_tensor = torch.as_tensor(boundary, device=device_name, dtype=torch.bool)
        epsilon_tensor = torch.as_tensor(epsilons, device=device_name, dtype=torch.float32)
        if device_name == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        result_tensor = _evaluate_proxy_robustness_torch(
            energy_tensor,
            margin_tensor,
            boundary_tensor,
            epsilon_tensor,
            torch,
            chunk_size=chunk_size,
        )
        if device_name == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000.0
        result = result_tensor.detach().cpu().numpy().astype(np.float64)
        reference = evaluate_proxy_robustness_numpy(
            energy,
            margin,
            boundary,
            epsilons[:8],
            chunk_size=8,
        )
        parity = float(np.max(np.abs(reference - result[:8])))
    else:
        started = perf_counter()
        result = evaluate_proxy_robustness_numpy(
            energy,
            margin,
            boundary,
            epsilons,
            chunk_size=chunk_size,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0

    authority = (
        f"ADVISORY · EMPIRICAL locked one-pair input · {backend} {device_name} "
        "regularizer robustness · not STAC DCT · not an evaluation"
    )
    return ProxyRobustnessReceipt(
        backend=backend,
        device=device_name,
        authority=authority,
        evidence_sha256=str(locked.manifest["artifact_sha256"]),
        pixel_count=int(energy.size),
        epsilon_count=int(epsilon_count),
        epsilon_range=(float(epsilons[0]), float(epsilons[-1])),
        elapsed_ms=float(elapsed_ms),
        parity_max_abs=parity,
        boundary_mass_range=(float(result[:, 0].min()), float(result[:, 0].max())),
        effective_sample_size_range=(
            float(result[:, 1].min()),
            float(result[:, 1].max()),
        ),
        top_pixel_mass_range=(
            float(result[:, 2].min()),
            float(result[:, 2].max()),
        ),
    )


def run_candidate_search(
    batch_size: int = 4096,
    *,
    use_accelerator: bool = True,
    height: int = 28,
    width: int = 42,
) -> CandidateSearchReceipt:
    """Run a deterministic NumPy or Torch candidate search and return custody."""

    candidates = deterministic_candidates(batch_size)
    device_name = "cpu"
    backend = "numpy"
    authority = "ADVISORY · deterministic NumPy proxy"
    parity: float | None = None

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        if use_accelerator and torch.cuda.is_available():
            device_name = "cuda"
        elif use_accelerator and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
        backend = "torch"
        authority = f"ADVISORY · torch {device_name} proxy"

        tensor = torch.as_tensor(candidates, device=device_name, dtype=torch.float32)
        if device_name == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        result_tensor = _evaluate_candidates_torch(tensor, torch, height=height, width=width)
        if device_name == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000.0
        result = result_tensor.detach().cpu().numpy().astype(np.float64)
        reference = evaluate_candidates_numpy(candidates[:32], height=height, width=width)
        parity = float(np.max(np.abs(reference - result[:32])))
    else:
        started = perf_counter()
        result = evaluate_candidates_numpy(candidates, height=height, width=width)
        elapsed_ms = (perf_counter() - started) * 1000.0

    best_index = int(np.argmin(result[:, 4]))
    return CandidateSearchReceipt(
        backend=backend,
        device=device_name,
        authority=authority,
        batch_size=int(batch_size),
        grid_shape=(int(height), int(width)),
        elapsed_ms=float(elapsed_ms),
        parity_max_abs=parity,
        best_index=best_index,
        best_parameters=tuple(float(value) for value in candidates[best_index]),
        best_metrics=tuple(float(value) for value in result[best_index]),
    )


__all__ = [
    "CandidateSearchReceipt",
    "ProxyRobustnessReceipt",
    "deterministic_candidates",
    "evaluate_candidates_numpy",
    "evaluate_proxy_robustness_numpy",
    "run_candidate_search",
    "run_proxy_robustness_sweep",
]
