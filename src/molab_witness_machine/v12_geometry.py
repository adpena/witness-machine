"""Deterministic geometry kernels for the public V12 notebook.

The functions here are deliberately small and inspectable.  They construct an
analytic signed-distance decision wall, its swept disagreement set, a literal
Morse double-well gradient system, and a receiver-side temporal warp.  They do
not call the comma.ai evaluator and none of their scalar outputs is a challenge
score.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class BoundarySweep:
    """One internally consistent analytic-SDF displacement construction."""

    x: Array
    y: Array
    phi_reference: Array
    phi_moved: Array
    reference_partition: Array
    moved_partition: Array
    swept_mask: Array
    exact_area_fraction: float
    raster_area_fraction: float
    first_order_area_fraction: float
    normalized_boundary_length: float
    displacement: float


@dataclass(frozen=True)
class CriticalPoint:
    """Analytically verified critical point of the double-well potential."""

    x: float
    y: float
    kind: str
    hessian_eigenvalues: tuple[float, float]


@dataclass(frozen=True)
class FlowTrace:
    """A sampled negative-gradient trajectory and its potential values."""

    points: Array
    energy: Array


def _skew(vector: Array) -> Array:
    """Return the 3x3 cross-product matrix of a three-vector."""

    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def se3_exp(twist: Array, theta: float = 1.0) -> Array:
    """Evaluate ``exp(hat(xi) * theta)`` for ``xi=(v, omega)``.

    The implementation uses the closed-form Rodrigues rotation and the
    corresponding translation Jacobian.  It is a geometric transport model;
    no interpretation of PoseNet's learned coordinates is asserted.
    """

    xi = np.asarray(twist, dtype=np.float64)
    amount = _finite_scalar(theta, name="theta")
    if xi.shape != (6,) or not np.all(np.isfinite(xi)):
        raise ValueError("twist must be a finite six-vector ordered (v, omega)")
    velocity = xi[:3]
    omega = xi[3:]
    angular_speed = float(np.linalg.norm(omega))
    transform = np.eye(4, dtype=np.float64)
    if angular_speed < 1e-12:
        transform[:3, 3] = amount * velocity
        return transform

    omega_hat = _skew(omega)
    phase = angular_speed * amount
    omega_hat_sq = omega_hat @ omega_hat
    rotation = (
        np.eye(3)
        + (np.sin(phase) / angular_speed) * omega_hat
        + ((1.0 - np.cos(phase)) / angular_speed**2) * omega_hat_sq
    )
    translation_jacobian = (
        amount * np.eye(3)
        + ((1.0 - np.cos(phase)) / angular_speed**2) * omega_hat
        + ((phase - np.sin(phase)) / angular_speed**3) * omega_hat_sq
    )
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_jacobian @ velocity
    return transform


def screw_trajectory(twist: Array, point: Array, thetas: Array) -> Array:
    """Transport one 3-D point through an explicit sequence of SE(3) poses."""

    source = np.asarray(point, dtype=np.float64)
    samples = np.asarray(thetas, dtype=np.float64)
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("point must be a finite three-vector")
    if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("thetas must be a finite one-dimensional array with at least two samples")
    homogeneous = np.concatenate((source, [1.0]))
    return np.stack([(se3_exp(twist, float(theta)) @ homogeneous)[:3] for theta in samples])


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def boundary_sweep(
    displacement: float,
    *,
    height: int = 96,
    width: int = 144,
    radius: float = 0.55,
    center: tuple[float, float] = (0.0, 0.0),
) -> BoundarySweep:
    """Move one circular SDF zero set and measure its swept disagreement area.

    Coordinates cover ``[-1, 1] x [-1, 1]`` and
    ``phi(x,y)=||(x,y)-center||-radius`` is an exact signed-distance field away
    from its medial point. Subtracting ``displacement`` therefore moves the zero
    set by that signed normal distance. The exact swept area and its first-order
    boundary integral are both closed form; the raster mask is retained only as
    a visible discretization check.
    """

    shift = _finite_scalar(displacement, name="displacement")
    base_radius = _finite_scalar(radius, name="radius")
    center_values = np.asarray(center, dtype=np.float64)
    if height < 8 or width < 8:
        raise ValueError("height and width must each be at least 8")
    if base_radius <= 0.0:
        raise ValueError("radius must be positive")
    if center_values.shape != (2,) or not np.all(np.isfinite(center_values)):
        raise ValueError("center must be a finite (x, y) pair")
    moved_radius = base_radius + shift
    if moved_radius <= 0.0:
        raise ValueError("displacement collapses or reverses the SDF circle")
    outer_radius = max(base_radius, moved_radius)
    if np.any(np.abs(center_values) + outer_radius >= 1.0):
        raise ValueError("the SDF circle must remain strictly inside the integration domain")

    x_line = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    y_line = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    x, y = np.meshgrid(x_line, y_line)
    phi_reference = np.hypot(x - center_values[0], y - center_values[1]) - base_radius
    phi_moved = phi_reference - shift
    reference = phi_reference <= 0.0
    moved = phi_moved <= 0.0
    swept = reference != moved

    boundary_length = float(2.0 * np.pi * base_radius)
    domain_area = 4.0
    exact_area = float(np.pi * abs(moved_radius**2 - base_radius**2) / domain_area)
    first_order = float(boundary_length * abs(shift) / domain_area)
    return BoundarySweep(
        x=x,
        y=y,
        phi_reference=phi_reference,
        phi_moved=phi_moved,
        reference_partition=reference,
        moved_partition=moved,
        swept_mask=swept,
        exact_area_fraction=exact_area,
        raster_area_fraction=float(np.mean(swept)),
        first_order_area_fraction=first_order,
        normalized_boundary_length=float(boundary_length / 2.0),
        displacement=shift,
    )


def soft_competition(fields: Array, temperature: float) -> tuple[Array, Array]:
    """Return stable soft assignments and entropy for a field stack.

    ``fields`` has shape ``(classes, ...)``.  Temperature is positive; lower
    values approach the tropical/max-plus argmax partition.
    """

    values = np.asarray(fields, dtype=np.float64)
    tau = _finite_scalar(temperature, name="temperature")
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("fields must contain at least two class fields")
    if tau <= 0.0:
        raise ValueError("temperature must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("fields must be finite")

    scaled = values / tau
    scaled -= np.max(scaled, axis=0, keepdims=True)
    weights = np.exp(scaled)
    weights /= np.sum(weights, axis=0, keepdims=True)
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-15)), axis=0)
    return weights, entropy


def double_well_potential(x: Array | float, y: Array | float) -> Array:
    """Evaluate ``U(x,y)=(x^2-1)^2+y^2``."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    return (x_values**2 - 1.0) ** 2 + y_values**2


def double_well_gradient(x: Array | float, y: Array | float) -> tuple[Array, Array]:
    """Return the analytic gradient of :func:`double_well_potential`."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    return 4.0 * x_values * (x_values**2 - 1.0), 2.0 * y_values


def double_well_critical_points() -> tuple[CriticalPoint, ...]:
    """Return the two minima and one saddle with analytic Hessian spectra."""

    return (
        CriticalPoint(-1.0, 0.0, "minimum", (8.0, 2.0)),
        CriticalPoint(0.0, 0.0, "saddle", (-4.0, 2.0)),
        CriticalPoint(1.0, 0.0, "minimum", (8.0, 2.0)),
    )


def integrate_negative_gradient(
    start: tuple[float, float],
    *,
    step: float = 0.025,
    steps: int = 160,
) -> FlowTrace:
    """Integrate ``x_dot=-grad U`` with deterministic bounded Euler steps."""

    if steps < 1:
        raise ValueError("steps must be positive")
    dt = _finite_scalar(step, name="step")
    if dt <= 0.0 or dt > 0.05:
        raise ValueError("step must be in (0, 0.05]")
    point = np.asarray(start, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("start must be a finite (x, y) pair")

    points = np.empty((steps + 1, 2), dtype=np.float64)
    energy = np.empty(steps + 1, dtype=np.float64)
    for index in range(steps + 1):
        points[index] = point
        energy[index] = float(double_well_potential(point[0], point[1]))
        if index == steps:
            break
        grad_x, grad_y = double_well_gradient(point[0], point[1])
        gradient = np.array([float(grad_x), float(grad_y)])
        proposal = point - dt * gradient
        proposal_energy = float(double_well_potential(proposal[0], proposal[1]))
        # The public construction should never show an integration artifact as
        # uphill flow.  Backtrack deterministically if Euler overshoots.
        local_dt = dt
        while proposal_energy > energy[index] + 1e-12 and local_dt > 1e-8:
            local_dt *= 0.5
            proposal = point - local_dt * gradient
            proposal_energy = float(double_well_potential(proposal[0], proposal[1]))
        point = proposal
    return FlowTrace(points=points, energy=energy)


def warp_nearest(values: Array, flow_yx: Array, *, fill_value: int | float = -1) -> Array:
    """Backward-warp a 2-D strategy map with a dense ``(dy, dx)`` field."""

    source = np.asarray(values)
    flow = np.asarray(flow_yx, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("values must be two-dimensional")
    if flow.shape != source.shape + (2,):
        raise ValueError("flow_yx must have shape values.shape + (2,)")
    if not np.all(np.isfinite(flow)):
        raise ValueError("flow_yx must be finite")

    height, width = source.shape
    yy, xx = np.mgrid[0:height, 0:width]
    source_y = np.rint(yy - flow[..., 0]).astype(np.int64)
    source_x = np.rint(xx - flow[..., 1]).astype(np.int64)
    valid = (source_y >= 0) & (source_y < height) & (source_x >= 0) & (source_x < width)
    output = np.full(source.shape, fill_value, dtype=np.result_type(source.dtype, type(fill_value)))
    output[valid] = source[source_y[valid], source_x[valid]]
    return output


def peak_signal_to_noise_ratio(reference: Array, candidate: Array, *, data_range: float = 1.0) -> float:
    """Compute PSNR for a declared numeric range; return infinity on equality."""

    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("reference and candidate must have identical shapes")
    span = _finite_scalar(data_range, name="data_range")
    if span <= 0.0:
        raise ValueError("data_range must be positive")
    mse = float(np.mean((first - second) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * log10((span**2) / mse)


__all__ = [
    "BoundarySweep",
    "CriticalPoint",
    "FlowTrace",
    "boundary_sweep",
    "double_well_critical_points",
    "double_well_gradient",
    "double_well_potential",
    "integrate_negative_gradient",
    "peak_signal_to_noise_ratio",
    "screw_trajectory",
    "se3_exp",
    "soft_competition",
    "warp_nearest",
]
