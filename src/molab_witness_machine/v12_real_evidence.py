"""Locked real frozen-scorer evidence for the V12 notebook.

This module deliberately stops short of calling the artifact a STAC
reproduction.  It validates and summarizes a real SegNet input-gradient
surface, then measures how much of that surface lies inside and outside
semantic-boundary annuli.  STAC's DCT-coefficient gradient, quantization
roundtrip, regional table selection, and temporal propagation are separate
experiments.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import struct
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "molab.v12.real_frozen_scorer_evidence.v1"
DISPLAY_SCHEMA = "molab.v12.real_frozen_scorer_display_derivative.v1"
DISPLAY_MANIFEST_SCHEMA = "molab.v12.real_frozen_scorer_display_manifest.v1"
MAX_DISPLAY_PACKAGE_BYTES = 300_000
MARGIN_PROXY_EPSILON = 1e-6
MARGIN_PROXY_FORMULA_DECLARATION = (
    "historical compatibility key: grad_energy / (margin^2 + 1e-6); "
    "not a probability or local flip radius"
)
REQUIRED_ARRAYS = {
    "argmax_u8",
    "boundary_u8",
    "flip_risk_f32",
    "grad_energy_f32",
    "gradient_magnitude_f32",
    "margin_f32",
    "sensitivity_display_u16",
}
DISPLAY_REQUIRED_ARRAYS = {
    "argmax_mode_u8",
    "boundary_coverage_u8",
    "flip_risk_log_mean_u8",
    "gradient_log_mean_u8",
}


class RealEvidenceError(ValueError):
    """Raised when a real-evidence artifact fails closed validation."""


@dataclass(frozen=True)
class LockedRealEvidence:
    """Validated arrays and provenance from one locked evidence artifact."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class LockedDisplayDerivative:
    """Validated compact display arrays subordinate to a locked parent artifact."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    manifest: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and embedded metadata."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Return a two-sided four-neighbour boundary mask for a label field."""

    field = np.asarray(labels)
    if field.ndim != 2 or field.size == 0:
        raise RealEvidenceError("labels must be a nonempty two-dimensional array")
    if not np.issubdtype(field.dtype, np.integer):
        raise RealEvidenceError("labels must have an integer dtype")

    boundary = np.zeros(field.shape, dtype=bool)
    vertical = field[1:, :] != field[:-1, :]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    horizontal = field[:, 1:] != field[:, :-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask by a square (Chebyshev) radius using NumPy only."""

    current = np.asarray(mask, dtype=bool)
    if current.ndim != 2 or current.size == 0:
        raise RealEvidenceError("mask must be a nonempty two-dimensional array")
    if not isinstance(radius, int) or radius < 0:
        raise RealEvidenceError("radius must be a nonnegative integer")
    current = current.copy()
    for _ in range(radius):
        padded = np.pad(current, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(current)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + current.shape[0], dx : dx + current.shape[1]]
        current = expanded
    return current


def _validated_sensitivity(values: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    sensitivity = np.asarray(values, dtype=np.float64)
    if sensitivity.shape != shape:
        raise RealEvidenceError("sensitivity and labels must have identical shapes")
    if not np.all(np.isfinite(sensitivity)):
        raise RealEvidenceError("sensitivity must contain only finite values")
    if np.any(sensitivity < 0):
        raise RealEvidenceError("sensitivity must be nonnegative")
    if float(sensitivity.sum(dtype=np.float64)) <= 0:
        raise RealEvidenceError("sensitivity must have positive total mass")
    return sensitivity


def _gini_nonnegative(values: np.ndarray) -> float:
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(flat.sum(dtype=np.float64))
    if flat.size == 0 or total <= 0:
        return 0.0
    indices = np.arange(1, flat.size + 1, dtype=np.float64)
    gini = 2.0 * float(np.sum(indices * flat, dtype=np.float64)) / (flat.size * total)
    gini -= (flat.size + 1.0) / flat.size
    return float(np.clip(gini, 0.0, 1.0))


def boundary_sensitivity_metrics(
    sensitivity: np.ndarray,
    labels: np.ndarray,
    *,
    annulus_widths: Sequence[int] = (0, 1, 3, 7),
    top_percentages: Sequence[float] = (1.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Measure boundary and off-boundary mass without implying STAC authority.

    The annulus at width zero is the two-sided four-neighbour semantic
    boundary.  Larger widths use Chebyshev dilation.  Every reported fraction
    is computed from the supplied nonnegative surface; no score is inferred.
    """

    label_field = np.asarray(labels)
    boundary = semantic_boundary_mask(label_field)
    values = _validated_sensitivity(sensitivity, shape=label_field.shape)
    widths = tuple(int(width) for width in annulus_widths)
    if not widths or any(width < 0 for width in widths) or len(set(widths)) != len(widths):
        raise RealEvidenceError("annulus_widths must be distinct nonnegative integers")
    percentages = tuple(float(value) for value in top_percentages)
    if any(not np.isfinite(value) or value <= 0 or value > 100 for value in percentages):
        raise RealEvidenceError("top_percentages must lie in (0, 100]")

    total_mass = float(values.sum(dtype=np.float64))
    annuli: dict[str, dict[str, float]] = {}
    for width in widths:
        annulus = dilate_chebyshev(boundary, width)
        inside_mass = float(values[annulus].sum(dtype=np.float64))
        outside_mass = total_mass - inside_mass
        inside_mean = float(values[annulus].mean(dtype=np.float64)) if annulus.any() else 0.0
        outside = ~annulus
        outside_mean = float(values[outside].mean(dtype=np.float64)) if outside.any() else 0.0
        annuli[str(width)] = {
            "annulus_pixel_fraction": float(annulus.mean(dtype=np.float64)),
            "inside_mass_fraction": inside_mass / total_mass,
            "outside_mass_fraction": outside_mass / total_mass,
            "inside_over_outside_mean_density": inside_mean / (outside_mean + 1e-30),
        }

    flat = values.reshape(-1)
    descending = np.sort(flat)[::-1]
    cumulative = np.cumsum(descending, dtype=np.float64)
    top_mass: dict[str, float] = {}
    for percentage in percentages:
        count = min(flat.size, max(1, int(np.ceil(flat.size * percentage / 100.0))))
        top_mass[f"{percentage:g}"] = float(cumulative[count - 1] / total_mass)

    return {
        "schema": "molab.v12.boundary_sensitivity_metrics.v1",
        "height": int(label_field.shape[0]),
        "width": int(label_field.shape[1]),
        "pixel_count": int(label_field.size),
        "boundary_pixel_fraction": float(boundary.mean(dtype=np.float64)),
        "total_mass": total_mass,
        "nonzero_fraction": float(np.count_nonzero(values) / values.size),
        "gini": _gini_nonnegative(values),
        "top_percent_mass": top_mass,
        "annuli_chebyshev_px": annuli,
    }


def margin_normalized_sensitivity_proxy(
    grad_energy: np.ndarray,
    margin: np.ndarray,
    *,
    epsilon: float = MARGIN_PROXY_EPSILON,
) -> np.ndarray:
    """Return the declared heuristic ``grad_energy / (margin**2 + epsilon)``.

    This is a formula-binding helper, not a calibrated flip probability.  A
    fixed additive ``epsilon`` makes the surface sensitive to both the chosen
    regularizer and logit scale, so callers must keep the parameter explicit.
    """

    energy = np.asarray(grad_energy, dtype=np.float64)
    local_margin = np.asarray(margin, dtype=np.float64)
    regularizer = float(epsilon)
    if energy.ndim != 2 or energy.size == 0 or local_margin.shape != energy.shape:
        raise RealEvidenceError("grad_energy and margin must be nonempty matching 2D arrays")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0):
        raise RealEvidenceError("grad_energy must be finite and nonnegative")
    if not np.all(np.isfinite(local_margin)) or np.any(local_margin < 0):
        raise RealEvidenceError("margin must be finite and nonnegative")
    if not np.isfinite(regularizer) or regularizer <= 0:
        raise RealEvidenceError("epsilon must be finite and strictly positive")
    return energy / (np.square(local_margin) + regularizer)


def normalized_log_uint16(values: np.ndarray, *, curvature: float = 4095.0) -> np.ndarray:
    """Return a deterministic log-normalized 16-bit display field."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise RealEvidenceError("values must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise RealEvidenceError("values must be finite and nonnegative")
    if not np.isfinite(curvature) or curvature <= 0:
        raise RealEvidenceError("curvature must be finite and positive")
    maximum = float(array.max(initial=0.0))
    if maximum <= 0:
        return np.zeros(array.shape, dtype=np.uint16)
    normalized = np.log1p(curvature * array / maximum) / np.log1p(curvature)
    return np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_deterministic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a byte-deterministic compressed NPZ with fixed ZIP metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(
            temporary_name,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in sorted(arrays):
                if not name or "/" in name or "\\" in name:
                    raise RealEvidenceError(f"unsafe NPZ array name: {name!r}")
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, _npy_bytes(np.asarray(arrays[name])), compresslevel=9)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_locked_evidence(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the deterministic NPZ and its external hash manifest atomically."""

    artifact = Path(artifact_path)
    manifest_destination = Path(manifest_path)
    metadata_dict = dict(metadata)
    if metadata_dict.get("schema") != SCHEMA:
        raise RealEvidenceError(f"metadata schema must be {SCHEMA!r}")
    metadata_payload = canonical_json_bytes(metadata_dict)
    payload_arrays = {name: np.asarray(value) for name, value in arrays.items()}
    payload_arrays["metadata_json"] = np.asarray(metadata_payload.decode("utf-8"))
    _validate_arrays(payload_arrays, metadata_dict)
    write_deterministic_npz(artifact, payload_arrays)

    manifest = {
        "schema": "molab.v12.real_frozen_scorer_evidence_manifest.v1",
        "artifact_file": artifact.name,
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256_file(artifact),
        "metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        "claim_class": "EMPIRICAL",
        "evidence_axis": metadata_dict.get("evidence_axis"),
        "score_claim": False,
        "stac_reproduction": False,
    }
    _atomic_write_bytes(manifest_destination, canonical_json_bytes(manifest) + b"\n")
    return manifest


def _validate_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    missing = REQUIRED_ARRAYS - set(arrays)
    if missing:
        raise RealEvidenceError(f"evidence artifact is missing arrays: {sorted(missing)}")
    labels = np.asarray(arrays["argmax_u8"])
    boundary = np.asarray(arrays["boundary_u8"])
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise RealEvidenceError("argmax_u8 must be a two-dimensional uint8 array")
    if labels.size == 0 or int(labels.max(initial=0)) > 4:
        raise RealEvidenceError("argmax_u8 must contain only class indices 0..4")
    if boundary.shape != labels.shape or boundary.dtype != np.uint8:
        raise RealEvidenceError("boundary_u8 must match argmax_u8 and use uint8")
    if not np.array_equal(boundary.astype(bool), semantic_boundary_mask(labels)):
        raise RealEvidenceError("boundary_u8 does not match the label-derived boundary")

    for name in ("flip_risk_f32", "grad_energy_f32", "gradient_magnitude_f32", "margin_f32"):
        array = np.asarray(arrays[name])
        if array.shape != labels.shape or array.dtype != np.float32:
            raise RealEvidenceError(f"{name} must match argmax_u8 and use float32")
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise RealEvidenceError(f"{name} must be finite and nonnegative")
    display = np.asarray(arrays["sensitivity_display_u16"])
    if display.shape != labels.shape or display.dtype != np.uint16:
        raise RealEvidenceError("sensitivity_display_u16 must match argmax_u8 and use uint16")
    if not np.allclose(
        np.square(np.asarray(arrays["gradient_magnitude_f32"], dtype=np.float64)),
        np.asarray(arrays["grad_energy_f32"], dtype=np.float64),
        rtol=2e-5,
        atol=1e-12,
    ):
        raise RealEvidenceError("gradient magnitude is inconsistent with grad energy")
    surface = metadata.get("surface")
    if not isinstance(surface, Mapping):
        raise RealEvidenceError("metadata must declare the margin-proxy surface formula")
    if surface.get("flip_risk_formula") != MARGIN_PROXY_FORMULA_DECLARATION:
        raise RealEvidenceError("metadata margin-proxy formula is missing or unsupported")
    expected_proxy = margin_normalized_sensitivity_proxy(
        arrays["grad_energy_f32"],
        arrays["margin_f32"],
    )
    if not np.allclose(
        np.asarray(arrays["flip_risk_f32"], dtype=np.float64),
        expected_proxy,
        rtol=2e-6,
        atol=1e-12,
    ):
        raise RealEvidenceError("flip_risk_f32 is inconsistent with the declared margin proxy")
    if metadata.get("score_claim") is not False or metadata.get("stac_reproduction") is not False:
        raise RealEvidenceError("metadata must explicitly refuse score and STAC reproduction claims")


def _assert_nested_close(expected: Any, actual: Any, *, path: str = "metrics") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise RealEvidenceError(f"{path} keys do not match recomputed metrics")
        for key in expected:
            _assert_nested_close(expected[key], actual[key], path=f"{path}.{key}")
        return
    if isinstance(expected, (int, np.integer)) and isinstance(actual, (int, np.integer)):
        if int(expected) != int(actual):
            raise RealEvidenceError(f"{path} does not match recomputed metrics")
        return
    if isinstance(expected, (float, int, np.floating, np.integer)) and isinstance(
        actual, (float, int, np.floating, np.integer)
    ):
        if not np.isclose(float(expected), float(actual), rtol=2e-6, atol=1e-12):
            raise RealEvidenceError(f"{path} does not match recomputed metrics")
        return
    if expected != actual:
        raise RealEvidenceError(f"{path} does not match recomputed metrics")


def load_locked_evidence(
    artifact_path: str | Path,
    manifest_path: str | Path,
) -> LockedRealEvidence:
    """Load, hash-check, and numerically revalidate a locked evidence artifact."""

    artifact = Path(artifact_path)
    manifest_source = Path(manifest_path)
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if manifest.get("schema") != "molab.v12.real_frozen_scorer_evidence_manifest.v1":
        raise RealEvidenceError("unsupported real-evidence manifest schema")
    if manifest.get("artifact_file") != artifact.name:
        raise RealEvidenceError("manifest artifact_file does not match the supplied artifact")
    if manifest.get("artifact_bytes") != artifact.stat().st_size:
        raise RealEvidenceError("artifact byte count does not match its manifest")
    if manifest.get("artifact_sha256") != sha256_file(artifact):
        raise RealEvidenceError("artifact SHA-256 does not match its manifest")

    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if "metadata_json" not in arrays:
        raise RealEvidenceError("artifact is missing embedded metadata_json")
    metadata_text = str(np.asarray(arrays.pop("metadata_json")).item())
    metadata_payload = metadata_text.encode("utf-8")
    if manifest.get("metadata_sha256") != hashlib.sha256(metadata_payload).hexdigest():
        raise RealEvidenceError("embedded metadata SHA-256 does not match its manifest")
    metadata = json.loads(metadata_text)
    if metadata.get("schema") != SCHEMA:
        raise RealEvidenceError("unsupported embedded evidence schema")
    _validate_arrays(arrays, metadata)

    configuration = metadata.get("metric_configuration", {})
    widths = tuple(configuration.get("annulus_widths", ()))
    top_percentages = tuple(configuration.get("top_percentages", ()))
    if not widths or not top_percentages:
        raise RealEvidenceError("metadata is missing metric configuration")
    recomputed = {
        "gradient_magnitude": boundary_sensitivity_metrics(
            arrays["gradient_magnitude_f32"],
            arrays["argmax_u8"],
            annulus_widths=widths,
            top_percentages=top_percentages,
        ),
        "flip_risk": boundary_sensitivity_metrics(
            arrays["flip_risk_f32"],
            arrays["argmax_u8"],
            annulus_widths=widths,
            top_percentages=top_percentages,
        ),
    }
    _assert_nested_close(metadata.get("metrics"), recomputed)
    return LockedRealEvidence(arrays=arrays, metadata=metadata, manifest=manifest)


def _block_view(array: np.ndarray, factor: int) -> np.ndarray:
    field = np.asarray(array)
    if field.ndim != 2 or field.size == 0:
        raise RealEvidenceError("display source arrays must be nonempty and two-dimensional")
    if not isinstance(factor, int) or factor <= 0:
        raise RealEvidenceError("display downsample factor must be a positive integer")
    height, width = field.shape
    if height % factor or width % factor:
        raise RealEvidenceError("display downsample factor must divide both source dimensions")
    return (
        field.reshape(height // factor, factor, width // factor, factor)
        .transpose(0, 2, 1, 3)
        .reshape(height // factor, width // factor, factor * factor)
    )


def block_mode_uint8(labels: np.ndarray, *, factor: int) -> np.ndarray:
    """Downsample labels by block mode, resolving ties to the lower class index."""

    field = np.asarray(labels)
    if not np.issubdtype(field.dtype, np.integer):
        raise RealEvidenceError("block-mode labels must use an integer dtype")
    if np.any(field < 0) or int(field.max(initial=0)) > 255:
        raise RealEvidenceError("block-mode labels must lie in uint8 range")
    blocks = _block_view(field.astype(np.uint8, copy=False), factor)
    maximum = int(field.max(initial=0))
    counts = np.stack([(blocks == label).sum(axis=-1) for label in range(maximum + 1)], axis=-1)
    return counts.argmax(axis=-1).astype(np.uint8)


def _block_mean_to_uint8(values: np.ndarray, *, factor: int, full_scale: int) -> np.ndarray:
    field = np.asarray(values)
    if not np.issubdtype(field.dtype, np.integer):
        raise RealEvidenceError("display block means require an integer source array")
    if full_scale <= 0 or np.any(field < 0) or int(field.max(initial=0)) > full_scale:
        raise RealEvidenceError("display block-mean source is outside its declared scale")
    blocks = _block_view(field, factor)
    denominator = blocks.shape[-1] * full_scale
    numerator = blocks.astype(np.uint64).sum(axis=-1, dtype=np.uint64) * 255
    return ((numerator + denominator // 2) // denominator).astype(np.uint8)


def _derive_display_arrays(
    parent: LockedRealEvidence,
    *,
    factor: int,
    log_curvature: float,
) -> dict[str, np.ndarray]:
    labels = np.asarray(parent.arrays["argmax_u8"])
    boundary = np.asarray(parent.arrays["boundary_u8"])
    gradient_log = normalized_log_uint16(
        parent.arrays["gradient_magnitude_f32"],
        curvature=log_curvature,
    )
    flip_risk_log = normalized_log_uint16(
        parent.arrays["flip_risk_f32"],
        curvature=log_curvature,
    )
    return {
        "argmax_mode_u8": block_mode_uint8(labels, factor=factor),
        "boundary_coverage_u8": _block_mean_to_uint8(
            boundary,
            factor=factor,
            full_scale=1,
        ),
        "gradient_log_mean_u8": _block_mean_to_uint8(
            gradient_log,
            factor=factor,
            full_scale=65535,
        ),
        "flip_risk_log_mean_u8": _block_mean_to_uint8(
            flip_risk_log,
            factor=factor,
            full_scale=65535,
        ),
    }


def _display_observation_summary(parent: LockedRealEvidence) -> dict[str, float]:
    metrics = parent.metadata.get("metrics", {})
    gradient = metrics.get("gradient_magnitude", {})
    flip_risk = metrics.get("flip_risk", {})
    try:
        return {
            "boundary_pixel_fraction": float(gradient["boundary_pixel_fraction"]),
            "gradient_mass_on_boundary": float(
                gradient["annuli_chebyshev_px"]["0"]["inside_mass_fraction"]
            ),
            "gradient_mass_outside_3px": float(
                gradient["annuli_chebyshev_px"]["3"]["outside_mass_fraction"]
            ),
            "gradient_mass_outside_15px": float(
                gradient["annuli_chebyshev_px"]["15"]["outside_mass_fraction"]
            ),
            "flip_risk_mass_on_boundary": float(
                flip_risk["annuli_chebyshev_px"]["0"]["inside_mass_fraction"]
            ),
            "flip_risk_mass_outside_3px": float(
                flip_risk["annuli_chebyshev_px"]["3"]["outside_mass_fraction"]
            ),
            "flip_risk_mass_outside_15px": float(
                flip_risk["annuli_chebyshev_px"]["15"]["outside_mass_fraction"]
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RealEvidenceError(
            "parent artifact lacks the locked annulus metrics required by the display derivative"
        ) from exc


def _validate_display_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    missing = DISPLAY_REQUIRED_ARRAYS - set(arrays)
    if missing:
        raise RealEvidenceError(f"display derivative is missing arrays: {sorted(missing)}")
    shapes = {np.asarray(arrays[name]).shape for name in DISPLAY_REQUIRED_ARRAYS}
    if len(shapes) != 1:
        raise RealEvidenceError("display derivative arrays must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 2 or not all(value > 0 for value in shape):
        raise RealEvidenceError("display derivative arrays must be nonempty and two-dimensional")
    for name in DISPLAY_REQUIRED_ARRAYS:
        if np.asarray(arrays[name]).dtype != np.uint8:
            raise RealEvidenceError(f"{name} must use uint8")
    if int(np.asarray(arrays["argmax_mode_u8"]).max(initial=0)) > 4:
        raise RealEvidenceError("argmax_mode_u8 must contain only class indices 0..4")
    declared_shape = tuple(metadata.get("dimensions", {}).get("display_height_width", ()))
    if declared_shape != shape:
        raise RealEvidenceError("display derivative shape does not match embedded metadata")
    if (
        metadata.get("claim_class") != "ADVISORY"
        or metadata.get("new_authority") is not False
        or metadata.get("score_claim") is not False
        or metadata.get("stac_reproduction") is not False
    ):
        raise RealEvidenceError("display derivative must explicitly refuse new authority claims")


def write_display_derivative(
    parent_artifact_path: str | Path,
    parent_manifest_path: str | Path,
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    factor: int = 2,
    log_curvature: float = 4095.0,
) -> dict[str, Any]:
    """Derive one compact advisory display artifact from an already locked parent."""

    parent_artifact = Path(parent_artifact_path)
    parent_manifest = Path(parent_manifest_path)
    destination = Path(artifact_path)
    manifest_destination = Path(manifest_path)
    parent = load_locked_evidence(parent_artifact, parent_manifest)
    arrays = _derive_display_arrays(parent, factor=factor, log_curvature=log_curvature)
    display_shape = next(iter(arrays.values())).shape
    source_shape = np.asarray(parent.arrays["argmax_u8"]).shape
    parent_lock = {
        "artifact_file": parent_artifact.name,
        "artifact_bytes": parent_artifact.stat().st_size,
        "artifact_sha256": sha256_file(parent_artifact),
        "manifest_file": parent_manifest.name,
        "manifest_sha256": sha256_file(parent_manifest),
        "metadata_sha256": parent.manifest["metadata_sha256"],
        "schema": parent.metadata["schema"],
    }
    limitations = list(parent.metadata.get("limitations", ())) + [
        "This factor-reduced, 8-bit display derivative is lossy and is not a measurement surface.",
        "Block-mode labels and block-mean log fields are for public rendering only; use the locked parent for analysis.",
        "The derivative creates no score, STAC, rank, promotion, or exact-eval authority.",
    ]
    array_semantics = {
        "argmax_mode_u8": (
            "factor-by-factor block mode of parent argmax_u8; equal-count ties choose the lower class index"
        ),
        "boundary_coverage_u8": (
            "rounded block mean of parent boundary_u8 mapped from [0,1] to [0,255]"
        ),
        "gradient_log_mean_u8": (
            "rounded block mean after global log1p normalization of parent gradient_magnitude_f32"
        ),
        "flip_risk_log_mean_u8": (
            "rounded block mean after global log1p normalization of the historical compatibility-named parent flip_risk_f32; this field is a margin-normalized summed-margin sensitivity proxy, not a probability"
        ),
    }
    downsample = {
        "factor_y": factor,
        "factor_x": factor,
        "log_curvature": log_curvature,
        "log_formula": (
            "log1p(log_curvature * value / global_max) / log1p(log_curvature)"
        ),
        "log_intermediate_quantization": "numpy.rint to uint16 [0,65535] (ties-to-even)",
        "mode_tie_break": "lowest_class_index",
        "continuous_reduce": (
            "sum uint16 block values, scale the block mean from [0,65535] to [0,255], "
            "then apply positive integer half-up rounding"
        ),
        "boundary_reduce": (
            "sum binary boundary block values, scale the block mean from [0,1] to [0,255], "
            "then apply positive integer half-up rounding"
        ),
        "final_rounding": "positive integer half-up",
        "source_only": "locked parent NPZ; no scorer, video decode, or Pact checkout consulted",
    }
    metadata = {
        "schema": DISPLAY_SCHEMA,
        "claim_class": "ADVISORY",
        "evidence_axis": parent.metadata.get("evidence_axis"),
        "artifact_role": "compact deterministic public-display derivative",
        "new_authority": False,
        "score_claim": False,
        "stac_reproduction": False,
        "parent": parent_lock,
        "pair_selection": parent.metadata.get("pair_selection"),
        "dimensions": {
            "source_height_width": [int(value) for value in source_shape],
            "display_height_width": [int(value) for value in display_shape],
        },
        "downsample": downsample,
        "array_semantics": array_semantics,
        "observation_summary_from_parent": _display_observation_summary(parent),
        "limitations": limitations,
    }
    metadata_payload = canonical_json_bytes(metadata)
    payload_arrays = {name: np.asarray(value) for name, value in arrays.items()}
    payload_arrays["metadata_json"] = np.asarray(metadata_payload.decode("utf-8"))
    _validate_display_arrays(payload_arrays, metadata)
    write_deterministic_npz(destination, payload_arrays)

    manifest = {
        "schema": DISPLAY_MANIFEST_SCHEMA,
        "artifact_file": destination.name,
        "artifact_bytes": destination.stat().st_size,
        "artifact_sha256": sha256_file(destination),
        "metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        "claim_class": "ADVISORY",
        "evidence_axis": metadata["evidence_axis"],
        "new_authority": False,
        "score_claim": False,
        "stac_reproduction": False,
        "parent": parent_lock,
        "pair_selection": metadata["pair_selection"],
        "dimensions": metadata["dimensions"],
        "downsample": downsample,
        "array_semantics": array_semantics,
        "observation_summary_from_parent": metadata["observation_summary_from_parent"],
        "limitations": limitations,
    }
    _atomic_write_bytes(manifest_destination, canonical_json_bytes(manifest) + b"\n")
    package_bytes = destination.stat().st_size + manifest_destination.stat().st_size
    if package_bytes >= MAX_DISPLAY_PACKAGE_BYTES:
        raise RealEvidenceError(
            f"display derivative package is {package_bytes} bytes; budget is "
            f"strictly below {MAX_DISPLAY_PACKAGE_BYTES}"
        )
    return manifest


def load_display_derivative(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    parent_artifact_path: str | Path | None = None,
    parent_manifest_path: str | Path | None = None,
) -> LockedDisplayDerivative:
    """Load a compact derivative, optionally proving it byte-for-byte from its parent."""

    artifact = Path(artifact_path)
    manifest_source = Path(manifest_path)
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if manifest.get("schema") != DISPLAY_MANIFEST_SCHEMA:
        raise RealEvidenceError("unsupported display derivative manifest schema")
    if manifest.get("artifact_file") != artifact.name:
        raise RealEvidenceError("display manifest artifact_file does not match supplied artifact")
    if manifest.get("artifact_bytes") != artifact.stat().st_size:
        raise RealEvidenceError("display artifact byte count does not match its manifest")
    if manifest.get("artifact_sha256") != sha256_file(artifact):
        raise RealEvidenceError("display artifact SHA-256 does not match its manifest")
    if artifact.stat().st_size + manifest_source.stat().st_size >= MAX_DISPLAY_PACKAGE_BYTES:
        raise RealEvidenceError("display derivative package exceeds its public byte budget")

    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if "metadata_json" not in arrays:
        raise RealEvidenceError("display derivative is missing embedded metadata_json")
    metadata_text = str(np.asarray(arrays.pop("metadata_json")).item())
    metadata_payload = metadata_text.encode("utf-8")
    if manifest.get("metadata_sha256") != hashlib.sha256(metadata_payload).hexdigest():
        raise RealEvidenceError("display metadata SHA-256 does not match its manifest")
    metadata = json.loads(metadata_text)
    if metadata.get("schema") != DISPLAY_SCHEMA:
        raise RealEvidenceError("unsupported embedded display derivative schema")
    _validate_display_arrays(arrays, metadata)
    mirrored_fields = (
        "claim_class",
        "evidence_axis",
        "new_authority",
        "score_claim",
        "stac_reproduction",
        "parent",
        "pair_selection",
        "dimensions",
        "downsample",
        "array_semantics",
        "observation_summary_from_parent",
        "limitations",
    )
    for field in mirrored_fields:
        if manifest.get(field) != metadata.get(field):
            raise RealEvidenceError(f"display manifest field {field!r} disagrees with metadata")

    if (parent_artifact_path is None) != (parent_manifest_path is None):
        raise RealEvidenceError("parent artifact and parent manifest must be supplied together")
    if parent_artifact_path is not None and parent_manifest_path is not None:
        parent_artifact = Path(parent_artifact_path)
        parent_manifest = Path(parent_manifest_path)
        lock = metadata["parent"]
        if sha256_file(parent_artifact) != lock["artifact_sha256"]:
            raise RealEvidenceError("supplied parent artifact does not match the derivative lock")
        if sha256_file(parent_manifest) != lock["manifest_sha256"]:
            raise RealEvidenceError("supplied parent manifest does not match the derivative lock")
        parent = load_locked_evidence(parent_artifact, parent_manifest)
        factor_y = metadata["downsample"]["factor_y"]
        factor_x = metadata["downsample"]["factor_x"]
        if factor_y != factor_x:
            raise RealEvidenceError("current display derivative requires an isotropic factor")
        recomputed = _derive_display_arrays(
            parent,
            factor=factor_y,
            log_curvature=float(metadata["downsample"]["log_curvature"]),
        )
        for name in DISPLAY_REQUIRED_ARRAYS:
            if not np.array_equal(arrays[name], recomputed[name]):
                raise RealEvidenceError(f"display array {name!r} does not match its locked parent")
    return LockedDisplayDerivative(arrays=arrays, metadata=metadata, manifest=manifest)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _rgb_png_bytes(rgb: np.ndarray) -> bytes:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise RealEvidenceError("embedded PNG input must be height-by-width-by-3 uint8 RGB")
    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + row.tobytes(order="C") for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _heatmap_rgb(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.uint16)
    if field.ndim != 2 or np.any(field > 255):
        raise RealEvidenceError("heatmap field must be a two-dimensional uint8-scale array")
    base = np.array([7, 12, 35], dtype=np.uint16)
    middle = np.array([0, 180, 200], dtype=np.uint16)
    high = np.array([255, 230, 80], dtype=np.uint16)
    output = np.empty(field.shape + (3,), dtype=np.uint16)
    lower = field <= 128
    lower_weight = np.minimum(field, 128)[..., None]
    upper_weight = np.maximum(field.astype(np.int32) - 128, 0).astype(np.uint16)[..., None]
    lower_rgb = (base * (128 - lower_weight) + middle * lower_weight + 64) // 128
    upper_rgb = (middle * (127 - upper_weight) + high * upper_weight + 63) // 127
    output[:] = upper_rgb
    output[lower] = lower_rgb[lower]
    return output.astype(np.uint8)


def display_derivative_svg(
    derivative: LockedDisplayDerivative,
    *,
    title: str | None = None,
    messages: Mapping[str, str] | None = None,
    locale: str = "en-US",
) -> str:
    """Emit a responsive, accessible, self-contained SVG with embedded PNG panels."""

    _validate_display_arrays(derivative.arrays, derivative.metadata)
    labels = np.asarray(derivative.arrays["argmax_mode_u8"])
    boundary = np.asarray(derivative.arrays["boundary_coverage_u8"], dtype=np.uint16)
    # comma10k-canonical class colors in the frozen scorer's class order
    # (0 Road, 1 Lane markings, 2 Undrivable, 3 Movable, 4 MyCar) — the same
    # encoding the temporal-transport figure uses, so one palette means one
    # thing everywhere in the notebook.
    palette = np.array(
        [
            [64, 32, 32],
            [255, 0, 0],
            [128, 128, 96],
            [0, 255, 102],
            [204, 0, 255],
        ],
        dtype=np.uint8,
    )
    semantic = palette[labels]
    boundary_alpha = (boundary * 3 // 5)[..., None]
    semantic = (
        (
            semantic.astype(np.uint16) * (255 - boundary_alpha)
            + 255 * boundary_alpha
            + 127
        )
        // 255
    ).astype(np.uint8)
    images = (
        semantic,
        _heatmap_rgb(derivative.arrays["gradient_log_mean_u8"]),
        _heatmap_rgb(derivative.arrays["flip_risk_log_mean_u8"]),
    )
    data_urls = [
        "data:image/png;base64," + base64.b64encode(_rgb_png_bytes(image)).decode("ascii")
        for image in images
    ]
    summary = derivative.metadata["observation_summary_from_parent"]
    pair = derivative.metadata["pair_selection"]["pair_index_zero_based"]
    axis = derivative.metadata["evidence_axis"]
    factor = derivative.metadata["downsample"]["factor_y"]
    defaults = {
        "real.visual.title": "Frozen-scorer sensitivity: field and margin proxy",
        "real.visual.description": (
            "Advisory factor-{factor} display of locked real pair {pair} on {axis}. "
            "Semantic boundaries occupy {boundary:.2f} percent of the parent grid. "
            "They carry {gradient:.2f} percent of raw gradient magnitude but "
            "{flip_risk:.2f} percent of a margin-normalized sensitivity proxy at epsilon=1e-6. "
            "The proxy is regularizer- and logit-scale-sensitive. This derivative is lossy, "
            "makes no score claim, and is not a STAC reproduction."
        ),
        "real.visual.semantic": "Semantic carrier",
        "real.visual.gradient": "Raw gradient magnitude",
        "real.visual.flip_risk": "Margin-normalized sensitivity proxy",
        "real.visual.semantic_aria": "Block-mode semantic argmax with boundary coverage brightened",
        "real.visual.gradient_aria": "Log-normalized block-mean raw gradient magnitude",
        "real.visual.flip_risk_aria": "Log-normalized block-mean margin-normalized sensitivity proxy",
        "real.visual.lower": "lower display value",
        "real.visual.higher": "higher display value",
        "real.visual.pair": "pair",
    }

    def text(key: str) -> str:
        return str(defaults[key] if messages is None else messages.get(key, defaults[key]))

    resolved_title = text("real.visual.title") if title is None else str(title)
    description = text("real.visual.description").format(
        factor=factor,
        pair=pair,
        axis=axis,
        boundary=100 * summary["boundary_pixel_fraction"],
        gradient=100 * summary["gradient_mass_on_boundary"],
        flip_risk=100 * summary["flip_risk_mass_on_boundary"],
    )
    embedded_metadata = html.escape(
        canonical_json_bytes(
            {
                "schema": derivative.metadata["schema"],
                "parent": derivative.metadata["parent"],
                "pair_selection": derivative.metadata["pair_selection"],
                "evidence_axis": axis,
                "new_authority": False,
            }
        ).decode("utf-8")
    )
    safe_title = html.escape(resolved_title)
    safe_description = html.escape(description)
    panel_labels = (
        text("real.visual.semantic"),
        text("real.visual.gradient"),
        text("real.visual.flip_risk"),
    )
    panel_aria = (
        text("real.visual.semantic_aria"),
        text("real.visual.gradient_aria"),
        text("real.visual.flip_risk_aria"),
    )
    panels: list[str] = []
    for index, (url, label, aria) in enumerate(zip(data_urls, panel_labels, panel_aria)):
        x = 32 + 360 * index
        rendering = "pixelated" if index == 0 else "auto"
        panels.append(
            f'<text x="{x}" y="67" class="panel-title">{html.escape(label)}</text>'
            f'<image x="{x}" y="82" width="336" height="252" '
            f'preserveAspectRatio="none" href="{url}" role="img" '
            f'style="image-rendering:{rendering}" '
            f'aria-label="{html.escape(aria)}"/>'
        )
    boundary_pct = 100 * summary["boundary_pixel_fraction"]
    flip_pct = 100 * summary["flip_risk_mass_on_boundary"]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xml:lang="{html.escape(locale)}" role="img" '
        'aria-labelledby="v12-display-title v12-display-desc" focusable="false" '
        'viewBox="0 0 1120 448" width="100%" preserveAspectRatio="xMidYMid meet">'
        f'<title id="v12-display-title">{safe_title}</title>'
        f'<desc id="v12-display-desc">{safe_description}</desc>'
        f'<metadata>{embedded_metadata}</metadata>'
        '<defs><linearGradient id="v12-heat" x1="0" x2="1">'
        '<stop offset="0" stop-color="#070c23"/><stop offset="0.5" stop-color="#00b4c8"/>'
        '<stop offset="1" stop-color="#ffe650"/></linearGradient></defs>'
        '<style>.panel-title{font:600 17px system-ui,sans-serif;fill:#f8fafc}'
        '.note{font:13px system-ui,sans-serif;fill:#cbd5e1}'
        '.stat{font:600 25px system-ui,sans-serif;fill:#f8fafc;font-variant-numeric:tabular-nums}'
        '.stat-label{font:13px system-ui,sans-serif;fill:#94a3b8}</style>'
        '<rect width="1120" height="448" rx="8" fill="#080d1f"/>'
        f'<text x="32" y="34" style="font:700 21px system-ui,sans-serif;fill:#fff">{safe_title}</text>'
        + "".join(panels)
        + f'<text x="32" y="384" class="stat">{boundary_pct:.1f}%</text>'
        f'<text x="32" y="404" class="stat-label">of pixels are boundary</text>'
        f'<text x="182" y="384" class="stat">{flip_pct:.1f}%</text>'
        f'<text x="182" y="404" class="stat-label">of sensitivity mass sits there</text>'
        '<rect x="392" y="366" width="336" height="8" fill="url(#v12-heat)"/>'
        f'<text x="392" y="394" class="note">{html.escape(text("real.visual.lower"))}</text>'
        f'<text x="728" y="394" text-anchor="end" class="note">{html.escape(text("real.visual.higher"))}</text>'
        f'<text x="1088" y="394" text-anchor="end" class="note">{html.escape(text("real.visual.pair"))} {pair} · {html.escape(axis)}</text>'
        "</svg>"
    )


__all__ = [
    "DISPLAY_MANIFEST_SCHEMA",
    "DISPLAY_SCHEMA",
    "LockedDisplayDerivative",
    "LockedRealEvidence",
    "MAX_DISPLAY_PACKAGE_BYTES",
    "MARGIN_PROXY_EPSILON",
    "MARGIN_PROXY_FORMULA_DECLARATION",
    "RealEvidenceError",
    "SCHEMA",
    "block_mode_uint8",
    "boundary_sensitivity_metrics",
    "canonical_json_bytes",
    "display_derivative_svg",
    "dilate_chebyshev",
    "load_display_derivative",
    "load_locked_evidence",
    "margin_normalized_sensitivity_proxy",
    "normalized_log_uint16",
    "semantic_boundary_mask",
    "sha256_file",
    "write_deterministic_npz",
    "write_display_derivative",
    "write_locked_evidence",
]
