"""Deterministic transport primitives for the V12 temporal witness display.

The functions in this module are deliberately small and NumPy-only.  They
transport a field with a supplied *forward* image-space flow, expose holes and
many-to-one collisions instead of hiding them, and rematerialize a dense
receiver field from compact integer strategy levels.

No function here estimates optical flow, runs PoseNet, runs SegNet, or computes
the comma.ai challenge score.  A caller that supplies Farneback flow therefore
has an advisory image-space construction, not a pose or exact-score result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ForwardWarpResult:
    """A transported field and the receiver-side support diagnostics."""

    values: Array
    valid: Array
    holes: Array
    collisions: Array
    contributor_count: Array
    weight_sum: Array

    @property
    def occlusion_or_collision(self) -> Array:
        """Return the conservative mask where transport is not one-to-one."""

        return self.holes | self.collisions


@dataclass(frozen=True)
class RefreshDiagnostics:
    """Continuous advisory signals for deciding when transport needs refresh."""

    valid_label_disagreement: float
    valid_boundary_disagreement: float
    hole_fraction: float
    collision_fraction: float
    refresh_pressure: float


@dataclass(frozen=True)
class LockedTemporalDisplay:
    """A hash-verified, non-authoritative temporal display artifact."""

    svg: str
    metadata: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_temporal_display(
    svg_path: str | Path,
    manifest_path: str | Path,
) -> LockedTemporalDisplay:
    """Load the public SVG only after its custody and claim refusals verify."""

    svg_source = Path(svg_path)
    manifest_source = Path(manifest_path)
    metadata = json.loads(manifest_source.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "v12_temporal_transport_display_v1":
        raise ValueError("unexpected temporal display schema")
    record = (metadata.get("outputs") or {}).get(svg_source.name)
    if not isinstance(record, dict):
        raise ValueError("temporal manifest does not lock the requested SVG")
    if svg_source.stat().st_size != record.get("bytes"):
        raise ValueError("temporal SVG byte count does not match its manifest")
    if _sha256_file(svg_source) != record.get("sha256"):
        raise ValueError("temporal SVG hash does not match its manifest")
    if metadata.get("authority") != "[ADVISORY]":
        raise ValueError("temporal display must remain advisory")
    for refusal in ("score_claim", "pose_claim", "full_stac_reproduction"):
        if metadata.get(refusal) is not False:
            raise ValueError(f"temporal display must explicitly refuse {refusal}")
    svg = svg_source.read_text(encoding="utf-8")
    if not svg.lstrip().startswith("<svg") or "<script" in svg.lower():
        raise ValueError("temporal display must be a script-free SVG")
    return LockedTemporalDisplay(svg=svg, metadata=metadata)


def localized_temporal_display_svg(
    display: LockedTemporalDisplay,
    messages: Mapping[str, str],
) -> str:
    """Localize only locked text nodes; image pixels and measurements stay fixed."""

    replacements = {
        "TEMPORAL STRATEGY TRANSPORT · ADVISORY": messages["temporal_svg.badge"],
        "Transport the strategy.": messages["temporal_svg.title_1"],
        "Measure when topology refuses.": messages["temporal_svg.title_2"],
        "One real consecutive pair. One explicit receiver map. No hidden certainty.": messages[
            "temporal_svg.dek"
        ],
        "Frozen partition · raw frame": messages["temporal_svg.frozen"],
        "frozen SegNet argmax [ADVISORY]": messages["temporal_svg.frozen_argmax"],
        "Transported partition": messages["temporal_svg.transported"],
        "forward nearest splat of dense Farneback flow": messages["temporal_svg.forward"],
        "Observed partition · raw frame": messages["temporal_svg.observed"],
        "next frozen SegNet argmax [ADVISORY]": messages["temporal_svg.next_argmax"],
        "Receiver support": messages["temporal_svg.support"],
        "green valid · amber collision · coral hole": messages["temporal_svg.support_legend"],
        "Rematerialized strategy": messages["temporal_svg.rematerialized"],
        "boundary / annulus / interior [DERIVATION]": messages["temporal_svg.strategy_legend"],
        "label debt": messages["temporal_svg.label_debt"],
        "boundary debt": messages["temporal_svg.boundary_debt"],
        "holes": messages["temporal_svg.holes"],
        "collisions": messages["temporal_svg.collisions"],
        "[ADVISORY] OpenCV Farneback image-space estimate": messages["temporal_svg.footer_1"],
        "[DERIVATION] illustrative table bank [1,4,12]": messages["temporal_svg.footer_2"],
        "Not PoseNet · not STAC DIS flow · not a STAC reproduction · not an exact or proxy challenge score": messages[
            "temporal_svg.refusal"
        ],
        "Transport the strategy; measure when topology refuses": messages["temporal_svg.accessible_title"],
    }
    localized = display.svg
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        localized = localized.replace(source, target)
    return localized


def _validate_field_and_flow(field: Array, flow_uv: Array) -> tuple[Array, Array]:
    values = np.asarray(field)
    flow = np.asarray(flow_uv, dtype=np.float64)
    if values.ndim not in (2, 3):
        raise ValueError("field must have shape (height, width) or (height, width, channels)")
    if flow.shape != values.shape[:2] + (2,):
        raise ValueError("flow_uv must have shape (height, width, 2)")
    if not np.all(np.isfinite(flow)):
        raise ValueError("flow_uv must contain only finite values")
    return values, flow


def _empty_output(field: Array, fill_value: float | int, *, dtype: np.dtype) -> Array:
    shape = field.shape
    return np.full(shape, fill_value, dtype=dtype)


def forward_warp_nearest(
    field: Array,
    flow_uv: Array,
    *,
    fill_value: float | int = 0,
) -> ForwardWarpResult:
    """Forward-splat ``field`` with deterministic nearest-neighbour placement.

    ``flow_uv[..., 0]`` is horizontal displacement and ``flow_uv[..., 1]`` is
    vertical displacement, both in pixels of ``field``.  Sources are visited in
    row-major order; at a collision, the earliest source wins while the
    collision remains explicit in ``contributor_count`` and ``collisions``.
    Coordinates use round-half-up placement via ``floor(x + 0.5)``.
    """

    values, flow = _validate_field_and_flow(field, flow_uv)
    height, width = values.shape[:2]
    output = _empty_output(values, fill_value, dtype=values.dtype)
    contributors = np.zeros((height, width), dtype=np.int32)

    for source_y in range(height):
        for source_x in range(width):
            target_x = int(np.floor(source_x + flow[source_y, source_x, 0] + 0.5))
            target_y = int(np.floor(source_y + flow[source_y, source_x, 1] + 0.5))
            if not (0 <= target_x < width and 0 <= target_y < height):
                continue
            if contributors[target_y, target_x] == 0:
                output[target_y, target_x] = values[source_y, source_x]
            contributors[target_y, target_x] += 1

    valid = contributors > 0
    holes = ~valid
    collisions = contributors > 1
    return ForwardWarpResult(
        values=output,
        valid=valid,
        holes=holes,
        collisions=collisions,
        contributor_count=contributors,
        weight_sum=valid.astype(np.float64),
    )


def forward_warp_bilinear(
    field: Array,
    flow_uv: Array,
    *,
    fill_value: float = 0.0,
) -> ForwardWarpResult:
    """Forward-splat a continuous field with normalized bilinear weights.

    Out-of-frame weight is discarded.  Remaining weight is normalized at each
    receiver pixel.  This is intended for continuous fields; categorical maps
    should use :func:`forward_warp_nearest`.
    """

    values, flow = _validate_field_and_flow(field, flow_uv)
    height, width = values.shape[:2]
    channels = () if values.ndim == 2 else (values.shape[2],)
    accumulator = np.zeros((height, width) + channels, dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    contributors = np.zeros((height, width), dtype=np.int32)

    for source_y in range(height):
        for source_x in range(width):
            target_x = source_x + flow[source_y, source_x, 0]
            target_y = source_y + flow[source_y, source_x, 1]
            x0 = int(np.floor(target_x))
            y0 = int(np.floor(target_y))
            dx = float(target_x - x0)
            dy = float(target_y - y0)
            for offset_y, weight_y in ((0, 1.0 - dy), (1, dy)):
                for offset_x, weight_x in ((0, 1.0 - dx), (1, dx)):
                    weight = weight_x * weight_y
                    target_ix = x0 + offset_x
                    target_iy = y0 + offset_y
                    if weight <= 0.0 or not (0 <= target_ix < width and 0 <= target_iy < height):
                        continue
                    accumulator[target_iy, target_ix] += values[source_y, source_x] * weight
                    weights[target_iy, target_ix] += weight
                    contributors[target_iy, target_ix] += 1

    valid = weights > 0.0
    output = _empty_output(values, fill_value, dtype=np.dtype(np.float64))
    if values.ndim == 2:
        output[valid] = accumulator[valid] / weights[valid]
    else:
        output[valid] = accumulator[valid] / weights[valid, None]
    holes = ~valid
    collisions = contributors > 1
    return ForwardWarpResult(
        values=output,
        valid=valid,
        holes=holes,
        collisions=collisions,
        contributor_count=contributors,
        weight_sum=weights,
    )


def interpolate_sparse_flow(
    sparse_flow_uv: Array,
    grid_xy: Array,
    *,
    output_shape: tuple[int, int],
    frame_shape: tuple[int, int],
) -> Array:
    """Interpolate a regular sparse flow grid into an output-resolution field.

    The grid coordinates and flow values are expressed in pixels of
    ``frame_shape``.  Interpolation is separable linear interpolation with edge
    clamping, followed by the exact scale change into ``output_shape`` pixels.
    It does not reconstruct the unavailable dense estimator output.
    """

    sparse = np.asarray(sparse_flow_uv, dtype=np.float64)
    grid = np.asarray(grid_xy, dtype=np.float64)
    if sparse.ndim != 3 or sparse.shape[-1] != 2 or grid.shape != sparse.shape:
        raise ValueError("sparse_flow_uv and grid_xy must share shape (grid_y, grid_x, 2)")
    if not np.all(np.isfinite(sparse)) or not np.all(np.isfinite(grid)):
        raise ValueError("sparse flow inputs must be finite")
    output_height, output_width = output_shape
    frame_height, frame_width = frame_shape
    if min(output_height, output_width, frame_height, frame_width) <= 1:
        raise ValueError("output_shape and frame_shape dimensions must exceed one")

    x_axis = grid[0, :, 0]
    y_axis = grid[:, 0, 1]
    if not np.allclose(grid[:, :, 0], x_axis[None, :]):
        raise ValueError("grid_xy x coordinates must form a regular separable grid")
    if not np.allclose(grid[:, :, 1], y_axis[:, None]):
        raise ValueError("grid_xy y coordinates must form a regular separable grid")
    if np.any(np.diff(x_axis) <= 0) or np.any(np.diff(y_axis) <= 0):
        raise ValueError("grid axes must be strictly increasing")

    target_x = np.linspace(0.0, frame_width - 1.0, output_width)
    target_y = np.linspace(0.0, frame_height - 1.0, output_height)
    dense = np.empty((output_height, output_width, 2), dtype=np.float64)
    for component in range(2):
        along_x = np.empty((sparse.shape[0], output_width), dtype=np.float64)
        for row in range(sparse.shape[0]):
            along_x[row] = np.interp(target_x, x_axis, sparse[row, :, component])
        for column in range(output_width):
            dense[:, column, component] = np.interp(target_y, y_axis, along_x[:, column])

    dense[..., 0] *= (output_width - 1.0) / (frame_width - 1.0)
    dense[..., 1] *= (output_height - 1.0) / (frame_height - 1.0)
    return dense


def nearest_resize(field: Array, output_shape: tuple[int, int]) -> Array:
    """Resize a 2-D or channel-last field by deterministic nearest sampling."""

    values = np.asarray(field)
    if values.ndim not in (2, 3):
        raise ValueError("field must be two-dimensional or channel-last three-dimensional")
    output_height, output_width = output_shape
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_shape values must be positive")
    source_height, source_width = values.shape[:2]
    ys = np.floor(np.linspace(0, source_height - 1, output_height) + 0.5).astype(np.int64)
    xs = np.floor(np.linspace(0, source_width - 1, output_width) + 0.5).astype(np.int64)
    return values[np.ix_(ys, xs)]


def decode_palette_labels(rgb: Array, palette: Array | Sequence[Sequence[int]]) -> Array:
    """Decode an exact RGB palette, failing closed on an unknown color."""

    image = np.asarray(rgb)
    colors = np.asarray(palette, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] > 255:
        raise ValueError("palette must have shape (classes, 3) with at most 255 classes")
    matches = np.all(image[:, :, None, :] == colors[None, None, :, :], axis=-1)
    known = np.any(matches, axis=-1)
    if not np.all(known):
        unknown = np.unique(image[~known].reshape(-1, 3), axis=0)
        raise ValueError(f"rgb contains colors outside the declared palette: {unknown[:4].tolist()}")
    return np.argmax(matches, axis=-1).astype(np.uint8)


def boundary_mask(labels: Array) -> Array:
    """Return the four-neighbour disagreement set of a categorical partition."""

    field = np.asarray(labels)
    if field.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    edge = np.zeros_like(field, dtype=bool)
    edge[:-1, :] |= field[:-1, :] != field[1:, :]
    edge[1:, :] |= field[:-1, :] != field[1:, :]
    edge[:, :-1] |= field[:, :-1] != field[:, 1:]
    edge[:, 1:] |= field[:, :-1] != field[:, 1:]
    return edge


def boundary_strategy_levels(labels: Array, *, annulus_radius: int = 1) -> Array:
    """Build a three-level boundary/annulus/interior display construction.

    Level 0 is the categorical boundary, level 1 is its ``annulus_radius``
    four-neighbour dilation, and level 2 is the remaining interior.  This is a
    geometric construction; it is not a learned STAC sensitivity map.
    """

    if annulus_radius < 0:
        raise ValueError("annulus_radius must be nonnegative")
    edge = boundary_mask(labels)
    annulus = edge.copy()
    for _ in range(annulus_radius):
        grown = annulus.copy()
        grown[:-1, :] |= annulus[1:, :]
        grown[1:, :] |= annulus[:-1, :]
        grown[:, :-1] |= annulus[:, 1:]
        grown[:, 1:] |= annulus[:, :-1]
        annulus = grown
    levels = np.full(edge.shape, 2, dtype=np.uint8)
    levels[annulus] = 1
    levels[edge] = 0
    return levels


def rematerialize_levels(levels: Array, table_bank: Array | Sequence[float]) -> Array:
    """Expand compact integer level indices through a receiver-held table bank."""

    selected = np.asarray(levels)
    tables = np.asarray(table_bank)
    if selected.ndim != 2 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("levels must be a two-dimensional integer array")
    if tables.ndim not in (1, 2) or tables.shape[0] < 1:
        raise ValueError("table_bank must have shape (levels,) or (levels, features)")
    if np.any(selected < 0) or np.any(selected >= tables.shape[0]):
        raise ValueError("levels contains an index outside table_bank")
    return tables[selected]


def topology_refresh_diagnostics(
    transported_labels: Array,
    target_labels: Array,
    *,
    valid: Array,
    collisions: Array,
) -> RefreshDiagnostics:
    """Compare transported and observed partitions without claiming a score.

    ``refresh_pressure`` is the conservative maximum of four normalized debts:
    valid-label disagreement, valid boundary-set disagreement, holes, and
    collisions.  It is a display/control signal in ``[0, 1]`` and is not a
    paper threshold, a SegNet score, or a comma.ai score component.
    """

    transported = np.asarray(transported_labels)
    target = np.asarray(target_labels)
    valid_mask = np.asarray(valid, dtype=bool)
    collision_mask = np.asarray(collisions, dtype=bool)
    if transported.shape != target.shape or transported.ndim != 2:
        raise ValueError("transported_labels and target_labels must share a two-dimensional shape")
    if valid_mask.shape != transported.shape or collision_mask.shape != transported.shape:
        raise ValueError("valid and collisions must match the label shape")

    if np.any(valid_mask):
        label_debt = float(np.mean(transported[valid_mask] != target[valid_mask]))
        transported_edge = boundary_mask(transported)
        target_edge = boundary_mask(target)
        boundary_debt = float(np.mean((transported_edge != target_edge)[valid_mask]))
    else:
        label_debt = 1.0
        boundary_debt = 1.0
    hole_fraction = float(np.mean(~valid_mask))
    collision_fraction = float(np.mean(collision_mask))
    pressure = max(label_debt, boundary_debt, hole_fraction, collision_fraction)
    return RefreshDiagnostics(
        valid_label_disagreement=label_debt,
        valid_boundary_disagreement=boundary_debt,
        hole_fraction=hole_fraction,
        collision_fraction=collision_fraction,
        refresh_pressure=pressure,
    )


__all__ = [
    "ForwardWarpResult",
    "LockedTemporalDisplay",
    "RefreshDiagnostics",
    "boundary_mask",
    "boundary_strategy_levels",
    "decode_palette_labels",
    "forward_warp_bilinear",
    "forward_warp_nearest",
    "interpolate_sparse_flow",
    "load_temporal_display",
    "localized_temporal_display_svg",
    "nearest_resize",
    "rematerialize_levels",
    "topology_refresh_diagnostics",
]
