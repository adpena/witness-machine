"""Real-frame demonstration figures for the V12 notebook.

Two locked-bundle artifacts feed this module:

- ``v12_real_frame_display.npz`` — the locked real pair's two frames,
  co-registered with the sealed frozen-scorer fields (384x512).
- ``v12_boundary_evolution.npz`` — the witness's realized + internal
  partitions on the same pair at the preserved per-stage EMA checkpoints of
  one real n600 training run, plus its rendered RGB frames and the sealed
  frozen-scorer partition.

Every number shown is read from the locked artifacts/manifests; this module
renders and never measures. Palette: comma10k-canonical class colors in the
frozen scorer's class order, identical to the sibling evidence figures.
"""
from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .v12_real_evidence import (
    RealEvidenceError,
    _rgb_png_bytes,
    normalized_log_uint16,
    sha256_file,
)

FRAME_SCHEMA = "molab.v12.real_frame_display.v1"
EVOLUTION_SCHEMA = "molab.v12.boundary_evolution.v1"

# comma10k-canonical class colors in the frozen scorer's class order
# (0 Road, 1 Lane markings, 2 Undrivable, 3 Movable, 4 MyCar).
CLASS_PALETTE = np.array(
    [
        [64, 32, 32],
        [255, 0, 0],
        [128, 128, 96],
        [0, 255, 102],
        [204, 0, 255],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class LockedRealFrames:
    frames: np.ndarray  # (2, 384, 512, 3) uint8
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class LockedBoundaryEvolution:
    realized_argmax: np.ndarray  # (E, 384, 512) uint8
    internal_argmax: np.ndarray  # (E, 384, 512) uint8
    witness_rgb: np.ndarray  # (E, 384, 512, 3) uint8
    sealed_argmax: np.ndarray  # (384, 512) uint8
    metadata: Mapping[str, Any]


def _load_locked(npz_path: str | Path, manifest_path: str | Path, schema: str):
    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_bytes") != npz_path.stat().st_size:
        raise RealEvidenceError(f"{npz_path.name}: byte count differs from its manifest")
    if manifest.get("artifact_sha256") != sha256_file(npz_path):
        raise RealEvidenceError(f"{npz_path.name}: SHA-256 differs from its manifest")
    data = np.load(npz_path)
    metadata = json.loads(str(data["metadata_json"]))
    if metadata.get("schema") != schema:
        raise RealEvidenceError(
            f"{npz_path.name}: expected schema {schema}, got {metadata.get('schema')}"
        )
    return data, metadata


def load_real_frames(
    npz_path: str | Path, manifest_path: str | Path
) -> LockedRealFrames:
    data, metadata = _load_locked(npz_path, manifest_path, FRAME_SCHEMA)
    frames = np.asarray(data["frames_rgb_u8"])
    if frames.shape != (2, 384, 512, 3) or frames.dtype != np.uint8:
        raise RealEvidenceError("frames_rgb_u8 must be (2, 384, 512, 3) uint8")
    return LockedRealFrames(frames=frames, metadata=metadata)


def load_boundary_evolution(
    npz_path: str | Path, manifest_path: str | Path
) -> LockedBoundaryEvolution:
    data, metadata = _load_locked(npz_path, manifest_path, EVOLUTION_SCHEMA)
    realized = np.asarray(data["realized_argmax_u8"])
    internal = np.asarray(data["internal_argmax_u8"])
    rgb = np.asarray(data["witness_rgb_u8"])
    sealed = np.asarray(data["sealed_argmax_u8"])
    checkpoints = metadata.get("checkpoints", [])
    expected = (len(checkpoints), 384, 512)
    if realized.shape != expected or realized.dtype != np.uint8:
        raise RealEvidenceError("realized_argmax_u8 shape/dtype mismatch")
    if internal.shape != expected or internal.dtype != np.uint8:
        raise RealEvidenceError("internal_argmax_u8 shape/dtype mismatch")
    if rgb.shape != expected + (3,) or rgb.dtype != np.uint8:
        raise RealEvidenceError("witness_rgb_u8 shape/dtype mismatch")
    if sealed.shape != (384, 512) or sealed.dtype != np.uint8:
        raise RealEvidenceError("sealed_argmax_u8 shape/dtype mismatch")
    for stack in (realized, internal, sealed):
        if int(stack.max()) > 4:
            raise RealEvidenceError("argmax fields must stay in the 5-class range")
    return LockedBoundaryEvolution(
        realized_argmax=realized,
        internal_argmax=internal,
        witness_rgb=rgb,
        sealed_argmax=sealed,
        metadata=metadata,
    )


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Two-sided four-neighbour argmax transition mask (same definition as
    the sealed evidence's boundary field)."""
    mask = np.zeros(labels.shape, dtype=bool)
    mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
    mask[1:, :] |= labels[1:, :] != labels[:-1, :]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return mask


def _partition_overlay(frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
    overlay = (
        0.55 * frame.astype(np.float64) + 0.45 * CLASS_PALETTE[labels].astype(np.float64)
    )
    overlay = np.clip(np.rint(overlay), 0, 255).astype(np.uint8)
    overlay[_boundary_mask(labels)] = 255
    return overlay


def _dilate_max(field: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev (max-pool) dilation for DISPLAY emphasis only."""
    out = field.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="edge")
        stacked = np.stack(
            [
                padded[dy : dy + field.shape[0], dx : dx + field.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
        out = stacked.max(axis=0)
    return out


def _annulus_overlay(frame: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    """Dimmed luma road with the margin-normalized sensitivity glowing
    coral-to-gold on the log display scale (glow widened 2 px for display)."""
    luma = (
        0.2126 * frame[..., 0].astype(np.float64)
        + 0.7152 * frame[..., 1].astype(np.float64)
        + 0.0722 * frame[..., 2].astype(np.float64)
    )
    # brightness-lift the night frame for legibility, then dim to ground
    lifted = np.clip(luma * 1.9 + 14.0, 0.0, 255.0) * 0.42
    base = np.repeat(lifted[..., None], 3, axis=-1)
    level = normalized_log_uint16(sensitivity).astype(np.float64) / 65535.0
    level = np.sqrt(_dilate_max(level, 2))
    coral = np.array([255.0, 116.0, 108.0])
    gold = np.array([255.0, 209.0, 102.0])
    ramp = coral[None, None, :] * (1.0 - level[..., None]) + gold[None, None, :] * level[..., None]
    alpha = level[..., None]
    blended = base * (1.0 - alpha) + ramp * alpha
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _display_gain(image: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(np.rint(image.astype(np.float64) * gain), 0, 255).astype(np.uint8)


def _data_url(image: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(_rgb_png_bytes(image)).decode("ascii")


def _fmt_pct(locale: str, fraction: float, digits: int = 2) -> str:
    rendered = f"{100.0 * fraction:.{digits}f}"
    if locale.lower().startswith("es"):
        rendered = rendered.replace(".", ",")
    return f"{rendered}%"


def real_road_partition_svg(
    frames: LockedRealFrames,
    evidence_arrays: Mapping[str, np.ndarray],
    evidence_metadata: Mapping[str, Any],
    *,
    messages: Mapping[str, str],
    locale: str = "en-US",
) -> str:
    """Three panels on the SAME real frame: the road, the frozen-scorer
    partition it induces, and the sensitivity annulus glowing on the road."""
    frame = np.asarray(frames.frames[1])
    labels = np.asarray(evidence_arrays["argmax_u8"]).astype(np.int64)
    sensitivity = np.asarray(evidence_arrays["flip_risk_f32"], dtype=np.float64)
    if labels.shape != frame.shape[:2] or sensitivity.shape != frame.shape[:2]:
        raise RealEvidenceError("real-frame panels must be co-registered at 384x512")
    panels = (
        (frame, messages["realframe.panel_road"], messages["realframe.panel_road_aria"]),
        (
            _partition_overlay(frame, labels),
            messages["realframe.panel_partition"],
            messages["realframe.panel_partition_aria"],
        ),
        (
            _annulus_overlay(frame, sensitivity),
            messages["realframe.panel_annulus"],
            messages["realframe.panel_annulus_aria"],
        ),
    )
    summary = evidence_metadata["metrics"]["flip_risk"]
    boundary_share = _fmt_pct(locale, summary["boundary_pixel_fraction"], 1)
    mass_share = _fmt_pct(
        locale, summary["annuli_chebyshev_px"]["0"]["inside_mass_fraction"], 1
    )
    axis = evidence_metadata["evidence_axis"]
    pair = evidence_metadata["pair_selection"]["pair_index_zero_based"]
    title = messages["realframe.title"]
    description = messages["realframe.description"].format(
        pair=pair, axis=axis, boundary=boundary_share, mass=mass_share
    )
    parts = []
    for index, (image, label, aria) in enumerate(panels):
        x = 32 + 360 * index
        parts.append(
            f'<text x="{x}" y="67" class="panel-title">{html.escape(label)}</text>'
            f'<image x="{x}" y="82" width="336" height="252" preserveAspectRatio="none" '
            f'href="{_data_url(image)}" role="img" aria-label="{html.escape(aria)}"/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xml:lang="{html.escape(locale)}" role="img" '
        'aria-labelledby="v12-realframe-title v12-realframe-desc" focusable="false" '
        'viewBox="0 0 1120 448" width="100%" preserveAspectRatio="xMidYMid meet">'
        f'<title id="v12-realframe-title">{html.escape(title)}</title>'
        f'<desc id="v12-realframe-desc">{html.escape(description)}</desc>'
        '<style>.panel-title{font:600 17px system-ui,sans-serif;fill:#f8fafc}'
        '.note{font:13px system-ui,sans-serif;fill:#cbd5e1}'
        '.stat{font:600 25px system-ui,sans-serif;fill:#f8fafc;font-variant-numeric:tabular-nums}'
        '.stat-label{font:13px system-ui,sans-serif;fill:#94a3b8}</style>'
        '<rect width="1120" height="448" rx="8" fill="#080d1f"/>'
        f'<text x="32" y="34" style="font:700 21px system-ui,sans-serif;fill:#fff">{html.escape(title)}</text>'
        + "".join(parts)
        + f'<text x="32" y="384" class="stat">{html.escape(boundary_share)}</text>'
        f'<text x="32" y="404" class="stat-label">{html.escape(messages["realframe.stat_boundary"])}</text>'
        f'<text x="262" y="384" class="stat">{html.escape(mass_share)}</text>'
        f'<text x="262" y="404" class="stat-label">{html.escape(messages["realframe.stat_mass"])}</text>'
        f'<text x="552" y="394" class="note">{html.escape(messages["realframe.scale_note"])}</text>'
        f'<text x="1088" y="394" text-anchor="end" class="note">'
        f'{html.escape(messages["real.visual.pair"])} {pair} · {html.escape(axis)}</text>'
        "</svg>"
    )


def witness_evolution_svg(
    evolution: LockedBoundaryEvolution,
    *,
    messages: Mapping[str, str],
    locale: str = "en-US",
    real_frame: np.ndarray | None = None,
) -> str:
    """Two rows across training: what the witness paints (its rendered
    frames), and the partition the frozen scorer reads from those frames,
    closing on the real frame and its sealed frozen-scorer partition."""
    checkpoints = list(evolution.metadata["checkpoints"])
    count = len(checkpoints)
    columns = count + 1
    panel_w, panel_h, gap, x0 = 200, 150, 12, 32
    width = x0 * 2 + columns * panel_w + (columns - 1) * gap
    axis = evolution.metadata["evidence_axis"]
    pair = evolution.metadata["pair_selection"]["pair_index_zero_based"]
    title = messages["evolution.title"]
    description = messages["evolution.description"].format(
        pair=pair, axis=axis, count=count
    )
    row1_y, row2_y = 96, 300
    parts = [
        f'<text x="{x0}" y="{row1_y - 10}" class="row-title">{html.escape(messages["evolution.row_witness"])}</text>',
        f'<text x="{x0}" y="{row2_y - 10}" class="row-title">{html.escape(messages["evolution.row_realized"])}</text>',
    ]
    for index in range(columns):
        x = x0 + index * (panel_w + gap)
        if index < count:
            row = checkpoints[index]
            disagreement = _fmt_pct(
                locale, row["pair196_realized_disagreement_vs_sealed_argmax"], 2
            )
            top = _display_gain(np.asarray(evolution.witness_rgb[index]), 2.2)
            top_aria = messages["evolution.row_witness_aria"]
            bottom = CLASS_PALETTE[evolution.realized_argmax[index].astype(np.int64)]
            caption = f'{messages["evolution.epoch"]} {row["epoch"]} · {row["stage"]}'
            stat = f'{messages["evolution.disagreement"]} {disagreement}'
        else:
            top = (
                None
                if real_frame is None
                else _display_gain(np.asarray(real_frame), 2.2)
            )
            top_aria = messages["evolution.reference_frame_aria"]
            bottom = CLASS_PALETTE[evolution.sealed_argmax.astype(np.int64)]
            caption = messages["evolution.reference"]
            stat = messages["evolution.reference_note"]
        if top is not None:
            parts.append(
                f'<image x="{x}" y="{row1_y}" width="{panel_w}" height="{panel_h}" '
                f'preserveAspectRatio="none" href="{_data_url(top)}" role="img" '
                f'aria-label="{html.escape(top_aria)}"/>'
            )
        parts.append(
            f'<image x="{x}" y="{row2_y}" width="{panel_w}" height="{panel_h}" '
            f'preserveAspectRatio="none" href="{_data_url(bottom)}" role="img" '
            f'style="image-rendering:pixelated" '
            f'aria-label="{html.escape(messages["evolution.row_realized_aria"])}"/>'
        )
        parts.append(
            f'<text x="{x}" y="{row2_y + panel_h + 22}" class="caption">{html.escape(caption)}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{row2_y + panel_h + 40}" class="stat-small">{html.escape(stat)}</text>'
        )
    footnote = str(messages["evolution.footnote"])
    footnote_lines = [footnote]
    if len(footnote) > 150:
        split_at = footnote.rfind(";", 0, 150)
        if split_at < 0:
            split_at = footnote.rfind(" ", 0, 150)
        if split_at > 0:
            footnote_lines = [footnote[: split_at + 1].rstrip(), footnote[split_at + 1 :].lstrip()]
    height = row2_y + panel_h + 72 + 18 * len(footnote_lines)
    footnote_parts = "".join(
        f'<text x="{x0}" y="{row2_y + panel_h + 66 + 18 * line_index}" class="note">{html.escape(line)}</text>'
        for line_index, line in enumerate(footnote_lines)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xml:lang="{html.escape(locale)}" role="img" '
        'aria-labelledby="v12-evolution-title v12-evolution-desc" focusable="false" '
        f'viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="xMidYMid meet">'
        f'<title id="v12-evolution-title">{html.escape(title)}</title>'
        f'<desc id="v12-evolution-desc">{html.escape(description)}</desc>'
        '<style>.row-title{font:600 17px system-ui,sans-serif;fill:#f8fafc}'
        '.caption{font:600 13px system-ui,sans-serif;fill:#e2e8f0;font-variant-numeric:tabular-nums}'
        '.stat-small{font:12px system-ui,sans-serif;fill:#94a3b8;font-variant-numeric:tabular-nums}'
        '.note{font:13px system-ui,sans-serif;fill:#cbd5e1}</style>'
        f'<rect width="{width}" height="{height}" rx="8" fill="#080d1f"/>'
        f'<text x="{x0}" y="38" style="font:700 21px system-ui,sans-serif;fill:#fff">{html.escape(title)}</text>'
        f'<text x="{width - x0}" y="38" text-anchor="end" class="note">'
        f'{html.escape(messages["real.visual.pair"])} {pair} · {html.escape(axis)}</text>'
        + "".join(parts)
        + footnote_parts
        + "</svg>"
    )


__all__ = [
    "CLASS_PALETTE",
    "EVOLUTION_SCHEMA",
    "FRAME_SCHEMA",
    "LockedBoundaryEvolution",
    "LockedRealFrames",
    "load_boundary_evolution",
    "load_real_frames",
    "real_road_partition_svg",
    "witness_evolution_svg",
]
