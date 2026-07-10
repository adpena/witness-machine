#!/usr/bin/env python3
"""Build the v12 real-frame display + boundary-evolution locked artifacts.

Produces two sealed-bundle artifacts (deterministic NPZ + manifest each):

1. ``v12_real_frame_display.npz`` — the locked real pair's two frames
   (source frames 392 and 393 of the public challenge video), decoded through
   the pinned upstream ``frame_utils.yuv420_to_rgb`` path and bilinearly
   resized to the frozen scorer's 384x512 input grid so they are
   pixel-co-registered with the sealed sensitivity/argmax fields in
   ``v12_real_frozen_scorer_evidence.npz``.

2. ``v12_boundary_evolution.npz`` — the witness's argmax partition on the
   same locked pair decoded from the preserved per-stage EMA checkpoints of
   one real n600 training run, so the notebook can show the level-set
   boundary evolving across training against the sealed frozen-scorer
   partition. Per-checkpoint pair-196 disagreement versus the sealed argmax
   is computed here, deterministically, from the same bytes that ship.

Honesty contract (mirrors the sibling evidence builders):
- No score, ranking, promotion, or exact-eval claim; ``score_claim=false``.
- The training run's own verdict axis is advisory
  ("[macOS-MLX training-gradient]/[macOS-CPU advisory]") and is copied into
  the manifest verbatim, never upgraded.
- Reads the Pact checkout strictly read-only.

Usage (from the witness-machine checkout, using the Pact venv for torch/av):

    /path/to/pact/.venv/bin/python scripts/build_v12_real_frames.py \
        --pact-root /path/to/pact \
        --run-dir experiments/results/levelset_n600_v2_attrclean_20260630T194549Z
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from molab_witness_machine.v12_real_evidence import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
    write_deterministic_npz,
)

PAIR_INDEX = 196
SOURCE_FRAME_INDICES = (392, 393)
SCORER_HW = (384, 512)
VIDEO_SHA256 = "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"
FRAME_SCHEMA = "molab.v12.real_frame_display.v1"
EVOLUTION_SCHEMA = "molab.v12.boundary_evolution.v1"
DEFAULT_RUN_DIR = "experiments/results/levelset_n600_v2_attrclean_20260630T194549Z"


def _discover_stage_checkpoints(run_dir: Path) -> tuple[tuple[str, str, int], ...]:
    """Find preserved per-stage EMA checkpoints, sorted by epoch.

    Decode-fidelity note: ``--structured-init`` and ``--lane-prior-phi1`` are
    INIT-time levers (they pretrain / shape the initial weights, which the
    trained checkpoint then carries); the render path itself stays the plain
    neural witness forward, so the TLI decode is faithful for these runs.
    """
    import re

    rows = []
    for path in sorted(run_dir.glob("levelset_ckpt_stage*_ep*.npz")):
        match = re.match(r"levelset_ckpt_stage(.+)_ep(\d+)\.npz$", path.name)
        if match:
            rows.append((match.group(1), path.name, int(match.group(2))))
    rows.sort(key=lambda row: row[2])
    if len(rows) < 2:
        raise SystemExit(f"run has fewer than 2 preserved stage checkpoints: {run_dir}")
    return tuple(rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_locked(path: Path, arrays: dict, metadata: dict) -> None:
    arrays = dict(arrays)
    arrays["metadata_json"] = np.array(
        canonical_json_bytes(metadata).decode("utf-8")
    )
    write_deterministic_npz(path, arrays)
    manifest = {
        "schema": metadata["schema"] + ".manifest",
        "artifact_file": path.name,
        "artifact_bytes": path.stat().st_size,
        "artifact_sha256": sha256_file(path),
        "metadata_sha256": _sha256_bytes(canonical_json_bytes(metadata)),
        "claim_class": metadata["claim_class"],
        "evidence_axis": metadata["evidence_axis"],
        "score_claim": False,
        "new_authority": False,
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print(f"wrote {path.name} ({manifest['artifact_bytes']:,} bytes) + manifest")


def _decode_scorer_frames(pact_root: Path) -> tuple[np.ndarray, dict]:
    """Decode frames 392/393 and resize to the frozen scorer's input grid."""
    video = pact_root / "upstream" / "videos" / "0.mkv"
    actual_sha = sha256_file(video)
    if actual_sha != VIDEO_SHA256:
        raise SystemExit(
            f"challenge video sha mismatch: expected {VIDEO_SHA256}, got {actual_sha}"
        )
    sys.path.insert(0, str(pact_root / "src"))
    sys.path.insert(0, str(pact_root / "upstream"))
    import torch
    from tac.analysis.score_exact_saliency import decode_real_pair_indices

    pair = decode_real_pair_indices(video, [PAIR_INDEX], device="cpu")
    # pair: (1, 2, 3, H, W) float32 camera-resolution tensor (canonical decode)
    frames_chw = pair[0]  # (2, 3, 874, 1164) float32
    camera_hw = [int(frames_chw.shape[-2]), int(frames_chw.shape[-1])]
    resized_t = torch.nn.functional.interpolate(
        frames_chw, size=SCORER_HW, mode="bilinear"
    )
    stack = (
        np.clip(np.rint(resized_t.permute(0, 2, 3, 1).numpy()), 0, 255).astype(np.uint8)
    )  # (2, 384, 512, 3)
    provenance = {
        "challenge_video": {
            "public_label": "upstream/videos/0.mkv",
            "bytes": video.stat().st_size,
            "sha256": actual_sha,
        },
        "decode_path": "upstream/frame_utils.yuv420_to_rgb via tac.analysis.score_exact_saliency.decode_real_pair_indices",
        "resize": "torch.nn.functional.interpolate mode=bilinear to (384, 512), matching the frozen scorer preprocess (upstream/modules.py), then round-clip to uint8",
        "camera_resolution_hw": camera_hw,
    }
    return stack, provenance


def _load_evidence_argmax(evidence_npz: Path) -> tuple[np.ndarray, str]:
    d = np.load(evidence_npz)
    return np.asarray(d["argmax_u8"], np.uint8), sha256_file(evidence_npz)


def _decode_witness_argmax(
    pact_root: Path, run_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    """Decode the witness at each stage checkpoint on the locked pair.

    Returns, per checkpoint:
    - the REALIZED partition (what the score reads): the witness's rendered RGB
      pushed through the contest-exact roundtrip R (bicubic up to camera,
      uint8, scorer bilinear down) and the frozen CPU-torch SegNet argmax;
    - the INTERNAL level-set argmax (argmax over the witness's own phi
      channels) — the representation chart, which the score never reads;
    - the witness's rendered RGB frame (render grid, uint8) for display.
    """
    import torch
    from tac.local_acceleration import torch_levelset_inflate as TLI
    from tac.boundary_math.seg_core import load_real_segnet

    sys.path.insert(0, str(pact_root / "experiments"))
    import train_witness_realized_through_R_mlx as base

    showcase_path = pact_root / "tools" / "build_witness_showcase.py"
    spec = importlib.util.spec_from_file_location("wm_showcase", showcase_path)
    showcase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(showcase)

    seg_cpu = load_real_segnet("cpu")
    realized_stack = []
    internal_stack = []
    rgb_stack = []
    rows = []
    render_hw = None
    for stage, name, epoch in _discover_stage_checkpoints(run_dir):
        ckpt = run_dir / name
        m, params, code = showcase._load_witness(ckpt)
        rh, rw = m["render_h"], m["render_w"]
        if render_hw is None:
            render_hw = (rh, rw)
        elif render_hw != (rh, rw):
            raise SystemExit("stage checkpoints disagree on render resolution")
        coords = TLI.coords_grid(rh, rw)
        bank = TLI.curvelet_B(
            m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"],
            m["bank_base"], m["bank_n_iso"], m["max_bank_freq"],
        )
        curv = TLI.curvelet_feats(coords, bank)
        td = torch.float32
        params_t = {k: torch.as_tensor(v, dtype=td) for k, v in params.items()}
        code_t = torch.as_tensor(code, dtype=td)
        phi, rgb = showcase._decode_frame_phi(
            TLI, m, params_t, code_t, curv, coords, PAIR_INDEX, td, torch
        )
        camera_u8 = base._torch_R_to_camera_uint8(rgb.astype(np.float32))
        _d_seg_list, realized = base.cpu_verdict_d_seg_argmax_batch(
            seg_cpu, [camera_u8], [np.zeros((rh, rw), np.int64)]
        )
        realized_stack.append(realized[0].astype(np.uint8))
        internal_stack.append(phi.argmax(-1).astype(np.uint8))
        rgb_stack.append(rgb)
        rows.append(
            {
                "stage": stage,
                "checkpoint_file": name,
                "checkpoint_sha256": sha256_file(ckpt),
                "epoch": epoch,
                "ema_epoch_recorded_in_checkpoint": m["epoch"],
            }
        )
        print(f"decoded {name} (stage {stage}, ep {epoch})")
    run_meta = {
        "run_dir_public_label": str(run_dir.relative_to(pact_root)),
        "render_hw": list(render_hw),
        "n_pairs": 600,
        "gt_cache": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        "realized_partition_path": (
            "TLI torch fp32 witness render -> _torch_R_to_camera_uint8 (bicubic up, "
            "uint8 at camera) -> frozen CPU-torch SegNet preprocess_input (contest "
            "bilinear down) -> argmax"
        ),
    }
    return (
        np.stack(realized_stack, axis=0),
        np.stack(internal_stack, axis=0),
        np.stack(rgb_stack, axis=0),
        rows,
        run_meta,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pact-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts" / "v12_public"
    )
    args = parser.parse_args(argv)
    pact_root = args.pact_root.resolve()
    run_dir = (pact_root / args.run_dir).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_npz = out_dir / "v12_real_frozen_scorer_evidence.npz"
    evidence_argmax, evidence_sha = _load_evidence_argmax(evidence_npz)

    # --- artifact 1: co-registered real frames -------------------------------
    frames, frame_provenance = _decode_scorer_frames(pact_root)
    frame_metadata = {
        "schema": FRAME_SCHEMA,
        "claim_class": "DISPLAY",
        "evidence_axis": "[public challenge video, deterministic decode]",
        "artifact_role": (
            "co-registered photographic substrate for the locked real pair; "
            "display only, carries no measurement"
        ),
        "pair_selection": {
            "pair_index_zero_based": PAIR_INDEX,
            "source_frame_indices_zero_based": list(SOURCE_FRAME_INDICES),
            "segnet_scored_source_frame_index_zero_based": SOURCE_FRAME_INDICES[1],
        },
        "co_registration": {
            "target_grid_hw": list(SCORER_HW),
            "co_registered_with": {
                "artifact_file": "v12_real_frozen_scorer_evidence.npz",
                "artifact_sha256": evidence_sha,
            },
        },
        "provenance": frame_provenance,
        "limitations": [
            "Two frames of one fixed pair only; this is a display substrate, not a dataset.",
            "The uint8 round after bilinear resize is a display quantization; measured fields come from the sealed evidence artifact, not from these pixels.",
            "No score, improvement, ranking, promotion, or exact-eval claim is made.",
        ],
        "score_claim": False,
        "new_authority": False,
    }
    _write_locked(
        out_dir / "v12_real_frame_display.npz",
        {"frames_rgb_u8": frames},
        frame_metadata,
    )

    # --- artifact 2: boundary evolution across training ----------------------
    realized_argmax, internal_argmax, witness_rgb, ckpt_rows, run_meta = (
        _decode_witness_argmax(pact_root, run_dir)
    )
    if realized_argmax.shape[1:] != evidence_argmax.shape:
        raise SystemExit(
            f"witness realized grid {realized_argmax.shape[1:]} does not match "
            f"the sealed evidence grid {evidence_argmax.shape}"
        )
    for row, realized, internal in zip(ckpt_rows, realized_argmax, internal_argmax):
        row["pair196_realized_disagreement_vs_sealed_argmax"] = float(
            (realized != evidence_argmax).mean()
        )
        row["pair196_internal_chart_disagreement_vs_sealed_argmax"] = float(
            (internal != evidence_argmax).mean()
        )
    train_result = json.loads(
        (run_dir / "levelset_train_result.json").read_text(encoding="utf-8")
    )
    all_history = [
        row for row in train_result.get("history", []) if "epoch" in row
    ]
    wanted_epochs = {0} | {row["epoch"] for row in ckpt_rows}
    history = []
    for target in sorted(wanted_epochs):
        nearest = min(all_history, key=lambda row: abs(row["epoch"] - target))
        if nearest not in history:
            history.append(nearest)
    best = json.loads((run_dir / "levelset_best.json").read_text(encoding="utf-8"))
    evolution_metadata = {
        "schema": EVOLUTION_SCHEMA,
        "claim_class": "EMPIRICAL",
        "evidence_axis": "[macOS-MLX training-gradient]/[macOS-CPU advisory]",
        "artifact_role": (
            "level-set witness partition evolution on the locked real pair, "
            "decoded from the preserved per-stage EMA checkpoints of one real "
            "n600 training run, against the sealed frozen-scorer partition; "
            "realized = SegNet reading the witness render through the contest "
            "roundtrip R; internal = the witness's own phi argmax chart"
        ),
        "pair_selection": {
            "pair_index_zero_based": PAIR_INDEX,
            "segnet_scored_source_frame_index_zero_based": SOURCE_FRAME_INDICES[1],
        },
        "training_run": run_meta,
        "checkpoints": ckpt_rows,
        "run_verdict_history_excerpt": {
            "note": (
                "full-run n600 d_seg verdicts from the run's own telemetry at the "
                "nearest recorded epochs; advisory axis, promotion_eligible=false"
            ),
            "rows": history,
            "best": best,
        },
        "sealed_reference": {
            "artifact_file": "v12_real_frozen_scorer_evidence.npz",
            "artifact_sha256": evidence_sha,
            "field": "argmax_u8",
        },
        "limitations": [
            "One fixed real pair only; per-checkpoint disagreement here is a single-pair fraction, not the n600 d_seg.",
            "The run's d_seg history rows are its own advisory-axis telemetry ([macOS-MLX training-gradient]/[macOS-CPU advisory]); they are not contest-CPU or contest-CUDA results.",
            "The realized partitions here are decoded through the torch fp32 witness forward, not the numpy fp32 byte-close reference (torch/numpy fp32 agree to ~1 ULP, 0-3 px per frame on this vehicle).",
            "The internal phi-argmax chart is the representation, not the scored object; the score-domain-trained witness constrains only the realized partition.",
            "No score, improvement, ranking, promotion, or exact-eval claim is made; the pointer is unmoved by this artifact.",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "new_authority": False,
    }
    _write_locked(
        out_dir / "v12_boundary_evolution.npz",
        {
            "realized_argmax_u8": realized_argmax,
            "internal_argmax_u8": internal_argmax,
            "witness_rgb_u8": witness_rgb,
            "sealed_argmax_u8": evidence_argmax,
        },
        evolution_metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
