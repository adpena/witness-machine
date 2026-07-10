"""MoLab Witness Machine starter utilities.

This package is intentionally small and transparent. It runs a toy task-space
witness in pure Python/Numpy, and it exposes adapters that can attach to a local
clone of the comma.ai compression challenge and adpena/comma-lab.
"""

__all__ = [
    "score_law",
    "sdf_geometry",
    "info_viz",
    "costate",
    "upstream_adapter",
    "run_replay",
    "v8_edge_decomp",
    "cathedral",
    "authority_wiring",
    "dsl_authority",
    "v8_probe",
    "mentor_packet_live",
    "exact_eval_card",
    "figure_lock",
    "release_candidate",
    "artifact_lock",
    "molt_pact",
    "camera_ready",
    "promised_land",
]

__version__ = "1.0.0"
EXACT_EVAL_CARD_SCHEMA = "exact-eval-card/v0.6"
FIGURE_LOCK_SCHEMA = "figure-lock/v0.6"
RELEASE_CANDIDATE_SCHEMA = "release-candidate/v0.7"

ARTIFACT_LOCK_SCHEMA = "artifact-lock/v0.8"
ARXIV_SOURCE_BUNDLE_SCHEMA = "arxiv-source-bundle/v0.8"

MOLT_PACT_SYNC_SCHEMA = "molt-pact-sync/v0.9"
CAMERA_READY_SCHEMA = "camera-ready/v0.9"
PROMISED_LAND_SCHEMA = "promised-land/v1.0"

# v1.0 promised-land workstation lock module is intentionally imported lazily.

try:
    from .final_sweep import FINAL_SWEEP_VERSION
except Exception:  # pragma: no cover
    FINAL_SWEEP_VERSION = "v1.1-final-ultimate"
