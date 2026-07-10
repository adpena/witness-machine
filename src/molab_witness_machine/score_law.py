"""Score law utilities for the comma.ai lossy video compression challenge.

The official evaluator computes:
    S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/original_bytes
with original_bytes = 37_545_489 for the public 0.mkv archive family.

These helpers are intentionally tiny and dependency-free so the marimo notebook
can use them even before the upstream challenge repository is present locally.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Mapping

ORIGINAL_BYTES = 37_545_489
SEG_WEIGHT = 100.0
POSE_INNER_WEIGHT = 10.0
RATE_WEIGHT = 25.0

# Back-compat aliases, useful if a previous notebook imported these names.
ORIGINAL_SIZE_BYTES = ORIGINAL_BYTES


@dataclass(frozen=True)
class ScoreComponents:
    d_seg: float
    d_pose: float
    archive_bytes: int
    original_bytes: int = ORIGINAL_BYTES

    @property
    def seg_term(self) -> float:
        return SEG_WEIGHT * float(self.d_seg)

    @property
    def pose_term(self) -> float:
        return math.sqrt(max(0.0, POSE_INNER_WEIGHT * float(self.d_pose)))

    @property
    def rate(self) -> float:
        return float(self.archive_bytes) / float(self.original_bytes)

    @property
    def rate_term(self) -> float:
        return RATE_WEIGHT * self.rate

    @property
    def score(self) -> float:
        return self.seg_term + self.pose_term + self.rate_term

    def to_dict(self) -> dict[str, float | int]:
        d = asdict(self)
        d.update(
            seg_term=self.seg_term,
            pose_term=self.pose_term,
            rate=self.rate,
            rate_term=self.rate_term,
            score=self.score,
        )
        return d

    def as_dict(self) -> dict[str, float | int]:
        return self.to_dict()


# Back-compatible class alias.
ScoreBreakdown = ScoreComponents


def score(d_seg: float, d_pose: float, archive_bytes: int, original_bytes: int = ORIGINAL_BYTES) -> float:
    return ScoreComponents(d_seg, d_pose, archive_bytes, original_bytes).score


def analytic_costates(d_pose: float | None, original_bytes: int = ORIGINAL_BYTES) -> dict[str, float | None]:
    """Exact local shadow prices of the score law.

    λ_seg is constant; λ_bytes is constant; λ_pose depends on the operating point
    and diverges as d_pose approaches zero.
    """
    lam_pose = None
    if d_pose is not None and d_pose > 0:
        lam_pose = 5.0 / math.sqrt(10.0 * float(d_pose))
    return {
        "lambda_d_seg": SEG_WEIGHT,
        "lambda_d_pose": lam_pose,
        "lambda_bytes": RATE_WEIGHT / float(original_bytes),
    }


def costates(d_pose: float, original_bytes: int = ORIGINAL_BYTES) -> dict[str, float | None]:
    return analytic_costates(d_pose, original_bytes)


def dseg_budget_for_score(target_score: float, d_pose: float, archive_bytes: int, original_bytes: int = ORIGINAL_BYTES) -> float:
    """Solve the score law for the maximum d_seg allowed at a target score."""
    pose_term = math.sqrt(max(0.0, POSE_INNER_WEIGHT * float(d_pose)))
    rate_term = RATE_WEIGHT * float(archive_bytes) / float(original_bytes)
    return (float(target_score) - pose_term - rate_term) / SEG_WEIGHT


def bytes_for_rate_term(rate_term: float, original_bytes: int = ORIGINAL_BYTES) -> int:
    return int(round(float(rate_term) * float(original_bytes) / RATE_WEIGHT))


def bytes_for_score_budget(target_score: float, d_seg: float, d_pose: float, original_bytes: int = ORIGINAL_BYTES) -> int:
    residue = float(target_score) - SEG_WEIGHT * float(d_seg) - math.sqrt(max(0.0, POSE_INNER_WEIGHT * float(d_pose)))
    return int((residue / RATE_WEIGHT) * original_bytes)


def summarize_operating_point(row: Mapping[str, float | int]) -> ScoreComponents:
    """Create a ScoreComponents row from common key names."""
    d_seg = float(row.get("d_seg", row.get("segnet_distortion", 0.0)))
    d_pose = float(row.get("d_pose", row.get("posenet_distortion", 0.0)))
    if "archive_bytes" in row:
        archive_bytes = int(row["archive_bytes"])
    elif "bytes" in row:
        archive_bytes = int(row["bytes"])
    elif "compressed_size" in row:
        archive_bytes = int(row["compressed_size"])
    else:
        archive_bytes = 0
    return ScoreComponents(d_seg=d_seg, d_pose=d_pose, archive_bytes=archive_bytes)


def table_rows(breakdowns: Mapping[str, ScoreComponents]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for name, b in breakdowns.items():
        row = {"name": name}
        row.update(b.to_dict())
        rows.append(row)
    return rows
