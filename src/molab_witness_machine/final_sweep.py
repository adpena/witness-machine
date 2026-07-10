"""v1.1 final cathedral sweep utilities.

This module is intentionally small.  It does not train, score, submit, push, or
promote claims.  It checks that the final handoff has the right structural
surfaces and that the Codex SOL prompt remains compact enough to paste directly
into an agent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

FINAL_SWEEP_VERSION = "v1.1-final-ultimate"
CODEX_PROMPT_LIMIT = 4000

REQUIRED_V11_FILES: tuple[str, ...] = (
    "docs/V11_FINAL_CATHEDRAL_SWEEP.md",
    "docs/THOUSAND_YEAR_ENDSTATE.md",
    "docs/RECURSIVE_ADVERSARIAL_REVIEW_FINAL.md",
    "docs/FINAL_EXIT_CRITERIA_V11.md",
    "docs/LATEST_PUBLIC_SYNC_V11_20260709.md",
    "docs/ELEGANCE_AND_ABSTRACTION_GUIDE_V11.md",
    "docs/WORK_SPEAKS_FOR_ITSELF_POLISH_PROTOCOL_V11.md",
    "docs/V11_NEXT_OPERATOR_RUNBOOK.md",
    "reports/live_research_notes.md",
    "reports/RECURSIVE_ADVERSARIAL_REVIEW_V11.json",
    "reports/RECURSIVE_ADVERSARIAL_REVIEW_V11.md",
    "reports/BROWSER_E2E_V11.md",
    "artifacts/LIVE_MENTOR_PACKET.md",
    "artifacts/release/arxiv_source_bundle_v08.zip",
    "artifacts/release/promised_land_bundle_v10.zip",
    "reports/figure_lock_v06.json",
    "reports/figure_lock_v08_generator_native.json",
    "paper/sections/generated_claims_table.tex",
    "paper/sections/generated_score_cards_table.tex",
    "configs/final_endstate_gates_v11.yaml",
    "configs/beauty_rigor_principles_v11.yaml",
    "configs/recursive_adversarial_review_v11.yaml",
    "configs/final_exit_criteria_v11.yaml",
    "prompts/CODEX_GPT56_SOL_GOAL_PROMPT.md",
    "prompts/recursive_adversarial_review_v11.md",
    "review/FINAL_REVIEW_BOARD_V11.md",
    "scripts/build_final_endstate.py",
    "scripts/check_v11_final.py",
    "scripts/run_recursive_adversarial_review_v11.py",
    "src/molab_witness_machine/final_sweep.py",
)

KEY_PRIOR_REPORTS: tuple[str, ...] = (
    "reports/PROMISED_LAND_V10.json",
    "reports/CAMERA_READY_V09.json",
    "reports/ARTIFACT_LOCK_V08.json",
    "reports/RELEASE_CANDIDATE_V07.json",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


@dataclass(frozen=True)
class V11FileStatus:
    relpath: str
    exists: bool
    size_bytes: int | None = None
    sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinalEndstateReport:
    version: str
    generated_at: str
    root: str
    required_files: tuple[V11FileStatus, ...]
    prior_reports: tuple[V11FileStatus, ...]
    codex_prompt_chars: int
    codex_prompt_ok: bool
    exact_score_cards: int
    pact_parity_evidence: int
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "root": self.root,
            "required_files": [f.as_dict() for f in self.required_files],
            "prior_reports": [f.as_dict() for f in self.prior_reports],
            "codex_prompt_chars": self.codex_prompt_chars,
            "codex_prompt_ok": self.codex_prompt_ok,
            "exact_score_cards": self.exact_score_cards,
            "pact_parity_evidence": self.pact_parity_evidence,
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "notes": list(self.notes),
            "ok": self.ok,
        }

    def write(self, json_path: Path, md_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(render_markdown(self), encoding="utf-8")


def _status(root: Path, rel: str) -> V11FileStatus:
    p = root / rel
    return V11FileStatus(rel, p.exists(), p.stat().st_size if p.exists() and p.is_file() else None, sha256_file(p))


def _count_exact_cards(root: Path) -> int:
    count = 0
    for p in (root / "reports").glob("**/*score*card*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        axis = str(data.get("evidence_axis") or data.get("device_axis") or data.get("axis") or "").lower()
        status = str(data.get("status") or data.get("evidence_status") or "").lower()
        if "exact" in axis or "exact" in status:
            count += 1
    return count


def _count_parity(root: Path) -> int:
    hints = ["parity", "candidate_outputs", "check_parity", "wasm"]
    count = 0
    for p in (root / "reports").glob("**/*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(h in name for h in hints):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")[:2000].lower()
            except Exception:
                txt = ""
            if "pass" in txt or p.suffix == ".npz":
                count += 1
    return count


def build_final_endstate_report(root: str | Path) -> FinalEndstateReport:
    root = Path(root).resolve()
    req = tuple(_status(root, rel) for rel in REQUIRED_V11_FILES)
    prior = tuple(_status(root, rel) for rel in KEY_PRIOR_REPORTS)
    blockers: list[str] = []
    warnings: list[str] = []
    missing = [f.relpath for f in req if not f.exists]
    if missing:
        blockers.extend(f"missing required v11 file: {m}" for m in missing)
    missing_prior = [f.relpath for f in prior if not f.exists]
    if missing_prior:
        warnings.extend(f"missing prior report: {m}" for m in missing_prior)
    prompt_path = root / "prompts/CODEX_GPT56_SOL_GOAL_PROMPT.md"
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
    prompt_chars = len(prompt_text)
    prompt_ok = 0 < prompt_chars < CODEX_PROMPT_LIMIT
    if not prompt_ok:
        blockers.append(f"Codex SOL prompt length must be 1..{CODEX_PROMPT_LIMIT-1} chars; got {prompt_chars}")
    exact_cards = _count_exact_cards(root)
    parity = _count_parity(root)
    notes = [
        "V11 is a structural final starter pack, not a promoted score claim.",
        "Exact score and Molt/Pact parity gates are expected to close on the operator workstation.",
        "Hard stops are operator GO for external side effects and credentials/login requirements.",
    ]
    return FinalEndstateReport(
        version=FINAL_SWEEP_VERSION,
        generated_at=utc_now_iso(),
        root=str(root),
        required_files=req,
        prior_reports=prior,
        codex_prompt_chars=prompt_chars,
        codex_prompt_ok=prompt_ok,
        exact_score_cards=exact_cards,
        pact_parity_evidence=parity,
        warnings=tuple(warnings),
        blockers=tuple(blockers),
        notes=tuple(notes),
    )


def render_markdown(report: FinalEndstateReport) -> str:
    lines = [
        "# Final endstate report v11",
        "",
        f"Generated: `{report.generated_at}`",
        f"Root: `{report.root}`",
        f"Version: `{report.version}`",
        "",
        "## Status",
        "",
        f"- OK: `{report.ok}`",
        f"- Codex prompt chars: `{report.codex_prompt_chars}` / `{CODEX_PROMPT_LIMIT}`",
        f"- Exact score cards detected: `{report.exact_score_cards}`",
        f"- Pact/Molt parity-like evidence detected: `{report.pact_parity_evidence}`",
        "",
        "## Required v11 files",
        "",
    ]
    for f in report.required_files:
        mark = "✅" if f.exists else "❌"
        lines.append(f"- {mark} `{f.relpath}`")
    if report.warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in report.warnings]
    if report.blockers:
        lines += ["", "## Blockers", ""] + [f"- {b}" for b in report.blockers]
    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    return "\n".join(lines) + "\n"
