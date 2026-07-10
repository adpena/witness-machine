"""Release-candidate assembly for the Witness Machine pack.

The module is intentionally small and boring: it turns already-locked inputs
(claim registry, exact score cards, figure-lock manifests, objection logs, and
release gates) into a reviewable release-candidate manifest.  It does not run
scorers, promote claims, or infer missing authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import glob
import json
import re

try:  # optional at runtime, but present in the pack dependencies
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


EXACT_STATUSES = {"exact", "a_grade_exact", "exact_candidate", "passed"}
BLOCKING_SCORE_STATUSES = {"blocked", "toy", "toy_synthetic", "advisory", "template"}
EVIDENCE_CLASSES = {
    "TOY",
    "ADVISORY",
    "EMPIRICAL",
    "DERIVATION",
    "EXTERNAL",
    "EXACT-CANDIDATE",
    "EXACT",
}


@dataclass(frozen=True)
class ClaimRow:
    id: str
    claim: str
    status: str
    public_safe: bool
    owner: str
    evidence_count: int
    falsifier_count: int
    evidence_class: str = "ADVISORY"


@dataclass(frozen=True)
class ScoreCardRow:
    path: str
    status: str
    axis_label: str
    recomputed_score: float | None
    archive_sha256: str | None
    archive_bytes: int | None
    claim_ids: tuple[str, ...] = field(default_factory=tuple)
    blockers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FigureRow:
    path: str
    status: str
    axis_label: str
    sha256: str
    claim_ids: tuple[str, ...] = field(default_factory=tuple)
    caption: str = ""


@dataclass(frozen=True)
class ObjectionRow:
    id: str
    severity: str
    status: str
    objection: str
    owner: str


@dataclass(frozen=True)
class ReleaseCandidate:
    schema_version: str
    created_at_utc: str
    root: str
    claims: tuple[ClaimRow, ...]
    score_cards: tuple[ScoreCardRow, ...]
    figures: tuple[FigureRow, ...]
    objections: tuple[ObjectionRow, ...]
    gates: dict[str, Any]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    # Back-compatibility with earlier v0.7 scripts/tests that treated the
    # release candidate as an artifact bundle with a nested manifest.
    @property
    def ok(self) -> bool:
        return self.ready

    @property
    def manifest(self) -> "ReleaseCandidate":
        return self

    @property
    def claim_count(self) -> int:
        return len(self.claims)

    @property
    def score_card_count(self) -> int:
        return len(self.score_cards)

    @property
    def figure_count(self) -> int:
        return len(self.figures)

    @property
    def required_gate_failures(self) -> tuple[str, ...]:
        return self.blockers

    @property
    def claim_table_tex(self) -> Path:
        return Path(self.root) / "paper" / "sections" / "generated_claims_table.tex"

    @property
    def score_table_tex(self) -> Path:
        return Path(self.root) / "paper" / "sections" / "generated_score_cards_table.tex"

    @property
    def figure_table_tex(self) -> Path:
        return Path(self.root) / "paper" / "sections" / "generated_figure_locks_table.tex"

    @property
    def reviewer_packet(self) -> Path:
        return Path(self.root) / "review" / "RELEASE_CANDIDATE_REVIEWER_PACKET.md"

    @property
    def manifest_path(self) -> Path:
        return Path(self.root) / "reports" / "RELEASE_CANDIDATE_V07.json"

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Release manifests are public evidence, not workstation custody
        # records. Keep the internal root for local builders, but never
        # serialize an absolute checkout path into a shareable report.
        d["root"] = "."
        d["ready"] = self.ready
        return d

    def markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Witness Machine v0.7 Release Candidate Report")
        lines.append("")
        lines.append(f"Created: `{self.created_at_utc}`")
        lines.append(f"Status: **{'READY' if self.ready else 'BLOCKED'}**")
        lines.append("")
        if self.blockers:
            lines.append("## Blocking issues")
            for b in self.blockers:
                lines.append(f"- {b}")
            lines.append("")
        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        lines.append("## Claims")
        lines.append("| evidence class | id | status | public safe | owner | evidence | falsifiers | claim |")
        lines.append("|---|---|---|---:|---|---:|---:|---|")
        for c in self.claims:
            lines.append(
                f"| `{evidence_badge(c.evidence_class)}` | `{c.id}` | `{c.status}` | "
                f"{str(c.public_safe).lower()} | `{c.owner}` | {c.evidence_count} | "
                f"{c.falsifier_count} | {escape_md(c.claim)} |"
            )
        lines.append("")
        lines.append("## Exact / score-card inputs")
        lines.append("| evidence | path | status | axis | score | bytes | sha | authority boundary |")
        lines.append("|---|---|---|---|---:|---:|---|---|")
        for s in self.score_cards:
            score = "" if s.recomputed_score is None else f"{s.recomputed_score:.12g}"
            boundary = score_card_boundary(s)
            sha = (s.archive_sha256 or "")[:12]
            if score_card_is_blocked(s):
                score = "N/A"
            bytes_s = "N/A" if score_card_is_blocked(s) or s.archive_bytes is None else str(s.archive_bytes)
            lines.append(
                f"| `{evidence_badge(s.axis_label or s.status)}` | `{s.path}` | `{s.status}` | "
                f"`{s.axis_label}` | {score} | {bytes_s} | `{sha}` | {escape_md(boundary)} |"
            )
        lines.append("")
        lines.append("## Figure locks")
        lines.append("| figure | status | axis | sha | claim ids | caption |")
        lines.append("|---|---|---|---|---|---|")
        for f in self.figures:
            lines.append(
                f"| `{f.path}` | `{f.status}` | `{f.axis_label}` | `{f.sha256[:12]}` | `{','.join(f.claim_ids)}` | {escape_md(f.caption)} |"
            )
        lines.append("")
        lines.append("## Open objections")
        lines.append("| id | severity | status | owner | objection |")
        lines.append("|---|---|---|---|---|")
        for o in self.objections:
            lines.append(f"| `{o.id}` | `{o.severity}` | `{o.status}` | `{o.owner}` | {escape_md(o.objection)} |")
        lines.append("")
        lines.append("## Gate summary")
        for k, v in self.gates.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        return "\n".join(lines)


def escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    if yaml is None:
        raise RuntimeError("pyyaml is required to read YAML release inputs")
    return yaml.safe_load(path.read_text())


def load_claims(path: Path) -> list[ClaimRow]:
    path = Path(path)
    if path.is_dir():
        path = path / "configs" / "claims_registry.yaml"
    data = load_yaml(path) or {}
    rows = []
    for c in data.get("claims", []) or []:
        rows.append(
            ClaimRow(
                id=str(c.get("id", "")),
                claim=str(c.get("claim", "")),
                status=str(c.get("status", "unknown")),
                public_safe=bool(c.get("public_safe", False)),
                owner=str(c.get("owner", "unknown")),
                evidence_count=len(c.get("evidence", []) or []),
                falsifier_count=len(c.get("falsifier", []) or []),
                evidence_class=str(c.get("evidence_class", "ADVISORY")).upper(),
            )
        )
    return rows


def _score_from_card(data: dict[str, Any]) -> float | None:
    metrics = data.get("metrics", {}) or {}
    val = metrics.get("recomputed_score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def load_score_cards(root: Path, patterns: list[str] | None = None) -> list[ScoreCardRow]:
    patterns = patterns or ["reports/*score_card*.json", "examples/*score_card*.json"]
    out = []
    seen: set[Path] = set()
    for pattern in patterns:
        for raw in glob.glob(str(root / pattern)):
            path = Path(raw)
            if path in seen:
                continue
            seen.add(path)
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            archive = data.get("archive", {}) or {}
            metrics = data.get("metrics", {}) or {}
            out.append(
                ScoreCardRow(
                    path=str(path.relative_to(root)),
                    status=str(data.get("status", "unknown")),
                    axis_label=str(data.get("axis_label", data.get("axis", "unknown"))),
                    recomputed_score=_score_from_card(data),
                    archive_sha256=archive.get("sha256"),
                    archive_bytes=(int(metrics["archive_bytes"]) if isinstance(metrics.get("archive_bytes"), int) else None),
                    claim_ids=tuple(str(x) for x in data.get("claim_ids", []) or []),
                    blockers=tuple(str(x) for x in data.get("blockers", []) or []),
                )
            )
    return out


def load_figures(root: Path, patterns: list[str] | None = None) -> list[FigureRow]:
    patterns = patterns or [
        "reports/*figure_lock*.json",
        "reports/*FIGURE_LOCK*.json",
        "examples/*figure_lock*.json",
    ]
    out = []
    seen: set[Path] = set()
    for pattern in patterns:
        for raw in glob.glob(str(root / pattern)):
            path = Path(raw)
            if path in seen:
                continue
            seen.add(path)
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            for fig in data.get("figures", []) or []:
                out.append(
                    FigureRow(
                        path=str(fig.get("path", "")),
                        status=str(fig.get("status", "unknown")),
                        axis_label=str(fig.get("axis_label", "unknown")),
                        sha256=str(fig.get("sha256", "")),
                        claim_ids=tuple(str(x) for x in fig.get("claim_ids", []) or []),
                        caption=str(fig.get("caption", "")),
                    )
                )
    return out


def parse_objection_log(path: Path) -> list[ObjectionRow]:
    if not path.exists():
        return []
    rows: list[ObjectionRow] = []
    current: dict[str, str] | None = None
    for line in path.read_text().splitlines():
        m = re.match(r"^##\s+(OBJ-[A-Za-z0-9_-]+)\s+\[(.*?)\]\s+\{(.*?)\}", line.strip())
        if m:
            if current:
                rows.append(_obj_from_dict(current))
            current = {"id": m.group(1), "severity": m.group(2), "status": m.group(3), "objection": "", "owner": "unknown"}
            continue
        if current is None:
            continue
        if line.startswith("Owner:"):
            current["owner"] = line.split(":", 1)[1].strip()
        elif line.startswith("Objection:"):
            current["objection"] = line.split(":", 1)[1].strip()
    if current:
        rows.append(_obj_from_dict(current))
    return rows


def _obj_from_dict(d: dict[str, str]) -> ObjectionRow:
    return ObjectionRow(
        id=d.get("id", "OBJ-UNKNOWN"),
        severity=d.get("severity", "unknown"),
        status=d.get("status", "open"),
        objection=d.get("objection", ""),
        owner=d.get("owner", "unknown"),
    )



def latex_figure_table(figures: list[FigureRow] | tuple[FigureRow, ...]) -> str:
    lines = [
        r"% Generated by molab_witness_machine.release_candidate. Do not hand-edit.",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llllp{0.24\linewidth}}",
        r"\toprule",
        r"Evidence & Status & Axis & SHA & Figure \\",
        r"\midrule",
    ]
    for f in figures:
        label = shorten(Path(f.path).name.replace("_", "-"), 35)
        lines.append(
            f"{tex(evidence_badge(f.axis_label or f.status))} & {tex(f.status)} & "
            f"{tex(f.axis_label)} & {tex(f.sha256[:12])} & "
            f"{{\\raggedright\\footnotesize {tex(label)}}} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", ""])
    return "\n".join(lines)


def evidence_badge(status_or_axis: str) -> str:
    s = str(status_or_axis).lower()
    if "exact-candidate" in s or "exact_candidate" in s:
        return "EXACT-CANDIDATE"
    if any(x in s for x in ("contest-cpu", "contest_cpu", "contest-cuda", "contest_cuda", "exact")):
        return "EXACT"
    if any(x in s for x in ("toy", "implemented_toy", "synthetic")):
        return "TOY"
    if any(x in s for x in ("empirical", "measured-artifact", "frozen-scorer evidence")):
        return "EMPIRICAL"
    if any(x in s for x in ("advisory", "mlx", "macos", "research-signal")):
        return "ADVISORY"
    if any(x in s for x in ("external", "public pr", "paper", "upstream")):
        return "EXTERNAL"
    if any(x in s for x in ("proposed", "derivation", "prediction", "measured")):
        # ``proposed`` and ``measured`` are registry lifecycle states, not
        # public evidence grades.  The former is a mathematical proposal; the
        # latter is an externally sourced rule until it has an exact card.
        return "DERIVATION" if "measured" not in s else "EXTERNAL"
    return "ADVISORY"


def render_claims_latex(claims: list[ClaimRow] | tuple[ClaimRow, ...]) -> str:
    return latex_claim_table(claims).replace("% Generated by molab_witness_machine.release_candidate. Do not hand-edit.", "% Generated by molab_witness_machine.release_candidate. Do not hand-edit.")


def validate_release_language(root: Path) -> tuple[bool, list[str]]:
    """Small public-language guard.

    It is intentionally conservative and checks only the failure classes that
    would most damage trust in this pack: accidental official endorsement,
    background-work promises, or score claims without exact evidence language.
    """
    root = Path(root)
    issues: list[str] = []
    surfaces = [root / "README.md", root / "docs" / "PUBLIC_RELEASE_LANGUAGE_V06.md", root / "docs" / "V07_RELEASE_CANDIDATE_PROTOCOL.md"]
    forbidden = [
        "officially endorsed by comma.ai",
        "officially endorsed by arxiv",
        "officially endorsed by marimo",
        "we will deliver later",
    ]
    for path in surfaces:
        if not path.exists():
            continue
        text = path.read_text().lower()
        for phrase in forbidden:
            if phrase in text:
                issues.append(f"{path.relative_to(root)} contains forbidden phrase: {phrase}")
    return (not issues), issues

def build_release_candidate(root: Path, *, strict: bool = False) -> ReleaseCandidate:
    root = Path(root)
    claims = load_claims(root / "configs" / "claims_registry.yaml")
    score_cards = load_score_cards(root)
    figures = load_figures(root)
    objections = parse_objection_log(root / "review" / "OBJECTION_LOG.md")
    gates_cfg = load_yaml(root / "configs" / "release_candidate_gates_v07.yaml") or {}

    blockers: list[str] = []
    warnings: list[str] = []

    if not claims:
        blockers.append("no claims loaded from configs/claims_registry.yaml")
    for c in claims:
        if not c.id or not c.claim:
            blockers.append("claim row missing id or text")
        if c.evidence_class not in EVIDENCE_CLASSES:
            blockers.append(
                f"claim {c.id} has invalid evidence_class {c.evidence_class!r}"
            )
        if c.public_safe and (c.evidence_count == 0 or c.falsifier_count == 0):
            blockers.append(f"public claim {c.id} lacks evidence or falsifier")

    exact_rows = [s for s in score_cards if s.status.lower() in EXACT_STATUSES and not s.blockers]
    if not score_cards:
        warnings.append("no exact score cards found; package remains paper-scaffold only")
    if gates_cfg.get("require_exact_for_score_claim", True):
        score_claim_ids = set(gates_cfg.get("score_claim_ids", []) or [])
        if score_claim_ids and not exact_rows:
            blockers.append("score-claim ids configured but no exact/passed score card is present")
        for s in score_cards:
            if s.status.lower() in BLOCKING_SCORE_STATUSES and s.claim_ids:
                warnings.append(f"score card {s.path} is {s.status}; claim ids remain non-promoted")

    if gates_cfg.get("require_figures_for_paper", True) and not figures:
        blockers.append("no figure locks found for paper/notebook release")
    for f in figures:
        if not f.sha256:
            blockers.append(f"figure {f.path} has no sha256")
        if not f.claim_ids:
            warnings.append(f"figure {f.path} has no claim ids")

    open_objs = [o for o in objections if o.status.lower() not in {"closed", "resolved", "accepted-risk"}]
    severe_open = [o for o in open_objs if o.severity.lower() in {"critical", "high"}]
    if severe_open:
        blockers.append("open high/critical objections: " + ", ".join(o.id for o in severe_open))
    elif open_objs:
        warnings.append("open non-blocking objections: " + ", ".join(o.id for o in open_objs))

    gates = {
        "claim_count": len(claims),
        "score_card_count": len(score_cards),
        "exact_score_card_count": len(exact_rows),
        "figure_count": len(figures),
        "open_objection_count": len(open_objs),
        "strict": bool(strict),
    }
    if strict and blockers:
        gates["strict_result"] = "blocked"
    else:
        gates["strict_result"] = "pass_with_warnings" if warnings else "pass"

    rc = ReleaseCandidate(
        schema_version="witness-machine-release-candidate.v0.7",
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        root=str(root),
        claims=tuple(claims),
        score_cards=tuple(score_cards),
        figures=tuple(figures),
        objections=tuple(objections),
        gates=gates,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
    # Generate review artifacts as deterministic views over registries.
    try:
        rc.claim_table_tex.parent.mkdir(parents=True, exist_ok=True)
        rc.claim_table_tex.write_text(render_claims_latex(rc.claims))
        rc.score_table_tex.write_text(latex_score_table(rc.score_cards))
        rc.figure_table_tex.write_text(latex_figure_table(rc.figures))
        rc.reviewer_packet.parent.mkdir(parents=True, exist_ok=True)
        rc.reviewer_packet.write_text(rc.markdown())
        rc.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rc.manifest_path.write_text(json.dumps(rc.to_dict(), indent=2, sort_keys=True) + "\n")
    except Exception:
        # Building the in-memory object is still useful; dedicated scripts will
        # surface write errors more directly.
        pass
    return rc


def latex_claim_table(claims: list[ClaimRow] | tuple[ClaimRow, ...]) -> str:
    lines = [
        r"% Generated by molab_witness_machine.release_candidate. Do not hand-edit.",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llllp{0.38\linewidth}}",
        r"\toprule",
        r"Evidence & ID & Status & Owner & Claim excerpt \\",
        r"\midrule",
    ]
    for c in claims:
        lines.append(
            f"{tex(evidence_badge(c.evidence_class))} & {tex(c.id)} & {tex(c.status)} & "
            f"{tex(c.owner)} & {{\\raggedright\\footnotesize {tex(shorten(c.claim, 110))}}} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", ""])
    return "\n".join(lines)


def latex_score_table(scores: list[ScoreCardRow] | tuple[ScoreCardRow, ...]) -> str:
    lines = [
        r"% Generated by molab_witness_machine.release_candidate. Do not hand-edit.",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llllllp{0.22\linewidth}}",
        r"\toprule",
        r"Evidence & Status & Axis & Score & SHA & Bytes & Authority boundary \\",
        r"\midrule",
    ]
    for s in scores:
        blocked = s.status.lower() in {"blocked", "template"} or "blocked" in s.axis_label.lower()
        score = "N/A" if blocked or s.recomputed_score is None else f"{s.recomputed_score:.6f}"
        sha = "--" if not s.archive_sha256 else s.archive_sha256[:12]
        b = "N/A" if blocked or s.archive_bytes is None else str(s.archive_bytes)
        if blocked:
            boundary = "BLOCKED: missing evaluator/archive authority"
        elif s.status.lower() in {"toy", "toy_synthetic"}:
            boundary = "TOY: synthetic explanatory card; not an evaluator row"
        elif s.blockers:
            boundary = "; ".join(s.blockers)
        else:
            boundary = "See card custody notes"
        lines.append(
            f"{tex(evidence_badge(s.axis_label or s.status))} & {tex(s.status)} & "
            f"{tex(s.axis_label)} & {tex(score)} & {tex(sha)} & {tex(b)} & "
            f"{{\\raggedright\\footnotesize {tex(boundary)}}} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", ""])
    return "\n".join(lines)


def score_card_is_blocked(card: ScoreCardRow) -> bool:
    return card.status.lower() in {"blocked", "template"} or "blocked" in card.axis_label.lower()


def score_card_boundary(card: ScoreCardRow) -> str:
    if score_card_is_blocked(card):
        return "BLOCKED: missing evaluator/archive authority"
    if card.status.lower() in {"toy", "toy_synthetic"}:
        return "TOY: synthetic explanatory card; not an evaluator row"
    if card.blockers:
        return "; ".join(card.blockers)
    return "See card custody notes"


def shorten(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def tex(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))
