"""Deterministic V12 figure locks and public release bundles.

The builders in this module package only source, compact derived evidence, and
reader-facing artifacts.  They never include upstream challenge video, model
weights, private run directories, or an exact score claim.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping
from xml.etree import ElementTree
import zipfile

import numpy as np

from .v12_copy import catalog
from .v12_control import notebook_toy_proposals, value_carrier_proposals
from .v12_geometry import boundary_sweep, integrate_negative_gradient, screw_trajectory
from .v12_real_evidence import display_derivative_svg, load_display_derivative
from .v12_stac import select_regional_strategy
from .v12_visuals import (
    edge_carrier_graph,
    evaluator_equivalence_scene,
    laguerre_cells,
    morse_flow_scene,
    score_law_balance,
    sdf_boundary_sweep,
    sensitivity_allocation_map,
    shadow_price_allocation,
    task_witness_hero,
    temporal_aperture_scene,
)


FIGURE_SCHEMA = "witness_machine.v12.figure_lock.v1"
RELEASE_SCHEMA = "witness_machine.v12.release_manifest.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _validate_svg(payload: str) -> None:
    root = ElementTree.fromstring(payload)
    if root.tag.split("}")[-1] != "svg":
        raise ValueError("figure payload is not SVG")
    if root.attrib.get("role") != "img" or "viewBox" not in root.attrib:
        raise ValueError("figure SVG must expose an accessible responsive contract")
    children = [child.tag.split("}")[-1] for child in root]
    if "title" not in children or "desc" not in children:
        raise ValueError("figure SVG must contain title and desc")


def _stac_figure(messages: Mapping[str, str]) -> str:
    height, width = 6, 9
    yy, xx = np.mgrid[0:height, 0:width]
    boundary_y = 4.2 - 0.25 * xx + 0.45 * np.sin(xx * 0.9)
    boundary = np.exp(-0.5 * ((yy - boundary_y) / 0.8) ** 2)
    hotspot = np.exp(-0.5 * (((xx - 1.5) / 0.9) ** 2 + ((yy - 0.8) / 0.7) ** 2))
    sensitivity = 0.08 + 0.92 * boundary + 1.35 * hotspot
    bank = np.array(
        [[0.125], [0.25], [0.5], [1.0], [2.0], [4.0], [8.0]],
        dtype=np.float64,
    )
    strategy = select_regional_strategy(
        sensitivity[:, :, None],
        bank,
        total_budget=12.0,
        region_shape=(2, 3),
    )
    if not strategy.feasible:
        raise ValueError("locked STAC construction exceeds its declared first-order budget")
    return sensitivity_allocation_map(
        sensitivity.tolist(),
        strategy.quant_steps[:, :, 0].tolist(),
        messages,
        id_prefix="wm12-lock-stac",
    )


def build_v12_figure_lock(
    root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Build deterministic SVG figures and their claim-aware hash manifest."""

    project = Path(root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else project / "artifacts" / "figures" / "v12"
    )
    destination.mkdir(parents=True, exist_ok=True)
    messages = catalog("en-US")
    sweep = boundary_sweep(0.10)
    traces = tuple(
        integrate_negative_gradient(start, steps=150).points.tolist()
        for start in ((-0.18, 0.88), (0.22, 0.78), (-0.35, -0.82), (0.38, -0.72))
    )
    screw_points = screw_trajectory(
        np.array([0.0, 0.0, 0.12, 0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.82, 0.0, 0.0], dtype=np.float64),
        np.linspace(0.0, 2.0 * np.pi, 81, dtype=np.float64),
    )
    controller_rows = value_carrier_proposals(
        notebook_toy_proposals(
            boundary_sweep_fraction=sweep.exact_area_fraction,
            texture_amplitude=0.65,
            refresh_pressure=0.35,
            normalized_budget=0.5,
        ),
        d_pose=0.02,
    )
    evidence_root = project / "artifacts" / "v12_public"
    display = load_display_derivative(
        evidence_root / "v12_real_frozen_scorer_display.npz",
        evidence_root / "v12_real_frozen_scorer_display.manifest.json",
    )
    temporal_display_svg = (
        evidence_root / "v12_temporal_transport_display.svg"
    ).read_text(encoding="utf-8")
    figures: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
        "01_task_witness_hero.svg": (
            task_witness_hero(messages, id_prefix="wm12-lock-hero"),
            "DERIVATION",
            "source to frozen receiver to compact witness thesis",
            ("C003",),
        ),
        "02_score_law_balance.svg": (
            score_law_balance(0.71, messages, id_prefix="wm12-lock-score"),
            "EXTERNAL",
            "official score-law structure with a conceptual controller dial",
            ("C001", "C015"),
        ),
        "03_stac_controller_construction.svg": (
            _stac_figure(messages),
            "TOY",
            "synthetic sensitivity field through the implemented STAC regional selector",
            ("C009",),
        ),
        "04_real_frozen_scorer_duality.svg": (
            display_derivative_svg(display, messages=messages, locale="en-US"),
            "EMPIRICAL",
            "compact lossy display of an EMPIRICAL one-pair frozen-SegNet parent artifact",
            ("C010", "C015"),
        ),
        "05_evaluator_equivalence.svg": (
            evaluator_equivalence_scene(0.45, messages, id_prefix="wm12-lock-equivalence"),
            "DERIVATION",
            "frozen-output equivalence-class construction",
            ("C003",),
        ),
        "06_sdf_boundary_sweep.svg": (
            sdf_boundary_sweep(
                sweep.displacement,
                sweep.exact_area_fraction,
                sweep.first_order_area_fraction,
                messages,
                id_prefix="wm12-lock-sdf",
            ),
            "DERIVATION",
            "internally consistent signed-distance boundary displacement construction",
            ("C011",),
        ),
        "07_literal_morse_flow.svg": (
            morse_flow_scene(traces, 0.25, messages, id_prefix="wm12-lock-morse"),
            "DERIVATION",
            "literal analytic double-well Morse gradient system",
            ("C012",),
        ),
        "08_laguerre_partition.svg": (
            laguerre_cells((0.35, -0.20, 0.10), messages, id_prefix="wm12-lock-laguerre"),
            "DERIVATION",
            "genuine equal-curvature power or Laguerre diagram",
            ("C013",),
        ),
        "09_v8_edge_carriers.svg": (
            edge_carrier_graph(messages, id_prefix="wm12-lock-edge"),
            "ADVISORY",
            "edge-centric Road-hub v8 representation hypothesis",
            ("C006",),
        ),
        "10_temporal_screw_aperture.svg": (
            temporal_aperture_scene(
                0.65,
                0.35,
                messages,
                screw_points=screw_points,
                id_prefix="wm12-lock-motion",
            ),
            "DERIVATION",
            "separated image-flow, literal exp(hat(xi) theta) screw model, texture, and refresh construction",
            ("C014",),
        ),
        "11_shadow_price_carriers.svg": (
            shadow_price_allocation(
                tuple(row.value_per_byte for row in controller_rows),
                messages,
                id_prefix="wm12-lock-control",
            ),
            "TOY",
            "explicit hypothetical carrier table valued by local score shadow prices, not a trajectory costate or evaluation",
            ("C004", "C015"),
        ),
        "12_real_temporal_transport.svg": (
            temporal_display_svg,
            "ADVISORY",
            "one locked consecutive pair with advisory Farneback strategy transport and explicit support debt",
            ("C017", "C015"),
        ),
    }

    rows: list[dict[str, object]] = []
    for filename, (svg, claim_class, scope, claim_ids) in figures.items():
        _validate_svg(svg)
        path = destination / filename
        _write_bytes(path, svg.encode("utf-8") + b"\n")
        rows.append(
            {
                "path": path.relative_to(project).as_posix()
                if path.is_relative_to(project)
                else path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "claim_class": claim_class,
                "status": claim_class,
                "axis_label": (
                    "[EMPIRICAL][macOS-CPU advisory][one locked pair][non-score-bearing]"
                    if filename == "04_real_frozen_scorer_duality.svg"
                    else "[ADVISORY][macOS-CPU advisory][one locked pair][non-score-bearing]"
                    if filename == "12_real_temporal_transport.svg"
                    else f"[{claim_class}][non-score-bearing]"
                ),
                "claim_ids": list(claim_ids),
                "caption": scope,
                "scope": scope,
                "exact_score": False,
            }
        )

    source_paths = (
        "notebooks/witness_machine_v12.py",
        "src/molab_witness_machine/v12_visuals.py",
        "src/molab_witness_machine/v12_copy.py",
        "src/molab_witness_machine/v12_control.py",
        "src/molab_witness_machine/v12_geometry.py",
        "src/molab_witness_machine/v12_stac.py",
        "src/molab_witness_machine/v12_policy_compare.py",
        "src/molab_witness_machine/v12_real_evidence.py",
        "src/molab_witness_machine/v12_temporal.py",
        "configs/v12_claim_sources.yaml",
    )
    manifest = {
        "schema": FIGURE_SCHEMA,
        "project_title": "The Witness Machine",
        "primary_paper_title": (
            "DNN-Driven Compressive Offloading for Edge-Assisted Semantic Video Segmentation"
        ),
        "research_date": "2026-07-09",
        "figure_count": len(rows),
        "figures": rows,
        "sources": [
            {
                "path": relative,
                "bytes": (project / relative).stat().st_size,
                "sha256": sha256_file(project / relative),
            }
            for relative in source_paths
        ],
        "real_display_parent_sha256": display.metadata["parent"]["artifact_sha256"],
        "score_claim": False,
        "operator_go_required_for_public_release": True,
    }
    manifest_path = destination / "figure_lock_manifest.json"
    manifest_bytes = _canonical_json(manifest) + b"\n"
    _write_bytes(manifest_path, manifest_bytes)
    if output_dir is None:
        _write_bytes(project / "reports" / "V12_FIGURE_LOCK.json", manifest_bytes)
    return manifest_path


def _zip_member_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_deterministic_zip(
    destination: str | Path,
    members: Mapping[str, bytes],
) -> Path:
    """Write a byte-identical ZIP from a mapping of POSIX member names."""

    text_suffixes = {
        ".bib", ".cff", ".css", ".html", ".json", ".md", ".py", ".srt",
        ".svg", ".tex", ".toml", ".txt", ".vtt", ".yaml", ".yml",
    }
    # Keep the guard's own source public-safe: spell signatures in fragments so
    # a literal scan of this module does not trip on its deny-list.
    forbidden_public_tokens = (
        b"/" + b"Users/",
        b"/" + b"Volumes/",
        b"local" + b"host",
        b"127" + b".0.0.1",
    )
    for name, payload in members.items():
        if Path(name).suffix.lower() in text_suffixes:
            for token in forbidden_public_tokens:
                if token in payload:
                    raise ValueError(
                        f"public text member {name!r} contains local-only token {token.decode('ascii')!r}"
                    )

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(members):
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
                raise ValueError(f"unsafe ZIP member name: {name!r}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name], compresslevel=9)
    return output


def _existing(project: Path, paths: Iterable[str]) -> list[str]:
    return [relative for relative in paths if (project / relative).is_file()]


def resolve_runtime_python_closure(
    project: str | Path,
    seed_paths: Iterable[str],
) -> list[str]:
    """Resolve every local import reachable from shipped Python seeds.

    The Molab archive must not depend on the builder's editable checkout.  This
    resolver walks imports syntactically (including imports inside functions)
    and fails closed when a package import does not resolve to a local module.
    The sorted result is deterministic and suitable for a ZIP manifest.
    """

    root = Path(project).resolve()
    source_root = root / "src"
    package_name = "molab_witness_machine"
    pending = sorted(
        {
            str(Path(path).as_posix())
            for path in seed_paths
            if Path(path).suffix == ".py"
        }
    )
    resolved: set[str] = set()

    while pending:
        relative = pending.pop(0)
        if relative in resolved:
            continue
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime Python seed is missing: {relative}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except SyntaxError as exc:  # pragma: no cover - Python reports the source line
            raise ValueError(f"cannot parse runtime module {relative}: {exc}") from exc

        if path.resolve().is_relative_to(source_root.resolve()):
            module_relative = path.resolve().relative_to(source_root.resolve())
            module_parts = list(module_relative.with_suffix("").parts)
            if module_parts[-1] == "__init__":
                module_parts.pop()
                current_package = module_parts
            else:
                current_package = module_parts[:-1]
        else:
            # The notebook is a seed too, but it is not a package module and
            # therefore cannot legally contain package-relative imports.
            current_package = []

        for node in ast.walk(tree):
            targets: list[list[str]] = []
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                if not current_package:
                    raise ValueError(
                        f"relative import outside package in {relative}:{node.lineno}"
                    )
                if node.level > len(current_package):
                    raise ValueError(
                        f"relative import escapes package in {relative}:{node.lineno}"
                    )
                base = current_package[: len(current_package) - node.level + 1]
                if node.module:
                    targets.append(base + node.module.split("."))
                else:
                    targets.extend(
                        base + alias.name.split(".") for alias in node.names
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                absolute = node.module.split(".")
                if absolute[0] == package_name:
                    targets.append(absolute)
                    if len(absolute) == 1:
                        for alias in node.names:
                            candidate = absolute + alias.name.split(".")
                            if source_root.joinpath(*candidate).with_suffix(
                                ".py"
                            ).is_file():
                                targets.append(candidate)
            elif isinstance(node, ast.Import):
                targets.extend(
                    alias.name.split(".")
                    for alias in node.names
                    if alias.name.split(".")[0] == package_name
                )
            if not targets:
                continue

            for target_parts in targets:
                if not target_parts or target_parts[0] != package_name:
                    continue
                module_path = source_root.joinpath(*target_parts).with_suffix(".py")
                package_path = source_root.joinpath(*target_parts, "__init__.py")
                if module_path.is_file():
                    dependency = module_path.relative_to(root).as_posix()
                elif package_path.is_file():
                    dependency = package_path.relative_to(root).as_posix()
                else:
                    dotted = ".".join(target_parts)
                    raise ValueError(
                        f"unresolved local import {dotted!r} "
                        f"in {relative}:{node.lineno}"
                    )
                if dependency not in resolved and dependency not in pending:
                    pending.append(dependency)
                    pending.sort()
        resolved.add(relative)

    return sorted(resolved)


def render_v12_paper_tables(project: Path, figure_manifest: Path) -> None:
    """Render the manuscript tables directly from the current V12 registries."""

    from .release_candidate import (
        evidence_badge,
        latex_claim_table,
        latex_score_table,
        load_claims,
        load_score_cards,
        tex,
    )

    sections = project / "paper" / "sections"
    sections.mkdir(parents=True, exist_ok=True)
    _write_bytes(
        sections / "generated_claims_table.tex",
        latex_claim_table(load_claims(project / "configs" / "claims_registry.yaml")).encode("utf-8"),
    )
    _write_bytes(
        sections / "generated_score_cards_table.tex",
        latex_score_table(load_score_cards(project)).encode("utf-8"),
    )

    manifest = json.loads(figure_manifest.read_text(encoding="utf-8"))
    lines = [
        r"% Generated from reports/V12_FIGURE_LOCK.json. Do not hand-edit.",
        r"\begin{center}",
        r"\footnotesize",
        r"\begin{tabular}{lllp{0.43\linewidth}}",
        r"\toprule",
        r"Evidence & Figure & SHA & Scope \\",
        r"\midrule",
    ]
    for row in manifest.get("figures", []):
        filename = Path(str(row["path"])).name.replace("_", "-")
        badge = evidence_badge(str(row.get("axis_label", row.get("claim_class", "ADVISORY"))))
        claim_ids = ",".join(str(value) for value in row.get("claim_ids", []))
        scope = f"{claim_ids}: {row.get('caption', '')}" if claim_ids else str(row.get("caption", ""))
        lines.append(
            f"{tex(badge)} & {tex(filename)} & {tex(str(row['sha256'])[:12])} & "
            f"{{\\raggedright {tex(scope)}}} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{center}", ""))
    _write_bytes(
        sections / "generated_figure_locks_table.tex",
        "\n".join(lines).encode("utf-8"),
    )


def build_v12_release_bundles(
    root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Build compact molab and fuller OSS/research ZIPs plus one custody manifest."""

    project = Path(root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else project / "artifacts" / "release"
    )
    destination.mkdir(parents=True, exist_ok=True)
    figure_manifest = build_v12_figure_lock(project)
    render_v12_paper_tables(project, figure_manifest)

    runtime_python_seeds = [
        "notebooks/witness_machine_v12.py",
        "src/molab_witness_machine/__init__.py",
        "src/molab_witness_machine/v12_copy.py",
        "src/molab_witness_machine/v12_control.py",
        "src/molab_witness_machine/v12_geometry.py",
        "src/molab_witness_machine/v12_gpu.py",
        "src/molab_witness_machine/v12_real_evidence.py",
        "src/molab_witness_machine/v12_release.py",
        "src/molab_witness_machine/v12_stac.py",
        "src/molab_witness_machine/v12_visuals.py",
        "src/molab_witness_machine/v12_temporal.py",
    ]
    runtime_files = resolve_runtime_python_closure(project, runtime_python_seeds) + [
        "artifacts/v12_public/v12_real_frozen_scorer_display.npz",
        "artifacts/v12_public/v12_real_frozen_scorer_display.manifest.json",
        "artifacts/v12_public/v12_real_frozen_scorer_evidence.npz",
        "artifacts/v12_public/v12_real_frozen_scorer_evidence.manifest.json",
        "artifacts/v12_public/v12_real_frozen_scorer_evidence.npz",
        "artifacts/v12_public/v12_real_frozen_scorer_evidence.manifest.json",
        "artifacts/v12_public/v12_temporal_transport_display.npz",
        "artifacts/v12_public/v12_temporal_transport_display.svg",
        "artifacts/v12_public/v12_temporal_transport_display.manifest.json",
        "notebooks/__marimo__/assets/witness_machine_v12/opengraph.png",
        "configs/v12_paper_lineage.yaml",
        "configs/v12_claim_sources.yaml",
        "docs/THIRD_PARTY_NOTICES_V12.md",
        "docs/V12_BUNDLE_README.md",
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
    ]
    runtime_files = sorted(set(_existing(project, runtime_files)))
    research_files = runtime_files + _existing(
        project,
        [
            "reports/live_research_notes.md",
            "reports/V12_PRIMARY_PAPER_FINAL_REDTEAM.md",
            "reports/V12_GEOMETRY_LINEAGE_FINAL_LOCK.md",
            "reports/V12_REAL_EVIDENCE_BUILD.md",
            "reports/V12_TEMPORAL_TRANSPORT_BUILD.md",
            "reports/V12_ACCELERATOR_PROXY_PARITY.json",
            "reports/V13_MATH_CLAIM_P0_REPAIR_20260709.md",
            "reports/MANIM_COMPANION_STORYBOARD_V12.md",
            "reports/V13_FILM_TASTE_RECEIPT.md",
            "reports/V13_ACCELERATOR_REAL_EVIDENCE.md",
            "reports/V13_ACCELERATOR_REAL_EVIDENCE.json",
            "reports/V13_JUDGE_FRESH_EYES_AUDIT.md",
            "reports/V13_JUDGE_FRESH_EYES_AUDIT.json",
            "reports/V13_PERFORMANCE_BENCHMARK.md",
            "reports/V13_PERFORMANCE_BENCHMARK.json",
            "reports/V13_JUDGE_FRESH_EYES_AUDIT.md",
            "reports/V13_JUDGE_FRESH_EYES_AUDIT.json",
            "reports/V12_SUBMISSION_FORM_CONSTRAINTS.json",
            "reports/V12_SUBMISSION_FORM_CONSTRAINTS.md",
            "reports/V12_FORM_UPLOAD_RECEIPT.json",
            "reports/V12_FORM_UPLOAD_RECEIPT.md",
            "reports/RECURSIVE_ADVERSARIAL_REVIEW_V12.md",
            "reports/RECURSIVE_ADVERSARIAL_REVIEW_V12.json",
            "review/MENTOR_PACKET_V12.md",
            "docs/V12_WORLD_CLASS_NOTEBOOK_REWRITE_PLAN.md",
            "configs/claims_registry.yaml",
            "paper/main.tex",
            "paper/references.bib",
            "paper/main.pdf",
            "paper/sections/generated_claims_table.tex",
            "paper/sections/generated_score_cards_table.tex",
            "paper/sections/generated_figure_locks_table.tex",
            "artifacts/figures/lane_sdf.png",
            "artifacts/figures/v8_generator_native_power_diagram.png",
            "artifacts/video/v12/captions_en-US.srt",
            "artifacts/video/v12/captions_en-US.vtt",
            "artifacts/video/v12/captions_es-US_DRAFT.srt",
            "artifacts/video/v12/captions_es-US_DRAFT.vtt",
            "artifacts/video/v12/transcript_en-US.txt",
            "reports/V12_FILM_PUBLIC_RECEIPT.json",
            "reports/V12_BROWSER_E2E.json",
            "reports/V12_BROWSER_E2E.md",
            "reports/V12_PACT_TAC_DRIFT_AUDIT.json",
            "reports/V12_PACT_TAC_DRIFT_AUDIT.md",
            "reports/V12_FINAL_REMEDIATION.json",
            "reports/V12_FINAL_REMEDIATION.md",
            "reports/V12_FINAL_REPORT.json",
            "reports/V12_FINAL_REPORT.md",
            "reports/V13_FILM_TASTE_RECEIPT.md",
            "reports/V13_MATH_CLAIM_P0_REPAIR_20260709.md",
            "reports/V13_PERFORMANCE_BENCHMARK.json",
            "reports/V13_PERFORMANCE_BENCHMARK.md",
            "artifacts/browser/v12_desktop_final_release.png",
            "artifacts/browser/v12_mobile_final_release.png",
            "artifacts/browser/v12_mobile_spanish_final_release.png",
            "src/molab_witness_machine/v8_edge_decomp.py",
            "scripts/check_v12_pact_drift.py",
            "scripts/benchmark_v13_interactions.py",
            "video/v12/beat_manifest.json",
            "video/v12/build_release_derivatives.py",
            "video/v12/build_form_upload.py",
            "tests/test_v12_claim_math_contract.py",
            "tests/test_v12_policy_compare.py",
            "tests/test_v13_performance_contract.py",
            "tests/test_v8_edge_decomp.py",
            "tests/test_v12_pact_drift.py",
            "artifacts/video/v12/final_slate.png",
            "artifacts/video/v12/contact_sheet_release_3x3.png",
        ],
    )
    figure_files = sorted(
        path.relative_to(project).as_posix()
        for path in figure_manifest.parent.glob("*")
        if path.is_file()
    )
    research_files.extend(figure_files)
    research_files = sorted(set(research_files))

    all_manifest_paths = sorted(set(runtime_files + research_files))
    companion_paths = _existing(
        project,
        [
            "artifacts/video/v12/witness_machine_v12_release_candidate_1080p30_en_burned.mp4",
            "artifacts/video/v12/witness_machine_v12_mobile_1080x1920_en_burned.mp4",
            "artifacts/video/v12/witness_machine_v12_animatic_480p15_en_burned.mp4",
            "artifacts/video/v12/witness_machine_v12_form_upload_480p15.flv",
        ],
    )
    manifest = {
        "schema": RELEASE_SCHEMA,
        "project_title": "The Witness Machine",
        "primary_paper_title": (
            "DNN-Driven Compressive Offloading for Edge-Assisted Semantic Video Segmentation"
        ),
        "research_date": "2026-07-09",
        "files": [
            {
                "path": relative,
                "bytes": (project / relative).stat().st_size,
                "sha256": sha256_file(project / relative),
            }
            for relative in all_manifest_paths
        ],
        "molab_runtime_files": runtime_files,
        "research_files": research_files,
        "companion_artifacts": [
            {
                "path": relative,
                "bytes": (project / relative).stat().st_size,
                "sha256": sha256_file(project / relative),
                "included_in_zip": False,
            }
            for relative in companion_paths
        ],
        "score_claim": False,
        "full_stac_reproduction": False,
        "public_submit_push_upload_pr": "BLOCKED_PENDING_OPERATOR_GO",
    }
    manifest_path = destination / "witness_machine_v12_release_manifest.json"
    manifest_bytes = _canonical_json(manifest) + b"\n"
    _write_bytes(manifest_path, manifest_bytes)

    def package(paths: Iterable[str], prefix: str) -> dict[str, bytes]:
        members = {
            f"{prefix}/{relative}": _zip_member_bytes(project / relative)
            for relative in sorted(set(paths))
        }
        members[f"{prefix}/README.md"] = _zip_member_bytes(
            project / "docs" / "V12_BUNDLE_README.md"
        )
        members[f"{prefix}/RELEASE_MANIFEST.json"] = manifest_bytes
        return members

    molab_zip = write_deterministic_zip(
        destination / "witness_machine_v12_molab_bundle.zip",
        package(runtime_files, "witness_machine_v12"),
    )
    research_zip = write_deterministic_zip(
        destination / "witness_machine_v12_research_bundle.zip",
        package(research_files, "witness_machine_v12"),
    )
    return molab_zip, research_zip, manifest_path


__all__ = [
    "FIGURE_SCHEMA",
    "RELEASE_SCHEMA",
    "build_v12_figure_lock",
    "build_v12_release_bundles",
    "render_v12_paper_tables",
    "resolve_runtime_python_closure",
    "sha256_file",
    "write_deterministic_zip",
]
