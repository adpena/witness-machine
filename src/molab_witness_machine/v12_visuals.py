"""Responsive inline-SVG figures for the V12 marimo narrative.

The functions in this module return self-contained SVG strings.  They have no
runtime dependency beyond Python, and every visible prose string can be
replaced through ``messages``.  Notebook cells can therefore wrap the result in
``mo.Html(...)`` without shipping fonts, images, JavaScript, or network assets.

The defaults describe mechanisms, not measured outcomes.  In particular, none
of these figures presents a candidate score.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import isfinite, log
import re
from typing import Mapping, Sequence

import numpy as np


MessageMap = Mapping[str, str]


@dataclass(frozen=True)
class VisualTheme:
    """A neutral instrument palette shared by the V12 figures.

    Graphite carries structure; color is reserved for semantic signal, budget,
    and failure.  The narrow palette keeps the figures closer to a scientific
    plate than a stack of product cards.
    """

    background: str = "var(--wm-bg, #F7F7F3)"
    panel: str = "var(--wm-panel, #FFFFFF)"
    panel_alt: str = "var(--wm-panel-2, #CBD2D8)"
    text: str = "var(--wm-text, #17191C)"
    muted: str = "var(--wm-muted, #4F5963)"
    cyan: str = "var(--wm-cyan, #006F8A)"
    gold: str = "var(--wm-gold, #8A5A00)"
    coral: str = "var(--wm-coral, #B73A37)"
    mint: str = "var(--wm-mint, #527A00)"
    violet: str = "var(--wm-violet, #5D4A9E)"
    dark_ink: str = "var(--wm-dark-ink, #101214)"


DEFAULT_THEME = VisualTheme()


DEFAULT_MESSAGES: dict[str, str] = {
    # Task-witness hero.
    "hero.title": "What must the machine see?",
    "hero.desc": (
        "A source driving frame passes through frozen perception. A compact "
        "witness keeps task evidence at decision boundaries instead of copying "
        "every pixel uniformly."
    ),
    "hero.source": "source video",
    "hero.judge": "frozen perception",
    "hero.witness": "compact witness",
    "hero.pixels": "many pixels",
    "hero.evidence": "task evidence",
    "hero.boundary": "decision boundary",
    "hero.question": "Preserve the evidence the judge consumes.",
    # Score balance.
    "score.title": "Three debts, one score",
    "score.desc": (
        "The official objective combines segmentation distance, pose distance, "
        "and archive rate. The dial is conceptual and reports no evaluation result."
    ),
    "score.seg": "segmentation",
    "score.pose": "pose",
    "score.rate": "archive rate",
    "score.task": "task evidence",
    "score.bytes": "fewer bytes",
    "score.balance": "representation balance",
    "score.prompt": "Move bytes toward evidence that changes the frozen judge.",
    # STAC-style frozen-network sensitivity.
    "stac.title": "Ask where compression hurts",
    "stac.desc": (
        "A frozen-network sensitivity map drives regional quantization. A highlighted "
        "off-boundary hotspot shows why semantic edges are a hypothesis, not a complete oracle."
    ),
    "stac.sensitivity": "network sensitivity",
    "stac.allocation": "finer quantization",
    "stac.boundary": "semantic boundary",
    "stac.hotspot": "off-boundary debt",
    "stac.note": "first-order controller · not an evaluated score",
    # v8 heterogeneous generator-native graph.
    "edge.title": "Heterogeneous generators carry the scene",
    "edge.desc": (
        "An illustrative generator-native graph: the Road/Undrivable interface is "
        "strictly edge-centric, while a lane band, movable sites, a static hood, "
        "and tiny tail edges keep their smallest natural coordinates."
    ),
    "edge.road": "Road",
    "edge.undriv": "Undrivable",
    "edge.lane": "Lane",
    "edge.mycar": "My car",
    "edge.movable": "Movable",
    "edge.shared": "illustrative generator-native composition",
    "edge.merge": "MERGE",
    "edge.diff": "DIFF",
    "edge.correct": "CORRECT",
    "edge.pipeline": "heterogeneous generators · de-shared edges · sparse repair",
    # Equal-curvature Laguerre cells.
    "laguerre.title": "Fields compete; a partition appears",
    "laguerre.desc": (
        "Equal-curvature quadratic fields form Laguerre cells. Changing a bias "
        "moves shared separatrices while the sites remain fixed."
    ),
    "laguerre.cell_a": "road",
    "laguerre.cell_b": "lane",
    "laguerre.cell_c": "movable",
    "laguerre.tie_locus": "shared tie locus",
    "laguerre.note": "argmax selects one cell at every point",
    # Evaluator-equivalence class.
    "equiv.title": "Different pictures, one machine answer",
    "equiv.desc": (
        "Three visibly different witnesses remain in one frozen-output cell, "
        "while a visually close fourth witness crosses the semantic decision wall."
    ),
    "equiv.source": "source",
    "equiv.witness_a": "less texture",
    "equiv.witness_b": "different color",
    "equiv.crossed": "wall crossed",
    "equiv.same": "same frozen output",
    "equiv.changed": "changed output",
    # SDF boundary sweep.
    "sdf.title": "A class change sweeps area",
    "sdf.desc": (
        "One analytic circular signed-distance zero set moves normally. The "
        "highlighted annulus is the exact sign-disagreement set."
    ),
    "sdf.field": "signed-distance field",
    "sdf.zero": "zero level set",
    "sdf.swept": "swept disagreement",
    "sdf.exact": "exact swept area",
    "sdf.first": "small-motion integral",
    "sdf.note": "DERIVATION · before topology changes",
    # Morse gradient system.
    "morse.title": "The wall is a dynamical separator",
    "morse.desc": (
        "Negative-gradient trajectories of an explicit double-well potential "
        "flow to two minima. The stable manifold of the saddle separates them."
    ),
    "morse.minimum": "minimum",
    "morse.saddle": "saddle",
    "morse.stable": "stable manifold",
    "morse.flow": "negative-gradient flow",
    "morse.soft_band": "soft 10–90% tie band",
    "morse.note": "literal system: U(x,y)=(x²−1)²+y²",
    # Temporal aperture.
    "motion.title": "Motion needs an aperture",
    "motion.desc": (
        "A semantic wall can be transported, but two-frame motion still needs "
        "observable texture. Image flow and the geometric transport model remain distinct."
    ),
    "motion.frame_a": "frame t",
    "motion.frame_b": "frame t+1",
    "motion.image_flow": "image-flow estimate",
    "motion.model": "geometric transport model",
    "motion.occlusion": "occlusion / refresh debt",
    "motion.screw": "screw-transport model",
    "motion.xi": "ξ=(v,ω)∈𝔰𝔢(3) → exp(ξ̂)∈SE(3)",
    "motion.note": "model coordinates are not asserted PoseNet semantics",
    # Shadow-price controller.
    "control.title": "Four carriers bid for the next bit",
    "control.desc": (
        "Boundary, motion, generator, and correction candidates are ranked by "
        "declared local value per counted unit. The bars are a controller construction."
    ),
    "control.boundary": "boundary",
    "control.motion": "motion",
    "control.generator": "generator",
    "control.residual": "correction",
    "control.winner": "next admitted carrier",
    "control.note": "shadow-price construction · not an exact evaluation",
}


_SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _prefix(value: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError("id_prefix must begin with a letter and contain only letters, digits, '_' or '-'")
    return value


def _text(messages: MessageMap | None, key: str) -> str:
    value = DEFAULT_MESSAGES[key] if messages is None else messages.get(key, DEFAULT_MESSAGES[key])
    return escape(str(value), quote=True)


def _theme_css(theme: VisualTheme, prefix: str) -> str:
    """CSS scoped by classes that are unique to a figure prefix."""

    p = prefix
    return f"""
    .{p}-bg {{ fill: {theme.background}; }}
    .{p}-panel {{ fill: {theme.panel}; stroke: {theme.panel_alt}; stroke-width: 1.5; }}
    .{p}-panel-alt {{ fill: {theme.panel}; stroke: {theme.panel_alt}; stroke-width: 1.5; }}
    .{p}-text {{ fill: {theme.text}; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", ui-sans-serif, system-ui, sans-serif; }}
    .{p}-muted {{ fill: {theme.muted}; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", ui-sans-serif, system-ui, sans-serif; }}
    .{p}-title {{ font-size: 34px; font-weight: 720; letter-spacing: -0.6px; }}
    .{p}-label {{ font-size: 22px; font-weight: 680; }}
    .{p}-small {{ font-size: 17px; font-weight: 560; }}
    .{p}-math {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .{p}-line {{ fill: none; stroke-linecap: square; stroke-linejoin: miter; vector-effect: non-scaling-stroke; }}
    .{p}-flow {{ stroke-dasharray: 8 10; animation: {p}-flow 8s linear infinite; animation-play-state: var(--wm-animation-state, paused); }}
    .{p}-pulse {{ transform-box: fill-box; transform-origin: center; animation: {p}-pulse 3.2s ease-in-out infinite; animation-play-state: var(--wm-animation-state, paused); }}
    svg:hover .{p}-flow, svg:focus .{p}-flow,
    svg:hover .{p}-pulse, svg:focus .{p}-pulse {{
      animation-play-state: var(--wm-interaction-animation-state, running);
    }}
    @keyframes {p}-flow {{ to {{ stroke-dashoffset: -72; }} }}
    @keyframes {p}-pulse {{ 0%, 100% {{ opacity: .62; }} 50% {{ opacity: 1; }} }}
    @media (prefers-reduced-motion: reduce) {{
      .{p}-flow, .{p}-pulse {{ animation: none !important; }}
    }}
    """


def _open_svg(
    *,
    prefix: str,
    visual: str,
    title: str,
    desc: str,
    view_box: str,
    theme: VisualTheme,
) -> str:
    return f"""<svg viewBox=\"{view_box}\" width=\"100%\" role=\"img\" tabindex=\"0\"
  aria-labelledby=\"{prefix}-title {prefix}-desc\" preserveAspectRatio=\"xMidYMid meet\"
  data-v12-visual=\"{visual}\"
  style=\"display:block;width:100%;max-width:100%;height:auto;overflow:visible\">
  <title id=\"{prefix}-title\">{title}</title>
  <desc id=\"{prefix}-desc\">{desc}</desc>
  <style>{_theme_css(theme, prefix)}</style>"""


def task_witness_hero(
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-hero",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Show the source → frozen judge → compact task-witness thesis."""

    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    return (
        _open_svg(
            prefix=p,
            visual="task-witness-hero",
            title=t("hero.title"),
            desc=t("hero.desc"),
            view_box="0 0 640 560",
            theme=theme,
        )
        + f"""
  <defs>
    <marker id=\"{p}-arrow\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\"
      markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\">
      <path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"{theme.cyan}\"/>
    </marker>
  </defs>
  <rect class=\"{p}-bg\" width=\"640\" height=\"560\"/>

  <g aria-hidden=\"true\">
    <rect x=\"36\" y=\"56\" width=\"244\" height=\"304\" rx=\"4\" fill=\"{theme.panel}\"
      stroke=\"{theme.panel_alt}\" stroke-width=\"2\"/>
    <path d=\"M36 238 L122 166 L188 207 L280 132 L280 360 L36 360Z\" fill=\"#122B37\"/>
    <path d=\"M36 360 L128 212 L194 212 L280 360Z\" fill=\"#233B50\"/>
    <path class=\"{p}-line\" d=\"M95 360 Q151 271 161 214 M224 360 Q193 270 161 214\"
      stroke=\"{theme.gold}\" stroke-width=\"3\" stroke-dasharray=\"12 12\"/>
    <rect x=\"106\" y=\"271\" width=\"48\" height=\"29\" rx=\"2\" fill=\"{theme.coral}\"/>
    <circle cx=\"116\" cy=\"302\" r=\"5\" fill=\"{theme.dark_ink}\"/>
    <circle cx=\"144\" cy=\"302\" r=\"5\" fill=\"{theme.dark_ink}\"/>
    <text class=\"{p}-text {p}-label\" x=\"52\" y=\"122\">{t("hero.source")}</text>
    <text class=\"{p}-muted {p}-small\" x=\"52\" y=\"342\">{t("hero.pixels")}</text>

    <path class=\"{p}-line {p}-flow\" d=\"M288 224 L347 224\" stroke=\"{theme.cyan}\"
      stroke-width=\"4\" marker-end=\"url(#{p}-arrow)\"/>
    <circle cx=\"380\" cy=\"224\" r=\"48\" fill=\"{theme.panel_alt}\" stroke=\"{theme.cyan}\" stroke-width=\"3\"/>
    <circle class=\"{p}-pulse\" cx=\"380\" cy=\"224\" r=\"24\" fill=\"none\" stroke=\"{theme.cyan}\" stroke-width=\"3\"/>
    <circle cx=\"380\" cy=\"224\" r=\"7\" fill=\"{theme.gold}\"/>
    <text class=\"{p}-text {p}-small\" x=\"380\" y=\"294\" text-anchor=\"middle\">{t("hero.judge")}</text>

    <path class=\"{p}-line {p}-flow\" d=\"M431 224 L474 224\" stroke=\"{theme.cyan}\"
      stroke-width=\"4\" marker-end=\"url(#{p}-arrow)\"/>
    <rect class=\"{p}-panel\" x=\"490\" y=\"126\" width=\"114\" height=\"196\" rx=\"4\"/>
    <path d=\"M504 286 Q526 238 548 245 Q573 251 590 174\" fill=\"none\" stroke=\"{theme.gold}\"
      stroke-width=\"11\" opacity=\".25\"/>
    <path class=\"{p}-line\" d=\"M504 286 Q526 238 548 245 Q573 251 590 174\" stroke=\"{theme.gold}\" stroke-width=\"3\"/>
    <circle cx=\"526\" cy=\"238\" r=\"5\" fill=\"{theme.coral}\"/>
    <circle cx=\"573\" cy=\"235\" r=\"5\" fill=\"{theme.mint}\"/>
    <text class=\"{p}-text {p}-small\" x=\"547\" y=\"350\" text-anchor=\"middle\">{t("hero.witness")}</text>
  </g>

  <g transform=\"translate(36 396)\">
    <rect class=\"{p}-panel\" width=\"568\" height=\"126\" rx=\"4\"/>
    <circle cx=\"36\" cy=\"40\" r=\"8\" fill=\"{theme.gold}\"/>
    <text class=\"{p}-text {p}-label\" x=\"58\" y=\"48\">{t("hero.boundary")}</text>
    <text class=\"{p}-muted {p}-small\" x=\"28\" y=\"83\">{t("hero.question")}</text>
    <text class=\"{p}-text {p}-small\" x=\"540\" y=\"104\" text-anchor=\"end\" fill=\"{theme.mint}\">{t("hero.evidence")}</text>
  </g>
</svg>"""
    )


def score_law_balance(
    task_focus: float = 0.62,
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-score",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Render the official score law beside a conceptual rate/task dial.

    ``task_focus`` only positions the unlabeled design dial; it is clamped to
    ``[0, 1]`` and is never displayed as an evaluation result.
    """

    if not isfinite(task_focus):
        raise ValueError("task_focus must be finite")
    focus = min(1.0, max(0.0, float(task_focus)))
    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    angle = -150.0 + 120.0 * focus
    return (
        _open_svg(
            prefix=p,
            visual="score-law-balance",
            title=t("score.title"),
            desc=t("score.desc"),
            view_box="0 0 640 500",
            theme=theme,
        )
        + f"""
  <rect class=\"{p}-bg\" width=\"640\" height=\"500\"/>

  <g transform=\"translate(36 56)\">
    <rect class=\"{p}-panel\" width=\"568\" height=\"146\" rx=\"4\"/>
    <text class=\"{p}-text {p}-label {p}-math\" x=\"284\" y=\"54\" text-anchor=\"middle\">
      S = 100 d<tspan baseline-shift=\"sub\" font-size=\"15\">seg</tspan> + √(10 d<tspan baseline-shift=\"sub\" font-size=\"15\">pose</tspan>) + 25 r
    </text>
    <g transform=\"translate(30 92)\">
      <circle cx=\"8\" cy=\"0\" r=\"7\" fill=\"{theme.cyan}\"/><text class=\"{p}-muted {p}-small\" x=\"24\" y=\"6\">{t("score.seg")}</text>
      <circle cx=\"196\" cy=\"0\" r=\"7\" fill=\"{theme.mint}\"/><text class=\"{p}-muted {p}-small\" x=\"212\" y=\"6\">{t("score.pose")}</text>
      <circle cx=\"374\" cy=\"0\" r=\"7\" fill=\"{theme.gold}\"/><text class=\"{p}-muted {p}-small\" x=\"390\" y=\"6\">{t("score.rate")}</text>
    </g>
  </g>

  <g transform=\"translate(320 345)\" aria-hidden=\"true\">
    <path class=\"{p}-line\" d=\"M-188 46 A194 194 0 0 1 188 46\" stroke=\"{theme.panel_alt}\" stroke-width=\"22\"/>
    <path class=\"{p}-line\" d=\"M-188 46 A194 194 0 0 1 0 -149\" stroke=\"{theme.gold}\" stroke-width=\"10\"/>
    <path class=\"{p}-line\" d=\"M0 -149 A194 194 0 0 1 188 46\" stroke=\"{theme.mint}\" stroke-width=\"10\"/>
    <g transform=\"rotate({angle:.3f})\">
      <path d=\"M0 12 L-7 -116 Q0 -135 7 -116Z\" fill=\"{theme.text}\"/>
    </g>
    <circle r=\"18\" fill=\"{theme.cyan}\"/><circle r=\"7\" fill=\"{theme.dark_ink}\"/>
  </g>
  <text class=\"{p}-text {p}-small\" x=\"108\" y=\"423\" text-anchor=\"middle\">{t("score.bytes")}</text>
  <text class=\"{p}-text {p}-small\" x=\"532\" y=\"423\" text-anchor=\"middle\">{t("score.task")}</text>
  <text class=\"{p}-muted {p}-small\" x=\"320\" y=\"456\" text-anchor=\"middle\">{t("score.balance")}</text>
  <text class=\"{p}-text {p}-small\" x=\"320\" y=\"482\" text-anchor=\"middle\">{t("score.prompt")}</text>
</svg>"""
    )


def sensitivity_allocation_map(
    sensitivity: Sequence[Sequence[float]],
    quant_steps: Sequence[Sequence[float]],
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-stac",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Render a STAC-style sensitivity field beside its regional allocation.

    The caller provides nonnegative controller values.  Smaller ``quant_steps``
    are displayed as finer allocation.  The fixed cyan curve is a semantic
    boundary reference; the explicit off-boundary hotspot prevents the visual
    from implying that task sensitivity is confined to edges.
    """

    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    sens_rows = [tuple(float(value) for value in row) for row in sensitivity]
    quant_rows = [tuple(float(value) for value in row) for row in quant_steps]
    if not sens_rows or not sens_rows[0]:
        raise ValueError("sensitivity must be a non-empty rectangular matrix")
    height, width = len(sens_rows), len(sens_rows[0])
    if any(len(row) != width for row in sens_rows) or len(quant_rows) != height:
        raise ValueError("sensitivity and quant_steps must be rectangular matrices of equal shape")
    if any(len(row) != width for row in quant_rows):
        raise ValueError("sensitivity and quant_steps must be rectangular matrices of equal shape")
    flat_sensitivity = [value for row in sens_rows for value in row]
    flat_steps = [value for row in quant_rows for value in row]
    if not all(isfinite(value) and value >= 0 for value in flat_sensitivity):
        raise ValueError("sensitivity values must be finite and nonnegative")
    if not all(isfinite(value) and value > 0 for value in flat_steps):
        raise ValueError("quant_steps values must be finite and positive")

    sensitivity_max = max(flat_sensitivity) or 1.0
    cell_width = 244.0 / width
    cell_height = 236.0 / height

    def cells(
        values: Sequence[float],
        *,
        x0: float,
        color: str,
        normalize,
        data_name: str,
    ) -> str:
        rectangles: list[str] = []
        for index, value in enumerate(values):
            row, column = divmod(index, width)
            opacity = 0.12 + 0.78 * normalize(value)
            rectangles.append(
                f'<rect x="{x0 + column * cell_width:.2f}" y="{146 + row * cell_height:.2f}" '
                f'width="{cell_width + 0.35:.2f}" height="{cell_height + 0.35:.2f}" '
                f'data-{data_name}="{value:.6g}" fill="{color}" fill-opacity="{opacity:.3f}"/>'
            )
        return "".join(rectangles)

    sensitivity_cells = cells(
        flat_sensitivity,
        x0=48.0,
        color=theme.coral,
        normalize=lambda value: value / sensitivity_max,
        data_name="sensitivity",
    )
    allocation_cells = cells(
        flat_steps,
        x0=348.0,
        color=theme.mint,
        # Fixed monotone transfer rather than per-image normalization: a
        # uniform change in absolute quantization step must remain visible.
        normalize=lambda value: 1.0 / (1.0 + value),
        data_name="quant-step",
    )
    return (
        _open_svg(
            prefix=p,
            visual="stac-sensitivity-allocation",
            title=t("stac.title"),
            desc=t("stac.desc"),
            view_box="0 0 640 520",
            theme=theme,
        )
        + f"""
  <rect class="{p}-bg" width="640" height="520"/>
  <g data-controller="stac-first-order" aria-hidden="true">
    <rect class="{p}-panel" x="34" y="60" width="272" height="356" rx="4"/>
    <rect class="{p}-panel" x="334" y="60" width="272" height="356" rx="4"/>
    <text class="{p}-text {p}-small" x="170" y="126" text-anchor="middle">{t("stac.sensitivity")}</text>
    <text class="{p}-text {p}-small" x="470" y="126" text-anchor="middle">{t("stac.allocation")}</text>
    {sensitivity_cells}
    {allocation_cells}
    <path class="{p}-line" d="M54 334 Q112 282 166 306 T286 224" stroke="{theme.cyan}" stroke-width="5"/>
    <path class="{p}-line" d="M354 334 Q412 282 466 306 T586 224" stroke="{theme.cyan}" stroke-width="5" opacity=".75"/>
    <circle cx="104" cy="184" r="13" fill="none" stroke="{theme.gold}" stroke-width="4"/>
    <path class="{p}-line" d="M116 176 L162 144" stroke="{theme.gold}" stroke-width="2"/>
    <text class="{p}-text {p}-small" x="168" y="148">{t("stac.hotspot")}</text>
    <text class="{p}-muted {p}-small" x="170" y="399" text-anchor="middle">{t("stac.boundary")}</text>
  </g>
  <line x1="70" y1="444" x2="570" y2="444" stroke="{theme.panel_alt}" stroke-width="2"/>
  <text class="{p}-muted {p}-small" x="320" y="473" text-anchor="middle">{t("stac.note")}</text>
</svg>"""
    )


def evaluator_equivalence_scene(
    texture_focus: float = 0.5,
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-equivalence",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Show a frozen evaluator cell rather than an RGB-fidelity ordering."""

    if not isfinite(texture_focus):
        raise ValueError("texture_focus must be finite")
    focus = min(1.0, max(0.0, float(texture_focus)))
    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    def road_tile(x: int, *, label: str, color: str, stripes: int, crossed: bool = False) -> str:
        dash_opacity = 0.20 + 0.75 * focus
        dash_lines = "".join(
            f'<line x1="{x + 70 + k * 12}" y1="266" x2="{x + 58 + k * 9}" y2="210" '
            f'stroke="{theme.gold}" stroke-width="3" opacity="{dash_opacity:.3f}"/>'
            for k in range(stripes)
        )
        boundary = (
            f'M{x + 22} 274 Q{x + 70} 214 {x + 126} 204'
            if not crossed
            else f'M{x + 22} 244 Q{x + 70} 186 {x + 126} 178'
        )
        return f"""
        <g data-witness="{'crossed' if crossed else 'same-cell'}">
          <rect x="{x}" y="118" width="136" height="180" rx="4" fill="{theme.panel}" stroke="{color}" stroke-width="2"/>
          <path d="M{x} 298 L{x + 38} 222 L{x + 100} 222 L{x + 136} 298Z" fill="{color}" opacity=".20"/>
          {dash_lines}
          <path class="{p}-line" d="{boundary}" stroke="{theme.cyan if not crossed else theme.coral}" stroke-width="4"/>
          <text class="{p}-text {p}-small" x="{x + 74}" y="328" text-anchor="middle">{label}</text>
        </g>"""

    return (
        _open_svg(
            prefix=p,
            visual="evaluator-equivalence-class",
            title=t("equiv.title"),
            desc=t("equiv.desc"),
            view_box="0 0 640 500",
            theme=theme,
        )
        + f"""
  <rect class="{p}-bg" width="640" height="500"/>
  <g aria-hidden="true" data-equivalence="frozen-output-cell">
    {road_tile(18, label=t("equiv.source"), color=theme.muted, stripes=5)}
    {road_tile(170, label=t("equiv.witness_a"), color=theme.mint, stripes=2)}
    {road_tile(322, label=t("equiv.witness_b"), color=theme.violet, stripes=4)}
    {road_tile(474, label=t("equiv.crossed"), color=theme.coral, stripes=5, crossed=True)}
    <path class="{p}-line" d="M36 386 H442" stroke="{theme.mint}" stroke-width="5"/>
    <text class="{p}-text {p}-small" x="238" y="418" text-anchor="middle">{t("equiv.same")}</text>
    <circle cx="542" cy="386" r="18" fill="none" stroke="{theme.coral}" stroke-width="4"/>
    <path class="{p}-line" d="M530 374 L554 398 M554 374 L530 398" stroke="{theme.coral}" stroke-width="4"/>
    <text class="{p}-text {p}-small" x="542" y="438" text-anchor="middle">{t("equiv.changed")}</text>
  </g>
</svg>"""
    )


def sdf_boundary_sweep(
    displacement: float,
    exact_area_fraction: float,
    first_order_area_fraction: float,
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-sdf",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Render an analytic circular SDF and its exact swept normal annulus."""

    values = (displacement, exact_area_fraction, first_order_area_fraction)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("SDF visual inputs must be finite")
    shift = max(-0.28, min(0.28, float(displacement)))
    exact = max(0.0, min(1.0, float(exact_area_fraction)))
    first = max(0.0, min(1.0, float(first_order_area_fraction)))
    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    center_x, center_y = 320.0, 247.0
    reference_radius = 102.0
    moved_radius = reference_radius + 180.0 * shift
    swept_radius = 0.5 * (reference_radius + moved_radius)
    swept_width = max(0.0, abs(moved_radius - reference_radius))
    return (
        _open_svg(
            prefix=p,
            visual="signed-distance-boundary-sweep",
            title=t("sdf.title"),
            desc=t("sdf.desc"),
            view_box="0 0 640 630",
            theme=theme,
        )
        + f"""
  <defs>
    <marker id="{p}-arrow" viewBox="0 0 10 10" refX="8" refY="5"
      markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{theme.gold}"/>
    </marker>
  </defs>
  <rect class="{p}-bg" width="640" height="630"/>
  <g data-construction="sdf-sweep" data-exact-area="{exact:.6f}" data-first-order-area="{first:.6f}" aria-hidden="true">
    <rect x="48" y="60" width="544" height="342" rx="4" fill="{theme.panel}" stroke="{theme.panel_alt}" stroke-width="2"/>
    <g data-field-contours="signed-distance" fill="none" stroke="{theme.panel_alt}" stroke-width="1.5" opacity=".9">
      <circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="48"/>
      <circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="74"/>
      <circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="132"/>
      <circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="158"/>
    </g>
    <circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="{swept_radius:.2f}"
            fill="none" stroke="{theme.gold}" stroke-width="{swept_width:.2f}" stroke-opacity=".38"/>
    <circle class="{p}-line" cx="{center_x:.1f}" cy="{center_y:.1f}" r="{reference_radius:.2f}"
            fill="none" stroke="{theme.cyan}" stroke-width="5"/>
    <circle class="{p}-line" cx="{center_x:.1f}" cy="{center_y:.1f}" r="{moved_radius:.2f}"
            fill="none" stroke="{theme.coral}" stroke-width="4" stroke-dasharray="10 8"/>
    <path d="M{center_x + reference_radius:.2f},{center_y:.2f} L{center_x + moved_radius:.2f},{center_y:.2f}"
          stroke="{theme.gold}" stroke-width="4" marker-end="url(#{p}-arrow)"/>
    <text class="{p}-muted {p}-small" x="66" y="122">{t("sdf.field")}</text>
    <text class="{p}-text {p}-small" x="438" y="160">{t("sdf.zero")}</text>
    <text class="{p}-text {p}-small" x="76" y="374">{t("sdf.swept")}</text>
  </g>
  <g transform="translate(48 432)" data-layout="two-metrics-plus-scope-note">
    <line x1="0" y1="0" x2="544" y2="0" stroke="{theme.panel_alt}" stroke-width="2"/>
    <line x1="270" y1="18" x2="270" y2="82" stroke="{theme.panel_alt}" stroke-width="1"/>
    <text class="{p}-muted {p}-small" x="0" y="30">{t("sdf.exact")}</text>
    <text class="{p}-text {p}-label" x="0" y="68">{100.0 * exact:.2f}%</text>
    <text class="{p}-muted {p}-small" x="294" y="30">{t("sdf.first")}</text>
    <text class="{p}-text {p}-label" x="294" y="68">{100.0 * first:.2f}%</text>
    <text class="{p}-muted {p}-small" x="0" y="116">{t("sdf.note")}</text>
  </g>
</svg>"""
    )


def morse_flow_scene(
    traces: Sequence[Sequence[Sequence[float]]],
    temperature: float,
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-morse",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Render trajectories from the literal double-well Morse gradient system."""

    if not isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    def project(point: Sequence[float]) -> tuple[float, float]:
        if len(point) != 2 or not all(isfinite(float(value)) for value in point):
            raise ValueError("each flow point must be a finite pair")
        x, y = float(point[0]), float(point[1])
        return 320.0 + 165.0 * x, 250.0 - 135.0 * y

    trace_paths: list[str] = []
    for trace_index, trace in enumerate(traces):
        sampled = list(trace)[:: max(1, len(trace) // 70)]
        coords = [project(point) for point in sampled]
        if len(coords) < 2:
            raise ValueError("each flow trace must contain at least two points")
        d = " ".join(("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}" for index, (x, y) in enumerate(coords))
        color = (theme.gold, theme.mint, theme.violet, theme.coral)[trace_index % 4]
        trace_paths.append(
            f'<path data-flow-trace="{trace_index}" class="{p}-line" d="{d}" stroke="{color}" stroke-width="3"/>'
        )

    tau = float(temperature)
    # For q_-(x)=-x and q_+(x)=x, softmax p_+(x) crosses 0.1 and
    # 0.9 at x=±tau*log(9)/2. The displayed band is that exact interval
    # under the scene's x projection (165 px per model unit).
    tie_width = min(310.0, 165.0 * tau * log(9.0))
    tie_x = 320.0 - 0.5 * tie_width
    return (
        _open_svg(
            prefix=p,
            visual="literal-morse-double-well",
            title=t("morse.title"),
            desc=t("morse.desc"),
            view_box="0 0 640 540",
            theme=theme,
        )
        + f"""
  <rect class="{p}-bg" width="640" height="540"/>
  <g data-potential="double-well-morse" aria-hidden="true">
    <ellipse cx="155" cy="250" rx="105" ry="136" fill="{theme.cyan}" fill-opacity=".16" stroke="{theme.cyan}" stroke-opacity=".36"/>
    <ellipse cx="485" cy="250" rx="105" ry="136" fill="{theme.violet}" fill-opacity=".16" stroke="{theme.violet}" stroke-opacity=".36"/>
    <rect data-soft-assignment="q-minus-vs-q-plus" data-temperature="{tau:.6f}"
          data-tie-width-model="{tau * log(9.0):.6f}" x="{tie_x:.2f}" y="92"
          width="{tie_width:.2f}" height="316" fill="{theme.gold}" fill-opacity=".16"/>
    <path class="{p}-line" d="M320 92 V408" stroke="{theme.coral}" stroke-width="4" stroke-dasharray="10 9"/>
    <path class="{p}-line" d="M155 250 H485" stroke="{theme.muted}" stroke-width="2"/>
    {''.join(trace_paths)}
    <circle data-critical-point="minimum" cx="155" cy="250" r="12" fill="{theme.cyan}"/>
    <circle data-critical-point="saddle" cx="320" cy="250" r="12" fill="{theme.gold}" transform="rotate(45 320 250)"/>
    <circle data-critical-point="minimum" cx="485" cy="250" r="12" fill="{theme.violet}"/>
    <text class="{p}-text {p}-small" x="155" y="282" text-anchor="middle">{t("morse.minimum")}</text>
    <text class="{p}-text {p}-small" x="320" y="282" text-anchor="middle">{t("morse.saddle")}</text>
    <text class="{p}-text {p}-small" x="485" y="282" text-anchor="middle">{t("morse.minimum")}</text>
    <text class="{p}-text {p}-small" x="334" y="112">{t("morse.stable")}</text>
    <text class="{p}-muted {p}-small" x="320" y="426" text-anchor="middle">{t("morse.soft_band")}</text>
    <text class="{p}-muted {p}-small" x="66" y="396">{t("morse.flow")}</text>
  </g>
  <line x1="76" y1="448" x2="564" y2="448" stroke="{theme.panel_alt}" stroke-width="2"/>
  <text class="{p}-text {p}-small {p}-math" x="320" y="481" text-anchor="middle">{t("morse.note")}</text>
</svg>"""
    )


def temporal_aperture_scene(
    texture_amplitude: float,
    refresh_pressure: float,
    messages: MessageMap | None = None,
    *,
    screw_points: Sequence[Sequence[float]] | None = None,
    id_prefix: str = "v12-motion",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Show temporal transport, texture observability, and a computed screw orbit."""

    if not isfinite(texture_amplitude) or not isfinite(refresh_pressure):
        raise ValueError("motion visual controls must be finite")
    texture = min(1.0, max(0.0, float(texture_amplitude)))
    pressure = min(1.0, max(0.0, float(refresh_pressure)))
    p = _prefix(id_prefix)

    if screw_points is None:
        # A deterministic geometric default keeps standalone visual calls
        # executable.  The notebook supplies the same construction explicitly.
        from .v12_geometry import screw_trajectory

        samples = np.linspace(0.0, 2.0 * np.pi, 81, dtype=np.float64)
        screw_array = screw_trajectory(
            np.array([0.0, 0.0, 0.12, 0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.82, 0.0, 0.0], dtype=np.float64),
            samples,
        )
    else:
        screw_array = np.asarray(screw_points, dtype=np.float64)
    if screw_array.ndim != 2 or screw_array.shape[1] != 3 or screw_array.shape[0] < 2:
        raise ValueError("screw_points must have shape (n, 3) with n >= 2")
    if not np.all(np.isfinite(screw_array)):
        raise ValueError("screw_points must be finite")

    # Fixed isometric projection, then affine fit into the visible plot.  Every
    # path vertex is therefore a projection of exp(hat(xi) theta) X0.
    projected = np.column_stack(
        (
            screw_array[:, 0] - 0.55 * screw_array[:, 1],
            0.35 * screw_array[:, 0] + 0.35 * screw_array[:, 1] - screw_array[:, 2],
        )
    )
    span = np.ptp(projected, axis=0)
    span[span < 1e-12] = 1.0
    normalized = (projected - np.min(projected, axis=0)) / span
    plot_points = np.column_stack((128.0 + 394.0 * normalized[:, 0], 382.0 + 96.0 * (1.0 - normalized[:, 1])))
    screw_path = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in plot_points)

    def t(key: str) -> str:
        return _text(messages, key)

    stripe_opacity = 0.08 + 0.90 * texture
    stripes_a = "".join(
        f'<line x1="{90 + index * 22}" y1="276" x2="{118 + index * 15}" y2="188" stroke="{theme.gold}" stroke-width="4" opacity="{stripe_opacity:.3f}"/>'
        for index in range(6)
    )
    stripes_b = "".join(
        f'<line x1="{374 + index * 22}" y1="276" x2="{406 + index * 15}" y2="188" stroke="{theme.gold}" stroke-width="4" opacity="{stripe_opacity:.3f}"/>'
        for index in range(6)
    )
    refresh_width = 512.0 * pressure
    return (
        _open_svg(
            prefix=p,
            visual="temporal-motion-aperture",
            title=t("motion.title"),
            desc=t("motion.desc"),
            view_box="0 0 640 680",
            theme=theme,
        )
        + f"""
  <defs>
    <marker id="{p}-motion-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0 0 L10 5 L0 10Z" fill="{theme.cyan}"/>
    </marker>
  </defs>
  <rect class="{p}-bg" width="640" height="680"/>
  <g data-motion-model="not-posenet-semantics" data-layout="frames-flow-screw-debt" aria-hidden="true">
    <rect class="{p}-panel" x="42" y="96" width="252" height="216" rx="4"/>
    <rect class="{p}-panel" x="346" y="96" width="252" height="216" rx="4"/>
    <text class="{p}-text {p}-small" x="66" y="126">{t("motion.frame_a")}</text>
    <text class="{p}-text {p}-small" x="370" y="126">{t("motion.frame_b")}</text>
    <path d="M62 306 L126 176 H226 L278 306Z" fill="{theme.panel_alt}"/>
    <path d="M366 306 L432 176 H532 L582 306Z" fill="{theme.panel_alt}"/>
    {stripes_a}{stripes_b}
    <path class="{p}-line" d="M70 284 Q146 230 268 204" stroke="{theme.cyan}" stroke-width="5"/>
    <path class="{p}-line" d="M374 278 Q450 220 574 194" stroke="{theme.cyan}" stroke-width="5"/>
    <path class="{p}-line {p}-flow" d="M286 214 C318 174 326 174 356 210" stroke="{theme.cyan}" stroke-width="4" marker-end="url(#{p}-motion-arrow)"/>
    <text class="{p}-muted {p}-small" x="320" y="78" text-anchor="middle">{t("motion.image_flow")}</text>
    <path class="{p}-line" d="M286 254 C316 286 326 286 356 252" stroke="{theme.gold}" stroke-width="4" stroke-dasharray="9 7"/>
    <text class="{p}-text {p}-small" x="320" y="340" text-anchor="middle">{t("motion.model")}</text>
    <path d="M520 96 L598 96 L598 174Z" fill="{theme.coral}" fill-opacity=".35"/>
  </g>
  <g data-lie-state="se3-twist-screw" data-trajectory-source="exp-hat-xi" data-sample-count="{screw_array.shape[0]}" aria-hidden="true">
    <line x1="110" y1="430" x2="530" y2="430" stroke="{theme.muted}" stroke-width="2"/>
    <path class="{p}-line" d="{screw_path}"
      stroke="{theme.gold}" stroke-width="5"/>
    <circle cx="{plot_points[0, 0]:.2f}" cy="{plot_points[0, 1]:.2f}" r="7" fill="{theme.coral}"/>
    <circle cx="{plot_points[-1, 0]:.2f}" cy="{plot_points[-1, 1]:.2f}" r="7" fill="{theme.mint}"/>
    <text class="{p}-text {p}-small" x="320" y="368" text-anchor="middle">{t("motion.screw")}</text>
    <text class="{p}-muted {p}-small {p}-math" x="320" y="520" text-anchor="middle">{t("motion.xi")}</text>
  </g>
  <g transform="translate(64 552)" data-refresh-pressure="{pressure:.6f}">
    <text class="{p}-muted {p}-small" x="0" y="22">{t("motion.occlusion")}</text>
    <rect x="0" y="40" width="512" height="18" fill="{theme.panel_alt}"/>
    <rect x="0" y="40" width="{refresh_width:.2f}" height="18" fill="{theme.coral}"/>
    <text class="{p}-muted {p}-small" x="0" y="94">{t("motion.note")}</text>
  </g>
</svg>"""
    )


def shadow_price_allocation(
    values: Sequence[float],
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-control",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Render four nonnegative local value-per-unit controller bids."""

    if len(values) != 4 or not all(isfinite(float(value)) and float(value) >= 0.0 for value in values):
        raise ValueError("values must contain four finite nonnegative bids")
    bids = tuple(float(value) for value in values)
    scale = max(bids) or 1.0
    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    labels = (t("control.boundary"), t("control.motion"), t("control.generator"), t("control.residual"))
    colors = (theme.cyan, theme.gold, theme.violet, theme.coral)
    bars: list[str] = []
    winner = int(max(range(4), key=lambda index: bids[index]))
    for index, (value, label, color) in enumerate(zip(bids, labels, colors, strict=True)):
        x = 72 + index * 138
        height = 210.0 * value / scale
        y = 356.0 - height
        bars.append(
            f'<g data-carrier-bid="{index}" data-selected="{str(index == winner).lower()}">'
            f'<rect x="{x}" y="{y:.2f}" width="84" height="{height:.2f}" rx="1" fill="{color}" fill-opacity=".78"/>'
            f'<circle cx="{x + 42}" cy="{y:.2f}" r="{9 if index == winner else 5}" fill="{theme.text}"/>'
            f'<text class="{p}-text {p}-small" x="{x + 42}" y="390" text-anchor="middle">{label}</text></g>'
        )
    return (
        _open_svg(
            prefix=p,
            visual="shadow-price-carrier-allocation",
            title=t("control.title"),
            desc=t("control.desc"),
            view_box="0 0 640 520",
            theme=theme,
        )
        + f"""
  <rect class="{p}-bg" width="640" height="520"/>
  <g data-controller="local-shadow-price-bids" aria-hidden="true">
    <line x1="54" y1="356" x2="586" y2="356" stroke="{theme.muted}" stroke-width="2"/>
    {''.join(bars)}
    <text class="{p}-text {p}-small" x="320" y="438" text-anchor="middle">{t("control.winner")}: {labels[winner]}</text>
  </g>
  <text class="{p}-muted {p}-small" x="320" y="486" text-anchor="middle">{t("control.note")}</text>
</svg>"""
    )


def edge_carrier_graph(
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-edge",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Render an illustrative heterogeneous, generator-native v8 carrier graph."""

    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    edges = (
        ("road-undriv", "M320 230 L122 160", "horizon-lateral", theme.cyan, 6, ""),
        ("road-lane", "M320 230 L518 160", "lane-band", theme.gold, 5, ""),
        ("road-mycar", "M320 230 L152 342", "hood-static", theme.violet, 5, ""),
        ("road-movable", "M320 230 L488 342", "movable-sites", theme.coral, 5, ""),
        ("undriv-movable", "M122 160 Q320 28 488 342", "movable-sites", theme.coral, 4, ""),
        ("lane-undriv", "M518 160 L122 160", "tiny-tail", theme.mint, 2, ' stroke-dasharray="8 7"'),
        ("lane-movable", "M518 160 L488 342", "tiny-tail", theme.mint, 2, ' stroke-dasharray="8 7"'),
        ("lane-mycar", "M518 160 Q350 430 152 342", "tiny-tail", theme.mint, 2, ' stroke-dasharray="8 7"'),
    )
    edge_svg = "\n".join(
        f'<g data-edge="{name}" data-carrier-family="{family}">'
        f'<path class="{p}-line {p}-flow" d="{path}" stroke="{color}" '
        f'stroke-width="{width}"{dash_attr}/></g>'
        for name, path, family, color, width, dash_attr in edges
    )
    nodes = (
        ("undriv", 122, 160, t("edge.undriv")),
        ("lane", 518, 160, t("edge.lane")),
        ("mycar", 152, 342, t("edge.mycar")),
        ("movable", 488, 342, t("edge.movable")),
    )
    node_svg = "\n".join(
        f'<g data-class-node="{name}"><circle cx="{x}" cy="{y}" r="50" fill="{theme.panel}" '
        f'stroke="{theme.muted}" stroke-width="2"/><text class="{p}-text {p}-small" x="{x}" y="{y + 6}" '
        f'text-anchor="middle">{label}</text></g>'
        for name, x, y, label in nodes
    )
    return (
        _open_svg(
            prefix=p,
            visual="v8-edge-carrier-graph",
            title=t("edge.title"),
            desc=t("edge.desc"),
            view_box="0 0 640 650",
            theme=theme,
        )
        + f"""
  <defs>
    <marker id=\"{p}-pipe-arrow\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\"
      markerWidth=\"7\" markerHeight=\"7\" orient=\"auto\">
      <path d=\"M0 0 L10 5 L0 10Z\" fill=\"{theme.muted}\"/>
    </marker>
  </defs>
  <rect class=\"{p}-bg\" width=\"640\" height=\"650\"/>
  <g data-model=\"heterogeneous-generator-native\" data-edge-centric-scope=\"shared-interface-only\" aria-hidden=\"true\">
    {edge_svg}
    {node_svg}
    <g data-class-node=\"road\" data-adjacency-hub=\"true\">
      <circle cx=\"320\" cy=\"230\" r=\"67\" fill=\"{theme.cyan}\" opacity=\".18\"/>
      <circle cx=\"320\" cy=\"230\" r=\"51\" fill=\"{theme.cyan}\"/>
      <text x=\"320\" y=\"238\" text-anchor=\"middle\" fill=\"{theme.dark_ink}\"
        font-family=\"ui-sans-serif, system-ui, sans-serif\" font-size=\"24\" font-weight=\"780\">{t("edge.road")}</text>
    </g>
    <line x1=\"194\" y1=\"394\" x2=\"446\" y2=\"394\" stroke=\"{theme.panel_alt}\" stroke-width=\"2\"/>
    <text class=\"{p}-text {p}-small\" x=\"320\" y=\"421\" text-anchor=\"middle\">{t("edge.shared")}</text>
  </g>

  <g transform=\"translate(38 482)\" data-pipeline=\"merge-diff-correct\">
    <rect class=\"{p}-panel\" width=\"564\" height=\"132\" rx=\"4\"/>
    <g data-stage=\"merge\"><rect x=\"24\" y=\"25\" width=\"132\" height=\"50\" rx=\"2\" fill=\"{theme.cyan}\"/>
      <text x=\"90\" y=\"58\" text-anchor=\"middle\" fill=\"{theme.dark_ink}\" font-size=\"20\" font-weight=\"800\">{t("edge.merge")}</text></g>
    <path class=\"{p}-line\" d=\"M164 50 L207 50\" stroke=\"{theme.muted}\" stroke-width=\"3\" marker-end=\"url(#{p}-pipe-arrow)\"/>
    <g data-stage=\"diff\"><rect x=\"216\" y=\"25\" width=\"132\" height=\"50\" rx=\"2\" fill=\"{theme.gold}\"/>
      <text x=\"282\" y=\"58\" text-anchor=\"middle\" fill=\"{theme.dark_ink}\" font-size=\"20\" font-weight=\"800\">{t("edge.diff")}</text></g>
    <path class=\"{p}-line\" d=\"M356 50 L399 50\" stroke=\"{theme.muted}\" stroke-width=\"3\" marker-end=\"url(#{p}-pipe-arrow)\"/>
    <g data-stage=\"correct\"><rect x=\"408\" y=\"25\" width=\"132\" height=\"50\" rx=\"2\" fill=\"{theme.mint}\"/>
      <text x=\"474\" y=\"58\" text-anchor=\"middle\" fill=\"{theme.dark_ink}\" font-size=\"20\" font-weight=\"800\">{t("edge.correct")}</text></g>
    <text class=\"{p}-muted {p}-small\" x=\"282\" y=\"108\" text-anchor=\"middle\">{t("edge.pipeline")}</text>
  </g>
</svg>"""
    )


Point = tuple[float, float]


def _clip_half_plane(polygon: Sequence[Point], a: float, b: float, c: float) -> list[Point]:
    """Clip ``polygon`` to ``a*x + b*y <= c`` (Sutherland-Hodgman)."""

    if not polygon:
        return []
    output: list[Point] = []
    previous = polygon[-1]
    previous_value = a * previous[0] + b * previous[1] - c
    for current in polygon:
        current_value = a * current[0] + b * current[1] - c
        previous_inside = previous_value <= 1e-9
        current_inside = current_value <= 1e-9
        if previous_inside != current_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-12:
                ratio = previous_value / denominator
                output.append(
                    (
                        previous[0] + ratio * (current[0] - previous[0]),
                        previous[1] + ratio * (current[1] - previous[1]),
                    )
                )
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
    return output


def _power_cells(sites: Sequence[Point], weights: Sequence[float], bounds: tuple[float, float, float, float]) -> list[list[Point]]:
    if len(sites) != len(weights) or len(sites) < 2:
        raise ValueError("sites and weights must have the same length of at least two")
    xmin, ymin, xmax, ymax = bounds
    cells: list[list[Point]] = []
    for i, (xi, yi) in enumerate(sites):
        polygon: list[Point] = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        for j, (xj, yj) in enumerate(sites):
            if i == j:
                continue
            # ||x-si||^2 - wi <= ||x-sj||^2 - wj
            a = 2.0 * (xj - xi)
            b = 2.0 * (yj - yi)
            c = xj * xj + yj * yj - weights[j] - (xi * xi + yi * yi - weights[i])
            polygon = _clip_half_plane(polygon, a, b, c)
        cells.append(polygon)
    return cells


def laguerre_cells(
    biases: Sequence[float] = (0.0, 0.0, 0.0),
    messages: MessageMap | None = None,
    *,
    id_prefix: str = "v12-laguerre",
    theme: VisualTheme = DEFAULT_THEME,
) -> str:
    """Render a genuine three-site equal-curvature Laguerre partition."""

    if len(biases) != 3 or not all(isfinite(float(value)) for value in biases):
        raise ValueError("biases must contain three finite numbers")
    weights = tuple(max(-1.5, min(1.5, float(value))) for value in biases)
    p = _prefix(id_prefix)

    def t(key: str) -> str:
        return _text(messages, key)

    sites: tuple[Point, ...] = ((1.25, 1.15), (3.75, 1.1), (2.65, 3.0))
    cells = _power_cells(sites, weights, (0.25, 0.25, 4.75, 3.75))
    scale, ox, oy = 112.0, 42.0, 52.0
    colors = (theme.cyan, theme.gold, theme.coral)
    labels = (t("laguerre.cell_a"), t("laguerre.cell_b"), t("laguerre.cell_c"))
    paths: list[str] = []
    points: list[str] = []
    for index, (polygon, site, color, label) in enumerate(zip(cells, sites, colors, labels, strict=True)):
        coords = [(ox + scale * x, oy + scale * y) for x, y in polygon]
        path = " ".join(("M" if k == 0 else "L") + f"{x:.2f},{y:.2f}" for k, (x, y) in enumerate(coords)) + " Z"
        paths.append(
            f'<path data-laguerre-cell="{index}" d="{path}" fill="{color}" fill-opacity=".23" '
            f'stroke="{color}" stroke-width="3" vector-effect="non-scaling-stroke"/>'
        )
        sx, sy = ox + scale * site[0], oy + scale * site[1]
        points.append(
            f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="8" fill="{color}"/>'
            f'<text class="{p}-text {p}-small" x="{sx:.2f}" y="{sy - 16:.2f}" text-anchor="middle">{label}</text>'
        )
    return (
        _open_svg(
            prefix=p,
            visual="equal-curvature-laguerre-cells",
            title=t("laguerre.title"),
            desc=t("laguerre.desc"),
            view_box="0 0 640 600",
            theme=theme,
        )
        + f"""
  <rect class=\"{p}-bg\" width=\"640\" height=\"600\"/>
  <g data-geometry=\"equal-curvature-power-diagram\" aria-hidden=\"true\">
    <rect x=\"70\" y=\"80\" width=\"504\" height=\"392\" rx=\"4\" fill=\"{theme.panel}\" stroke=\"{theme.panel_alt}\"/>
    {''.join(paths)}
    {''.join(points)}
  </g>
  <line x1=\"54\" y1=\"510\" x2=\"586\" y2=\"510\" stroke=\"{theme.panel_alt}\" stroke-width=\"2\"/>
  <text class=\"{p}-text {p}-small {p}-math\" x=\"320\" y=\"544\" text-anchor=\"middle\">φc(x) = −‖x − μc‖² + bc</text>
  <text class=\"{p}-muted {p}-small\" x=\"320\" y=\"576\" text-anchor=\"middle\">{t("laguerre.note")}</text>
</svg>"""
    )


__all__ = [
    "DEFAULT_MESSAGES",
    "DEFAULT_THEME",
    "VisualTheme",
    "edge_carrier_graph",
    "evaluator_equivalence_scene",
    "laguerre_cells",
    "morse_flow_scene",
    "score_law_balance",
    "sdf_boundary_sweep",
    "sensitivity_allocation_map",
    "shadow_price_allocation",
    "task_witness_hero",
    "temporal_aperture_scene",
]
