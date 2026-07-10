"""Small, validated message catalogs for the V12 public notebook.

The narrative remains English-authoritative, but prose, controls, takeaways,
and accessible descriptions use stable message IDs.  Equations and source data
stay language-neutral.  Missing translations fail in tests and fall back to
English at runtime so a partial locale can never break the notebook.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


DEFAULT_LOCALE = "en-US"
SUPPORTED_LOCALES = ("en-US", "es-US")


_EN_US = {
    "meta.language": "Language",
    "meta.english": "English",
    "meta.spanish": "Español",
    "accessibility.help": (
        "Light, dark, color-vision-safe, monochrome, and motion preferences change "
        "presentation only; the computed geometry is invariant."
    ),
    "dev.status": "Interactive paper implementation · move the mathematics, inspect the evidence",
    "nav.score": "01 · FROZEN RECEIVER",
    "nav.paper": "02 · PAPER CONTROLLER",
    "nav.real": "REAL FROZEN-SCORER EVIDENCE",
    "nav.equivalence": "03 · EQUIVALENCE CLASS",
    "nav.boundary": "04 · BOUNDARY LENS",
    "nav.topology": "05 · TOPOLOGY",
    "nav.generator": "06 · GENERATOR LENS",
    "nav.shared": "07 · SHARED CARRIERS",
    "nav.motion": "08 · TEMPORAL APERTURE",
    "nav.control": "09 · SHADOW-PRICE CONTROL",
    "nav.close": "10 · WHAT FELL OUT",
    "route.aria": "Five-minute judge path",
    "route.label": "5-MINUTE PATH",
    "route.paper": "paper experiment",
    "route.real": "real evidence",
    "route.wall": "move the wall",
    "route.gpu": "stress the proxy",
    "route.close": "conclusion",
    "hero.eyebrow": "COMPRESSION FOR A FROZEN MACHINE",
    "hero.title": "The Witness Machine",
    "hero.subtitle": (
        "What is the smallest video that makes a frozen driving machine reach "
        "the same decision?"
    ),
    "hero.paper_prefix": "Interactive implementation and extension of",
    "hero.paper_working": (
        "DNN-Driven Compressive Offloading for Edge-Assisted Semantic Video Segmentation"
    ),
    "hero.lede": (
        "STAC asks where compression hurts. The Witness Machine asks which compact "
        "geometric coordinate should carry that debt. This challenge preserves a "
        "semantic partition, a two-frame motion signal, and a counted archive—not "
        "every pixel uniformly. Move one control and watch the representation follow the receiver."
    ),
    "control.task_pressure": "Task pressure",
    "control.task_pressure_help": (
        "A conceptual controller setting: move capacity from uniform appearance "
        "toward evidence that changes the frozen judge. It is not an evaluated score."
    ),
    "control.loss_budget": "Allowed first-order loss budget B",
    "control.loss_budget_help": (
        "The paper controller spends this declared local budget across coefficients. "
        "Larger B permits coarser regional tables; it is not a measured task loss or score."
    ),
    "control.loss_budget_prompt": (
        "[TOY][DERIVATION] START HERE · Move B. One reactive state recomputes the "
        "receiver balance, the paper-controller table allocation, and the final "
        "carrier bids. Follow the same pressure from equation to witness."
    ),
    "control.boundary_focus": "Outline precision ratio ρ",
    "control.boundary_focus_help": (
        "At ρ=1 the outline-only policy is uniform. Larger ρ spends more of the same "
        "declared precision proxy on the semantic outline and less away from it."
    ),
    "control.boundary_focus_prompt": (
        "[TOY][DERIVATION] Try ρ=1, then ρ=8. At matched Σ1/q, does outline-only "
        "still beat uniform?"
    ),
    "hero.desc": (
        "A source driving frame passes through frozen perception. A compact witness "
        "keeps task evidence where the frozen receiver is sensitive—on and away "
        "from decision boundaries—instead of copying every pixel uniformly."
    ),
    "hero.source": "source video",
    "hero.judge": "frozen perception",
    "hero.witness": "compact witness",
    "hero.pixels": "many pixels",
    "hero.evidence": "task evidence",
    "hero.boundary": "decision boundary",
    "hero.question": "Preserve the evidence the judge consumes.",
    "score.heading": "One receiver, three debts",
    "score.body": (
        "Segmentation disagreement is linear, pose disagreement enters through a "
        "square root, and every archive byte pays rate. The equation is official; "
        "the dial is an explorable design question, not a candidate result."
    ),
    "score.takeaway": "A byte has value only through the decision it protects.",
    "score.title": "Three debts, one score",
    "score.desc": (
        "The official objective combines segmentation distance, pose distance, and "
        "archive rate. The dial is conceptual and reports no evaluation result."
    ),
    "score.seg": "segmentation",
    "score.pose": "pose",
    "score.rate": "archive rate",
    "score.task": "task evidence",
    "score.bytes": "fewer bytes",
    "score.balance": "representation balance",
    "score.prompt": "Move bytes toward evidence that changes the frozen judge.",
    "stac.title": "Ask where compression hurts",
    "stac.desc": (
        "A frozen-network sensitivity map drives regional quantization. A highlighted "
        "off-boundary hotspot shows why semantic edges are a hypothesis, not a complete oracle."
    ),
    "stac.sensitivity": "network sensitivity",
    "stac.allocation": "finer quantization",
    "stac.boundary": "semantic boundary",
    "stac.hotspot": "off-boundary debt",
    "stac.note": "TOY · SYNTHETIC CONSTRUCTION · first-order controller · not an evaluated score",
    "stac.heading": "The network, not the outline, sets the price",
    "stac.body": (
        "STAC differentiates a frozen segmentation loss with respect to transform "
        "coefficients. Its first-order bound assigns smaller quantization steps where "
        "the loss is more sensitive, then approximates that ideal with regional table choices "
        "(§III, Eq. 2 and Algorithm 1; §IV, Algorithm 2)."
    ),
    "stac.takeaway": (
        "A boundary is a powerful coordinate, but frozen-network sensitivity can also live "
        "away from it. The residual is part of the model, not an embarrassment."
    ),
    "stac.scope": (
        "This first view is an analytic sensitivity field passed through the implemented "
        "STAC allocation kernel. It demonstrates the controller; it is not a SegNet backward pass."
    ),
    "stac.requested": "declared B",
    "stac.realized": "realized bound",
    "stac.slack": "remaining slack",
    "stac.feasible": "finite-bank gate",
    "stac.feasible_yes": "PASS · ≤ B",
    "stac.feasible_no": "REFUSE · bank infeasible",
    "stac.policy_caption": "Matched-effort counterexample: uniform, outline-only, and sensitivity-projected policies",
    "stac.policy": "policy",
    "stac.policy_bound": "first-order bound",
    "stac.policy_precision": "matched Σ1/q",
    "stac.policy_uniform": "uniform",
    "stac.policy_boundary": "outline-only",
    "stac.policy_sensitivity": "sensitivity-projected",
    "stac.policy_result_loses": "At this field and matched proxy, outline-only loses to uniform.",
    "stac.policy_result_wins": "At this field and matched proxy, outline-only beats uniform.",
    "stac.policy_result_ties": "At ρ=1, outline-only collapses exactly to uniform.",
    "stac.policy_scope": "TOY · same analytic field and matched reciprocal-step precision proxy · not JPEG · not an evaluated score",
    "real.heading": "One frozen scorer, two meanings of sensitivity",
    "real.body": (
        "On one fixed real comma.ai pair, the raw top-two-margin input gradient is broad. "
        "At the declared 1e-6 regularizer, normalization by the squared local decision margin "
        "makes a heuristic sensitivity proxy collapse toward the semantic wall. The result is "
        "regularizer- and logit-scale-sensitive, and it is not a calibrated flip probability "
        "because the summed-output gradient couples receptive fields."
    ),
    "real.raw_metric": "raw gradient mass outside a 3 px boundary annulus",
    "real.flip_metric": "margin-normalized sensitivity-proxy mass on the exact boundary",
    "real.takeaway": (
        "The same real surface rejects edge-only saliency and shows how a declared small-margin "
        "heuristic can concentrate on a boundary at one fixed regularizer. It does not predict "
        "a flip probability or define a scale-invariant statistic."
    ),
    "real.scope": (
        "[EMPIRICAL][macOS-CPU advisory][one fixed preselected pair] · pixel-space "
        "summed-margin sensitivity proxy at epsilon=1e-6 · parameter-sensitive · no score · "
        "not STAC DCT sensitivity"
    ),
    "real.duality_aria": "Two boundary-sensitivity observations from one locked real pair",
    "real.visual.title": "Frozen-scorer sensitivity: field and margin proxy",
    "real.visual.description": (
        "Advisory factor-{factor} display of locked real pair {pair} on {axis}. Semantic "
        "boundaries occupy {boundary:.2f} percent of the parent grid. They carry "
        "{gradient:.2f} percent of raw gradient magnitude but {flip_risk:.2f} percent of the "
        "margin-normalized sensitivity proxy. This derivative is lossy, makes no score claim, and "
        "is not a STAC reproduction."
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
    "equiv.heading": "Preserve the answer, not the appearance",
    "equiv.body": (
        "A frozen evaluator partitions videos into cells. Inside one cell, color and texture "
        "may change without changing the receiver. A small boundary crossing can leave the cell."
    ),
    "equiv.takeaway": "A witness is one compact representative of a frozen-output equivalence class.",
    "equiv.title": "Different pictures, one machine answer",
    "equiv.desc": (
        "Three visibly different witnesses remain in one frozen-output cell, while a "
        "visually close fourth witness crosses the semantic decision wall."
    ),
    "equiv.source": "source",
    "equiv.witness_a": "less texture",
    "equiv.witness_b": "different color",
    "equiv.crossed": "wall crossed",
    "equiv.same": "same frozen output",
    "equiv.changed": "changed output",
    "control.texture_trade": "Appearance retained",
    "control.boundary_shift": "Normal boundary displacement",
    "control.boundary_shift_help": (
        "Move one analytic circular SDF zero set. The partition, swept annulus, exact area, "
        "and first variation use the same normal displacement."
    ),
    "sdf.heading": "A moving decision wall sweeps task area",
    "sdf.body": (
        "A signed-distance field stores a region by its zero set and normal direction. "
        "For a small normal displacement with no topological event, disagreement is the thin "
        "strip swept by that wall."
    ),
    "sdf.takeaway": "Class changes begin in area swept by a decision boundary; births and merges need more.",
    "sdf.title": "A class change sweeps area",
    "sdf.desc": (
        "One analytic circular signed-distance zero set moves normally. The highlighted "
        "annulus is the exact sign-disagreement set of this construction."
    ),
    "sdf.field": "signed-distance field",
    "sdf.zero": "zero level set",
    "sdf.swept": "swept disagreement",
    "sdf.exact": "exact swept area",
    "sdf.first": "small-motion integral",
    "sdf.note": "DERIVATION · before topology changes",
    "control.temperature": "Soft-max temperature",
    "morse.heading": "From a soft tie to a literal separatrix",
    "morse.body": (
        "Cooling log-sum-exp reveals a max-plus partition. Separately, the explicit potential "
        "U(x,y)=(x²−1)²+y² has two minima and one saddle: its stable manifold is literally a "
        "Morse–Smale separatrix. A class tie earns that name only when this dynamical condition holds."
    ),
    "morse.takeaway": "Topology tells the generator when a wall may move—and when a new component must be born.",
    "morse.title": "The wall is a dynamical separator",
    "morse.desc": (
        "Negative-gradient trajectories of an explicit double-well potential flow to two "
        "minima. The stable manifold of the saddle separates them."
    ),
    "morse.minimum": "minimum",
    "morse.saddle": "saddle",
    "morse.stable": "stable manifold",
    "morse.flow": "negative-gradient flow",
    "morse.soft_band": "soft 10–90% tie band",
    "morse.note": "literal system: U(x,y)=(x²−1)²+y²",
    "generator.heading": "Store causes, not paint",
    "generator.body": (
        "Equal-curvature quadratic fields form a genuine Laguerre partition. "
        "Changing one additive weight moves a shared wall without repainting every pixel."
    ),
    "control.road_weight": "Road weight",
    "control.lane_weight": "Lane weight",
    "control.movable_weight": "Movable weight",
    "generator.takeaway": (
        "The exact power-diagram construction is a clean model; learned SDF walls "
        "may curve and require residual correction."
    ),
    "laguerre.title": "Fields compete; a partition appears",
    "laguerre.desc": (
        "Equal-curvature quadratic fields form Laguerre cells. Changing a bias moves "
        "shared tie loci while the sites remain fixed."
    ),
    "laguerre.cell_a": "road",
    "laguerre.cell_b": "lane",
    "laguerre.cell_c": "movable",
    "laguerre.tie_locus": "shared tie locus",
    "laguerre.note": "argmax selects one cell at every point",
    "edge.heading": "One scene, several irreducible coordinates",
    "edge.body": (
        "v8 is not five naive independent class fields. It is a heterogeneous "
        "generator-native program: Road↔Undrivable is strictly edge-centric; the lane "
        "band, movable-site carrier, hood model, and tiny tail edges keep their smallest "
        "natural coordinates before merge, diff, and correction."
    ),
    "edge.takeaway": "Generate the horizon, lane band, moving sites, and hood once; merge, diff, correct.",
    "edge.title": "Heterogeneous generators carry the scene",
    "edge.desc": (
        "An illustrative generator-native graph: the Road/Undrivable interface is "
        "strictly edge-centric, while a lane band, movable sites, a static hood, and "
        "tiny tail edges keep their smallest natural coordinates."
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
    "control.texture": "Temporal texture aperture",
    "control.refresh": "Refresh pressure",
    "temporal_evidence.kicker": "REAL TEMPORAL EVIDENCE",
    "temporal_evidence.heading": "Transport the strategy; measure when topology refuses",
    "temporal_evidence.body": (
        "A corrected custody pass uses true consecutive raw frames 392→393. Frozen SegNet "
        "partitions and a compact three-level strategy are forward-warped; holes, collisions, "
        "and disagreement stay visible instead of being painted over."
    ),
    "temporal_evidence.scope": (
        "[ADVISORY] One downsampled Farneback image-flow construction. Not PoseNet, not "
        "STAC's DIS flow or refresh rule, and not an exact or proxy challenge score."
    ),
    "temporal_evidence.takeaway": (
        "Transport the compact cause; refresh where receiver support or topology breaks."
    ),
    "temporal_svg.badge": "TEMPORAL STRATEGY TRANSPORT · ADVISORY",
    "temporal_svg.title_1": "Transport the strategy.",
    "temporal_svg.title_2": "Measure when topology refuses.",
    "temporal_svg.dek": "One real consecutive pair. One explicit receiver map. No hidden certainty.",
    "temporal_svg.frozen": "Frozen partition · raw frame",
    "temporal_svg.frozen_argmax": "frozen SegNet argmax [ADVISORY]",
    "temporal_svg.transported": "Transported partition",
    "temporal_svg.forward": "forward nearest splat of dense Farneback flow",
    "temporal_svg.observed": "Observed partition · raw frame",
    "temporal_svg.next_argmax": "next frozen SegNet argmax [ADVISORY]",
    "temporal_svg.support": "Receiver support",
    "temporal_svg.support_legend": "green valid · amber collision · coral hole",
    "temporal_svg.rematerialized": "Rematerialized strategy",
    "temporal_svg.strategy_legend": "boundary / annulus / interior [DERIVATION]",
    "temporal_svg.label_debt": "label debt",
    "temporal_svg.boundary_debt": "boundary debt",
    "temporal_svg.holes": "holes",
    "temporal_svg.collisions": "collisions",
    "temporal_svg.footer_1": "[ADVISORY] OpenCV Farneback image-space estimate",
    "temporal_svg.footer_2": "[DERIVATION] illustrative table bank [1,4,12]",
    "temporal_svg.refusal": (
        "Not PoseNet · not STAC DIS flow · not a STAC reproduction · "
        "not an exact or proxy challenge score"
    ),
    "temporal_svg.accessible_title": "Transport the strategy; measure when topology refuses",
    "motion.heading": "A perfect partition can still starve motion",
    "motion.body": (
        "STAC transports a cached segmentation and regional strategy through dense flow, then "
        "refreshes when the propagated state becomes stale. Our geometry can carry the wall; "
        "two-frame perception still needs observable texture and occlusion repair."
    ),
    "motion.takeaway": "Transport structure, preserve an aperture, and refresh at events the warp cannot explain.",
    "motion.title": "Motion needs an aperture",
    "motion.desc": (
        "A semantic wall can be transported, but two-frame motion still needs observable "
        "texture. Image flow and the geometric transport model remain distinct."
    ),
    "motion.frame_a": "frame t",
    "motion.frame_b": "frame t+1",
    "motion.image_flow": "image-flow estimate",
    "motion.model": "geometric transport model",
    "motion.occlusion": "occlusion / refresh debt",
    "motion.screw": "screw-transport model",
    "motion.xi": "ξ=(v,ω)∈𝔰𝔢(3) → exp(ξ̂)∈SE(3)",
    "motion.note": "model coordinates are not asserted PoseNet semantics",
    "control.heading": "Let local shadow prices rank the next carrier",
    "control.body": (
        "The score law supplies exact local derivatives with respect to its declared debts. "
        "A model then estimates how one counted unit changes each debt. Their product ranks "
        "boundary, motion, generator, and correction actions at this operating point."
    ),
    "control.takeaway": "Edges are a compact basis; a declared local shadow-price proxy asks whether they earn the next byte.",
    "control.title": "Four carriers bid for the next bit",
    "control.desc": (
        "Boundary, motion, generator, and correction candidates are ranked by declared "
        "local value per counted unit. The bars are a controller construction."
    ),
    "control.boundary": "boundary",
    "control.motion": "motion",
    "control.generator": "generator",
    "control.residual": "correction",
    "control.winner": "next admitted carrier",
    "control.note": "TOY · local shadow-price construction · not a trajectory costate · not an evaluation",
    "control.scope": "[TOY] Explicit hypothetical reductions at d_pose={d_pose:.2f}; every row still owes a real carrier and receiver round trip.",
    "control.table_caption": "Declared carrier reductions and first-order score-drop proxy",
    "control.table_carrier": "carrier",
    "gpu.heading": "Stress the proxy where it is most fragile",
    "gpu.body": (
        "An optional batched Torch CUDA, MPS, or CPU sweep recomputes the declared "
        "margin-normalized sensitivity proxy on the locked real one-pair evidence across "
        "256 regularizers and checks a deterministic slice against NumPy. It measures "
        "robustness of an advisory proxy, not STAC's DCT gradient or a challenge score."
    ),
    "gpu.run": "Run 256 × 196,608 locked-evidence robustness sweep",
    "gpu.idle": (
        "[EXTERNAL] In Molab, attach a GPU from notebook specs. [ADVISORY] Without "
        "one, the same sweep reports CPU; the complete story remains available on NumPy."
    ),
    "gpu.running": "The accelerator probe is running off the reactive path; the notebook remains interactive.",
    "gpu.device": "device",
    "gpu.batch": "ε × pixels",
    "gpu.elapsed": "elapsed",
    "gpu.parity": "NumPy drift",
    "gpu.epsilon": "regularizer ε",
    "gpu.boundary_mass": "boundary mass",
    "gpu.ess": "effective pixels",
    "gpu.top_pixel": "top-pixel mass",
    "close.heading": "What the notebook establishes",
    "close.body": (
        "STAC supplies the local price; the extension proposes the compact coordinate that "
        "should carry it. Geometry generates a candidate witness, but the frozen receiver "
        "still decides what survives the round trip."
    ),
    "close.pipeline_1": "FROZEN SENSITIVITY → SHARED GEOMETRIC CARRIERS",
    "close.pipeline_2": "→ TEXTURE / POSE / REPAIR → ROUND TRIP R",
    "close.pipeline_3": "→ FROZEN SEGMENTATION + TWO-FRAME MOTION RECEIVER",
    "close.results_aria": "Three scoped notebook results",
    "close.result_paper": (
        "[TOY][DERIVATION] At matched proxy effort, outline-only can lose to uniform "
        "on the declared analytic field."
    ),
    "close.result_real": (
        "[EMPIRICAL][macOS-CPU advisory][one locked pair] The margin-normalized "
        "frozen-scorer proxy changes materially across the declared ε range."
    ),
    "close.result_proposal": (
        "[TOY] Edge carriers and local shadow prices are a generator proposal, not an "
        "exact score result."
    ),
    "close.provenance": "PROVENANCE · v7.5 shared trunk → v8 generator-native heterogeneous decomposition",
    "close.takeaway": "Compress the evidence the frozen judge needs, and let geometry regenerate the rest.",
    "sources.heading": "Sources, scope, and reproducibility",
    "sources.body": (
        "This preview makes no exact score claim. A challenge score requires exact archive "
        "custody and an unmodified evaluator transcript. Mathematical constructions are "
        "identified by their hypotheses; public baselines remain attributed to their authors."
    ),
    "sources.paper_search": (
        "Primary paper: Xiao et al. (INFOCOM 2022). This notebook implements its "
        "first-order allocation and regional-table kernel; full frozen-DNN, transform "
        "round-trip, and temporal gates remain explicitly scoped. SPARC, SA-ICM, "
        "task-aware encoder control, task rate-distortion theory, and topology-preserving "
        "compression remain explicit supporting lineage."
    ),
    "sources.geometry_intro": (
        "Foundational constellation: STAC asks where task debt lies; the Witness Machine "
        "asks which compact geometric coordinate should carry it. Level-set transport, shape "
        "adjoints, literal Morse–Smale dynamics, Lie/screw motion, Laguerre cells, max-plus "
        "algebra, and generator-native video are separately attributed below."
    ),
    "sources.primary_paper": "Primary alphaXiv paper: DNN-Driven Compressive Offloading for Edge-Assisted Semantic Video Segmentation",
    "sources.arxiv": "Primary arXiv record and paper PDF",
    "sources.geometry": "Geometry and topology source lineage",
    "sources.claims": "Claim, source, and falsifier manifest",
    "sources.challenge": "Official comma.ai video compression challenge",
    "sources.competition": "Official molab Notebook Competition #2",
    "sources.rubric": "Official judging rubric",
    "sources.form": "Official submission form",
}


_ES_US = {
    "meta.language": "Idioma",
    "meta.english": "English",
    "meta.spanish": "Español",
    "accessibility.help": (
        "Las preferencias de luz, oscuridad, visión cromática, monocromo y movimiento "
        "solo cambian la presentación; la geometría calculada permanece invariante."
    ),
    "dev.status": "Implementación interactiva del artículo · mueve las matemáticas, inspecciona la evidencia",
    "nav.score": "01 · RECEPTOR CONGELADO",
    "nav.paper": "02 · CONTROLADOR DEL ARTÍCULO",
    "nav.real": "EVIDENCIA REAL DEL EVALUADOR CONGELADO",
    "nav.equivalence": "03 · CLASE DE EQUIVALENCIA",
    "nav.boundary": "04 · LENTE DE FRONTERA",
    "nav.topology": "05 · TOPOLOGÍA",
    "nav.generator": "06 · LENTE GENERATIVA",
    "nav.shared": "07 · PORTADORES COMPARTIDOS",
    "nav.motion": "08 · APERTURA TEMPORAL",
    "nav.control": "09 · CONTROL POR PRECIO SOMBRA",
    "nav.close": "10 · LO QUE EMERGIÓ",
    "route.aria": "Ruta de cinco minutos para el jurado",
    "route.label": "RUTA DE 5 MINUTOS",
    "route.paper": "experimento del artículo",
    "route.real": "evidencia real",
    "route.wall": "mueve la frontera",
    "route.gpu": "tensiona el proxy",
    "route.close": "conclusión",
    "hero.eyebrow": "COMPRESIÓN PARA UNA MÁQUINA CONGELADA",
    "hero.title": "La máquina testigo",
    "hero.subtitle": (
        "¿Cuál es el video más pequeño que lleva a una máquina de conducción congelada "
        "a la misma decisión?"
    ),
    "hero.paper_prefix": "Implementación interactiva y extensión de",
    "hero.paper_working": (
        "DNN-Driven Compressive Offloading for Edge-Assisted Semantic Video Segmentation"
    ),
    "hero.lede": (
        "STAC pregunta dónde duele la compresión. La máquina testigo pregunta qué "
        "coordenada geométrica compacta debe cargar esa deuda. Este reto preserva una "
        "partición semántica, una señal de movimiento entre dos cuadros y un archivo contado, "
        "no cada píxel por igual. Mueve un control y observa cómo la representación sigue al receptor."
    ),
    "control.task_pressure": "Presión de la tarea",
    "control.task_pressure_help": (
        "Un ajuste conceptual: desplaza capacidad desde la apariencia uniforme hacia "
        "la evidencia que cambia al juez congelado. No es una puntuación evaluada."
    ),
    "control.loss_budget": "Presupuesto permitido de pérdida de primer orden B",
    "control.loss_budget_help": (
        "El controlador del artículo reparte este presupuesto local declarado entre coeficientes. "
        "Un B mayor permite tablas regionales más gruesas; no es pérdida medida ni puntuación."
    ),
    "control.loss_budget_prompt": (
        "[TOY][DERIVATION] EMPIEZA AQUÍ · Mueve B. Un único estado reactivo "
        "recalcula el balance del receptor, la asignación de tablas del controlador "
        "del artículo y las pujas finales de los portadores. Sigue la misma presión "
        "desde la ecuación hasta el testigo."
    ),
    "control.boundary_focus": "Razón de precisión del contorno ρ",
    "control.boundary_focus_help": (
        "Con ρ=1 la política de solo contorno es uniforme. Un ρ mayor dedica más del mismo "
        "proxy de precisión declarado al contorno semántico y menos fuera de él."
    ),
    "control.boundary_focus_prompt": (
        "[TOY][DERIVATION] Prueba ρ=1 y luego ρ=8. Con Σ1/q igualada, ¿sigue "
        "superando la política de contorno a la uniforme?"
    ),
    "hero.desc": (
        "Un cuadro de conducción atraviesa una percepción congelada. Un testigo compacto "
        "conserva evidencia donde el receptor congelado es sensible—sobre y lejos de las "
        "fronteras de decisión—sin copiar cada píxel."
    ),
    "hero.source": "video fuente",
    "hero.judge": "percepción congelada",
    "hero.witness": "testigo compacto",
    "hero.pixels": "muchos píxeles",
    "hero.evidence": "evidencia de tarea",
    "hero.boundary": "frontera de decisión",
    "hero.question": "Preserva la evidencia que consume el juez.",
    "score.heading": "Un receptor, tres deudas",
    "score.body": (
        "El desacuerdo de segmentación es lineal, el de pose entra mediante una raíz "
        "cuadrada y cada byte del archivo paga tasa. La ecuación es oficial; el dial "
        "es una pregunta de diseño, no un resultado candidato."
    ),
    "score.takeaway": "Un byte solo vale por la decisión que protege.",
    "score.title": "Tres deudas, una puntuación",
    "score.desc": (
        "El objetivo oficial combina distancia de segmentación, distancia de pose y "
        "tasa del archivo. El dial es conceptual y no informa un resultado evaluado."
    ),
    "score.seg": "segmentación",
    "score.pose": "pose",
    "score.rate": "tasa del archivo",
    "score.task": "evidencia de tarea",
    "score.bytes": "menos bytes",
    "score.balance": "equilibrio de representación",
    "score.prompt": "Mueve bytes hacia la evidencia que cambia al juez congelado.",
    "stac.title": "Pregunta dónde duele la compresión",
    "stac.desc": (
        "Un mapa de sensibilidad de la red congelada guía la cuantización regional. "
        "Un foco fuera del borde muestra que los contornos no son un oráculo completo."
    ),
    "stac.sensitivity": "sensibilidad de la red",
    "stac.allocation": "cuantización más fina",
    "stac.boundary": "frontera semántica",
    "stac.hotspot": "deuda fuera del borde",
    "stac.note": "TOY · CONSTRUCCIÓN SINTÉTICA · controlador de primer orden · no es una puntuación evaluada",
    "stac.heading": "La red, no el contorno, fija el precio",
    "stac.body": (
        "STAC diferencia una pérdida de segmentación congelada respecto a los coeficientes "
        "de transformada. Su cota de primer orden usa pasos menores donde la pérdida es más "
        "sensible y aproxima ese ideal mediante tablas regionales (§III, ec. 2 y algoritmo 1; "
        "§IV, algoritmo 2)."
    ),
    "stac.takeaway": (
        "Una frontera es una coordenada potente, pero la sensibilidad de la red también puede "
        "vivir lejos de ella. El residuo es parte del modelo, no una vergüenza."
    ),
    "stac.scope": (
        "Esta primera vista es un campo analítico de sensibilidad que pasa por el núcleo de "
        "asignación STAC implementado. Demuestra el controlador; no es una retropropagación de SegNet."
    ),
    "stac.requested": "B declarado",
    "stac.realized": "cota realizada",
    "stac.slack": "holgura restante",
    "stac.feasible": "compuerta del banco",
    "stac.feasible_yes": "PASA · ≤ B",
    "stac.feasible_no": "RECHAZA · banco inviable",
    "stac.policy_caption": "Contraejemplo con esfuerzo igualado: políticas uniforme, de solo contorno y proyectada por sensibilidad",
    "stac.policy": "política",
    "stac.policy_bound": "cota de primer orden",
    "stac.policy_precision": "Σ1/q igualada",
    "stac.policy_uniform": "uniforme",
    "stac.policy_boundary": "solo contorno",
    "stac.policy_sensitivity": "proyectada por sensibilidad",
    "stac.policy_result_loses": "En este campo y con el proxy igualado, solo contorno pierde ante uniforme.",
    "stac.policy_result_wins": "En este campo y con el proxy igualado, solo contorno supera a uniforme.",
    "stac.policy_result_ties": "Con ρ=1, solo contorno coincide exactamente con uniforme.",
    "stac.policy_scope": "TOY · mismo campo analítico y proxy de precisión recíproca igualado · no es JPEG · no es una puntuación evaluada",
    "real.heading": "Un evaluador congelado, dos significados de sensibilidad",
    "real.body": (
        "En un par real fijo de comma.ai, el gradiente de entrada del margen entre las dos "
        "clases principales es amplio. Con el regularizador declarado 1e-6, la normalización por "
        "el margen de decisión local al cuadrado concentra un proxy heurístico en la pared. El "
        "resultado depende del regularizador y de la escala de los logits; no es una probabilidad "
        "calibrada porque el gradiente de salida sumada acopla campos receptivos."
    ),
    "real.raw_metric": "masa del gradiente crudo fuera de un anillo fronterizo de 3 px",
    "real.flip_metric": "masa del proxy de sensibilidad normalizado sobre la frontera exacta",
    "real.takeaway": (
        "La misma superficie real rechaza una saliencia solo de bordes y muestra cómo una "
        "heurística declarada de margen pequeño puede concentrarse en una frontera con un "
        "regularizador fijo. No predice una probabilidad ni define un estadístico invariante de escala."
    ),
    "real.scope": (
        "[EMPIRICAL][macOS-CPU advisory][un par fijo preseleccionado] · proxy de sensibilidad "
        "de margen sumado en píxeles con epsilon=1e-6 · depende del parámetro · sin puntuación · "
        "no es sensibilidad DCT de STAC"
    ),
    "real.duality_aria": "Dos observaciones de sensibilidad fronteriza de un par real bloqueado",
    "real.visual.title": "Sensibilidad del evaluador congelado: campo y proxy de margen",
    "real.visual.description": (
        "Vista orientativa con factor {factor} del par real bloqueado {pair} en {axis}. Las "
        "fronteras semánticas ocupan {boundary:.2f} por ciento de la malla. Contienen "
        "{gradient:.2f} por ciento del gradiente crudo y {flip_risk:.2f} por ciento del proxy "
        "de sensibilidad normalizado. La derivada visual es con pérdida, no afirma puntuación y no reproduce STAC."
    ),
    "real.visual.semantic": "Portador semántico",
    "real.visual.gradient": "Magnitud del gradiente crudo",
    "real.visual.flip_risk": "Proxy de sensibilidad normalizado",
    "real.visual.semantic_aria": "Argmax semántico por moda de bloque con cobertura fronteriza iluminada",
    "real.visual.gradient_aria": "Media de bloque log-normalizada de la magnitud del gradiente crudo",
    "real.visual.flip_risk_aria": "Media de bloque log-normalizada del proxy de sensibilidad",
    "real.visual.lower": "valor visual menor",
    "real.visual.higher": "valor visual mayor",
    "real.visual.pair": "par",
    "equiv.heading": "Preserva la respuesta, no la apariencia",
    "equiv.body": (
        "Un evaluador congelado divide los videos en celdas. Dentro de una celda, el color y "
        "la textura pueden cambiar sin alterar el receptor. Un pequeño cruce de frontera puede salir de ella."
    ),
    "equiv.takeaway": "Un testigo es un representante compacto de una clase de equivalencia de salida congelada.",
    "equiv.title": "Imágenes distintas, una respuesta de la máquina",
    "equiv.desc": (
        "Tres testigos visualmente distintos permanecen en una celda de salida congelada; un "
        "cuarto, parecido, cruza la pared de decisión semántica."
    ),
    "equiv.source": "fuente",
    "equiv.witness_a": "menos textura",
    "equiv.witness_b": "otro color",
    "equiv.crossed": "pared cruzada",
    "equiv.same": "misma salida congelada",
    "equiv.changed": "salida cambiada",
    "control.texture_trade": "Apariencia conservada",
    "control.boundary_shift": "Desplazamiento normal de la frontera",
    "control.boundary_shift_help": (
        "Mueve el conjunto cero de una SDF circular analítica. La partición, el anillo barrido, "
        "el área exacta y la primera variación usan el mismo desplazamiento normal."
    ),
    "sdf.heading": "Una pared de decisión móvil barre área de tarea",
    "sdf.body": (
        "Un campo de distancia con signo almacena una región mediante su conjunto cero y su "
        "normal. Para un desplazamiento pequeño sin evento topológico, el desacuerdo es la "
        "franja delgada barrida por la pared."
    ),
    "sdf.takeaway": "Los cambios de clase comienzan en el área barrida; nacimientos y fusiones exigen más.",
    "sdf.title": "Un cambio de clase barre área",
    "sdf.desc": (
        "Un conjunto cero circular analítico se mueve en la normal. El anillo resaltado es "
        "el desacuerdo exacto de signos de esta construcción."
    ),
    "sdf.field": "campo de distancia con signo",
    "sdf.zero": "conjunto de nivel cero",
    "sdf.swept": "desacuerdo barrido",
    "sdf.exact": "área barrida exacta",
    "sdf.first": "integral de pequeño movimiento",
    "sdf.note": "DERIVATION · antes de cambios topológicos",
    "control.temperature": "Temperatura del máximo suave",
    "morse.heading": "De un empate suave a una separatriz literal",
    "morse.body": (
        "Enfriar log-sum-exp revela una partición max-plus. Por separado, el potencial explícito "
        "U(x,y)=(x²−1)²+y² tiene dos mínimos y una silla: su variedad estable es literalmente "
        "una separatriz de Morse–Smale. Un empate de clases merece ese nombre solo bajo esta condición."
    ),
    "morse.takeaway": "La topología dice cuándo mover una pared y cuándo debe nacer un componente.",
    "morse.title": "La pared es un separador dinámico",
    "morse.desc": (
        "Trayectorias de gradiente negativo de un potencial de doble pozo fluyen hacia dos "
        "mínimos. La variedad estable de la silla los separa."
    ),
    "morse.minimum": "mínimo",
    "morse.saddle": "silla",
    "morse.stable": "variedad estable",
    "morse.flow": "flujo de gradiente negativo",
    "morse.soft_band": "banda suave de empate 10–90%",
    "morse.note": "sistema literal: U(x,y)=(x²−1)²+y²",
    "generator.heading": "Guarda causas, no pintura",
    "generator.body": (
        "Campos cuadráticos con la misma curvatura forman una partición de Laguerre "
        "auténtica. Cambiar un peso aditivo mueve una pared compartida sin repintar cada píxel."
    ),
    "control.road_weight": "Peso de la calzada",
    "control.lane_weight": "Peso del carril",
    "control.movable_weight": "Peso móvil",
    "generator.takeaway": (
        "El diagrama de potencia exacto es un modelo limpio; las paredes SDF aprendidas "
        "pueden curvarse y exigir una corrección residual."
    ),
    "laguerre.title": "Los campos compiten; aparece una partición",
    "laguerre.desc": (
        "Campos cuadráticos con igual curvatura forman celdas de Laguerre. Cambiar un "
        "sesgo mueve lugares de empate compartidos mientras los sitios permanecen fijos."
    ),
    "laguerre.cell_a": "calzada",
    "laguerre.cell_b": "carril",
    "laguerre.cell_c": "móvil",
    "laguerre.tie_locus": "lugar de empate compartido",
    "laguerre.note": "argmax elige una celda en cada punto",
    "edge.heading": "Una escena, varias coordenadas irreducibles",
    "edge.body": (
        "v8 no son cinco campos de clase ingenuos e independientes. Es un programa "
        "heterogéneo y nativo de generadores: Road↔Undrivable es estrictamente céntrico "
        "en el borde; la banda de carril, los sitios móviles, el capó y las aristas de "
        "cola conservan sus coordenadas naturales mínimas antes de fusionar, diferenciar y corregir."
    ),
    "edge.takeaway": "Genera una vez horizonte, banda de carril, sitios móviles y capó; fusiona, diferencia, corrige.",
    "edge.title": "Generadores heterogéneos sostienen la escena",
    "edge.desc": (
        "Un grafo ilustrativo nativo de generadores: la interfaz Road/Undrivable es "
        "estrictamente céntrica en el borde; la banda de carril, los sitios móviles, "
        "el capó estático y las aristas de cola conservan sus coordenadas naturales mínimas."
    ),
    "edge.road": "Calzada",
    "edge.undriv": "No transitable",
    "edge.lane": "Carril",
    "edge.mycar": "Mi vehículo",
    "edge.movable": "Móvil",
    "edge.shared": "composición ilustrativa nativa de generadores",
    "edge.merge": "FUSIONA",
    "edge.diff": "DIFERENCIA",
    "edge.correct": "CORRIGE",
    "edge.pipeline": "generadores heterogéneos · bordes sin duplicar · reparación dispersa",
    "control.texture": "Apertura de textura temporal",
    "control.refresh": "Presión de refresco",
    "temporal_evidence.kicker": "EVIDENCIA TEMPORAL REAL",
    "temporal_evidence.heading": "Transporta la estrategia; mide cuando la topología se resiste",
    "temporal_evidence.body": (
        "Una revisión de custodia corregida usa cuadros crudos realmente consecutivos 392→393. "
        "Las particiones de SegNet congelada y una estrategia compacta de tres niveles se "
        "transportan hacia delante; huecos, colisiones y desacuerdo permanecen visibles."
    ),
    "temporal_evidence.scope": (
        "[ADVISORY] Una construcción de flujo de imagen Farneback submuestreada. No es PoseNet, "
        "ni el flujo DIS o la regla de refresco de STAC, ni una puntuación exacta o proxy."
    ),
    "temporal_evidence.takeaway": (
        "Transporta la causa compacta; refresca donde se rompen el soporte o la topología."
    ),
    "temporal_svg.badge": "TRANSPORTE DE ESTRATEGIA TEMPORAL · ADVISORY",
    "temporal_svg.title_1": "Transporta la estrategia.",
    "temporal_svg.title_2": "Mide cuando la topología se resiste.",
    "temporal_svg.dek": "Un par real consecutivo. Un mapa receptor explícito. Sin certeza oculta.",
    "temporal_svg.frozen": "Partición congelada · cuadro crudo",
    "temporal_svg.frozen_argmax": "argmax de SegNet congelada [ADVISORY]",
    "temporal_svg.transported": "Partición transportada",
    "temporal_svg.forward": "splat frontal vecino-cercano con flujo Farneback denso",
    "temporal_svg.observed": "Partición observada · cuadro crudo",
    "temporal_svg.next_argmax": "siguiente argmax de SegNet congelada [ADVISORY]",
    "temporal_svg.support": "Soporte del receptor",
    "temporal_svg.support_legend": "verde válido · ámbar colisión · coral hueco",
    "temporal_svg.rematerialized": "Estrategia rematerializada",
    "temporal_svg.strategy_legend": "frontera / anillo / interior [DERIVATION]",
    "temporal_svg.label_debt": "deuda de etiqueta",
    "temporal_svg.boundary_debt": "deuda de frontera",
    "temporal_svg.holes": "huecos",
    "temporal_svg.collisions": "colisiones",
    "temporal_svg.footer_1": "[ADVISORY] estimación de flujo de imagen OpenCV Farneback",
    "temporal_svg.footer_2": "[DERIVATION] banco ilustrativo [1,4,12]",
    "temporal_svg.refusal": (
        "No es PoseNet · no es flujo DIS de STAC · no reproduce STAC · "
        "no es puntuación exacta ni proxy del reto"
    ),
    "temporal_svg.accessible_title": (
        "Transporta la estrategia; mide cuando la topología se resiste"
    ),
    "motion.heading": "Una partición perfecta aún puede privar al movimiento",
    "motion.body": (
        "STAC transporta una segmentación y una estrategia regional mediante flujo denso y "
        "refresca cuando el estado propagado envejece. Nuestra geometría lleva la pared; la "
        "percepción entre cuadros aún necesita textura observable y reparación de oclusiones."
    ),
    "motion.takeaway": "Transporta estructura, conserva una apertura y refresca ante eventos que el warp no explica.",
    "motion.title": "El movimiento necesita una apertura",
    "motion.desc": (
        "Una pared semántica puede transportarse, pero el movimiento entre dos cuadros aún "
        "necesita textura observable. El flujo de imagen y el modelo geométrico se mantienen distintos."
    ),
    "motion.frame_a": "cuadro t",
    "motion.frame_b": "cuadro t+1",
    "motion.image_flow": "estimación de flujo de imagen",
    "motion.model": "modelo geométrico de transporte",
    "motion.occlusion": "deuda de oclusión / refresco",
    "motion.screw": "modelo de transporte helicoidal",
    "motion.xi": "ξ=(v,ω)∈𝔰𝔢(3) → exp(ξ̂)∈SE(3)",
    "motion.note": "las coordenadas del modelo no se atribuyen a la semántica de PoseNet",
    "control.heading": "Deja que los precios sombra locales ordenen el próximo portador",
    "control.body": (
        "La ley de puntuación da derivadas locales exactas respecto a sus deudas declaradas. "
        "Un modelo estima cuánto cambia cada deuda por unidad contada. Su producto ordena acciones "
        "de frontera, movimiento, generador y corrección en este punto operativo."
    ),
    "control.takeaway": "Los bordes son una base compacta; un proxy local declarado de precio sombra pregunta si merecen el siguiente byte.",
    "control.title": "Cuatro portadores pujan por el siguiente bit",
    "control.desc": (
        "Frontera, movimiento, generador y corrección se ordenan por valor local declarado "
        "por unidad contada. Las barras son una construcción del controlador."
    ),
    "control.boundary": "frontera",
    "control.motion": "movimiento",
    "control.generator": "generador",
    "control.residual": "corrección",
    "control.winner": "próximo portador admitido",
    "control.note": "TOY · construcción local de precio sombra · no es un coestado de trayectoria ni una evaluación",
    "control.scope": "[TOY] Reducciones hipotéticas explícitas en d_pose={d_pose:.2f}; cada fila aún debe realizar un portador y el ciclo del receptor.",
    "control.table_caption": "Reducciones declaradas por portador y proxy de caída de puntuación de primer orden",
    "control.table_carrier": "portador",
    "gpu.heading": "Tensiona el proxy donde es más frágil",
    "gpu.body": (
        "Un barrido opcional con Torch CUDA, MPS o CPU recalcula por lotes el proxy de "
        "sensibilidad normalizado por margen sobre la evidencia real bloqueada de un par "
        "para 256 regularizadores y compara una parte determinista con NumPy. Mide la "
        "robustez de un proxy consultivo, no el gradiente DCT de STAC ni una puntuación."
    ),
    "gpu.run": "Ejecutar barrido de robustez 256 × 196.608 con evidencia bloqueada",
    "gpu.idle": (
        "[EXTERNAL] En Molab, conecta una GPU desde las especificaciones del cuaderno. "
        "[ADVISORY] Sin GPU, el mismo barrido informa CPU; la historia completa sigue "
        "disponible con NumPy."
    ),
    "gpu.running": "La sonda del acelerador se ejecuta fuera de la ruta reactiva; el cuaderno sigue interactivo.",
    "gpu.device": "dispositivo",
    "gpu.batch": "ε × píxeles",
    "gpu.elapsed": "tiempo",
    "gpu.parity": "deriva frente a NumPy",
    "gpu.epsilon": "regularizador ε",
    "gpu.boundary_mass": "masa en frontera",
    "gpu.ess": "píxeles efectivos",
    "gpu.top_pixel": "masa del píxel máximo",
    "close.heading": "Lo que establece el cuaderno",
    "close.body": (
        "STAC aporta el precio local; la extensión propone la coordenada compacta que debe "
        "transportarlo. La geometría genera un testigo candidato, pero el receptor congelado "
        "aún decide qué sobrevive al ciclo completo."
    ),
    "close.pipeline_1": "SENSIBILIDAD CONGELADA → PORTADORES GEOMÉTRICOS COMPARTIDOS",
    "close.pipeline_2": "→ TEXTURA / POSE / REPARACIÓN → CICLO R",
    "close.pipeline_3": "→ RECEPTOR DE SEGMENTACIÓN Y MOVIMIENTO DE DOS CUADROS",
    "close.results_aria": "Tres resultados delimitados del cuaderno",
    "close.result_paper": (
        "[TOY][DERIVATION] Con el mismo esfuerzo proxy, el contorno puede perder frente "
        "a la política uniforme en el campo analítico declarado."
    ),
    "close.result_real": (
        "[EMPIRICAL][macOS-CPU advisory][un par bloqueado] El proxy del evaluador "
        "congelado normalizado por margen cambia materialmente en el intervalo ε declarado."
    ),
    "close.result_proposal": (
        "[TOY] Los portadores de borde y los precios sombra locales son una propuesta "
        "generativa, no un resultado de puntuación exacta."
    ),
    "close.provenance": "PROCEDENCIA · tronco compartido v7.5 → descomposición heterogénea nativa de generadores v8",
    "close.takeaway": "Comprime la evidencia que necesita el juez congelado y deja que la geometría regenere el resto.",
    "sources.heading": "Fuentes, alcance y reproducibilidad",
    "sources.body": (
        "Esta vista previa no afirma una puntuación exacta. Una puntuación del reto exige "
        "custodia exacta del archivo y una transcripción del evaluador sin modificar. "
        "Las construcciones matemáticas declaran sus hipótesis y las líneas base públicas "
        "conservan su atribución."
    ),
    "sources.paper_search": (
        "Artículo principal: Xiao et al. (INFOCOM 2022). Este cuaderno implementa su "
        "asignación de primer orden y el núcleo de tablas regionales; la DNN congelada, "
        "la ida y vuelta de transformada y las compuertas temporales quedan delimitadas. "
        "SPARC, SA-ICM, control del codificador, teoría tasa-distorsión y compresión "
        "topológica permanecen como linaje explícito."
    ),
    "sources.geometry_intro": (
        "Constelación fundacional: STAC pregunta dónde vive la deuda de tarea; la Máquina "
        "Testigo pregunta qué coordenada geométrica compacta debe transportarla. El transporte "
        "por conjuntos de nivel, los adjuntos de forma, Morse–Smale literal, Lie/tornillo, "
        "Laguerre, max-plus y el video generativo se atribuyen por separado."
    ),
    "sources.primary_paper": "Artículo principal en alphaXiv: DNN-Driven Compressive Offloading for Edge-Assisted Semantic Video Segmentation",
    "sources.arxiv": "Registro principal de arXiv y PDF",
    "sources.geometry": "Linaje de fuentes de geometría y topología",
    "sources.claims": "Manifiesto de afirmaciones, fuentes y falsadores",
    "sources.challenge": "Reto oficial de compresión de video de comma.ai",
    "sources.competition": "Competencia oficial de cuadernos molab #2",
    "sources.rubric": "Rúbrica oficial de evaluación",
    "sources.form": "Formulario oficial de entrega",
}


_CATALOGS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "en-US": MappingProxyType(_EN_US),
        "es-US": MappingProxyType(_ES_US),
    }
)


def catalog(locale: str) -> Mapping[str, str]:
    """Return a complete catalog, falling back to the English source locale."""

    selected = _CATALOGS.get(locale, _CATALOGS[DEFAULT_LOCALE])
    if selected is _CATALOGS[DEFAULT_LOCALE]:
        return selected
    return MappingProxyType({**_CATALOGS[DEFAULT_LOCALE], **selected})


def translate(locale: str, key: str) -> str:
    """Resolve one stable message ID, raising for a programming error."""

    messages = catalog(locale)
    try:
        return messages[key]
    except KeyError as exc:
        raise KeyError(f"unknown V12 message id: {key}") from exc


def missing_keys(locale: str) -> tuple[str, ...]:
    """Return source-locale keys absent from ``locale`` before fallback."""

    if locale not in _CATALOGS:
        return tuple(sorted(_CATALOGS[DEFAULT_LOCALE]))
    return tuple(sorted(set(_CATALOGS[DEFAULT_LOCALE]) - set(_CATALOGS[locale])))


def format_percent(locale: str, fraction: float, *, digits: int = 2) -> str:
    """Format a finite fraction as a compact locale-aware percentage."""

    value = float(fraction)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("fraction must be finite")
    if digits < 0 or digits > 6:
        raise ValueError("digits must be between 0 and 6")
    rendered = f"{100.0 * value:.{digits}f}"
    if locale.lower().startswith("es"):
        rendered = rendered.replace(".", ",")
    return f"{rendered}%"


__all__ = [
    "DEFAULT_LOCALE",
    "SUPPORTED_LOCALES",
    "catalog",
    "format_percent",
    "missing_keys",
    "translate",
]
