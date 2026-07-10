# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.23.0",
#   "numpy>=1.25",
# ]
# [tool.marimo.opengraph]
# title = "The Witness Machine"
# description = "An interactive, shadow-price-guided geometric extension of STAC for frozen driving perception."
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full", app_title="The Witness Machine")


@app.cell(hide_code=True)
def _(escape, format_percent, locale, messages, mo, real_display, task_witness_hero):
    hero_svg = task_witness_hero(messages, id_prefix="wm12-hero-main")
    hero_observation = real_display.metadata["observation_summary_from_parent"]
    hero_boundary_share = format_percent(
        locale, hero_observation["boundary_pixel_fraction"], digits=1
    )
    hero_mass_share = format_percent(
        locale, hero_observation["flip_risk_mass_on_boundary"], digits=1
    )
    if locale.lower().startswith("es"):
        hero_boundary_label = "de los píxeles son frontera de decisión"
        hero_mass_label = "de la masa de sensibilidad se concentra allí"
        hero_result_scope = (
            "Resultado principal, medido en un par real fijo de comma.ai a través "
            "del SegNet congelado · [macOS-CPU advisory]"
        )
        hero_result_aria = "Resultado principal medido"
    else:
        hero_boundary_label = "of pixels are decision boundary"
        hero_mass_label = "of the sensitivity mass sits there"
        hero_result_scope = (
            "Headline result, measured on one locked real comma.ai pair through "
            "the frozen SegNet · [macOS-CPU advisory]"
        )
        hero_result_aria = "Headline measured result"
    mo.Html(
        f"""
        <header class="wm-shell" lang="{escape(locale)}" aria-label="The Witness Machine interactive notebook opening">
          <section class="wm-hero-grid" aria-labelledby="wm-main-title">
            <div class="wm-copy">
              <p class="wm-dev">{escape(messages['dev.status'])}</p>
              <p class="wm-kicker">{escape(messages['hero.eyebrow'])}</p>
              <h1 class="wm-title" id="wm-main-title">{escape(messages['hero.title'])}</h1>
              <p class="wm-subtitle">{escape(messages['hero.subtitle'])}</p>
              <p class="wm-paper">
                {escape(messages['hero.paper_prefix'])}
                <a href="https://www.alphaxiv.org/abs/2203.14481" target="_blank" rel="noreferrer">
                  <strong>{escape(messages['hero.paper_working'])}</strong>
                </a>.
              </p>
              <p class="wm-body">{escape(messages['hero.lede'])}</p>
              <div class="wm-duality" aria-label="{escape(hero_result_aria)}">
                <div><strong>{escape(hero_boundary_share)}</strong><span>{escape(hero_boundary_label)}</span></div>
                <div><strong>{escape(hero_mass_share)}</strong><span>{escape(hero_mass_label)}</span></div>
              </div>
              <p class="wm-scope">{escape(hero_result_scope)}</p>
            </div>
            <div class="wm-visual">{hero_svg}</div>
          </section>
        </header>
        """
    )
    return


@app.cell(hide_code=True)
def _(
    escape,
    locale,
    messages,
    mo,
    real_evidence_full,
    real_frames_display,
    real_road_partition_svg,
):
    road_svg = real_road_partition_svg(
        real_frames_display,
        real_evidence_full.arrays,
        real_evidence_full.metadata,
        messages=messages,
        locale=locale,
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-road-heading">
          <p class="wm-kicker">{escape(messages['realsection.kicker'])}</p>
          <h2 id="wm-road-heading">{escape(messages['realsection.heading'])}</h2>
          <p class="wm-body">{escape(messages['realsection.body'])}</p>
          <div class="wm-evidence">{road_svg}</div>
          <p class="wm-scope">{escape(messages['realsection.scope'])}</p>
          <p class="wm-takeaway">{escape(messages['realsection.takeaway'])}</p>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(escape, locale, messages, mo):
    route_items = (
        ("#wm-real-heading", messages["route.real"]),
        ("#wm-paper-experiment", messages["route.paper"]),
        ("#wm-wall-experiment", messages["route.wall"]),
        ("#wm-gpu-experiment", messages["route.gpu"]),
        ("#wm-close-heading", messages["route.close"]),
    )
    route_html = '<span class="wm-route-separator" aria-hidden="true">→</span>'.join(
        f'<a href="{target}">{escape(label)}</a>' for target, label in route_items
    )
    mo.Html(
        f"""
        <nav class="wm-shell wm-route" lang="{escape(locale)}"
             aria-label="{escape(messages['route.aria'])}">
          <span class="wm-route-label">{escape(messages['route.label'])}</span>
          {route_html}
        </nav>
        """
    )
    return


@app.cell(hide_code=True)
def _(
    display_derivative_svg,
    escape,
    format_percent,
    locale,
    messages,
    mo,
    real_display,
):
    observation = real_display.metadata["observation_summary_from_parent"]
    raw_outside = format_percent(locale, observation["gradient_mass_outside_3px"])
    flip_on_boundary = format_percent(locale, observation["flip_risk_mass_on_boundary"])
    real_svg = display_derivative_svg(
        real_display,
        messages=messages,
        locale=locale,
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-real-heading">
          <p class="wm-kicker">{escape(messages['nav.real'])}</p>
          <h2 id="wm-real-heading">{escape(messages['real.heading'])}</h2>
          <p class="wm-body">{escape(messages['real.body'])}</p>
          <code class="wm-equation">s<sub>margin</sub>(x) = ‖∇<sub>x</sub>Σ<sub>p</sub>m(p)‖² / (m(x)² + ε)</code>
          <div class="wm-duality" aria-label="{escape(messages['real.duality_aria'])}">
            <div><strong>{escape(raw_outside)}</strong><span>{escape(messages['real.raw_metric'])}</span></div>
            <div><strong>{escape(flip_on_boundary)}</strong><span>{escape(messages['real.flip_metric'])}</span></div>
          </div>
          <div class="wm-evidence">{real_svg}</div>
          <p class="wm-scope">{escape(messages['real.scope'])}</p>
          <p class="wm-takeaway">{escape(messages['real.takeaway'])}</p>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(
    boundary_evolution,
    escape,
    locale,
    messages,
    mo,
    real_frames_display,
    witness_evolution_svg,
):
    evolution_svg = witness_evolution_svg(
        boundary_evolution,
        messages=messages,
        locale=locale,
        real_frame=real_frames_display.frames[1],
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-evolution-heading">
          <p class="wm-kicker">{escape(messages['evosection.kicker'])}</p>
          <h2 id="wm-evolution-heading">{escape(messages['evosection.heading'])}</h2>
          <p class="wm-body">{escape(messages['evosection.body'])}</p>
          <div class="wm-evidence">{evolution_svg}</div>
          <p class="wm-scope">{escape(messages['evosection.scope'])}</p>
          <p class="wm-takeaway">{escape(messages['evosection.takeaway'])}</p>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(
    appearance_picker,
    color_encoding_picker,
    escape,
    locale_picker,
    loss_budget,
    messages,
    mo,
    motion_picker,
):
    accessibility_controls = mo.vstack(
        [
            mo.hstack(
                [appearance_picker, color_encoding_picker, motion_picker],
                justify="start",
                gap=1.0,
            ),
            mo.Html(
                f"<p class='wm-control-note'>{escape(messages['accessibility.help'])}</p>"
            ),
        ],
        gap=0.45,
    )
    mo.vstack(
        [
            mo.hstack([locale_picker], justify="start"),
            loss_budget,
            mo.Html(
                f"<p class='wm-control-note'>{escape(messages['control.loss_budget_help'])}</p>"
            ),
            mo.Html(
                f"<p class='wm-takeaway'>{escape(messages['control.loss_budget_prompt'])}</p>"
            ),
            mo.accordion(
                {"Display & accessibility / Pantalla y accesibilidad": accessibility_controls},
                lazy=False,
            ),
        ],
        gap=0.55,
    )
    return


@app.cell(hide_code=True)
def _(escape, locale, loss_budget, messages, mo, score_law_balance):
    task_focus = 1.0 - (float(loss_budget.value) - 4.0) / 28.0
    score_svg = score_law_balance(
        task_focus,
        messages,
        id_prefix="wm12-score-main",
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-score-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.score'])}</p>
              <h2 id="wm-score-heading">{escape(messages['score.heading'])}</h2>
              <p class="wm-body">{escape(messages['score.body'])}</p>
              <p class="wm-takeaway" aria-live="polite">{escape(messages['score.takeaway'])}</p>
            </div>
            <div class="wm-visual">{score_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(escape, messages, mo):
    boundary_focus_ratio = mo.ui.slider(
        start=1.0,
        stop=8.0,
        step=0.25,
        value=4.0,
        label=messages["control.boundary_focus"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    mo.vstack(
        [
            mo.Html('<span id="wm-paper-experiment" aria-hidden="true"></span>'),
            boundary_focus_ratio,
            mo.Html(
                f"<p class='wm-control-note'>{escape(messages['control.boundary_focus_help'])}</p>"
            ),
            mo.Html(
                f"<p class='wm-scope'>{escape(messages['control.boundary_focus_prompt'])}</p>"
            ),
        ],
        gap=0.35,
    )
    return (boundary_focus_ratio,)


@app.cell(hide_code=True)
def _(
    escape,
    locale,
    messages,
    mo,
    policy_comparison,
    sensitivity,
    sensitivity_allocation_map,
    stac_strategy,
):
    realized_bound = f"{stac_strategy.total_bound:.3f}"
    requested_budget = f"{stac_strategy.requested_total_budget:.3f}"
    budget_slack = f"{stac_strategy.budget_slack:.3f}"
    if locale.lower().startswith("es"):
        realized_bound = realized_bound.replace(".", ",")
        requested_budget = requested_budget.replace(".", ",")
        budget_slack = budget_slack.replace(".", ",")
    feasibility = messages[
        "stac.feasible_yes" if stac_strategy.feasible else "stac.feasible_no"
    ]
    stac_svg = sensitivity_allocation_map(
        sensitivity.tolist(),
        stac_strategy.quant_steps[:, :, 0].tolist(),
        messages,
        id_prefix="wm12-stac-main",
    )
    policy_rows = (
        (messages["stac.policy_uniform"], policy_comparison.uniform),
        (messages["stac.policy_boundary"], policy_comparison.boundary_only),
        (
            messages["stac.policy_sensitivity"],
            policy_comparison.sensitivity_projected,
        ),
    )
    policy_table_rows = "".join(
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{row.first_order_bound:.4f}</td>"
        f"<td>{row.precision_proxy:.2f}</td>"
        "</tr>"
        for label, row in policy_rows
    )
    bound_delta = (
        policy_comparison.boundary_only.first_order_bound
        - policy_comparison.uniform.first_order_bound
    )
    if abs(bound_delta) <= 1e-12:
        policy_result = messages["stac.policy_result_ties"]
    elif bound_delta > 0.0:
        policy_result = messages["stac.policy_result_loses"]
    else:
        policy_result = messages["stac.policy_result_wins"]
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-stac-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.paper'])}</p>
              <h2 id="wm-stac-heading">{escape(messages['stac.heading'])}</h2>
              <p class="wm-body">{escape(messages['stac.body'])}</p>
              <code class="wm-equation">
                ΔQ<sub>max</sub> = Σ |g<sub>i</sub>| q<sub>i</sub> / 2 ≤ B<br/>
                q<sub>i</sub><sup>*</sup> = 2B / (M |g<sub>i</sub>|)
              </code>
              <dl class="wm-receipt" aria-live="polite">
                <div><dt>{escape(messages['stac.requested'])}</dt><dd>{escape(requested_budget)}</dd></div>
                <div><dt>{escape(messages['stac.realized'])}</dt><dd>{escape(realized_bound)}</dd></div>
                <div><dt>{escape(messages['stac.slack'])}</dt><dd>{escape(budget_slack)}</dd></div>
                <div><dt>{escape(messages['stac.feasible'])}</dt><dd>{escape(feasibility)}</dd></div>
              </dl>
              <p class="wm-scope">{escape(messages['stac.scope'])}</p>
              <table class="wm-proposal-table">
                <caption>{escape(messages['stac.policy_caption'])}</caption>
                <thead><tr>
                  <th>{escape(messages['stac.policy'])}</th>
                  <th>{escape(messages['stac.policy_bound'])}</th>
                  <th>{escape(messages['stac.policy_precision'])}</th>
                </tr></thead>
                <tbody>{policy_table_rows}</tbody>
              </table>
              <p class="wm-scope" aria-live="polite">{escape(policy_result)}</p>
              <p class="wm-control-note">{escape(messages['stac.policy_scope'])}</p>
              <p class="wm-takeaway">{escape(messages['stac.takeaway'])}</p>
            </div>
            <div class="wm-visual">{stac_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(
    escape,
    locale,
    localized_temporal_display_svg,
    messages,
    mo,
    temporal_display,
):
    temporal_svg = localized_temporal_display_svg(temporal_display, messages)
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}"
                 aria-labelledby="wm-temporal-evidence-heading">
          <div class="wm-copy">
            <p class="wm-kicker">{escape(messages['temporal_evidence.kicker'])}</p>
            <h2 id="wm-temporal-evidence-heading">{escape(messages['temporal_evidence.heading'])}</h2>
            <p class="wm-body">{escape(messages['temporal_evidence.body'])}</p>
            <p class="wm-scope">{escape(messages['temporal_evidence.scope'])}</p>
            <p class="wm-takeaway">{escape(messages['temporal_evidence.takeaway'])}</p>
          </div>
          <div class="wm-evidence">{temporal_svg}</div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(messages, mo):
    texture_trade = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.45,
        label=messages["control.texture_trade"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    texture_trade
    return (texture_trade,)


@app.cell(hide_code=True)
def _(
    escape,
    evaluator_equivalence_scene,
    locale,
    messages,
    mo,
    texture_trade,
):
    equivalence_svg = evaluator_equivalence_scene(
        float(texture_trade.value),
        messages,
        id_prefix="wm12-equivalence-main",
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-equivalence-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.equivalence'])}</p>
              <h2 id="wm-equivalence-heading">{escape(messages['equiv.heading'])}</h2>
              <p class="wm-body">{escape(messages['equiv.body'])}</p>
              <code class="wm-equation">
                [V]<sub>F</sub> = &#123; V̂ : F(R(V̂)) = F(R(V)) &#125;
              </code>
              <p class="wm-takeaway">{escape(messages['equiv.takeaway'])}</p>
            </div>
            <div class="wm-visual">{equivalence_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(messages, mo):
    boundary_shift = mo.ui.slider(
        start=-0.22,
        stop=0.22,
        step=0.01,
        value=0.10,
        label=messages["control.boundary_shift"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    mo.vstack(
        [
            mo.Html('<span id="wm-wall-experiment" aria-hidden="true"></span>'),
            boundary_shift,
            mo.md(messages["control.boundary_shift_help"]),
        ]
    )
    return (boundary_shift,)


@app.cell(hide_code=True)
def _(
    boundary_state,
    escape,
    locale,
    messages,
    mo,
    sdf_boundary_sweep,
):
    sdf_svg = sdf_boundary_sweep(
        boundary_state.displacement,
        boundary_state.exact_area_fraction,
        boundary_state.first_order_area_fraction,
        messages,
        id_prefix="wm12-sdf-main",
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-sdf-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.boundary'])}</p>
              <h2 id="wm-sdf-heading">{escape(messages['sdf.heading'])}</h2>
              <p class="wm-body">{escape(messages['sdf.body'])}</p>
              <code class="wm-equation">
                φ(x)=±dist(x,Γ), &nbsp; Γ=&#123;φ=0&#125;<br/>
                ∂<sub>t</sub>φ + v<sub>n</sub>‖∇φ‖ = 0<br/>
                |Ω Δ Ω̂| ≈ ∫<sub>Γ</sub>|δn(s)| ds
              </code>
              <p class="wm-takeaway" aria-live="polite">{escape(messages['sdf.takeaway'])}</p>
            </div>
            <div class="wm-visual">{sdf_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(messages, mo):
    soft_temperature = mo.ui.slider(
        start=0.05,
        stop=1.20,
        step=0.05,
        value=0.25,
        label=messages["control.temperature"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    soft_temperature
    return (soft_temperature,)


@app.cell(hide_code=True)
def _(
    escape,
    locale,
    messages,
    mo,
    morse_flow_scene,
    morse_traces,
    soft_temperature,
):
    morse_svg = morse_flow_scene(
        morse_traces,
        float(soft_temperature.value),
        messages,
        id_prefix="wm12-morse-main",
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-morse-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.topology'])}</p>
              <h2 id="wm-morse-heading">{escape(messages['morse.heading'])}</h2>
              <p class="wm-body">{escape(messages['morse.body'])}</p>
              <code class="wm-equation">
                τ log Σ<sub>c</sub> exp(q<sub>c</sub>/τ) → max<sub>c</sub> q<sub>c</sub><br/>
                ẋ = −∇U(x)
              </code>
              <p class="wm-takeaway">{escape(messages['morse.takeaway'])}</p>
            </div>
            <div class="wm-visual">{morse_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(messages, mo):
    road_weight = mo.ui.slider(
        start=-1.5,
        stop=1.5,
        step=0.05,
        value=0.0,
        label=messages["control.road_weight"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    lane_weight = mo.ui.slider(
        start=-1.5,
        stop=1.5,
        step=0.05,
        value=0.0,
        label=messages["control.lane_weight"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    movable_weight = mo.ui.slider(
        start=-1.5,
        stop=1.5,
        step=0.05,
        value=0.0,
        label=messages["control.movable_weight"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    mo.vstack([road_weight, lane_weight, movable_weight])
    return lane_weight, movable_weight, road_weight


@app.cell(hide_code=True)
def _(
    escape,
    laguerre_cells,
    lane_weight,
    locale,
    messages,
    mo,
    movable_weight,
    road_weight,
):
    laguerre_svg = laguerre_cells(
        (float(road_weight.value), float(lane_weight.value), float(movable_weight.value)),
        messages,
        id_prefix="wm12-laguerre-main",
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-generator-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.generator'])}</p>
              <h2 id="wm-generator-heading">{escape(messages['generator.heading'])}</h2>
              <p class="wm-body">{escape(messages['generator.body'])}</p>
              <p class="wm-takeaway" aria-live="polite">{escape(messages['generator.takeaway'])}</p>
            </div>
            <div class="wm-visual">{laguerre_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(edge_carrier_graph, escape, locale, messages, mo):
    edge_svg = edge_carrier_graph(messages, id_prefix="wm12-edge-main")
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-edge-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.shared'])}</p>
              <h2 id="wm-edge-heading">{escape(messages['edge.heading'])}</h2>
              <p class="wm-body">{escape(messages['edge.body'])}</p>
              <p class="wm-takeaway">{escape(messages['edge.takeaway'])}</p>
            </div>
            <div class="wm-visual">{edge_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(messages, mo):
    texture_amplitude = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.65,
        label=messages["control.texture"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    refresh_pressure = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.35,
        label=messages["control.refresh"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    mo.vstack([texture_amplitude, refresh_pressure])
    return refresh_pressure, texture_amplitude


@app.cell(hide_code=True)
def _(
    escape,
    locale,
    messages,
    mo,
    refresh_pressure,
    screw_points,
    temporal_aperture_scene,
    texture_amplitude,
):
    motion_svg = temporal_aperture_scene(
        float(texture_amplitude.value),
        float(refresh_pressure.value),
        messages,
        screw_points=screw_points,
        id_prefix="wm12-motion-main",
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-motion-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.motion'])}</p>
              <h2 id="wm-motion-heading">{escape(messages['motion.heading'])}</h2>
              <p class="wm-body">{escape(messages['motion.body'])}</p>
              <code class="wm-equation">
                Ẋ = ξ̂X, &nbsp; ξ ∈ 𝔰𝔢(3)<br/>
                T(θ)=exp(ξ̂θ) &nbsp; — screw transport model
              </code>
              <p class="wm-scope">{escape(messages['motion.note'])}</p>
              <p class="wm-takeaway">{escape(messages['motion.takeaway'])}</p>
            </div>
            <div class="wm-visual">{motion_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(
    controller_bids,
    controller_rows,
    escape,
    locale,
    messages,
    mo,
    shadow_price_allocation,
    toy_operating_d_pose,
):
    control_svg = shadow_price_allocation(
        controller_bids,
        messages,
        id_prefix="wm12-control-main",
    )
    carrier_labels = {
        "boundary": messages["control.boundary"],
        "motion": messages["control.motion"],
        "generator": messages["control.generator"],
        "correction": messages["control.residual"],
    }
    controller_table_rows = "".join(
        "<tr>"
        f"<td>{escape(carrier_labels[row.proposal.name])}</td>"
        f"<td>{row.proposal.seg_debt_reduction:.5f}</td>"
        f"<td>{row.proposal.pose_debt_reduction:.5f}</td>"
        f"<td>{row.proposal.added_bytes:,}</td>"
        f"<td>{row.score_drop_proxy:+.4f}</td>"
        "</tr>"
        for row in controller_rows
    )
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-control-heading">
          <div class="wm-scene-grid">
            <div class="wm-copy">
              <p class="wm-kicker">{escape(messages['nav.control'])}</p>
              <h2 id="wm-control-heading">{escape(messages['control.heading'])}</h2>
              <p class="wm-body">{escape(messages['control.body'])}</p>
              <code class="wm-equation">
                λ<sub>seg</sub>=100, &nbsp;
                λ<sub>pose</sub>=5/√(10d<sub>pose</sub>) for d<sub>pose</sub>&gt;0, &nbsp;
                λ<sub>byte</sub>=25/N<br/>
                ΔS<sub>drop</sub><sup>(1)</sup>=λ<sub>seg</sub>r<sub>seg</sub>+λ<sub>pose</sub>r<sub>pose</sub>−λ<sub>byte</sub>ΔB, &nbsp;
                𝒱=ΔS<sub>drop</sub><sup>(1)</sup>/ΔB
              </code>
              <p class="wm-scope">{escape(messages['control.scope'].format(d_pose=toy_operating_d_pose))}</p>
              <table class="wm-proposal-table">
                <caption class="sr-only">{escape(messages['control.table_caption'])}</caption>
                <thead><tr>
                  <th>{escape(messages['control.table_carrier'])}</th>
                  <th>r<sub>seg</sub></th>
                  <th>r<sub>pose</sub></th>
                  <th>ΔB</th>
                  <th>ΔS<sub>drop</sub><sup>(1)</sup></th>
                </tr></thead>
                <tbody>{controller_table_rows}</tbody>
              </table>
              <p class="wm-takeaway">{escape(messages['control.takeaway'])}</p>
            </div>
            <div class="wm-visual">{control_svg}</div>
          </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(messages, mo):
    gpu_run = mo.ui.button(
        label=messages["gpu.run"],
        value=0,
        on_click=lambda count: int(count or 0) + 1,
        kind="success",
        full_width=True,
    )
    mo.vstack(
        [
            mo.Html('<span id="wm-gpu-experiment" aria-hidden="true"></span>'),
            gpu_run,
        ],
        gap=0.0,
    )
    return (gpu_run,)


@app.cell(hide_code=True)
def _(escape, get_accelerator_state, locale, messages, mo):
    accelerator_state = get_accelerator_state()
    accelerator_receipt = accelerator_state["receipt"]
    if accelerator_state["status"] == "running":
        accelerator_detail = (
            f"<p class='wm-scope' role='status' aria-live='polite'>{escape(messages['gpu.running'])}</p>"
        )
    elif accelerator_state["status"] == "error":
        accelerator_detail = (
            "<p class='wm-scope' role='alert'>ADVISORY · accelerator probe failed closed · "
            f"{escape(str(accelerator_state['error']))}</p>"
        )
    elif accelerator_receipt is None:
        accelerator_detail = (
            f"<p class='wm-scope'>{escape(messages['gpu.idle'])}</p>"
        )
    else:
        parity = (
            "n/a"
            if accelerator_receipt.parity_max_abs is None
            else f"{accelerator_receipt.parity_max_abs:.3g}"
        )
        accelerator_detail = f"""
          <dl class="wm-receipt" aria-label="Accelerator probe receipt">
            <div><dt>{escape(messages['gpu.device'])}</dt><dd>{escape(accelerator_receipt.backend)} · {escape(accelerator_receipt.device)}</dd></div>
            <div><dt>{escape(messages['gpu.batch'])}</dt><dd>{accelerator_receipt.epsilon_count:,} × {accelerator_receipt.pixel_count:,}</dd></div>
            <div><dt>{escape(messages['gpu.elapsed'])}</dt><dd>{accelerator_receipt.elapsed_ms:.2f} ms</dd></div>
            <div><dt>{escape(messages['gpu.parity'])}</dt><dd>{escape(parity)}</dd></div>
          </dl>
          <dl class="wm-receipt" aria-label="Proxy robustness range">
            <div><dt>{escape(messages['gpu.epsilon'])}</dt><dd>{accelerator_receipt.epsilon_range[0]:.0e} → {accelerator_receipt.epsilon_range[1]:.0e}</dd></div>
            <div><dt>{escape(messages['gpu.boundary_mass'])}</dt><dd>{100.0 * accelerator_receipt.boundary_mass_range[0]:.2f}% → {100.0 * accelerator_receipt.boundary_mass_range[1]:.2f}%</dd></div>
            <div><dt>{escape(messages['gpu.ess'])}</dt><dd>{accelerator_receipt.effective_sample_size_range[0]:.1f} → {accelerator_receipt.effective_sample_size_range[1]:.1f}</dd></div>
            <div><dt>{escape(messages['gpu.top_pixel'])}</dt><dd>{100.0 * accelerator_receipt.top_pixel_mass_range[0]:.2f}% → {100.0 * accelerator_receipt.top_pixel_mass_range[1]:.2f}%</dd></div>
          </dl>
          <p class="wm-scope">{escape(accelerator_receipt.authority)} · exact_score=false</p>
        """
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-gpu-heading">
          <h2 id="wm-gpu-heading">{escape(messages['gpu.heading'])}</h2>
          <p class="wm-body">{escape(messages['gpu.body'])}</p>
          {accelerator_detail}
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(escape, locale, messages, mo):
    mo.Html(
        f"""
        <section class="wm-shell wm-section" lang="{escape(locale)}" aria-labelledby="wm-close-heading">
          <p class="wm-kicker">{escape(messages['nav.close'])}</p>
          <h2 id="wm-close-heading">{escape(messages['close.heading'])}</h2>
          <p class="wm-body">{escape(messages['close.body'])}</p>
          <code class="wm-equation">
            {escape(messages['close.pipeline_1'])}<br/>
            {escape(messages['close.pipeline_2'])}<br/>
            {escape(messages['close.pipeline_3'])}
          </code>
          <div aria-label="{escape(messages['close.results_aria'])}">
            <p class="wm-scope">{escape(messages['close.result_paper'])}</p>
            <p class="wm-scope">{escape(messages['close.result_real'])}</p>
            <p class="wm-scope">{escape(messages['close.result_proposal'])}</p>
          </div>
          <p class="wm-dev">{escape(messages['close.provenance'])}</p>
          <p class="wm-takeaway">{escape(messages['close.takeaway'])}</p>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _(escape, locale, messages, mo):
    sources = mo.Html(
        f"""
        <div class="wm-shell" lang="{escape(locale)}">
          <p class="wm-body">{escape(messages['sources.body'])}</p>
          <p class="wm-body">{escape(messages['sources.paper_search'])}</p>
          <p class="wm-body">{escape(messages['sources.geometry_intro'])}</p>
          <ul class="wm-source-links">
            <li><a href="https://www.alphaxiv.org/abs/2203.14481" target="_blank" rel="noreferrer">{escape(messages['sources.primary_paper'])}</a></li>
            <li><a href="https://arxiv.org/abs/2203.14481" target="_blank" rel="noreferrer">{escape(messages['sources.arxiv'])}</a></li>
            <li><a href="https://doi.org/10.1016/0021-9991(88)90002-2" target="_blank" rel="noreferrer">Osher &amp; Sethian — Fronts Propagating with Curvature-Dependent Speed</a></li>
            <li><a href="https://doi.org/10.1016/j.jcp.2003.09.032" target="_blank" rel="noreferrer">Allaire, Jouve &amp; Toader — Structural Optimization Using Sensitivity Analysis and a Level-Set Method</a></li>
            <li><a href="https://doi.org/10.2307/1970311" target="_blank" rel="noreferrer">Smale — On Gradient Dynamical Systems</a></li>
            <li><a href="https://arxiv.org/abs/2409.17346" target="_blank" rel="noreferrer">Li et al. — Preserving Discrete Morse–Smale Complexes in Error-Bounded Lossy Compression</a></li>
            <li><a href="https://arxiv.org/abs/1812.01537" target="_blank" rel="noreferrer">Solà, Deray &amp; Atchuthan — A Micro Lie Theory for State Estimation in Robotics</a></li>
            <li><a href="https://doi.org/10.1137/0216006" target="_blank" rel="noreferrer">Aurenhammer — Power Diagrams: Properties, Algorithms and Applications</a></li>
            <li><a href="https://arxiv.org/abs/math/0308254" target="_blank" rel="noreferrer">Develin &amp; Sturmfels — Tropical Convexity</a></li>
            <li><a href="https://arxiv.org/abs/2110.13903" target="_blank" rel="noreferrer">Chen et al. — NeRV: Neural Representations for Videos</a></li>
            <li><a href="https://arxiv.org/abs/2112.11312" target="_blank" rel="noreferrer">Zhang et al. — Implicit Neural Video Compression</a></li>
            <li><a href="https://github.com/commaai/comma_video_compression_challenge" target="_blank" rel="noreferrer">{escape(messages['sources.challenge'])}</a></li>
            <li><a href="https://marimo.io/pages/events/notebook-competition-2" target="_blank" rel="noreferrer">{escape(messages['sources.competition'])}</a></li>
            <li><a href="https://marimo.io/blog/reintroducing-molab" target="_blank" rel="noreferrer">Molab GPU compute plane</a></li>
            <li><a href="https://docs.google.com/spreadsheets/d/131Kz9fz99oEE_xFIYOW6XM89iFfWF84Csel-97pdxwk/edit?gid=620363524" target="_blank" rel="noreferrer">{escape(messages['sources.rubric'])}</a></li>
            <li><a href="https://form.jotform.com/261457442234152" target="_blank" rel="noreferrer">{escape(messages['sources.form'])}</a></li>
          </ul>
        </div>
        """
    )
    mo.accordion({messages["sources.heading"]: sources}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        r"""
        <style>
          :root {
            color-scheme: light;
            --wm-bg: #f7f7f3;
            --wm-panel: #ffffff;
            --wm-panel-2: #cbd2d8;
            --wm-text: #17191c;
            --wm-muted: #4f5963;
            --wm-cyan: #006f8a;
            --wm-gold: #8a5a00;
            --wm-coral: #b73a37;
            --wm-mint: #527a00;
            --wm-violet: #5d4a9e;
            --wm-dark-ink: #101214;
            --wm-rule: color-mix(in srgb, var(--wm-muted) 26%, transparent);
          }

          @media (prefers-color-scheme: dark) {
            :root {
              color-scheme: dark;
              --wm-bg: #0b0d10;
              --wm-panel: #11161c;
              --wm-panel-2: #2a333d;
              --wm-text: #f5f7f2;
              --wm-muted: #b7c0c9;
              --wm-cyan: #8bd5ff;
              --wm-gold: #ffd166;
              --wm-coral: #ff746c;
              --wm-mint: #b8f26a;
              --wm-violet: #b9a7ff;
              --wm-dark-ink: #0b0d10;
            }
          }

          body, marimo-app, marimo-app .app {
            background: var(--wm-bg) !important;
          }

          .wm-shell {
            width: min(100%, 1220px);
            margin-inline: auto;
            padding-inline: clamp(1rem, 3vw, 2.5rem);
            color: var(--wm-text);
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, ui-sans-serif, system-ui, sans-serif;
          }

          .wm-hero-grid,
          .wm-scene-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(18rem, 1fr);
            align-items: center;
            gap: clamp(1.2rem, 4vw, 4rem);
            min-width: 0;
          }

          .wm-scene-grid {
            grid-template-columns: minmax(0, 0.78fr) minmax(20rem, 1.22fr);
          }

          .wm-copy,
          .wm-visual {
            min-width: 0;
          }

          .wm-visual > svg {
            width: 100%;
            max-width: 100%;
            height: auto;
          }

          .wm-proposal-table {
            width: 100%;
            margin-top: 1rem;
            border-collapse: collapse;
            font-variant-numeric: tabular-nums;
            font-size: clamp(0.7rem, 1.3vw, 0.84rem);
          }

          .wm-proposal-table th,
          .wm-proposal-table td {
            padding: 0.48rem 0.38rem;
            border-bottom: 1px solid color-mix(in srgb, var(--wm-muted) 24%, transparent);
            text-align: right;
          }

          .wm-proposal-table th:first-child,
          .wm-proposal-table td:first-child {
            text-align: left;
          }

          .wm-proposal-table caption {
            margin-block-end: 0.55rem;
            color: var(--wm-muted);
            font-size: 0.82rem;
            line-height: 1.4;
            text-align: start;
          }


          .wm-kicker {
            margin: 0 0 0.75rem;
            color: var(--wm-mint);
            font-family: "SFMono-Regular", Consolas, ui-monospace, monospace;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.16em;
          }

          .wm-title {
            margin: 0;
            max-width: 12ch;
            color: var(--wm-text);
            font-size: clamp(3rem, 6.8vw, 5.9rem);
            line-height: 0.94;
            letter-spacing: -0.065em;
          }

          .wm-subtitle {
            max-width: 27ch;
            margin-block: 1.25rem 1rem;
            color: var(--wm-text);
            font-size: clamp(1.25rem, 2.3vw, 2rem);
            font-weight: 620;
            line-height: 1.2;
            letter-spacing: -0.025em;
          }

          .wm-paper,
          .wm-body {
            max-width: 64ch;
            color: var(--wm-muted);
            font-size: clamp(1rem, 1.45vw, 1.16rem);
            line-height: 1.62;
          }

          .wm-paper {
            padding-inline-start: 0.9rem;
            border-inline-start: 3px solid var(--wm-gold);
          }

          .wm-paper a,
          .wm-inline-link {
            color: var(--wm-text);
            text-decoration-color: var(--wm-gold);
            text-decoration-thickness: 2px;
            text-underline-offset: 0.22em;
          }

          .wm-dev {
            display: block;
            width: fit-content;
            margin-block: 0.5rem 1.5rem;
            padding-block-start: 0.55rem;
            border-block-start: 1px solid var(--wm-gold);
            color: var(--wm-muted);
            font-family: "SFMono-Regular", Consolas, ui-monospace, monospace;
            font-size: 0.72rem;
            letter-spacing: 0.04em;
            line-height: 1.35;
          }

          .wm-section {
            margin-block: clamp(2.75rem, 5vw, 4.75rem);
            scroll-margin-top: 2rem;
          }

          .wm-section h2 {
            margin: 0 0 1rem;
            color: var(--wm-text);
            font-size: clamp(2rem, 4vw, 3.7rem);
            line-height: 1;
            letter-spacing: -0.045em;
          }

          .wm-takeaway {
            margin-block-start: 1.25rem;
            padding: 0.85rem 0 0.15rem 1rem;
            border-inline-start: 2px solid var(--wm-mint);
            background: transparent;
            color: var(--wm-text);
            font-size: clamp(1.02rem, 1.7vw, 1.28rem);
            font-weight: 650;
            line-height: 1.45;
          }

          .wm-control-note {
            margin-block-start: 0.6rem;
            color: var(--wm-muted);
            font-size: 0.92rem;
            line-height: 1.45;
          }

          .wm-scope {
            margin-block-start: 1rem;
            padding: 0.2rem 0 0.2rem 0.85rem;
            border-inline-start: 1px solid var(--wm-coral);
            color: var(--wm-muted);
            font-size: 0.9rem;
            line-height: 1.5;
          }

          .wm-receipt {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin-block-start: 1.1rem;
          }

          .wm-receipt > div {
            min-width: 0;
            padding-block: 0.65rem;
            border-block-start: 2px solid var(--wm-panel-2);
          }

          .wm-receipt dt {
            color: var(--wm-muted);
            font-size: 0.76rem;
            line-height: 1.25;
          }

          .wm-receipt dd {
            margin: 0.3rem 0 0;
            overflow-wrap: anywhere;
            color: var(--wm-text);
            font-family: "SFMono-Regular", Consolas, ui-monospace, monospace;
            font-size: 0.9rem;
          }

          .wm-evidence {
            margin-block-start: 1.4rem;
          }

          .wm-evidence > svg {
            display: block;
            width: 100%;
            max-width: 100%;
            height: auto;
          }

          .wm-duality {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-block: 1.25rem;
          }

          .wm-duality > div {
            min-width: 0;
            padding-inline-start: 1rem;
            border-inline-start: 3px solid var(--wm-cyan);
          }

          .wm-duality > div:last-child {
            border-inline-start-color: var(--wm-gold);
          }

          .wm-duality strong {
            display: block;
            color: var(--wm-text);
            font-size: clamp(1.5rem, 3vw, 2.5rem);
            line-height: 1;
          }

          .wm-duality span {
            display: block;
            margin-block-start: 0.45rem;
            color: var(--wm-muted);
            line-height: 1.35;
          }

          .wm-equation {
            display: block;
            max-width: 100%;
            margin-block: 1.15rem;
            padding: 0.85rem 0;
            overflow-wrap: anywhere;
            border-block: 1px solid var(--wm-rule);
            background: transparent;
            color: var(--wm-text);
            font-family: "SFMono-Regular", Consolas, ui-monospace, monospace;
            font-size: clamp(0.84rem, 1.4vw, 1.05rem);
            line-height: 1.6;
          }

          :where(a, button, input, select, textarea, [tabindex]):focus-visible {
            outline: 3px solid var(--wm-gold) !important;
            outline-offset: 4px !important;
          }

          .wm-source-links {
            display: grid;
            gap: 0.7rem;
            padding: 0;
            list-style: none;
          }

          .wm-source-links a {
            color: var(--wm-cyan);
            text-underline-offset: 0.2em;
          }

          .wm-route {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 0.55rem 0.8rem;
            margin-block: 1.4rem 0.25rem;
            padding-block: 0.8rem;
            border-block: 1px solid var(--wm-rule);
            font-size: 0.86rem;
            line-height: 1.45;
          }

          .wm-route-label {
            color: var(--wm-muted);
            font-family: "SFMono-Regular", Consolas, ui-monospace, monospace;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
          }

          .wm-route a {
            color: var(--wm-text);
            font-weight: 650;
            text-decoration-color: var(--wm-gold);
            text-underline-offset: 0.2em;
          }

          .wm-route-separator {
            color: var(--wm-muted);
          }

          @media (max-width: 900px) {
            .wm-shell {
              padding-inline: clamp(0.9rem, 4vw, 1.5rem);
            }

            .wm-hero-grid,
            .wm-scene-grid {
              grid-template-columns: minmax(0, 1fr);
              gap: 1.2rem;
            }

            .wm-title {
              max-width: 10ch;
              font-size: clamp(2.9rem, 15vw, 5rem);
            }

            .wm-subtitle {
              max-width: 31ch;
            }

            .wm-visual {
              width: min(100%, 35rem);
              margin-inline: auto;
            }

            .wm-section {
              margin-block: 3.25rem;
            }

            .wm-receipt {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .wm-duality {
              grid-template-columns: minmax(0, 1fr);
            }

            .wm-proposal-table {
              font-size: clamp(0.66rem, 2.7vw, 0.82rem);
            }

            .wm-proposal-table th,
            .wm-proposal-table td {
              padding-inline: 0.18rem;
            }
          }

          @media (max-width: 430px) {
            .wm-receipt {
              grid-template-columns: minmax(0, 1fr);
            }

            .wm-section h2 {
              font-size: clamp(2rem, 12vw, 3.1rem);
            }
          }

          @media (prefers-contrast: more) {
            :root {
              --wm-muted: var(--wm-text) !important;
              --wm-panel-2: var(--wm-text) !important;
              --wm-rule: var(--wm-text) !important;
            }
          }

          @media (forced-colors: active) {
            .wm-kicker,
            .wm-dev,
            .wm-paper,
            .wm-takeaway,
            .wm-scope {
              border-color: CanvasText;
              color: CanvasText;
            }
          }

          @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
              scroll-behavior: auto !important;
              animation-duration: 0.001ms !important;
              animation-iteration-count: 1 !important;
              transition-duration: 0.001ms !important;
            }
          }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _(appearance_picker, color_encoding_picker, mo, motion_picker):
    light_tokens = {
        "scheme": "light",
        "bg": "#f7f7f3",
        "panel": "#ffffff",
        "panel_2": "#cbd2d8",
        "text": "#17191c",
        "muted": "#4f5963",
        "cyan": "#006f8a",
        "gold": "#8a5a00",
        "coral": "#b73a37",
        "mint": "#527a00",
        "violet": "#5d4a9e",
        "ink": "#101214",
    }
    dark_tokens = {
        "scheme": "dark",
        "bg": "#0b0d10",
        "panel": "#11161c",
        "panel_2": "#2a333d",
        "text": "#f5f7f2",
        "muted": "#b7c0c9",
        "cyan": "#8bd5ff",
        "gold": "#ffd166",
        "coral": "#ff746c",
        "mint": "#b8f26a",
        "violet": "#b9a7ff",
        "ink": "#0b0d10",
    }
    selected_appearance = str(appearance_picker.value or "light")
    tokens = None if selected_appearance == "auto" else (
        light_tokens if selected_appearance == "light" else dark_tokens
    )
    selected_encoding = str(color_encoding_picker.value or "semantic")
    auto_dark_color_tokens = None
    if selected_encoding == "cvd":
        cvd_light = {
            "cyan": "#005a8d",
            "gold": "#704900",
            "coral": "#a53a00",
            "mint": "#006b50",
            "violet": "#713d78",
        }
        cvd_dark = {
            "cyan": "#56b4e9",
            "gold": "#f0c15c",
            "coral": "#ff8a5b",
            "mint": "#65d2b1",
            "violet": "#e58fd8",
        }
        if selected_appearance == "auto":
            color_tokens = cvd_light
            auto_dark_color_tokens = cvd_dark
        else:
            color_tokens = cvd_light if selected_appearance == "light" else cvd_dark
    elif selected_encoding == "monochrome":
        monochrome_light = {
            "cyan": "#17191c",
            "gold": "#59616a",
            "coral": "#17191c",
            "mint": "#59616a",
            "violet": "#17191c",
        }
        monochrome_dark = {
            "cyan": "#f5f7f2",
            "gold": "#b7c0c9",
            "coral": "#f5f7f2",
            "mint": "#b7c0c9",
            "violet": "#f5f7f2",
        }
        if selected_appearance == "auto":
            color_tokens = monochrome_light
            auto_dark_color_tokens = monochrome_dark
        else:
            color_tokens = (
                monochrome_light if selected_appearance == "light" else monochrome_dark
            )
    else:
        color_tokens = None
    selected_motion = str(motion_picker.value or "focus")
    base_animation_state = "running" if selected_motion == "continuous" else "paused"
    interaction_animation_state = "paused" if selected_motion == "still" else "running"
    if tokens is None:
        token_rules = ""
    else:
        token_rules = f"""
            color-scheme: {tokens['scheme']};
            --wm-bg: {tokens['bg']};
            --wm-panel: {tokens['panel']};
            --wm-panel-2: {tokens['panel_2']};
            --wm-text: {tokens['text']};
            --wm-muted: {tokens['muted']};
            --wm-cyan: {tokens['cyan']};
            --wm-gold: {tokens['gold']};
            --wm-coral: {tokens['coral']};
            --wm-mint: {tokens['mint']};
            --wm-violet: {tokens['violet']};
            --wm-dark-ink: {tokens['ink']};
        """
    if color_tokens is None:
        color_rules = ""
    else:
        color_rules = f"""
            --wm-cyan: {color_tokens['cyan']};
            --wm-gold: {color_tokens['gold']};
            --wm-coral: {color_tokens['coral']};
            --wm-mint: {color_tokens['mint']};
            --wm-violet: {color_tokens['violet']};
        """
    if auto_dark_color_tokens is None:
        auto_dark_color_rules = ""
    else:
        auto_dark_color_rules = f"""
          @media (prefers-color-scheme: dark) {{
            :root {{
              --wm-cyan: {auto_dark_color_tokens['cyan']};
              --wm-gold: {auto_dark_color_tokens['gold']};
              --wm-coral: {auto_dark_color_tokens['coral']};
              --wm-mint: {auto_dark_color_tokens['mint']};
              --wm-violet: {auto_dark_color_tokens['violet']};
            }}
          }}
        """
    appearance_style = f"""
        <style data-wm-explicit-appearance="{selected_appearance}"
               data-wm-color-encoding="{selected_encoding}"
               data-wm-motion="{selected_motion}">
          :root {{
            {token_rules}
            {color_rules}
            --wm-animation-state: {base_animation_state};
            --wm-interaction-animation-state: {interaction_animation_state};
            --wm-rule: color-mix(in srgb, var(--wm-muted) 26%, transparent);
          }}
          {auto_dark_color_rules}
        </style>
    """
    mo.Html(appearance_style)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from html import escape
    import hashlib
    import importlib
    import os
    from pathlib import Path, PurePosixPath
    import shutil
    import sys
    import tempfile
    import urllib.request
    import zipfile

    BUNDLE_URL = "https://github.com/adpena/witness-machine/releases/download/v1.3.0/witness_machine_v12_molab_bundle.zip"
    BUNDLE_BYTES = 6_251_044
    BUNDLE_SHA256 = "f9ed972157d70e7fda63431dcf7cb422a24cec2ded0ce51b860921d2240835c8"
    BUNDLE_TOP_LEVEL = "witness_machine_v12"
    BUNDLE_MARKER = ".bundle-sha256"
    REQUIRED_ROOT_FILES = (
        "src/molab_witness_machine/__init__.py",
        "src/molab_witness_machine/score_law.py",
        "src/molab_witness_machine/v12_copy.py",
        "src/molab_witness_machine/v12_control.py",
        "src/molab_witness_machine/v12_geometry.py",
        "src/molab_witness_machine/v12_gpu.py",
        "src/molab_witness_machine/v12_policy_compare.py",
        "src/molab_witness_machine/v12_real_evidence.py",
        "src/molab_witness_machine/v12_real_frames.py",
        "src/molab_witness_machine/v12_stac.py",
        "src/molab_witness_machine/v12_temporal.py",
        "src/molab_witness_machine/v12_visuals.py",
        "artifacts/v12_public/v12_temporal_transport_display.svg",
        "artifacts/v12_public/v12_temporal_transport_display.manifest.json",
        "artifacts/v12_public/v12_real_frozen_scorer_display.npz",
        "artifacts/v12_public/v12_real_frozen_scorer_display.manifest.json",
        "artifacts/v12_public/v12_real_frozen_scorer_evidence.npz",
        "artifacts/v12_public/v12_real_frozen_scorer_evidence.manifest.json",
        "artifacts/v12_public/v12_real_frame_display.npz",
        "artifacts/v12_public/v12_real_frame_display.manifest.json",
        "artifacts/v12_public/v12_boundary_evolution.npz",
        "artifacts/v12_public/v12_boundary_evolution.manifest.json",
    )

    def _fail_bundle(message: str) -> None:
        raise RuntimeError(f"Molab runtime bootstrap failed for {BUNDLE_URL}: {message}")

    def _missing_required_root_files(root: Path) -> tuple[str, ...]:
        return tuple(
            relative_path
            for relative_path in REQUIRED_ROOT_FILES
            if not (root / relative_path).is_file()
        )

    def _is_valid_root(root: Path) -> bool:
        return not _missing_required_root_files(root)

    def _candidate_roots(notebook_file: str | Path) -> tuple[Path, ...]:
        seen: set[Path] = set()
        candidates: list[Path] = []
        for start in (Path(notebook_file).resolve().parent, Path.cwd().resolve()):
            for candidate in (start, *start.parents):
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
        return tuple(candidates)

    def _cache_base() -> Path:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            return Path(xdg_cache_home).expanduser().resolve() / "witness-machine"
        return Path.home().expanduser().resolve() / ".cache" / "witness-machine"

    def _cache_slot() -> Path:
        return _cache_base() / BUNDLE_SHA256

    def _cache_root(cache_slot: Path) -> Path:
        return cache_slot / BUNDLE_TOP_LEVEL

    def _cache_marker_path(cache_slot: Path) -> Path:
        return cache_slot / BUNDLE_MARKER

    def _is_valid_cache_slot(cache_slot: Path) -> bool:
        marker_path = _cache_marker_path(cache_slot)
        try:
            marker_text = marker_path.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        return marker_text == BUNDLE_SHA256 and _is_valid_root(_cache_root(cache_slot))

    def _remove_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _download_bundle_bytes(*, urlopen=None) -> bytes:
        opener = urllib.request.urlopen if urlopen is None else urlopen
        try:
            with opener(BUNDLE_URL, timeout=30) as response:
                payload = response.read(BUNDLE_BYTES + 1)
        except OSError as exc:
            _fail_bundle(f"download failed: {exc}")
        actual_bytes = len(payload)
        if actual_bytes != BUNDLE_BYTES:
            actual_label = str(actual_bytes) if actual_bytes <= BUNDLE_BYTES else f"at least {actual_bytes}"
            _fail_bundle(f"expected {BUNDLE_BYTES} bytes, got {actual_label}")
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != BUNDLE_SHA256:
            _fail_bundle(f"expected SHA-256 {BUNDLE_SHA256}, got {actual_sha256}")
        return payload

    def _validate_zip_members(members: list[zipfile.ZipInfo]) -> None:
        if not members:
            _fail_bundle("archive is empty")
        for info in members:
            member_path = PurePosixPath(info.filename)
            raw_parts = info.filename.split("/")
            if raw_parts and raw_parts[-1] == "":
                raw_parts = raw_parts[:-1]
            if member_path.is_absolute():
                _fail_bundle(f"unsafe ZIP member {info.filename!r}")
            if not raw_parts or any(part in ("", ".", "..") for part in raw_parts):
                _fail_bundle(f"unsafe ZIP member {info.filename!r}")
            if raw_parts[0] != BUNDLE_TOP_LEVEL:
                _fail_bundle(
                    f"expected top-level directory {BUNDLE_TOP_LEVEL!r}, got {raw_parts[0]!r}"
                )

    def _materialize_cache_root(*, urlopen=None) -> Path:
        cache_slot = _cache_slot()
        if _is_valid_cache_slot(cache_slot):
            return _cache_root(cache_slot)
        if cache_slot.exists():
            _remove_path(cache_slot)
        cache_parent = cache_slot.parent
        cache_parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix="witness-machine-", dir=cache_parent))
        staging_slot = staging_dir / BUNDLE_SHA256
        staging_slot.mkdir()
        try:
            archive_path = staging_dir / "bundle.zip"
            archive_path.write_bytes(_download_bundle_bytes(urlopen=urlopen))
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                _validate_zip_members(members)
                archive.extractall(staging_slot)
            _cache_marker_path(staging_slot).write_text(f"{BUNDLE_SHA256}\n", encoding="utf-8")
            missing_files = _missing_required_root_files(_cache_root(staging_slot))
            if missing_files:
                _fail_bundle(f"extracted bundle is missing {missing_files[0]}")
            try:
                staging_slot.replace(cache_slot)
            except FileExistsError:
                if not _is_valid_cache_slot(cache_slot):
                    _remove_path(cache_slot)
                    staging_slot.replace(cache_slot)
            if not _is_valid_cache_slot(cache_slot):
                missing_files = _missing_required_root_files(_cache_root(cache_slot))
                if missing_files:
                    _fail_bundle(f"cache is missing {missing_files[0]}")
                _fail_bundle("cache marker does not match the sealed bundle SHA-256")
            return _cache_root(cache_slot)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _resolve_project_root(notebook_file: str | Path, *, urlopen=None) -> Path:
        for candidate in _candidate_roots(notebook_file):
            if _is_valid_root(candidate):
                return candidate
        return _materialize_cache_root(urlopen=urlopen)

    project_root = _resolve_project_root(__file__)
    source_root = project_root / "src"
    if not sys.path or sys.path[0] != str(source_root):
        if str(source_root) in sys.path:
            sys.path.remove(str(source_root))
        sys.path.insert(0, str(source_root))

    import numpy as np

    copy_module = importlib.import_module("molab_witness_machine.v12_copy")
    control_module = importlib.import_module("molab_witness_machine.v12_control")
    geometry_module = importlib.import_module("molab_witness_machine.v12_geometry")
    policy_compare_module = importlib.import_module("molab_witness_machine.v12_policy_compare")
    real_evidence_module = importlib.import_module("molab_witness_machine.v12_real_evidence")
    real_frames_module = importlib.import_module("molab_witness_machine.v12_real_frames")
    stac_module = importlib.import_module("molab_witness_machine.v12_stac")
    temporal_module = importlib.import_module("molab_witness_machine.v12_temporal")
    visuals_module = importlib.import_module("molab_witness_machine.v12_visuals")

    boundary_sweep = geometry_module.boundary_sweep
    catalog = copy_module.catalog
    compare_matched_precision_policies = (
        policy_compare_module.compare_matched_precision_policies
    )
    display_derivative_svg = real_evidence_module.display_derivative_svg
    edge_carrier_graph = visuals_module.edge_carrier_graph
    evaluator_equivalence_scene = visuals_module.evaluator_equivalence_scene
    format_percent = copy_module.format_percent
    integrate_negative_gradient = geometry_module.integrate_negative_gradient
    laguerre_cells = visuals_module.laguerre_cells
    load_display_derivative = real_evidence_module.load_display_derivative
    load_temporal_display = temporal_module.load_temporal_display
    localized_temporal_display_svg = temporal_module.localized_temporal_display_svg
    morse_flow_scene = visuals_module.morse_flow_scene
    notebook_toy_proposals = control_module.notebook_toy_proposals
    score_law_balance = visuals_module.score_law_balance
    select_regional_strategy = stac_module.select_regional_strategy
    screw_trajectory = geometry_module.screw_trajectory
    sdf_boundary_sweep = visuals_module.sdf_boundary_sweep
    sensitivity_allocation_map = visuals_module.sensitivity_allocation_map
    shadow_price_allocation = visuals_module.shadow_price_allocation
    task_witness_hero = visuals_module.task_witness_hero
    temporal_aperture_scene = visuals_module.temporal_aperture_scene
    value_carrier_proposals = control_module.value_carrier_proposals

    temporal_display = load_temporal_display(
        project_root / "artifacts/v12_public/v12_temporal_transport_display.svg",
        project_root / "artifacts/v12_public/v12_temporal_transport_display.manifest.json",
    )

    real_road_partition_svg = real_frames_module.real_road_partition_svg
    witness_evolution_svg = real_frames_module.witness_evolution_svg
    real_frames_display = real_frames_module.load_real_frames(
        project_root / "artifacts/v12_public/v12_real_frame_display.npz",
        project_root / "artifacts/v12_public/v12_real_frame_display.manifest.json",
    )
    boundary_evolution = real_frames_module.load_boundary_evolution(
        project_root / "artifacts/v12_public/v12_boundary_evolution.npz",
        project_root / "artifacts/v12_public/v12_boundary_evolution.manifest.json",
    )
    real_evidence_full = real_evidence_module.load_locked_evidence(
        project_root / "artifacts/v12_public/v12_real_frozen_scorer_evidence.npz",
        project_root / "artifacts/v12_public/v12_real_frozen_scorer_evidence.manifest.json",
    )

    return (
        boundary_evolution,
        boundary_sweep,
        catalog,
        compare_matched_precision_policies,
        display_derivative_svg,
        edge_carrier_graph,
        evaluator_equivalence_scene,
        escape,
        format_percent,
        integrate_negative_gradient,
        laguerre_cells,
        load_display_derivative,
        load_temporal_display,
        localized_temporal_display_svg,
        morse_flow_scene,
        notebook_toy_proposals,
        np,
        project_root,
        real_evidence_full,
        real_frames_display,
        real_road_partition_svg,
        score_law_balance,
        select_regional_strategy,
        screw_trajectory,
        sdf_boundary_sweep,
        sensitivity_allocation_map,
        shadow_price_allocation,
        task_witness_hero,
        temporal_aperture_scene,
        temporal_display,
        value_carrier_proposals,
        witness_evolution_svg,
    )


@app.cell(hide_code=True)
def _(mo):
    locale_picker = mo.ui.dropdown(
        options={"English": "en-US", "Español": "es-US"},
        value="English",
        label="Language / Idioma",
        full_width=False,
    )
    return (locale_picker,)


@app.cell(hide_code=True)
def _(mo):
    appearance_picker = mo.ui.dropdown(
        options={
            "Light · Claro": "light",
            "Dark · Oscuro": "dark",
            "Auto": "auto",
        },
        value="Light · Claro",
        label="Appearance / Apariencia",
        full_width=False,
    )
    return (appearance_picker,)


@app.cell(hide_code=True)
def _(mo):
    color_encoding_picker = mo.ui.dropdown(
        options={
            "Semantic color · Color semántico": "semantic",
            "Color-vision safe · Visión cromática": "cvd",
            "Monochrome · Monocromo": "monochrome",
        },
        value="Semantic color · Color semántico",
        label="Color encoding / Codificación",
        full_width=False,
    )
    return (color_encoding_picker,)


@app.cell(hide_code=True)
def _(mo):
    motion_picker = mo.ui.dropdown(
        options={
            "Wake on focus · Activa al enfocar": "focus",
            "Still · Estático": "still",
            "Continuous · Continuo": "continuous",
        },
        value="Wake on focus · Activa al enfocar",
        label="Motion / Movimiento",
        full_width=False,
    )
    return (motion_picker,)


@app.cell(hide_code=True)
def _(catalog, locale_picker):
    locale = str(locale_picker.value or "en-US")
    messages = catalog(locale)
    return locale, messages


@app.cell(hide_code=True)
def _(messages, mo):
    loss_budget = mo.ui.slider(
        start=4.0,
        stop=32.0,
        step=1.0,
        value=12.0,
        label=messages["control.loss_budget"],
        show_value=True,
        debounce=False,
        full_width=True,
    )
    return (loss_budget,)


@app.cell(hide_code=True)
def _(
    boundary_focus_ratio,
    compare_matched_precision_policies,
    loss_budget,
    np,
    select_regional_strategy,
):
    height, width = 6, 9
    yy, xx = np.mgrid[0:height, 0:width]
    boundary_y = 4.2 - 0.25 * xx + 0.45 * np.sin(xx * 0.9)
    boundary_sensitivity = np.exp(-0.5 * ((yy - boundary_y) / 0.8) ** 2)
    off_boundary_hotspot = np.exp(-0.5 * (((xx - 1.5) / 0.9) ** 2 + ((yy - 0.8) / 0.7) ** 2))
    sensitivity = (
        0.08
        + 0.92 * boundary_sensitivity
        + 1.35 * off_boundary_hotspot
    )
    table_bank = np.array(
        [[0.125], [0.25], [0.5], [1.0], [2.0], [4.0], [8.0]],
        dtype=np.float64,
    )
    stac_strategy = select_regional_strategy(
        sensitivity[:, :, None],
        table_bank,
        total_budget=float(loss_budget.value),
        region_shape=(2, 3),
    )
    semantic_outline = np.abs(yy - boundary_y) <= 1.0
    policy_comparison = compare_matched_precision_policies(
        sensitivity,
        semantic_outline,
        precision_budget=2.0 * float(loss_budget.value),
        boundary_focus_ratio=float(boundary_focus_ratio.value),
    )
    return policy_comparison, sensitivity, stac_strategy


@app.cell(hide_code=True)
def _(load_display_derivative, project_root):
    evidence_root = project_root / "artifacts" / "v12_public"
    real_display = load_display_derivative(
        evidence_root / "v12_real_frozen_scorer_display.npz",
        evidence_root / "v12_real_frozen_scorer_display.manifest.json",
    )
    return (real_display,)


@app.cell(hide_code=True)
def _(boundary_shift, boundary_sweep):
    boundary_state = boundary_sweep(float(boundary_shift.value))
    return (boundary_state,)


@app.cell(hide_code=True)
def _(integrate_negative_gradient):
    morse_traces = tuple(
        integrate_negative_gradient(start, steps=150).points.tolist()
        for start in ((-0.18, 0.88), (0.22, 0.78), (-0.35, -0.82), (0.38, -0.72))
    )
    return (morse_traces,)


@app.cell(hide_code=True)
def _(np, screw_trajectory):
    screw_twist = np.array([0.0, 0.0, 0.12, 0.0, 0.0, 1.0], dtype=np.float64)
    screw_seed_point = np.array([0.82, 0.0, 0.0], dtype=np.float64)
    screw_thetas = np.linspace(0.0, 2.0 * np.pi, 81, dtype=np.float64)
    screw_points = screw_trajectory(screw_twist, screw_seed_point, screw_thetas)
    return (screw_points,)


@app.cell(hide_code=True)
def _(
    boundary_state,
    loss_budget,
    notebook_toy_proposals,
    refresh_pressure,
    texture_amplitude,
    value_carrier_proposals,
):
    normalized_budget = (float(loss_budget.value) - 4.0) / 28.0
    toy_operating_d_pose = 0.02
    controller_proposals = notebook_toy_proposals(
        boundary_sweep_fraction=boundary_state.exact_area_fraction,
        texture_amplitude=float(texture_amplitude.value),
        refresh_pressure=float(refresh_pressure.value),
        normalized_budget=normalized_budget,
    )
    controller_rows = value_carrier_proposals(
        controller_proposals,
        d_pose=toy_operating_d_pose,
    )
    controller_bids = tuple(row.value_per_byte for row in controller_rows)
    return controller_bids, controller_rows, toy_operating_d_pose


@app.cell(hide_code=True)
def _(mo):
    get_accelerator_state, set_accelerator_state = mo.state(
        {"status": "idle", "request_id": 0, "receipt": None, "error": None}
    )
    return get_accelerator_state, set_accelerator_state


@app.cell(hide_code=True)
def _(gpu_run, mo, project_root, set_accelerator_state):
    accelerator_thread = None
    accelerator_request_id = int(gpu_run.value or 0)
    if accelerator_request_id > 0:
        set_accelerator_state(
            {
                "status": "running",
                "request_id": accelerator_request_id,
                "receipt": None,
                "error": None,
            }
        )

        def _run_accelerator_probe():
            worker = mo.current_thread()
            try:
                import importlib

                gpu_module = importlib.import_module("molab_witness_machine.v12_gpu")
                run_proxy_robustness_sweep = gpu_module.run_proxy_robustness_sweep

                evidence_root = project_root / "artifacts" / "v12_public"
                receipt = run_proxy_robustness_sweep(
                    evidence_root / "v12_real_frozen_scorer_evidence.npz",
                    evidence_root
                    / "v12_real_frozen_scorer_evidence.manifest.json",
                    epsilon_count=256,
                    use_accelerator=True,
                    chunk_size=32,
                )
                if not worker.should_exit:
                    set_accelerator_state(
                        {
                            "status": "complete",
                            "request_id": accelerator_request_id,
                            "receipt": receipt,
                            "error": None,
                        }
                    )
            except Exception as exc:
                if not worker.should_exit:
                    set_accelerator_state(
                        {
                            "status": "error",
                            "request_id": accelerator_request_id,
                            "receipt": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        accelerator_thread = mo.Thread(
            target=_run_accelerator_probe,
            name=f"wm-accelerator-{accelerator_request_id}",
            daemon=True,
        )
        accelerator_thread.start()
    return (accelerator_thread,)


if __name__ == "__main__":
    app.run()
