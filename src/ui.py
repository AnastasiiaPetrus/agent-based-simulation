import os
from datetime import datetime
from html import escape
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

from src.charts import line_chart
from src.constants import (
    CHECK_DESCRIPTIONS,
    DEFAULT_TRUE_HIGH_RISK_RATE,
    DISTRICTS,
    LLM_MODEL,
    DEFAULT_LLM_MODEL_OPTIONS,
    MAX_RUN_LOG_SIZE,
    POLICY_DESCRIPTIONS,
    POLICY_ORDER,
    POLICIES,
    POPULATION_DOT_ANIMATION_SECONDS,
    POPULATION_DOT_STAGGER_GROUP,
    POPULATION_DOT_STAGGER_SECONDS,
    RESULT_METRIC_DESCRIPTIONS,
    RUN_METRIC_LABELS,
    SETTING_DESCRIPTIONS,
)
from src.llm import (
    DEFAULT_SYSTEM_PROMPT,
    build_llm_simulation_prompt,
    compact_parameter_summary,
    friendly_llm_error,
    run_openai_json,
)
from src.simulation import (
    clamp_count,
    clean_llm_district_results,
    clean_llm_run_results,
    compact_aggregate_metrics,
    normalize_llm_metrics,
    risk_signal_counts,
    validate_llm_tables,
)
from src.state import (
    add_llm_run_log_entry,
    attach_model_label,
    attach_policy_label,
    initialize_llm_state,
    latest_result_has_current_schema,
    normalize_representative_agents,
    trim_llm_run_log,
)
from src.tables import average_results_table, combined_policy_totals_table


def unique_values(values):
    return list(dict.fromkeys(value for value in values if value))


def csv_values(raw_value):
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def llm_model_options():
    configured_models = csv_values(os.environ.get("LLM_AGENT_MODELS", ""))
    return unique_values(configured_models or [LLM_MODEL, *DEFAULT_LLM_MODEL_OPTIONS])


def default_llm_agent_models(model_options):
    preferred_models = unique_values([LLM_MODEL, "gpt-4.1-mini"])
    defaults = [model for model in preferred_models if model in model_options]
    return defaults or model_options[:1]


def glossary_markdown(items):
    lines = []
    for item, meaning in items.items():
        lines.append(f"- **{item}:** {meaning}")
    return "\n".join(lines)


def baseline_children(settings):
    population_size = int(settings["population_size"])
    true_high = clamp_count(settings["true_high_risk_rate"] * population_size, population_size)
    false_positive, false_negative, flagged = risk_signal_counts(true_high, settings)
    true_positive = max(0, true_high - false_negative)
    low_unflagged = max(0, population_size - true_positive - false_negative - false_positive)

    children = (
        [{"base": "high", "flagged": True}] * true_positive
        + [{"base": "high", "flagged": False}] * false_negative
        + [{"base": "low", "flagged": True}] * false_positive
        + [{"base": "low", "flagged": False}] * low_unflagged
    )
    seed = population_size * 17 + true_high * 31 + flagged * 43
    rng = np.random.default_rng(seed)
    rng.shuffle(children)
    return children, {
        "high": true_high,
        "low": population_size - true_high,
        "flagged": flagged,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def policy_transition_metrics(run_results, policy, settings):
    if run_results is None or run_results.empty or "policy" not in run_results.columns:
        return None

    policy_rows = run_results[run_results["policy"] == policy]
    if policy_rows.empty:
        return None

    population_size = int(settings["population_size"])
    averages = policy_rows.mean(numeric_only=True)
    baseline = clamp_count(averages.get("baseline_crimes"), population_size)
    false_positives = clamp_count(averages.get("false_positives"), population_size)
    false_negatives = clamp_count(averages.get("false_negatives"), population_size)
    flagged = clamp_count(averages.get("children_flagged"), population_size)
    if flagged == 0:
        flagged = clamp_count(baseline - false_negatives + false_positives, population_size)

    return {
        "prevented": clamp_count(averages.get("crimes_prevented"), population_size),
        "harmed": clamp_count(averages.get("children_harmed"), population_size),
        "baseline": baseline,
        "flagged": flagged,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def seeded_subset(indices, count, seed_text):
    indices = list(indices)
    count = min(max(0, int(count)), len(indices))
    if count == 0:
        return set()

    seed = sum(ord(char) for char in seed_text) + len(indices) * 97 + count * 193
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return set(indices[:count])


def policy_panel_html(children, policy, metrics, panel_index):
    high_flagged = [index for index, child in enumerate(children) if child["base"] == "high" and child["flagged"]]
    low_flagged = [index for index, child in enumerate(children) if child["base"] == "low" and child["flagged"]]

    has_result = metrics is not None
    prevented = min(metrics["prevented"], len(high_flagged)) if has_result else 0
    prevented_indices = seeded_subset(high_flagged, prevented, f"{policy}-prevented")
    # harmed pool = all flagged children not already counted as prevented
    high_flagged_remaining = [i for i in high_flagged if i not in prevented_indices]
    harmed_pool = low_flagged + high_flagged_remaining
    harmed = min(metrics["harmed"], len(harmed_pool)) if has_result else 0
    harmed_indices = seeded_subset(harmed_pool, harmed, f"{policy}-harmed")

    dots = []
    final_high = 0
    final_low = 0
    final_harmed = 0
    for index, child in enumerate(children):
        delay = (index % POPULATION_DOT_STAGGER_GROUP) * POPULATION_DOT_STAGGER_SECONDS
        base_class = "base-high" if child["base"] == "high" else "base-low"
        final_class = base_class.replace("base-", "final-")
        change_class = ""

        if index in prevented_indices:
            final_class = "final-low"
            change_class = "changed-prevented"
        elif index in harmed_indices:
            final_class = "final-harmed"
            change_class = "changed-harmed"

        if final_class == "final-harmed":
            final_harmed += 1
        elif final_class == "final-high":
            final_high += 1
        else:
            final_low += 1

        flagged_class = " flagged-dot" if child["flagged"] else ""
        dots.append(
            f'<span class="life-dot {base_class} {final_class} {change_class}{flagged_class}" '
            f'style="--delay:{delay:.3f}s"></span>'
        )

    if has_result:
        harmed_fp = sum(1 for i in harmed_indices if i in set(low_flagged))
        harmed_tp = len(harmed_indices) - harmed_fp
        summary = (
            f"{prevented} prevented (red→green); "
            f"{harmed_fp} wrongly flagged & harmed (green→orange); "
            f"{harmed_tp} flagged & harmed (red→orange); "
            f"{final_high} red remain."
        )
        ready = "true"
    else:
        summary = "Waiting for this policy result. Baseline is shown."
        ready = "false"

    return f"""
    <section class="policy-panel" data-ready="{ready}">
      <div class="policy-panel-title">{escape(policy)}</div>
      <div class="policy-panel-summary">{escape(summary)}</div>
      <div class="policy-grid" data-panel="{panel_index}">
        {''.join(dots)}
        <div class="lc-spotlight"></div>
      </div>
    </section>
    """


def population_animation_html(run_results, settings, title, animation_key=""):
    children, baseline_counts = baseline_children(settings)
    seed_text = f"{title}-{animation_key}"
    animation_id = abs(sum(ord(char) for char in seed_text) + len(children) * 17) % 100000
    root_id = f"policyTransition{animation_id}"
    panels = [
        policy_panel_html(
            children,
            policy,
            policy_transition_metrics(run_results, policy, settings),
            panel_index,
        )
        for panel_index, policy in enumerate(POLICY_ORDER)
    ]

    return f"""
<style>
.life-course-card {{
  border: 1px solid #e3e6ee;
  border-radius: 8px;
  padding: 14px 16px;
  margin: 10px 0 16px;
  background: #ffffff;
  overflow-x: auto;
}}
.life-course-header {{
  margin-bottom: 10px;
}}
.life-course-title {{
  font-weight: 700;
  color: #2d3142;
}}
.policy-panels {{
  display: grid;
  grid-template-columns: repeat(3, minmax(660px, 1fr));
  gap: 18px;
  min-width: 2100px;
}}
.policy-panel {{
  border: 1px solid #edf0f6;
  border-radius: 8px;
  padding: 12px;
  background: #fbfcfe;
}}
.policy-panel-title {{
  min-height: 42px;
  color: #2d3142;
  font-size: 0.92rem;
  font-weight: 700;
  line-height: 1.25;
}}
.policy-panel-summary {{
  min-height: 34px;
  color: #697287;
  font-size: 0.78rem;
  line-height: 1.25;
  margin-bottom: 8px;
}}
.policy-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(11px, 1fr));
  grid-auto-rows: 15px;
  gap: 5px;
  align-items: center;
  justify-items: center;
  padding: 12px 10px;
  position: relative;
  overflow: hidden;
}}
.life-dot {{
  width: 11px;
  height: 11px;
  border-radius: 999px;
  display: inline-block;
  justify-self: center;
  box-sizing: border-box;
  opacity: 0;
  background: var(--base-color);
  transform: scale(0.45);
  --base-scale: 1;
  --final-scale: 1;
  animation: revealRiskDot {POPULATION_DOT_ANIMATION_SECONDS:.2f}s cubic-bezier(.22,.61,.19,1) forwards;
  animation-delay: var(--delay);
  transition: background 0.75s ease, transform 0.75s ease, box-shadow 0.75s ease, filter 0.15s ease;
  position: relative;
  z-index: 1;
}}
.life-dot:hover {{
  filter: brightness(1.25);
  transform: scale(calc(var(--final-scale) * 1.18));
}}
.base-low {{
  --base-color: #25a55f;
  --base-scale: 0.92;
}}
.base-high {{
  --base-color: #d64b3c;
  --base-scale: 1.28;
}}
.final-low {{
  --final-color: #25a55f;
  --final-scale: 0.92;
}}
.final-high {{
  --final-color: #d64b3c;
  --final-scale: 1.28;
}}
.final-harmed {{
  --final-color: #e57c23;
  --final-scale: 1.12;
}}
.flagged-dot {{
  outline: 3px solid #f2b705;
  outline-offset: 1px;
}}
@keyframes revealRiskDot {{
  0%   {{ opacity: 0; transform: scale(0.45); }}
  100% {{ opacity: 1; transform: scale(var(--base-scale)); }}
}}
.policy-panel.show-final .life-dot {{
  background: var(--final-color);
  transform: scale(var(--final-scale));
}}
.policy-panel.show-final .life-dot.changed-prevented {{
  box-shadow: 0 0 0 3px rgba(37,165,95,0.22);
}}
.policy-panel.show-final .life-dot.changed-harmed {{
  box-shadow: 0 0 0 3px rgba(229,124,35,0.24);
}}
.lc-spotlight {{
  position: absolute;
  top: 0;
  left: 0;
  width: 200px;
  height: 200px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0.10) 38%, transparent 68%);
  pointer-events: none;
  opacity: 0;
  transform: translate3d(-220px, -220px, 0);
  transition: opacity 0.18s ease;
  will-change: transform, opacity;
  z-index: 2;
}}
.life-course-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-top: 10px;
  color: #4c5568;
  font-size: 0.88rem;
}}
.legend-item {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.legend-dot {{
  width: 9px;
  height: 9px;
  border-radius: 999px;
  display: inline-block;
  background: var(--legend-color);
}}
.legend-dot.flagged-dot {{
  background: transparent;
  outline-width: 2px;
}}
.legend-low {{ --legend-color: #25a55f; }}
.legend-high {{ --legend-color: #d64b3c; }}
.legend-harmed {{ --legend-color: #e57c23; }}
</style>
<div class="life-course-card" id="{root_id}">
  <div class="life-course-header">
    <div class="life-course-title">{escape(title)}</div>
  </div>
  <div class="policy-panels">
    {''.join(panels)}
  </div>
  <div class="life-course-legend">
    <span class="legend-item"><span class="legend-dot legend-low"></span>green: no modeled offense / prevented</span>
    <span class="legend-item"><span class="legend-dot legend-high"></span>red: true high-risk / offense remains</span>
    <span class="legend-item"><span class="legend-dot legend-harmed"></span>orange: wrongly flagged and harmed</span>
    <span class="legend-item"><span class="legend-dot flagged-dot"></span>yellow outline: flagged by prediction ({baseline_counts['flagged']})</span>
  </div>
</div>
<script>
(function() {{
  var root = document.getElementById('{root_id}');
  if (!root) return;

  setTimeout(function() {{
    root.querySelectorAll('.policy-panel[data-ready="true"]').forEach(function(panel) {{
      panel.classList.add('show-final');
    }});
  }}, 420);

  root.querySelectorAll('.policy-grid').forEach(function(grid) {{
    var spot = grid.querySelector('.lc-spotlight');
    var rafId = null;
    var curX = -220, curY = -220;
    var tgtX = -220, tgtY = -220;
    var LERP = 0.13;

    function step() {{
      var dx = tgtX - curX;
      var dy = tgtY - curY;
      curX += dx * LERP;
      curY += dy * LERP;
      var speed = Math.sqrt(dx * dx + dy * dy);
      var scale = 1 + Math.min(speed * 0.005, 0.45);
      spot.style.transform = 'translate3d(' + (curX - 100) + 'px,' + (curY - 100) + 'px,0) scale(' + scale + ')';
      if (speed > 0.5) {{
        rafId = requestAnimationFrame(step);
      }} else {{
        spot.style.transform = 'translate3d(' + (tgtX - 100) + 'px,' + (tgtY - 100) + 'px,0) scale(1)';
        rafId = null;
      }}
    }}

    grid.addEventListener('pointermove', function(e) {{
      var r = grid.getBoundingClientRect();
      tgtX = e.clientX - r.left;
      tgtY = e.clientY - r.top;
      spot.style.opacity = '1';
      if (!rafId) rafId = requestAnimationFrame(step);
    }}, {{ passive: true }});
    grid.addEventListener('pointerleave', function() {{
      spot.style.opacity = '0';
    }});
  }});
}})();
</script>
"""


def render_population_animation(container, run_results, settings, title, animation_key=""):
    html = population_animation_html(run_results, settings, title, animation_key)
    with container:
        st.html(html, unsafe_allow_javascript=True)


def render_reference_guide():
    with st.expander("How this works", expanded=False):
        st.markdown("### Three policies")
        st.markdown(glossary_markdown(POLICY_DESCRIPTIONS))

        st.markdown("### What you can adjust")
        st.markdown(glossary_markdown(SETTING_DESCRIPTIONS))

        st.markdown("### What the results show")
        st.markdown(glossary_markdown(RESULT_METRIC_DESCRIPTIONS))

        st.markdown("### Key trade-offs")
        st.markdown(glossary_markdown(CHECK_DESCRIPTIONS))


def display_average_table(average_table):
    display_table = average_table.copy()

    def format_value(value):
        if pd.isna(value):
            return "Not applicable"
        return f"{value:,.3f}"

    display_table["Average per synthetic run"] = display_table["Average per synthetic run"].map(format_value)
    st.dataframe(display_table, use_container_width=True, hide_index=True)


def display_combined_policy_totals_table(run_results, population_size):
    st.dataframe(
        combined_policy_totals_table(run_results, population_size),
        use_container_width=True,
        hide_index=True,
    )


def render_charts(run_results, district_results):
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.pyplot(
            line_chart(
                run_results,
                "run",
                "crimes_prevented",
                "Offenses prevented by run",
                "Offenses prevented",
            ),
            clear_figure=True,
        )

    if "children_harmed" in run_results.columns and run_results["children_harmed"].sum() > 0:
        with chart_right:
            st.pyplot(
                line_chart(
                    run_results,
                    "run",
                    "children_harmed",
                    "Children exposed to harmful intervention by run",
                    "Children exposed",
                ),
                clear_figure=True,
            )


def render_interpretation(policy, average_table, bias_against_district_c):
    values = dict(zip(average_table["Metric"], average_table["Average per synthetic run"]))
    crimes_prevented = values.get("Offenses prevented", 0.0)
    false_positives = values.get("Wrongly flagged", 0.0)
    children_helped = values.get("Received support", 0.0)
    children_harmed = values.get("Harmed by intervention", 0.0)

    if policy == "Coercive preventive intervention for high-risk children":
        st.warning(
            f"On average, {crimes_prevented:.0f} offenses are prevented per run — "
            f"but {children_harmed:.0f} children are restricted before committing any offense, "
            f"including {false_positives:.0f} who would not have offended at all."
        )
    elif policy == "Targeted support for high-risk children":
        st.write(
            f"On average, {crimes_prevented:.0f} offenses are prevented per run "
            f"and {children_helped:.0f} children receive help. "
            f"Of those, {false_positives:.0f} are wrongly flagged and receive unnecessary support."
        )
    elif policy == "Surveillance of high-risk children":
        st.warning(
            f"On average, {crimes_prevented:.0f} offenses are prevented per run, "
            f"but {children_harmed:.0f} children are monitored — including {false_positives:.0f} "
            "who would not have offended and have no basis to be watched."
        )


def render_llm_run_log(max_entries):
    trim_llm_run_log(max_entries)
    run_log = st.session_state["llm_agent_run_log"]
    if not run_log:
        return

    st.subheader("Previous runs")

    if st.button("Clear LLM-agent run log"):
        st.session_state["llm_agent_run_log"] = []
        st.write("LLM-agent run log cleared for this session.")
        return

    log_for_csv = []
    for entry in run_log:
        metrics = entry["aggregate_metrics"]
        log_for_csv.append(
            {
                "timestamp": entry["timestamp"],
                "policy_summary": entry.get("policy_summary", ""),
                "parameter_summary": entry["parameter_summary"],
                "representative_agent_count": entry["representative_agent_count"],
                "llm_model": entry["llm_model"],
                "crimes_prevented": metrics.get("crimes_prevented"),
                "false_positives": metrics.get("false_positives"),
                "false_negatives": metrics.get("false_negatives"),
                "children_helped": metrics.get("children_helped"),
                "children_harmed": metrics.get("children_harmed"),
                "debrief_text": entry["debrief_text"],
            }
        )

    csv_buffer = StringIO()
    pd.DataFrame(log_for_csv).to_csv(csv_buffer, index=False)
    st.download_button(
        "Download LLM-agent run log as CSV",
        data=csv_buffer.getvalue(),
        file_name="llm_agent_run_log.csv",
        mime="text/csv",
    )

    for entry in run_log:
        title = f"{entry['timestamp']} | {entry['policy_summary']} | {entry['llm_model']}"
        with st.expander(title):
            st.write(entry["debrief_text"])
            st.caption(entry["parameter_summary"])
            st.json(entry["aggregate_metrics"])


def render_llm_agent_section(settings):
    initialize_llm_state()
    st.subheader("Simulation")
    total_calls = len(settings["llm_agent_models"]) * len(POLICIES)
    st.caption(f"{total_calls} LLM call(s) — one per policy × model.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning(
            "LLM-agent simulation is enabled, but OPENAI_API_KEY is not configured. Add it as an environment "
            "variable or Railway secret."
        )

    selected_models = settings["llm_agent_models"]
    if not selected_models:
        st.warning("Select at least one LLM model agent in the sidebar.")

    parameter_summary = compact_parameter_summary(settings)
    system_prompt = DEFAULT_SYSTEM_PROMPT
    latest_result = st.session_state.get("llm_agent_latest_result")
    initial_population_results = None
    initial_population_title = "Live synthetic population view"
    if latest_result and latest_result_has_current_schema(latest_result):
        initial_population_results = latest_result["run_results"]
        initial_population_title = "Latest synthetic population view"

    live_population = st.empty()
    render_population_animation(
        live_population,
        initial_population_results,
        settings,
        initial_population_title,
        "initial",
    )
    progress_slot = st.empty()
    live_status = st.empty()
    live_debrief = st.empty()

    st.sidebar.caption(
        f"Current run: {len(POLICIES)} policies × {len(selected_models)} model agent(s) = "
        f"{total_calls} LLM call(s). Each call generates {int(settings['llm_simulation_runs'])} "
        f"synthetic run(s) over {int(settings['population_size']):,} synthetic children."
    )
    settings_invalid = settings["true_high_risk_rate"] == 0
    if settings_invalid:
        st.sidebar.warning("Set 'Percentage of true high-risk children' above 0% to run a meaningful simulation.")

    run_requested = st.sidebar.button(
        "Run simulation",
        disabled=not bool(api_key) or not selected_models or settings_invalid,
        type="primary",
        use_container_width=True,
    )

    if run_requested:
        model_results = []
        model_errors = []
        completed_calls = 0
        progress_bar = progress_slot.progress(0.0)
        live_status.write("Starting LLM-agent simulation...")
        render_population_animation(
            live_population,
            None,
            settings,
            "Waiting for the first LLM-agent result",
            "waiting",
        )

        with st.spinner(f"Running {len(POLICIES)} policy(ies) × {len(selected_models)} model agent(s)..."):
            for model in selected_models:
                for policy in POLICY_ORDER:
                    policy_settings = dict(settings)
                    policy_settings["policy"] = policy

                    live_status.write(f"Running **{model}** on **{policy}**...")
                    user_prompt = build_llm_simulation_prompt(policy_settings)
                    try:
                        raw_result = run_openai_json(system_prompt, user_prompt, model=model)
                        run_results = clean_llm_run_results(raw_result.get("run_results", []))
                        district_results = clean_llm_district_results(raw_result.get("district_results", []))
                        validate_llm_tables(run_results, district_results, policy_settings)
                        run_results, district_results = normalize_llm_metrics(
                            run_results,
                            district_results,
                            policy_settings,
                        )
                        run_results, district_results = attach_model_label(
                            run_results, district_results, model
                        )
                        run_results, district_results = attach_policy_label(
                            run_results, district_results, policy
                        )
                        representative_agents = normalize_representative_agents(
                            raw_result.get("representative_agents", [])[
                                : int(settings["llm_representative_agents"])
                            ],
                            model,
                        )
                        debrief_text = str(raw_result.get("debrief_text", "")).strip()
                        completed_calls += 1
                        progress_bar.progress(completed_calls / max(total_calls, 1))
                        live_status.write(f"Received **{model}** result for **{policy}**.")
                        result_entry = {
                            "llm_model": model,
                            "policy": policy,
                            "run_results": run_results,
                            "district_results": district_results,
                            "representative_agents": representative_agents,
                            "aggregate_metrics": compact_aggregate_metrics(run_results),
                            "debrief_text": debrief_text,
                        }
                        model_results.append(result_entry)
                        partial_run_results = pd.concat(
                            [result["run_results"] for result in model_results],
                            ignore_index=True,
                        )
                        render_population_animation(
                            live_population,
                            partial_run_results,
                            settings,
                            "Policy outcome transitions",
                            f"result-{completed_calls}",
                        )
                        if debrief_text:
                            live_debrief.info(f"{model} | {policy}: {debrief_text}")
                    except Exception as error:
                        completed_calls += 1
                        progress_bar.progress(completed_calls / max(total_calls, 1))
                        live_status.write(f"Could not generate **{model}** result for **{policy}**.")
                        model_errors.append((f"{model} | {policy}", friendly_llm_error(error)))

        progress_slot.empty()
        live_status.empty()
        live_debrief.empty()

        for error_label, error_message in model_errors:
            st.error(f"{error_label}: {error_message}")

        if model_results:
            combined_run_results = pd.concat(
                [result["run_results"] for result in model_results], ignore_index=True
            )
            combined_district_results = pd.concat(
                [result["district_results"] for result in model_results], ignore_index=True
            )
            all_representative_agents = [
                agent
                for result in model_results
                for agent in result["representative_agents"]
            ]
            debrief_text = "\n\n".join(
                f"{result['llm_model']} | {result['policy']}: {result['debrief_text']}"
                for result in model_results
                if result["debrief_text"]
            )
            aggregate_metrics = compact_aggregate_metrics(combined_run_results)

            st.session_state["llm_agent_latest_result"] = {
                "run_results": combined_run_results,
                "district_results": combined_district_results,
                "model_results": model_results,
                "representative_agents": all_representative_agents,
                "debrief_text": debrief_text,
            }
            render_population_animation(
                live_population,
                combined_run_results,
                settings,
                "Latest synthetic population view",
                f"final-{datetime.now().timestamp()}",
            )

            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "policy_summary": "All policies",
                "parameter_summary": parameter_summary,
                "representative_agent_count": len(all_representative_agents),
                "llm_model": ", ".join(result["llm_model"] for result in model_results),
                "aggregate_metrics": aggregate_metrics,
                "debrief_text": debrief_text,
            }
            add_llm_run_log_entry(entry, MAX_RUN_LOG_SIZE)
            if model_errors:
                st.warning("LLM-agent simulation generated for the successful model agents.")
            else:
                st.success("LLM-agent simulation generated.")

    latest_result = st.session_state.get("llm_agent_latest_result")
    if latest_result and not latest_result_has_current_schema(latest_result):
        st.session_state.pop("llm_agent_latest_result", None)
        latest_result = None
        st.write("Previous in-session results used an older metric schema. Run the simulation again.")

    if latest_result:
        latest_run_results = latest_result["run_results"]
        latest_district_results = latest_result["district_results"]

        st.subheader("Policy comparison")
        st.caption("Average outcomes per run. Use this to compare policies side by side.")
        display_combined_policy_totals_table(latest_run_results, settings["population_size"])

        policy_tabs = st.tabs(POLICY_ORDER)
        for policy, tab in zip(POLICY_ORDER, policy_tabs):
            with tab:
                policy_runs = latest_run_results[latest_run_results["policy"] == policy]
                policy_districts = latest_district_results[latest_district_results["policy"] == policy]
                if policy_runs.empty or policy_districts.empty:
                    st.caption("No results for this policy in the current session.")
                    continue

                st.subheader("Averages")
                display_average_table(average_results_table(policy_runs))
                render_charts(policy_runs, policy_districts)

                # Interpretation uses per-run averages across selected models for this policy.
                render_interpretation(
                    policy,
                    average_results_table(policy_runs),
                    settings["bias_against_district_c"],
                )

        st.markdown("**Latest LLM-agent explanations**")
        model_results = latest_result.get("model_results", [])
        if model_results:
            for result in model_results:
                with st.expander(
                    f"{result['llm_model']} | {result.get('policy', 'Unknown policy')} explanation",
                    expanded=False,
                ):
                    st.write(result["debrief_text"])
                    st.json(result["aggregate_metrics"])
        else:
            st.write(latest_result["debrief_text"])

        representative_agents = latest_result.get("representative_agents", [])
        if representative_agents:
            with st.expander("Representative LLM synthetic agents used", expanded=False):
                st.dataframe(pd.DataFrame(representative_agents), use_container_width=True, hide_index=True)
    render_llm_run_log(MAX_RUN_LOG_SIZE)


def sidebar_inputs():
    st.sidebar.header("Simulation settings")
    st.sidebar.caption("Synthetic population: **1 000 children** (fixed).")
    population_size = 1000
    true_high_risk_rate = st.sidebar.slider(
        "Percentage of true high-risk children (%)",
        0, 100, int(DEFAULT_TRUE_HIGH_RISK_RATE * 100), step=1,
        help=SETTING_DESCRIPTIONS["Percentage of true high-risk children (%)"],
    ) / 100
    prediction_noise = st.sidebar.slider(
        "Prediction error rate (%)",
        0, 100, 5, step=1,
        help=SETTING_DESCRIPTIONS["Prediction error rate (%)"],
    ) / 100
    settings = {
        "population_size": population_size,
        "true_high_risk_rate": true_high_risk_rate,
        "prediction_noise": prediction_noise,
        "bias_against_district_c": 0.0,
        "policy_effect_strength": st.sidebar.select_slider(
            "Intervention strength",
            options=["Low", "Medium", "High"],
            value="Medium",
            help=SETTING_DESCRIPTIONS["Intervention strength"],
        ),
        "llm_simulation_runs": 5,
        "llm_representative_agents": 2,
    }
    true_high_risk_count = clamp_count(true_high_risk_rate * population_size, population_size)
    fp, fn, flagged_count = risk_signal_counts(true_high_risk_count, settings)
    settings["high_risk_threshold"] = flagged_count / population_size

    model_options = llm_model_options()
    settings["llm_agent_models"] = st.sidebar.multiselect(
        "LLM model agent(s)",
        options=model_options,
        default=default_llm_agent_models(model_options),
    )

    return settings


def render_app():
    st.set_page_config(page_title="Predictive Justice Thought Experiment", layout="wide")

    st.title("Predictive Justice Thought Experiment")
    st.markdown(
        "Suppose we could reliably predict, at age 10, who will become a violent criminal by age 30. "
        "What should we do with that information? This simulation compares three policy responses using "
        "synthetic LLM-generated life-course trajectories for 1 000 children."
    )
    render_reference_guide()

    settings = sidebar_inputs()
    render_llm_agent_section(settings)
    latest_result = st.session_state.get("llm_agent_latest_result")
    if latest_result and latest_result_has_current_schema(latest_result):
        csv_buffer = StringIO()
        latest_result["run_results"].to_csv(csv_buffer, index=False)
        st.download_button(
            "Download LLM-agent results as CSV",
            data=csv_buffer.getvalue(),
            file_name="llm_agent_simulation_results.csv",
            mime="text/csv",
        )
