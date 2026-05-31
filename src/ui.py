import json
import os
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from src.achievements import ACHIEVEMENT_INDEX, ACHIEVEMENTS, check_achievements
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
    optimize_result_frames,
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


_LIVE_GRID_ID = "livePopGrid"


def render_global_styles():
    st.html(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
  --surface: #ffffff;
  --surface-muted: #f6f8fb;
  --surface-panel: #fbfcfe;
  --line: #dce3ec;
  --line-soft: #e8edf3;
  --text: #1a2232;
  --text-muted: #5e6e85;
  --primary: #145c58;
  --primary-light: rgba(20, 92, 88, 0.08);
  --primary-strong: #0f4744;
  --accent: #d95f47;
  --amber: #d6a21d;
  --success: #23875a;
  --radius-sm: 6px;
  --radius: 10px;
  --shadow-xs: 0 1px 2px rgba(24, 33, 51, 0.06);
  --shadow-sm: 0 2px 8px rgba(24, 33, 51, 0.08), 0 1px 3px rgba(24, 33, 51, 0.05);
  --shadow-md: 0 8px 24px rgba(24, 33, 51, 0.10);
}

/* ── Base ─────────────────────────────────── */
html, body, .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.stApp {
  background:
    linear-gradient(150deg, rgba(20, 92, 88, 0.04) 0%, transparent 32%),
    #f7f9fc;
  color: var(--text);
}

/* ── Topbar ───────────────────────────────── */
[data-testid="stHeader"] {
  background: rgba(247, 249, 252, 0.90);
  border-bottom: 1px solid rgba(220, 227, 236, 0.60);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}

/* ── Main content ─────────────────────────── */
[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 1200px;
  padding-top: 2.5rem;
  padding-bottom: 4rem;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
  border-right: 1px solid var(--line-soft);
  box-shadow: 4px 0 24px rgba(24, 33, 51, 0.04);
}

[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, #ffffff 0%, #f8fafd 100%);
  padding-top: 1.75rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > p {
  font-size: 0.82rem;
}

/* ── Typography ───────────────────────────── */
h1, h2, h3 {
  color: var(--text);
  letter-spacing: -0.02em;
  font-weight: 700;
}

h1 {
  max-width: 840px;
  margin-bottom: 0.5rem;
  font-size: 3.2rem;
  line-height: 1.05;
}

h2 {
  margin-top: 1.5rem;
  font-size: 1.4rem;
}

h3 {
  font-size: 1.05rem;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color: var(--text-muted);
  line-height: 1.65;
  font-size: 0.94rem;
}

[data-testid="stMarkdownContainer"] strong {
  color: var(--text);
  font-weight: 600;
}

[data-testid="stCaptionContainer"] {
  color: #7d8a98;
  font-size: 0.81rem;
}

/* ── Expanders ────────────────────────────── */
[data-testid="stExpander"] details {
  background: var(--surface);
  border: 1px solid var(--line-soft);
  border-radius: var(--radius);
  box-shadow: var(--shadow-xs);
  transition: box-shadow 180ms ease, border-color 180ms ease;
}

[data-testid="stExpander"] details:hover {
  box-shadow: var(--shadow-sm);
  border-color: var(--line);
}

[data-testid="stExpander"] details[open] {
  border-color: rgba(20, 92, 88, 0.22);
  box-shadow: var(--shadow-sm);
}

[data-testid="stExpander"] summary {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--text);
  padding: 0.85rem 1rem;
  letter-spacing: -0.01em;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
  min-height: 2.55rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--line);
  font-weight: 500;
  font-size: 0.88rem;
  box-shadow: var(--shadow-xs);
  transition: border-color 140ms ease, box-shadow 140ms ease, transform 120ms ease;
}

.stButton > button:hover:not(:disabled) {
  border-color: rgba(20, 92, 88, 0.38);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.stButton > button:active:not(:disabled) {
  transform: translateY(0);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(175deg, #1a6e69 0%, var(--primary-strong) 100%);
  border-color: var(--primary-strong);
  color: #ffffff;
  box-shadow: 0 2px 8px rgba(20, 92, 88, 0.24), 0 1px 2px rgba(20, 92, 88, 0.12);
}

.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p {
  color: #ffffff !important;
  font-weight: 600;
}

.stButton > button[kind="primary"]:hover:not(:disabled) {
  background: linear-gradient(175deg, #1e7872 0%, #145c58 100%);
  box-shadow: 0 6px 18px rgba(20, 92, 88, 0.28), 0 2px 6px rgba(20, 92, 88, 0.14);
}

/* ── Form controls ────────────────────────── */
[data-baseweb="select"] > div {
  border-radius: var(--radius-sm) !important;
}

[data-testid="stSlider"] [role="slider"] {
  background-color: var(--primary);
  border-color: #ffffff;
  box-shadow: 0 0 0 3px rgba(20, 92, 88, 0.14), 0 2px 5px rgba(20, 92, 88, 0.22);
}

[data-testid="stSlider"] div[data-testid="stTickBar"] div {
  background: var(--line);
}

/* ── Tabs ─────────────────────────────────── */
[data-baseweb="tab-list"] {
  gap: 0.2rem;
  border-bottom: 2px solid var(--line-soft);
}

[data-baseweb="tab"] {
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
  color: var(--text-muted);
  font-weight: 500;
  font-size: 0.88rem;
  padding: 0.6rem 1.1rem;
  transition: color 140ms ease, background 140ms ease;
}

[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  color: var(--text);
  background: var(--surface-muted);
}

[data-baseweb="tab"][aria-selected="true"] {
  background: var(--surface);
  color: var(--primary);
  font-weight: 650;
  box-shadow: inset 0 -2px 0 var(--primary);
}

/* ── Metric cards ─────────────────────────── */
div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMetric"]) {
  background: var(--surface);
  border: 1px solid var(--line-soft);
  border-radius: var(--radius);
  padding: 1rem;
  box-shadow: var(--shadow-xs);
  transition: box-shadow 150ms ease;
}

div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMetric"]):hover {
  box-shadow: var(--shadow-sm);
}

/* ── Data tables ──────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--line-soft);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow-xs);
}

/* ── Alerts ───────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius);
  border: 1px solid var(--line-soft);
  box-shadow: var(--shadow-xs);
  font-size: 0.9rem;
}

/* ── Charts ───────────────────────────────── */
[data-testid="stPlotlyChart"],
[data-testid="stImage"],
[data-testid="stPyplot"] {
  border-radius: var(--radius);
}

/* ── Sidebar toggle always visible ───────── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button {
  opacity: 1 !important;
}

/* ── Responsive ───────────────────────────── */
@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"],
  .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 1.25rem;
  }

  h1 {
    font-size: 2.2rem;
    line-height: 1.08;
  }

  [data-baseweb="tab"] {
    padding-left: 0.65rem;
    padding-right: 0.65rem;
    font-size: 0.82rem;
  }
}
</style>
        """
    )


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


def _panel_changes(children, policy, metrics):
    """Compute which dots change state for a given policy result.

    Returns (prevented_indices, harmed_indices, summary, ready).
    """
    high_flagged = [i for i, c in enumerate(children) if c["base"] == "high" and c["flagged"]]
    low_flagged = [i for i, c in enumerate(children) if c["base"] == "low" and c["flagged"]]
    low_flagged_set = set(low_flagged)

    if metrics is None:
        return set(), set(), "Waiting for this policy result. Baseline is shown.", False

    prevented = min(metrics["prevented"], len(high_flagged))
    prevented_indices = seeded_subset(high_flagged, prevented, f"{policy}-prevented")
    high_flagged_remaining = [i for i in high_flagged if i not in prevented_indices]
    harmed_pool = low_flagged + high_flagged_remaining
    harmed = min(metrics["harmed"], len(harmed_pool))
    harmed_indices = seeded_subset(harmed_pool, harmed, f"{policy}-harmed")

    harmed_fp = sum(1 for i in harmed_indices if i in low_flagged_set)
    harmed_tp = len(harmed_indices) - harmed_fp
    final_high = sum(
        1 for i, c in enumerate(children)
        if c["base"] == "high" and i not in prevented_indices and i not in harmed_indices
    )
    harmed_total = harmed_fp + harmed_tp
    summary = (
        f"{len(prevented_indices)} prevented (red→green); "
        f"{harmed_total} harmed by intervention (→red, outlined); "
        f"{final_high} red remain."
    )
    return prevented_indices, harmed_indices, summary, True


def policy_panel_html(children, policy, metrics, panel_index):
    prevented_indices, harmed_indices, summary, has_result = _panel_changes(children, policy, metrics)

    dots = []
    for index, child in enumerate(children):
        delay = (index % POPULATION_DOT_STAGGER_GROUP) * POPULATION_DOT_STAGGER_SECONDS
        base_class = "base-high" if child["base"] == "high" else "base-low"
        final_class = base_class.replace("base-", "final-")
        change_class = ""
        if index in prevented_indices:
            final_class = "final-low"
            change_class = "changed-prevented"
        elif index in harmed_indices:
            final_class = "final-high"
            change_class = "changed-harmed"
        flagged_class = " flagged-dot" if child["flagged"] else ""
        dots.append(
            f'<span class="life-dot {base_class} {final_class} {change_class}{flagged_class}" '
            f'style="--delay:{delay:.3f}s"></span>'
        )

    ready = "true" if has_result else "false"
    return f"""
    <section class="policy-panel" data-ready="{ready}">
      <div class="policy-panel-title">{escape(policy)}</div>
      <div class="policy-panel-summary">{escape(summary)}</div>
      <div class="policy-grid" data-panel="{panel_index}">
        {''.join(dots)}
      </div>
    </section>
    """


def population_update_script(root_id, policy, panel_index, children, metrics):
    """Return a <script> snippet that updates one policy panel's dot colours in-place."""
    prevented_indices, harmed_indices, summary, _ = _panel_changes(children, policy, metrics)
    return f"""<script>
(function() {{
  var root = document.getElementById({json.dumps(root_id)});
  if (!root) return;
  var panel = root.querySelectorAll('.policy-panel')[{panel_index}];
  if (!panel) return;
  var dots = Array.from(panel.querySelectorAll('.life-dot'));
  dots.forEach(function(dot) {{
    dot.classList.remove('final-low','final-high','final-harmed','changed-prevented','changed-harmed');
    dot.classList.add(dot.classList.contains('base-high') ? 'final-high' : 'final-low');
  }});
  {json.dumps(sorted(prevented_indices))}.forEach(function(i) {{
    if (!dots[i]) return;
    dots[i].classList.remove('final-high');
    dots[i].classList.add('final-low','changed-prevented');
  }});
  {json.dumps(sorted(harmed_indices))}.forEach(function(i) {{
    if (!dots[i]) return;
    dots[i].classList.remove('final-low');
    dots[i].classList.add('final-high','changed-harmed');
  }});
  var el = panel.querySelector('.policy-panel-summary');
  if (el) el.textContent = {json.dumps(summary)};
  panel.dataset.ready = 'true';
  panel.classList.add('show-final');
  var pg = panel.querySelector('.policy-grid');
  if (pg) pg.dispatchEvent(new CustomEvent('resetWaveCache'));
}})();
</script>"""


def population_animation_html(run_results, settings, title, animation_key="", root_id=None):
    children, baseline_counts = baseline_children(settings)
    if root_id is None:
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
  border: 1px solid #dce3ec;
  border-radius: 12px;
  padding: 18px 20px 16px;
  margin: 12px 0 20px;
  background: #ffffff;
  box-shadow: 0 2px 12px rgba(24,33,51,0.08), 0 1px 3px rgba(24,33,51,0.05);
}}
.life-course-header {{
  margin-bottom: 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e8edf3;
}}
.life-course-title {{
  font-weight: 700;
  font-size: 0.95rem;
  color: #1a2232;
  letter-spacing: -0.01em;
}}
.policy-panels {{
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.policy-panel {{
  border: 1px solid #e8edf3;
  border-left: 3px solid #e8edf3;
  border-radius: 8px;
  padding: 12px 14px;
  background: #fafbfd;
  box-shadow: 0 1px 2px rgba(24,33,51,0.04);
  transition: border-color 200ms ease, box-shadow 200ms ease;
}}
.policy-panel:nth-child(n) {{ border-left-color: #145c58; }}
.policy-panel[data-ready="true"] {{
  border-color: rgba(20,92,88,0.22);
  border-left-width: 3px;
  box-shadow: 0 2px 8px rgba(20,92,88,0.07);
}}
.policy-panel-title {{
  color: #1a2232;
  font-size: 0.88rem;
  font-weight: 650;
  line-height: 1.3;
  margin-bottom: 4px;
  letter-spacing: -0.01em;
}}
.policy-panel-summary {{
  color: #5e6e85;
  font-size: 0.75rem;
  line-height: 1.4;
  margin-bottom: 10px;
}}
.policy-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(10px, 1fr));
  grid-auto-rows: 14px;
  gap: 4px;
  align-items: center;
  justify-items: center;
  padding: 12px 10px;
  position: relative;
  overflow: hidden;
  border-radius: 6px;
  background: #f5f7fa;
  border: 1px solid #edf0f4;
}}
.life-dot {{
  width: 10px;
  height: 10px;
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
  transition: background 0.75s ease, transform 0.75s ease, box-shadow 0.75s ease, outline-color 0.75s ease;
  position: relative;
  z-index: 1;
}}
.base-low {{
  --base-color: #23875a;
  --base-scale: 0.92;
}}
.base-high {{
  --base-color: #d95f47;
  --base-scale: 1.28;
}}
.final-low {{
  --final-color: #23875a;
  --final-scale: 0.92;
}}
.final-high {{
  --final-color: #d95f47;
  --final-scale: 1.28;
}}
.flagged-dot {{
  outline: 3px solid #d6a21d;
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
  box-shadow: 0 0 0 3px rgba(35,135,90,0.24);
}}
.policy-panel.show-final .life-dot.changed-harmed {{
  box-shadow: 0 0 0 3px rgba(217,95,71,0.30);
}}
.life-course-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid #e8edf3;
  color: #5e6e85;
  font-size: 0.8rem;
  line-height: 1.4;
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
.legend-low {{ --legend-color: #23875a; }}
.legend-high {{ --legend-color: #d95f47; }}
@media (max-width: 700px) {{
  .life-course-card {{
    padding: 14px;
    border-radius: 10px;
  }}

  .policy-grid {{
    grid-template-columns: repeat(auto-fill, minmax(8px, 1fr));
    grid-auto-rows: 12px;
    gap: 3px;
    padding: 10px 8px;
  }}

  .life-dot {{
    width: 8px;
    height: 8px;
  }}
}}
</style>
<div class="life-course-card" id="{root_id}">
  <div class="life-course-header">
    <div class="life-course-title">{escape(title)}</div>
  </div>
  <div class="policy-panels">
    {''.join(panels)}
  </div>
  <div class="life-course-legend">
    <span class="legend-item"><span class="legend-dot legend-low"></span>green: low-risk / offense prevented</span>
    <span class="legend-item"><span class="legend-dot legend-high"></span>red: high-risk / offense remains or harmed by intervention</span>
    <span class="legend-item"><span class="legend-dot flagged-dot"></span>yellow outline: flagged by prediction</span>
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

}})();
</script>
"""


def render_population_animation(container, run_results, settings, title, animation_key="", root_id=None):
    html = population_animation_html(run_results, settings, title, animation_key, root_id=root_id)
    with container:
        st.html(html, unsafe_allow_javascript=True)


def render_population_update(update_slot, root_id, policy, panel_index, children, metrics):
    script = population_update_script(root_id, policy, panel_index, children, metrics)
    with update_slot:
        st.html(script, unsafe_allow_javascript=True)


def scroll_to_anchor(anchor_id):
    st.html(
        f"""<script>
setTimeout(function() {{
  var target = document.getElementById({json.dumps(anchor_id)});
  if (target) {{
    target.scrollIntoView({{behavior: 'smooth', block: 'center'}});
  }}
}}, 100);
</script>""",
        unsafe_allow_javascript=True,
    )


def render_sidebar_achievements():
    earned = st.session_state.get("earned_achievements", set())
    count = len(earned)
    label = f"🏆 Achievements  {count} / {len(ACHIEVEMENTS)}"
    with st.expander(label, expanded=False):
        for ach in ACHIEVEMENTS:
            if ach["id"] in earned:
                st.markdown(f"{ach['icon']} **{ach['name']}**")
            else:
                st.caption(f"🔒 {ach['name']}")


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

    parameter_summary = compact_parameter_summary(settings)
    system_prompt = DEFAULT_SYSTEM_PROMPT
    latest_result = st.session_state.get("llm_agent_latest_result")
    initial_population_results = None
    initial_population_title = "Live synthetic population view"
    if latest_result and latest_result_has_current_schema(latest_result):
        initial_population_results = latest_result["run_results"]
        initial_population_title = "Latest synthetic population view"

    live_population = st.empty()
    update_slot = st.empty()
    render_population_animation(
        live_population,
        initial_population_results,
        settings,
        initial_population_title,
        "initial",
        root_id=_LIVE_GRID_ID,
    )
    st.html('<div id="simulation-progress-anchor" style="height: 1px;"></div>')
    progress_slot = st.empty()
    live_status = st.empty()
    live_debrief = st.empty()

    st.sidebar.caption(
        f"Current run: {len(POLICIES)} policies × {len(selected_models)} model agent(s) = "
        f"{total_calls} LLM call(s). Each call generates {int(settings['llm_simulation_runs'])} "
        f"synthetic run(s) over {int(settings['population_size']):,} synthetic children."
    )
    settings_invalid = settings["true_high_risk_rate"] == 0
    if not selected_models:
        st.sidebar.warning("Select at least one LLM model agent.")
    if settings_invalid:
        st.sidebar.warning("Set 'Percentage of true high-risk children' above 0% to run a meaningful simulation.")

    run_disabled = not bool(api_key) or not selected_models or settings_invalid
    run_button_slot = st.sidebar.empty()
    if st.session_state.get("simulation_running", False):
        with run_button_slot:
            st.button(
                "Run simulation",
                key="run_simulation_running",
                disabled=True,
                type="primary",
                use_container_width=True,
            )
        run_requested = False
    else:
        with run_button_slot:
            run_requested = st.button(
                "Run simulation",
                key="run_simulation_start",
                disabled=run_disabled,
                type="primary",
                use_container_width=True,
            )

    if run_requested:
        st.session_state["simulation_running"] = True
        st.session_state["simulation_pending"] = True
        st.rerun()

    for level, message in st.session_state.pop("last_run_notices", []):
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.success(message)

    execute_run = st.session_state.pop("simulation_pending", False)
    if execute_run:
        with run_button_slot:
            st.button(
                "Run simulation",
                key="run_simulation_active",
                disabled=True,
                type="primary",
                use_container_width=True,
            )
        run_frames = []
        district_frames = []
        model_summaries = []
        all_representative_agents = []
        debrief_parts = []
        model_errors = []
        completion_notices = []
        completed_calls = 0
        progress_bar = progress_slot.progress(0.0)
        scroll_to_anchor("simulation-progress-anchor")
        live_status.write("Starting LLM-agent simulation...")
        children, _ = baseline_children(settings)
        render_population_animation(
            live_population,
            None,
            settings,
            "Simulating…",
            "waiting",
            root_id=_LIVE_GRID_ID,
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
                        aggregate_metrics = compact_aggregate_metrics(run_results)
                        run_frames.append(run_results)
                        district_frames.append(district_results)
                        all_representative_agents.extend(representative_agents)
                        if debrief_text:
                            debrief_parts.append(f"{model} | {policy}: {debrief_text}")
                        model_summaries.append(
                            {
                                "llm_model": model,
                                "policy": policy,
                                "aggregate_metrics": aggregate_metrics,
                                "debrief_text": debrief_text,
                            }
                        )
                        metrics = policy_transition_metrics(run_results, policy, settings)
                        render_population_update(
                            update_slot, _LIVE_GRID_ID, policy, POLICY_ORDER.index(policy), children, metrics
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
            completion_notices.append(("error", f"{error_label}: {error_message}"))

        if run_frames:
            combined_run_results = pd.concat(run_frames, ignore_index=True, copy=False)
            combined_district_results = pd.concat(district_frames, ignore_index=True, copy=False)
            combined_run_results, combined_district_results = optimize_result_frames(
                combined_run_results,
                combined_district_results,
            )
            debrief_text = "\n\n".join(debrief_parts)
            aggregate_metrics = compact_aggregate_metrics(combined_run_results)

            st.session_state["llm_agent_latest_result"] = {
                "run_results": combined_run_results,
                "district_results": combined_district_results,
                "model_results": model_summaries,
                "representative_agents": all_representative_agents,
                "debrief_text": debrief_text,
            }

            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "policy_summary": "All policies",
                "parameter_summary": parameter_summary,
                "representative_agent_count": len(all_representative_agents),
                "llm_model": ", ".join(unique_values(summary["llm_model"] for summary in model_summaries)),
                "aggregate_metrics": aggregate_metrics,
                "debrief_text": debrief_text,
            }
            add_llm_run_log_entry(entry, MAX_RUN_LOG_SIZE)

            simulation_count = st.session_state.get("simulation_run_count", 0) + 1
            st.session_state["simulation_run_count"] = simulation_count
            newly_earned = check_achievements(combined_run_results, settings, simulation_count)
            previously_earned = st.session_state.get("earned_achievements", set())
            st.session_state["earned_achievements"] = previously_earned | newly_earned
            for ach_id in newly_earned - previously_earned:
                ach = ACHIEVEMENT_INDEX[ach_id]
                completion_notices.append(
                    ("success", f"🏆 Achievement unlocked: {ach['icon']} **{ach['name']}** — {ach['description']}")
                )

            if model_errors:
                completion_notices.append(("warning", "LLM-agent simulation generated for the successful model agents."))
            else:
                completion_notices.append(("success", "LLM-agent simulation generated."))
            del run_frames, district_frames
        elif model_errors:
            completion_notices.append(("warning", "No LLM-agent simulation results were generated."))

        st.session_state["last_run_notices"] = completion_notices
        st.session_state["simulation_running"] = False
        st.rerun()

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
    render_global_styles()

    st.title("Predictive Justice Thought Experiment")
    st.markdown(
        "Suppose we could reliably predict, at age 10, who will become a violent criminal by age 30. "
        "What should we do with that information? This simulation compares three policy responses using "
        "synthetic LLM-generated life-course trajectories for 1 000 children."
    )
    render_reference_guide()

    achievements_slot = st.sidebar.container()
    settings = sidebar_inputs()
    render_llm_agent_section(settings)
    with achievements_slot:
        render_sidebar_achievements()
