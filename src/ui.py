import hashlib
import json
import os
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from html import escape
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import streamlit as st

from src.achievements import ACHIEVEMENT_INDEX, ACHIEVEMENTS, check_achievements
from src.charts import line_chart_png
from src.constants import (
    CHECK_DESCRIPTIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_PREDICTION_ERROR_RATE,
    DEFAULT_TRUE_HIGH_RISK_RATE,
    LLM_MODEL,
    DEFAULT_LLM_MODEL_OPTIONS,
    MAX_LLM_MODEL_AGENTS,
    MAX_PARALLEL_LLM_CALLS,
    MAX_RUN_LOG_SIZE,
    POLICY_DESCRIPTIONS,
    POLICIES,
    POPULATION_DOT_ANIMATION_SECONDS,
    POPULATION_DOT_STAGGER_GROUP,
    POPULATION_DOT_STAGGER_SECONDS,
    PREDICTION_ERROR_RATE_MAX,
    PREDICTION_ERROR_RATE_MIN,
    PROGRESS_NOTE_MIN_SECONDS,
    RESULT_METRIC_DESCRIPTIONS,
    SETTING_DESCRIPTIONS,
    TRUE_HIGH_RISK_RATE_MAX,
    TRUE_HIGH_RISK_RATE_MIN,
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
    clean_llm_policy_effects,
    compact_aggregate_metrics,
    derived_flagged_count,
    normalize_llm_policy_effects,
    optimize_result_frames,
    risk_signal_counts,
    run_results_from_policy_effects,
    validate_llm_policy_effects,
    validate_llm_tables,
)
from src.state import (
    add_llm_run_log_entry,
    attach_model_label,
    attach_policy_label,
    initialize_llm_state,
    latest_result_has_current_schema,
    normalize_representative_agents,
    simulation_settings_signature,
    trim_llm_run_log,
)
from src.tables import average_results_table, combined_policy_totals_table, formatted_average_results_table


_LIVE_GRID_ID = "livePopGrid"
_HOW_THIS_WORKS_EXPANDED_KEY = "how_this_works_expanded"
# Show the aggregate bubble population view.
SHOW_POPULATION_DOT_VIEW = True


def render_app_header():
    st.html(
        f"""
<header class="app-topbar">
  <div class="app-brand">
    <div class="brand-mark">PJ</div>
    <div>
      <div class="brand-kicker">Agent_Lab // v0.1</div>
      <div class="brand-title">Predictive Justice Simulator</div>
    </div>
  </div>
  <div class="app-status-chips">
    <span class="badge-chip">N = {DEFAULT_POPULATION_SIZE:,}</span>
    <span class="badge-chip">{len(POLICIES)} policies</span>
    <span class="badge-chip badge-live badge-live-status">live</span>
  </div>
</header>
        """
    )


def render_hero_summary(settings):
    population_size = int(settings["population_size"])
    true_high_risk_count = clamp_count(settings["true_high_risk_rate"] * population_size, population_size)
    false_positives, false_negatives, flagged_count = risk_signal_counts(true_high_risk_count, settings)
    cards = [
        ("Synthetic population", f"{population_size:,}", "Children in each run", ""),
        ("True high-risk children", f"{true_high_risk_count:,}", "Without intervention", "primary"),
        ("Prediction error", f"{false_positives + false_negatives:,}", "False positives + false negatives", "warning"),
        ("Positive predictions", f"{flagged_count:,}", "Policy exposure group", ""),
    ]
    card_html = []
    for label, value, caption, tone in cards:
        tone_class = f" hero-stat-{tone}" if tone else ""
        card_html.append(
            f"""
<div class="hero-stat-card">
  <div class="hero-stat-label">{escape(label)}</div>
  <div class="hero-stat-value{tone_class}">{escape(value)}</div>
  <div class="hero-stat-caption">{escape(caption)}</div>
</div>
            """
        )
    st.html(f'<div class="hero-stat-grid">{"".join(card_html)}</div>')


def render_hero_statement():
    st.html(
        f"""
<div class="hero-badges">
  <span class="badge-chip badge-live">Thought Experiment</span>
  <span class="badge-chip">Statistical</span>
  <span class="badge-chip">LLM</span>
  <span class="badge-chip">Agentic AI</span>
</div>
<section class="hero-copy">
  <h1 class="hero-question">
    <span class="hero-nowrap">Suppose we could reliably predict, at age <span class="hero-accent">10</span></span>,<br><span class="hero-nowrap">who will commit a serious harmful act by age <span class="hero-accent">30</span></span>.
  </h1>
  <p class="hero-subtitle">
    <em>What should we do with that information?</em><br>Run {DEFAULT_POPULATION_SIZE:,} synthetic lives through three policy responses &mdash; and watch the trade-offs come alive: prevented outcomes, false positives, false negatives, benefit, and harm.
  </p>
</section>
        """
    )


def render_population_view_overview(settings):
    population_size = int(settings["population_size"])
    policy_count = len(POLICIES)
    st.html(
        f"""
<section class="population-overview">
  <div class="population-overview-title">LIVE_SYNTHETIC_POPULATION_VIEW</div>
  <div class="population-overview-heading">{population_size:,} synthetic children &middot; {policy_count} parallel policies</div>
  <div class="population-legend">
    <span class="population-legend-item">
      <span class="population-legend-dot is-safe"></span>
      <span class="population-legend-text">
        <span class="population-legend-label">Not flagged &middot; low-risk</span>
        <span class="population-legend-desc">Not on the predicted-outcome path and not subject to policy action</span>
      </span>
    </span>
    <span class="population-legend-item">
      <span class="population-legend-dot is-diverted"></span>
      <span class="population-legend-text">
        <span class="population-legend-label">True positive &middot; predicted outcome prevented</span>
        <span class="population-legend-desc">On the predicted-outcome path, flagged, and shifted to low-risk</span>
      </span>
    </span>
    <span class="population-legend-item">
      <span class="population-legend-dot is-outcome-remains"></span>
      <span class="population-legend-text">
        <span class="population-legend-label">True positive &middot; predicted outcome remains</span>
        <span class="population-legend-desc">On the predicted-outcome path, flagged, and still high-risk after policy</span>
      </span>
    </span>
    <span class="population-legend-item">
      <span class="population-legend-dot is-missed"></span>
      <span class="population-legend-text">
        <span class="population-legend-label">False negative</span>
        <span class="population-legend-desc">Would have the predicted outcome but was not flagged — receives no intervention</span>
      </span>
    </span>
    <span class="population-legend-item">
      <span class="population-legend-dot is-wrong"></span>
      <span class="population-legend-text">
        <span class="population-legend-label">False positive</span>
        <span class="population-legend-desc">Not on the predicted-outcome path but flagged and subject to policy action</span>
      </span>
    </span>
  </div>
</section>
        """
    )


ACHIEVEMENT_ICON_BY_ID = {
    "first_run": "rocket",
    "do_no_harm": "dove",
    "full_comparison": "puzzle",
    "false_alarm": "siren",
    "tinkerer": "microscope",
    "crime_preventer": "hero",
    "crime_crusher": "burst",
    "base_rate_trap": "trap",
    "schrodinger": "puzzle",
    "helping_hundreds": "handshake",
    "overreaction": "shocked",
    "sharp_signal": "target",
    "high_risk_world": "flame",
    "night_owl": "owl",
}

_VALID_ICON_KEYS = frozenset({
    "rocket", "microscope", "dove", "siren", "hero", "burst",
    "trap", "target", "cat", "flame", "shocked", "handshake",
    "puzzle", "owl", "lock", "trophy",
})

ACHIEVEMENT_ICON_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "achievement-icons"


def achievement_icon_key(achievement_id: str | None, locked: bool = False) -> str:
    if locked:
        return "lock"
    return ACHIEVEMENT_ICON_BY_ID.get(achievement_id or "", "trophy")


@lru_cache(maxsize=None)
def achievement_icon_src(icon_key: str) -> str:
    valid_key = icon_key if icon_key in _VALID_ICON_KEYS else "trophy"
    payload = (ACHIEVEMENT_ICON_ASSET_DIR / f"{valid_key}.png").read_bytes()
    return f"data:image/png;base64,{b64encode(payload).decode('ascii')}"


def achievement_icon_html(icon_key: str, extra_class: str = "") -> str:
    icon_key = icon_key if icon_key in _VALID_ICON_KEYS or icon_key == "chevron" else "trophy"
    class_attr = f"achievement-icon achievement-icon-{icon_key}"
    if extra_class:
        class_attr = f"{class_attr} {extra_class}"
    if icon_key == "chevron":
        return f'<span class="{class_attr}" aria-hidden="true"></span>'
    return (
        f'<span class="{class_attr}" aria-hidden="true">'
        f'<img class="achievement-icon-img" src="{achievement_icon_src(icon_key)}" alt="" decoding="async">'
        "</span>"
    )


def render_achievement_notifications(notifications):
    if not notifications:
        return

    cards = []
    for index, achievement in enumerate(notifications):
        icon = achievement_icon_html(achievement_icon_key(achievement.get("id")))
        cards.append(
            f"""
<div class="achievement-toast" style="--toast-index:{index}">
  <span class="achievement-toast-check" aria-hidden="true"></span>
  {icon}
  <span>
    <span class="achievement-toast-title">{escape(achievement['name'])}</span>
    <span class="achievement-toast-desc">{escape(achievement['description'])}</span>
  </span>
</div>
            """
        )
    st.html(f'<div class="achievement-toast-stack">{"".join(cards)}</div>')


def render_global_styles():
    st.html(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --surface: #ffffff;
  --surface-muted: #f8faf7;
  --surface-panel: #fbfdfb;
  --line: #d6e1db;
  --line-soft: rgba(53, 88, 72, 0.13);
  --text: #071823;
  --text-muted: #48616a;
  --primary: #00a757;
  --primary-light: rgba(0, 167, 87, 0.10);
  --primary-strong: #008846;
  --accent: #dd2538;
  --amber: #f08a00;
  --success: #00a757;
  --frame-border: rgba(53, 88, 72, 0.17);
  --frame-border-soft: rgba(53, 88, 72, 0.12);
  --frame-border-strong: #b0bfba;
  --frame-bg: rgba(255, 255, 255, 0.86);
  --frame-bg-soft: rgba(255, 255, 255, 0.78);
  --frame-bg-strong: rgba(255, 255, 255, 0.94);
  --primary-border: rgba(0, 167, 87, 0.32);
  --amber-light: rgba(240, 138, 0, 0.07);
  --amber-border: rgba(240, 138, 0, 0.28);
  --accent-light: rgba(221, 37, 56, 0.06);
  --accent-border: rgba(221, 37, 56, 0.22);
  --risk-safe: rgba(157, 190, 168, 0.62);
  --risk-safe-soft: rgba(157, 190, 168, 0.58);
  --disabled-fill: #c8d4cf;
  --disabled-border: #b0bfba;
  --disabled-surface: rgba(53, 88, 72, 0.035);
  --disabled-surface-active: rgba(53, 88, 72, 0.065);
  --disabled-border-soft: rgba(53, 88, 72, 0.12);
  --disabled-border-active: rgba(53, 88, 72, 0.22);
  --disabled-text: rgba(72, 97, 106, 0.44);
  --disabled-text-active: rgba(72, 97, 106, 0.62);
  --disabled-track: rgba(200, 212, 207, 0.55);
  --disabled-tag: rgba(200, 212, 207, 0.50);
  --achievement-blue: #4f9fe8;
  --achievement-green: #6faf4f;
  --achievement-purple: #7a55d8;
  --radius-sm: 6px;
  --radius: 8px;
  --frame-padding: 1.15rem 1.35rem 1.25rem;
  --sidebar-toggle-top: 0.7rem;
  --sidebar-toggle-edge: 0.7rem;
  --sidebar-toggle-size: 2.2rem;
  --mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
  --shadow-xs: none;
  --shadow-sm: none;
  --shadow-md: none;
}

/* ── Base ─────────────────────────────────── */
html, body, .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.stApp {
  background:
    linear-gradient(rgba(0, 167, 87, 0.075) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 167, 87, 0.075) 1px, transparent 1px),
    linear-gradient(115deg, rgba(232, 255, 242, 0.92) 0%, rgba(247, 252, 249, 0.96) 49%, rgba(255, 247, 238, 0.93) 100%);
  background-size: 32px 32px, 32px 32px, auto;
  background-attachment: fixed;
  color: var(--text);
}

/* ── Streamlit chrome ─────────────────────── */
[data-testid="stHeader"] {
  height: 0;
  min-height: 0;
  background: transparent;
  overflow: visible;
  pointer-events: none;
}

[data-testid="stDecoration"],
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
#MainMenu {
  display: none !important;
}

[data-testid="stToolbar"] {
  background: transparent;
  pointer-events: none;
}

[data-testid="stExpandSidebarButton"] {
  position: fixed !important;
  top: var(--sidebar-toggle-top) !important;
  left: var(--sidebar-toggle-edge) !important;
  z-index: 999999;
  display: inline-grid !important;
  width: var(--sidebar-toggle-size);
  height: var(--sidebar-toggle-size);
  place-items: center;
  border: 1px solid var(--frame-border);
  border-radius: 8px;
  background: var(--frame-bg);
  box-shadow: var(--shadow-xs);
  pointer-events: auto;
}

[data-testid="stExpandSidebarButton"] * {
  pointer-events: auto;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  z-index: 999999 !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
  position: absolute !important;
  top: var(--sidebar-toggle-top) !important;
  right: var(--sidebar-toggle-edge) !important;
  margin: 0 !important;
  transform: none !important;
}

[data-testid="collapsedControl"] {
  position: fixed !important;
  top: var(--sidebar-toggle-top) !important;
  left: var(--sidebar-toggle-edge) !important;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
  width: var(--sidebar-toggle-size) !important;
  min-width: var(--sidebar-toggle-size) !important;
  height: var(--sidebar-toggle-size) !important;
  min-height: var(--sidebar-toggle-size) !important;
  margin: 0 !important;
}

/* ── Main content ─────────────────────────── */
[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 1232px;
  padding-top: 1.2rem;
  padding-bottom: 3rem;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
  position: relative;
  border-right: 1px solid var(--line-soft);
  box-shadow: none;
}

[data-testid="stSidebar"] > div:first-child {
  background:
    linear-gradient(rgba(0, 167, 87, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 167, 87, 0.05) 1px, transparent 1px),
    var(--frame-bg);
  background-size: 28px 28px, 28px 28px, auto;
  backdrop-filter: blur(14px);
  padding-top: 1.75rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > p {
  font-size: 0.82rem;
}

/* ── Typography ───────────────────────────── */
h1, h2, h3 {
  color: var(--text);
  letter-spacing: 0;
  font-weight: 700;
}

h1 {
  max-width: 760px;
  margin-top: 1.1rem;
  margin-bottom: 1rem;
  font-size: 4.1rem;
  line-height: 1.04;
  font-weight: 750;
}

h2 {
  margin-top: 1.1rem;
  font-size: 1.28rem;
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.18em;
}

h3 {
  margin-top: 1.05rem;
  font-size: 0.98rem;
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.16em;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color: var(--text-muted);
  line-height: 1.6;
  font-size: 1rem;
}

[data-testid="stMarkdownContainer"] strong {
  color: var(--text);
  font-weight: 600;
}

[data-testid="stCaptionContainer"] {
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.77rem;
  letter-spacing: 0.04em;
}

/* ── Lovable-style app chrome ─────────────── */
.app-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.8rem 0 0.95rem;
  margin: -0.25rem 0 1.35rem;
  border-bottom: 1px solid var(--line-soft);
}

.app-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.brand-mark {
  display: grid;
  width: 2rem;
  height: 2rem;
  place-items: center;
  border: 1px solid var(--primary-border);
  border-radius: 7px;
  background: var(--primary-light);
  color: var(--primary);
  font-family: var(--mono);
  font-size: 0.8rem;
  font-weight: 700;
}

.brand-kicker,
.badge-chip {
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.22em;
}

.brand-kicker {
  color: var(--text-muted);
  font-size: 0.66rem;
  line-height: 1.1;
}

.brand-title {
  color: var(--text);
  font-size: 0.98rem;
  font-weight: 700;
  line-height: 1.25;
}

.app-status-chips,
.hero-badges {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin: 0 0 0.65rem;
}

.hero-copy {
  max-width: 980px;
  margin-top: 0;
}

.hero-question {
  margin: 0 0 0;
  color: var(--text);
  font-size: clamp(1.4rem, 3.2vw, 2.8rem);
  font-weight: 750;
  line-height: 1.12;
  letter-spacing: 0;
}

.hero-question .hero-accent {
  color: var(--primary);
}

.hero-nowrap {
  white-space: nowrap;
}

.hero-subtitle {
  max-width: 760px;
  margin: 0.65rem 0 0;
  color: var(--text-muted);
  font-size: 1.28rem;
  line-height: 1.55;
}

.badge-chip {
  display: inline-flex;
  align-items: center;
  min-height: 1.55rem;
  padding: 0.25rem 0.7rem;
  border: 1px solid var(--line-soft);
  border-radius: 999px;
  background: var(--frame-bg-soft);
  color: var(--text-muted);
  font-size: 0.67rem;
  font-weight: 600;
}

.badge-live {
  border-color: var(--primary-border);
  color: var(--primary);
}

.badge-live-status {
  animation: liveStatusFade 3.2s ease-in-out infinite;
}

@keyframes liveStatusFade {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.18;
  }
}

@media (prefers-reduced-motion: reduce) {
  .badge-live-status {
    animation: none;
  }
}

.section-gap {
  height: 0.85rem;
}

.section-gap-lg {
  height: 1.15rem;
}

.section-gap-results {
  height: 1.8rem;
}

.section-gap-explanations {
  height: 1.35rem;
}

.section-gap-history {
  height: 2.05rem;
}

.hero-stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.7rem;
  margin: 1.55rem 0 0.95rem;
}

.hero-stat-card {
  min-height: 7.2rem;
  padding: 1.05rem 1.2rem;
  border: 1px solid var(--frame-border);
  border-radius: 8px;
  background: var(--frame-bg-soft);
  box-shadow: var(--shadow-xs);
}

.hero-stat-label,
.hero-stat-caption {
  color: var(--text-muted);
}

.hero-stat-label {
  font-family: var(--mono);
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  line-height: 1.35;
  text-transform: uppercase;
}

.hero-stat-value {
  margin-top: 0.55rem;
  color: var(--text);
  font-family: var(--mono);
  font-size: 2rem;
  line-height: 1;
  letter-spacing: 0;
}

.hero-stat-primary {
  color: var(--primary);
}

.hero-stat-warning {
  color: var(--amber);
}

.hero-stat-caption {
  margin-top: 0.55rem;
  font-size: 0.8rem;
  line-height: 1.35;
}

.population-overview {
  margin: 0.75rem 0 0.8rem;
  padding: var(--frame-padding);
  border: 1px solid var(--frame-border);
  border-radius: 8px;
  background: var(--frame-bg);
  box-shadow: var(--shadow-xs);
}

.population-overview-title {
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.24em;
  line-height: 1.35;
  text-transform: uppercase;
}

.population-overview-heading {
  margin-top: 0.32rem;
  color: var(--text);
  font-size: 1.55rem;
  font-weight: 750;
  line-height: 1.2;
}

.population-legend {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(18rem, 1fr));
  gap: 0.7rem 1.2rem;
  margin-top: 0.95rem;
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.86rem;
  line-height: 1.35;
}

.population-legend-item {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
}

.population-legend-text {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}

.population-legend-label {
  color: var(--text-primary, inherit);
  font-weight: 500;
}

.population-legend-desc {
  font-size: 0.77rem;
  opacity: 0.72;
  line-height: 1.3;
}

.population-legend-dot {
  display: inline-block;
  position: relative;
  overflow: hidden;
  width: 0.72rem;
  height: 0.72rem;
  flex: 0 0 auto;
  border-radius: 999px;
  box-sizing: border-box;
  margin-top: 0.18rem;
}

.population-legend-dot::after {
  content: none;
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: repeating-linear-gradient(
    135deg,
    rgba(0, 0, 0, 0.36) 0,
    rgba(0, 0, 0, 0.36) 1.5px,
    transparent 1.5px,
    transparent 5px
  );
}

.population-legend-dot.is-safe {
  background: var(--risk-safe);
}

.population-legend-dot.is-diverted {
  background: var(--primary);
}

.population-legend-dot.is-diverted::after,
.population-legend-dot.is-outcome-remains::after,
.population-legend-dot.is-wrong::after {
  content: "";
}

.population-legend-dot.is-outcome-remains {
  background: var(--accent);
}

.population-legend-dot.is-missed {
  background: var(--accent);
}

.population-legend-dot.is-wrong {
  background: var(--risk-safe);
}

.achievement-toast-stack {
  position: fixed;
  right: 1.5rem;
  bottom: 1.5rem;
  z-index: 999999;
  display: flex;
  flex-direction: column-reverse;
  gap: 0.8rem;
  width: min(24rem, calc(100vw - 2rem));
  pointer-events: none;
}

.achievement-toast {
  display: grid;
  grid-template-columns: 1.55rem 1.55rem minmax(0, 1fr);
  align-items: center;
  gap: 0.7rem;
  padding: 1rem 1.1rem;
  border: 1px solid var(--frame-border-soft);
  border-radius: 8px;
  background: var(--frame-bg-strong);
  box-shadow: none;
  animation: achievementToast 7s ease both;
  animation-delay: calc(var(--toast-index) * 130ms);
}

.achievement-toast-check {
  display: grid;
  width: 1.55rem;
  height: 1.55rem;
  place-items: center;
  border-radius: 999px;
  background: var(--text);
}

.achievement-toast-check::before {
  content: "";
  width: 0.72rem;
  height: 0.44rem;
  border-left: 2px solid var(--surface);
  border-bottom: 2px solid var(--surface);
  transform: rotate(-45deg) translate(1px, -1px);
}

.achievement-toast .achievement-icon {
  width: 1.55rem;
  height: 1.55rem;
}

.achievement-toast .achievement-icon-img {
  width: 1.55rem;
  height: 1.55rem;
}

.achievement-toast-title {
  display: block;
  color: var(--text);
  font-size: 1.05rem;
  font-weight: 750;
  line-height: 1.25;
}

.achievement-toast-desc {
  display: block;
  margin-top: 0.22rem;
  color: var(--text-muted);
  font-size: 0.92rem;
  line-height: 1.35;
}

@keyframes achievementToast {
  0% {
    opacity: 0;
    transform: translateY(18px) scale(0.98);
  }
  10%, 78% {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
  100% {
    opacity: 0;
    transform: translateY(10px) scale(0.98);
  }
}

/* ── Terminal progress ────────────────────── */
.terminal-progress-card {
  margin: 0.25rem 0 1rem;
  padding: 1rem 1.15rem 1.1rem;
  border: 1px solid var(--frame-border);
  border-radius: 8px;
  background: var(--frame-bg);
  box-shadow: var(--shadow-xs);
}

.terminal-progress-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.7rem;
  color: var(--primary);
  font-family: var(--mono);
}

.terminal-progress-title-wrap {
  display: inline-flex;
  align-items: center;
  gap: 0.62rem;
  min-width: 0;
}

.terminal-progress-dot {
  width: 0.62rem;
  height: 0.62rem;
  flex: 0 0 auto;
  border-radius: 999px;
  background: var(--primary);
}

.terminal-progress-title {
  font-size: 0.9rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  line-height: 1.2;
  text-transform: uppercase;
}

.terminal-progress-pct {
  flex: 0 0 auto;
  font-size: 0.9rem;
  font-weight: 700;
}

.terminal-progress-track {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.8rem 0.85rem;
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  background: var(--primary-light);
}

.terminal-progress-bracket {
  color: var(--primary);
  font-family: var(--mono);
  font-size: 0.82rem;
  font-weight: 700;
}

.terminal-progress-blocks {
  display: grid;
  grid-template-columns: repeat(32, minmax(0, 1fr));
  gap: 0.25rem;
  flex: 1 1 auto;
  min-width: 0;
}

.terminal-progress-block {
  height: 0.9rem;
  border-radius: 3px;
  background: var(--primary-light);
}

.terminal-progress-block.is-filled {
  background: var(--primary);
}

.terminal-progress-block.is-active {
  box-shadow: none;
}

.terminal-progress-status {
  margin-top: 0.7rem;
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.9rem;
  line-height: 1.4;
}

.terminal-progress-note {
  margin-top: 0.65rem;
  padding: 0.78rem 0.9rem;
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  background: var(--primary-light);
  color: var(--text-muted);
  font-size: 0.86rem;
  line-height: 1.45;
}

.terminal-progress-case {
  margin-top: 0.4rem;
}

.terminal-progress-prompt {
  color: var(--primary);
  font-weight: 700;
}

.terminal-progress-cursor {
  display: inline-block;
  width: 0.52rem;
  height: 0.92rem;
  margin-left: 0.25rem;
  background: var(--primary);
  vertical-align: -0.12rem;
  animation: terminalCursorBlink 1s steps(2, jump-none) infinite;
}

@keyframes terminalCursorBlink {
  0%, 45% { opacity: 1; }
  46%, 100% { opacity: 0; }
}

.achievement-shell {
  overflow: hidden;
  margin-bottom: 0.75rem;
  border: 1px solid var(--frame-border-strong);
  border-radius: 8px;
  background: var(--frame-bg);
  box-shadow: var(--shadow-xs);
  filter: none !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}

.achievement-shell summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  min-height: 3.45rem;
  padding: 0.9rem 1rem;
  cursor: pointer;
  list-style: none;
  border-bottom: 1px solid var(--line-soft);
  filter: none !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}

.achievement-shell summary::-webkit-details-marker {
  display: none;
}

.achievement-summary-main {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  min-width: 0;
}

.achievement-title {
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  white-space: nowrap;
}

.achievement-count {
  color: var(--text);
  font-family: var(--mono);
  font-size: 0.95rem;
  white-space: nowrap;
}

.achievement-count span {
  color: var(--text-muted);
}

[data-testid="stSidebar"] .achievement-shell {
  border-color: var(--frame-border-strong) !important;
}

[data-testid="stSidebar"] .achievement-shell,
[data-testid="stSidebar"] .achievement-shell summary {
  pointer-events: auto !important;
}

.achievement-toggle {
  display: inline-grid;
  width: 1.35rem;
  height: 1.35rem;
  place-items: center;
  color: var(--text-muted);
  line-height: 1;
  transition: transform 160ms ease;
}

.achievement-shell:not([open]) .achievement-toggle {
  transform: rotate(180deg);
}

.achievement-shell:not([open]) summary {
  border-bottom: 0;
}

.achievement-list {
  display: flex;
  flex-direction: column;
}

.achievement-row {
  display: grid;
  grid-template-columns: 1.6rem minmax(0, 1fr);
  gap: 0.75rem;
  padding: 0.85rem 1rem;
  border-bottom: 1px solid var(--line-soft);
}

.achievement-row:last-child {
  border-bottom: 0;
}

.achievement-row.is-locked {
  opacity: 0.52;
}

.achievement-icon {
  display: inline-grid;
  width: 1.45rem;
  height: 1.45rem;
  place-items: center;
  color: var(--amber);
  line-height: 1;
}

.achievement-icon-img {
  display: block;
  width: 1.38rem;
  height: 1.38rem;
  object-fit: contain;
}

.achievement-icon-rocket,
.achievement-icon-trap,
.achievement-icon-lock {
  color: var(--text-muted);
}

.achievement-icon-microscope,
.achievement-icon-target,
.achievement-icon-shocked {
  color: var(--achievement-blue);
}

.achievement-icon-dove,
.achievement-icon-handshake {
  color: var(--achievement-green);
}

.achievement-icon-siren,
.achievement-icon-flame {
  color: var(--accent);
}

.achievement-icon-hero,
.achievement-icon-puzzle {
  color: var(--achievement-purple);
}

.achievement-icon-burst,
.achievement-icon-cat,
.achievement-icon-owl,
.achievement-icon-trophy {
  color: var(--amber);
}

.achievement-icon-chevron {
  color: var(--text-muted);
}

.achievement-toggle::before {
  content: "";
  display: block;
  width: 0.68rem;
  height: 0.68rem;
  border-right: 2px solid currentColor;
  border-bottom: 2px solid currentColor;
  transform: rotate(225deg);
}

.achievement-row.is-locked .achievement-icon {
  color: var(--text-muted);
}

.achievement-name {
  color: var(--text);
  font-size: 0.92rem;
  font-weight: 700;
  line-height: 1.25;
}

.achievement-desc {
  margin-top: 0.18rem;
  color: var(--text-muted);
  font-size: 0.78rem;
  line-height: 1.35;
}

/* -- Sidebar simulation params --------------- */
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.settings-panel-title) {
  overflow: hidden;
  margin-bottom: 0.75rem;
  border: 1px solid var(--frame-border);
  border-radius: 8px;
  background: var(--frame-bg);
  box-shadow: var(--shadow-xs);
}

[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.settings-panel-title) > div {
  padding: 0.9rem;
}

[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.settings-panel-title) [data-testid="stVerticalBlock"] {
  gap: 0.42rem;
}

.settings-panel-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.55rem;
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.73rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  line-height: 1.35;
  text-transform: uppercase;
}

.settings-panel-dot {
  width: 0.48rem;
  height: 0.48rem;
  border-radius: 999px;
  background: var(--primary);
  box-shadow: 0 0 0 4px var(--primary-light);
}

.settings-field-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.75rem;
  margin-top: 0.38rem;
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  line-height: 1.35;
  text-transform: uppercase;
}

.settings-field-value {
  color: var(--primary);
  font-size: 1.15rem;
  letter-spacing: 0;
  white-space: nowrap;
}

.settings-field-copy {
  margin: 0.12rem 0 0.32rem;
  color: var(--text-muted);
  font-size: 0.78rem;
  line-height: 1.35;
}

.settings-panel-divider {
  height: 1px;
  margin: 0.52rem 0 0.18rem;
  background: var(--frame-border-soft);
}

.settings-kv {
  display: grid;
  gap: 0.38rem;
  margin-bottom: 0.2rem;
}

.settings-kv-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.82rem;
  line-height: 1.35;
}

.settings-kv-row strong {
  color: var(--text);
  font-weight: 500;
  text-align: right;
  white-space: nowrap;
}

/* ── Expanders ────────────────────────────── */
[data-testid="stExpander"] details {
  background: var(--frame-bg-soft);
  border: 1px solid var(--frame-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-xs);
  transition: box-shadow 180ms ease, border-color 180ms ease;
}

[data-testid="stExpander"] details:hover {
  box-shadow: var(--shadow-sm);
  border-color: var(--line);
}

[data-testid="stExpander"] details[open] {
  border-color: var(--primary-border);
  box-shadow: var(--shadow-sm);
}

[data-testid="stExpander"] summary {
  font-family: var(--mono);
  font-weight: 600;
  font-size: 0.75rem;
  color: var(--text);
  padding: 0.72rem 0.9rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
  min-height: 2.55rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  font-weight: 500;
  font-size: 0.88rem;
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  box-shadow: var(--shadow-xs);
  transition: border-color 140ms ease, box-shadow 140ms ease, transform 120ms ease;
}

.stButton > button:hover:not(:disabled) {
  border-color: var(--primary-border);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.stButton > button:active:not(:disabled) {
  transform: translateY(0);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, var(--primary) 0%, var(--primary-strong) 100%);
  border-color: var(--primary-strong);
  color: var(--surface);
  box-shadow: none;
}

.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p {
  color: var(--surface) !important;
  font-weight: 600;
}

.stButton > button[kind="primary"]:hover:not(:disabled) {
  background: linear-gradient(180deg, var(--primary) 0%, var(--primary-strong) 100%);
  box-shadow: none;
}

/* ── Form controls ────────────────────────── */
[data-baseweb="select"] > div {
  border-radius: var(--radius-sm) !important;
}

[data-testid="stSlider"] [role="slider"] {
  background-color: var(--primary);
  border-color: var(--surface);
  box-shadow: 0 0 0 3px var(--primary-light);
}

[data-testid="stSlider"] div[data-testid="stTickBar"] div {
  background: var(--line);
}

[data-testid="stSidebar"] [data-testid="stSlider"] {
  margin-top: -0.3rem;
  margin-bottom: 0.05rem;
}

[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
  background: var(--surface);
  border: 1px solid var(--primary);
}

[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
[data-testid="stSidebar"] [data-testid="stSliderTickBar"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBar"] {
  display: none;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] [data-baseweb="button-group"],
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
  width: 100%;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] [data-baseweb="button-group"],
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] [role="radiogroup"] {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  width: 100%;
  border: 0 !important;
  overflow: visible;
}

[data-testid="stSidebar"] [data-testid="stSegmentedControl"] label {
  width: 100%;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"],
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] label > div {
  justify-content: center;
  width: 100%;
  min-height: 2.5rem;
  min-width: 0;
  padding-left: 0.18rem;
  padding-right: 0.18rem;
  border: 0 !important;
  border-radius: 8px;
  background: var(--frame-bg-soft);
  box-shadow: inset 0 0 0 1px var(--line);
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  overflow: visible;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"] > div,
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"] span,
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"] div[data-testid="stMarkdownContainer"] {
  max-width: none;
  min-width: 0;
  overflow: visible;
  text-overflow: clip;
  white-space: nowrap;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"] p {
  color: inherit;
  font-family: var(--mono);
  font-size: 0.64rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  line-height: 1;
  text-transform: uppercase;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] label:has(input:checked) > div {
  border-color: transparent;
  background: var(--primary-light);
  box-shadow: inset 0 0 0 1.5px var(--primary);
  color: var(--primary);
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
  border-color: var(--line);
  background: var(--frame-bg-soft);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.9rem;
  min-height: 3.25rem;
  margin-top: 0.4rem;
  padding: 0.65rem 1.2rem;
  border-radius: 8px;
  background: var(--primary);
  border-color: var(--primary-strong);
  box-shadow: none;
  transition: none !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover:not(:disabled),
[data-testid="stSidebar"] .stButton > button[kind="primary"]:active:not(:disabled),
[data-testid="stSidebar"] .stButton > button[kind="primary"]:focus:not(:disabled) {
  background: var(--primary) !important;
  box-shadow: none !important;
  transform: none !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] > div {
  flex: 0 0 auto;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] p {
  font-size: 0.95rem;
  letter-spacing: 0.16em;
  line-height: 1;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled {
  background: var(--disabled-fill) !important;
  border-color: var(--disabled-border) !important;
  color: var(--disabled-text-active) !important;
  box-shadow: none !important;
  opacity: 1 !important;
  cursor: not-allowed !important;
  pointer-events: auto !important;
  transform: none !important;
  transition: none !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled *,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled p {
  color: var(--disabled-text-active) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled:active {
  background: var(--disabled-fill) !important;
  box-shadow: none !important;
  transform: none !important;
}

/* ── Disabled state (simulation running) ──── */
.st-key-simulation_params_panel:has(input:disabled) .settings-field-head {
  opacity: 0.45;
}

.st-key-simulation_params_panel:has(input:disabled) .settings-field-copy {
  opacity: 0.38;
}

.st-key-simulation_params_panel:has(input:disabled) .settings-panel-title {
  color: var(--disabled-text-active);
}

.st-key-simulation_params_panel:has(input:disabled) .settings-panel-dot {
  background: var(--disabled-fill);
  box-shadow: 0 0 0 4px var(--disabled-surface-active);
}

.st-key-simulation_params_panel:has(input:disabled) .settings-panel-divider {
  background: var(--disabled-border-soft);
}

.st-key-simulation_params_panel:has(input:disabled) .settings-kv-row {
  color: var(--disabled-text);
}

.st-key-simulation_params_panel:has(input:disabled) .settings-kv-row strong {
  color: var(--disabled-text-active);
}

[data-testid="stSidebar"] [data-testid="stSlider"]:has(input:disabled) [role="slider"] {
  background: var(--disabled-fill) !important;
  border-color: var(--disabled-border) !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"]:has(input:disabled) > div > div {
  background: var(--disabled-track) !important;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"]:disabled,
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"][disabled],
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"][aria-disabled="true"],
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] label:has(input:disabled) > div {
  background: var(--disabled-surface) !important;
  box-shadow: inset 0 0 0 1px var(--disabled-border-soft) !important;
  color: var(--disabled-text) !important;
  cursor: not-allowed !important;
  opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind="segmented_controlActive"]:disabled,
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind="segmented_controlActive"][disabled],
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind="segmented_controlActive"][aria-disabled="true"],
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] label:has(input:disabled:checked) > div {
  background: var(--disabled-surface-active) !important;
  box-shadow: inset 0 0 0 1.5px var(--disabled-border-active) !important;
  color: var(--disabled-text-active) !important;
  cursor: not-allowed !important;
  opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"]:disabled p,
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"][disabled] p,
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind^="segmented_control"][aria-disabled="true"] p {
  color: inherit !important;
}

[data-testid="stSidebar"] [data-testid="stMultiSelect"]:has(input:disabled) [data-baseweb="select"] > div {
  background: var(--disabled-surface) !important;
  border-color: var(--disabled-border-soft) !important;
  color: var(--disabled-text) !important;
  cursor: not-allowed !important;
}

[data-testid="stSidebar"] [data-testid="stMultiSelect"]:has(input:disabled) [data-baseweb="tag"] {
  background: var(--disabled-tag) !important;
  color: var(--disabled-text-active) !important;
  opacity: 0.70;
}

/* Achievements are never disabled, even while simulation inputs are locked. */
[data-testid="stSidebar"] .achievement-shell {
  background: var(--frame-bg) !important;
  color: var(--text) !important;
  filter: none !important;
  opacity: 1 !important;
}

[data-testid="stSidebar"] .achievement-shell,
[data-testid="stSidebar"] .achievement-shell summary,
[data-testid="stSidebar"] .achievement-shell .achievement-summary-main,
[data-testid="stSidebar"] .achievement-shell .achievement-title,
[data-testid="stSidebar"] .achievement-shell .achievement-count,
[data-testid="stSidebar"] .achievement-shell .achievement-count span,
[data-testid="stSidebar"] .achievement-shell .achievement-toggle,
[data-testid="stSidebar"] .achievement-shell summary > .achievement-icon {
  filter: none !important;
  opacity: 1 !important;
}

[data-testid="stSidebar"] .achievement-shell .achievement-title,
[data-testid="stSidebar"] .achievement-shell .achievement-count span,
[data-testid="stSidebar"] .achievement-shell .achievement-toggle,
[data-testid="stSidebar"] .achievement-shell .achievement-icon-chevron {
  color: var(--text-muted) !important;
}

[data-testid="stSidebar"] .achievement-shell .achievement-count,
[data-testid="stSidebar"] .achievement-shell .achievement-name {
  color: var(--text) !important;
}

[data-testid="stSidebar"] .achievement-shell .achievement-desc {
  color: var(--text-muted) !important;
}

[data-testid="stSidebar"] .achievement-shell summary .achievement-icon-trophy {
  color: var(--amber) !important;
}

/* Keep the achievements summary color stable in every app state. */
[data-testid="stSidebar"] .achievement-shell summary .achievement-title {
  color: #48616a !important;
}

[data-testid="stSidebar"] .achievement-shell summary .achievement-count {
  color: #071823 !important;
}

[data-testid="stSidebar"] .achievement-shell summary .achievement-count span {
  color: #48616a !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--text-muted);
  font-size: 0.74rem;
  line-height: 1.35;
}

/* ── Tabs ─────────────────────────────────── */
[data-baseweb="tab-list"] {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  width: 100%;
  margin: 0.9rem 0 1.2rem;
  border-bottom: 0;
  overflow: visible;
}

[data-baseweb="tab-border"],
[data-baseweb="tab-highlight"] {
  display: none;
}

[data-baseweb="tab"] {
  justify-content: center;
  width: 100%;
  min-width: 0;
  min-height: 2.5rem;
  padding: 0.5rem;
  border-radius: 8px;
  background: var(--frame-bg-soft);
  box-shadow: inset 0 0 0 1px var(--line);
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.10em;
  line-height: 1.35;
  text-align: center;
  text-transform: uppercase;
  white-space: normal;
  opacity: 1 !important;
  transition: none !important;
}

[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  color: var(--text);
  background: var(--frame-bg-strong);
}

[data-baseweb="tab"][aria-selected="true"] {
  background: var(--primary-light) !important;
  color: var(--primary) !important;
  box-shadow: inset 0 0 0 1.5px var(--primary) !important;
}

[data-baseweb="tab"] *,
[data-baseweb="tab"] p {
  color: inherit !important;
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.10em;
  line-height: 1.35;
  text-align: center;
  text-transform: uppercase;
  white-space: normal;
  opacity: 1 !important;
  visibility: visible !important;
}

/* ── Policy segmented control (main content) ─ */
[data-testid="stMain"] [data-testid="stButtonGroup"] [data-baseweb="button-group"],
[data-testid="stMain"] [data-testid="stSegmentedControl"] [role="radiogroup"] {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  width: 100%;
  border: 0 !important;
  overflow: visible;
}

[data-testid="stMain"] [data-testid="stSegmentedControl"] label {
  width: 100%;
}

[data-testid="stMain"] [data-testid="stButtonGroup"] button[kind^="segmented_control"],
[data-testid="stMain"] [data-testid="stSegmentedControl"] label > div {
  justify-content: center;
  text-align: center;
  width: 100%;
  min-height: 2.5rem;
  min-width: 0;
  padding: 0.5rem 0.5rem;
  border: 0 !important;
  border-radius: 8px;
  background: var(--frame-bg-soft);
  box-shadow: inset 0 0 0 1px var(--line);
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  white-space: normal;
  line-height: 1.35;
  overflow: visible;
}

[data-testid="stMain"] [data-testid="stButtonGroup"] button[kind^="segmented_control"] p,
[data-testid="stMain"] [data-testid="stSegmentedControl"] label > div p {
  color: inherit;
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.10em;
  line-height: 1.35;
  text-transform: uppercase;
  white-space: normal;
}

[data-testid="stMain"] [data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stMain"] [data-testid="stSegmentedControl"] label:has(input:checked) > div {
  background: var(--primary-light) !important;
  box-shadow: inset 0 0 0 1.5px var(--primary) !important;
  color: var(--primary) !important;
}

/* ── Metric cards ─────────────────────────── */
div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMetric"]) {
  background: var(--frame-bg);
  border: 1px solid var(--frame-border);
  border-radius: var(--radius);
  padding: 0.9rem 1rem;
  box-shadow: var(--shadow-xs);
  transition: box-shadow 150ms ease;
}

div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMetric"]):hover {
  box-shadow: var(--shadow-sm);
}

/* ── Data tables ──────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--frame-border);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow-xs);
}

/* ── Alerts ───────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius);
  box-shadow: var(--shadow-xs);
  font-size: 0.9rem;
}

[data-testid="stAlert"][kind="info"],
[data-testid="stAlert"] [kind="info"],
[data-baseweb="notification"][kind="info"] {
  background: var(--primary-light) !important;
  border-color: var(--primary-border) !important;
}

[data-testid="stAlert"][kind="success"],
[data-testid="stAlert"] [kind="success"],
[data-baseweb="notification"][kind="positive"] {
  background: var(--primary-light) !important;
  border-color: var(--primary-border) !important;
}

[data-testid="stAlert"][kind="warning"],
[data-testid="stAlert"] [kind="warning"],
[data-baseweb="notification"][kind="warning"] {
  background: var(--amber-light) !important;
  border-color: var(--amber-border) !important;
}

[data-testid="stAlert"][kind="error"],
[data-testid="stAlert"] [kind="error"],
[data-baseweb="notification"][kind="negative"] {
  background: var(--accent-light) !important;
  border-color: var(--accent-border) !important;
}

/* ── Results content panel (tables + charts only) ── */
.st-key-results_content_panel {
  box-sizing: border-box;
  overflow: hidden;
  margin: 0.75rem 0 0.8rem;
  padding: var(--frame-padding) !important;
  border: 1px solid var(--frame-border) !important;
  outline: 0 !important;
  border-radius: 8px !important;
  background: var(--surface) !important;
  box-shadow: var(--shadow-xs) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"].st-key-results_content_panel,
div[data-testid="stVerticalBlockBorderWrapper"].st-key-results_content_panel > div,
.st-key-results_content_panel > div {
  outline: 0 !important;
  background: var(--surface) !important;
  box-shadow: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"].st-key-results_content_panel > div,
.st-key-results_content_panel > div[data-testid="stVerticalBlock"] {
  padding: 0 !important;
}

.st-key-results_content_panel [data-testid="stVerticalBlock"] {
  gap: 0.65rem;
}

.st-key-results_content_panel h3 {
  margin-top: 0 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.14em !important;
  line-height: 1.35 !important;
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
    padding-top: 1rem;
  }

  h1 {
    font-size: 2.35rem;
    line-height: 1.08;
  }

  [data-baseweb="tab-list"] {
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }

  [data-baseweb="tab"] {
    min-height: 2.35rem;
    padding-left: 0.65rem;
    padding-right: 0.65rem;
  }

  .app-topbar {
    align-items: flex-start;
    flex-direction: column;
    margin-bottom: 1rem;
  }

  .app-status-chips {
    gap: 0.4rem;
  }

  .hero-stat-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .hero-question {
    font-size: 2.55rem;
    line-height: 1.08;
  }

  .hero-subtitle {
    margin-top: 0.8rem;
    font-size: 1.05rem;
  }

  .population-overview {
    padding: 1rem;
  }

  .population-overview-heading {
    font-size: 1.25rem;
  }

  .terminal-progress-card {
    padding: 1rem;
  }

  .terminal-progress-header {
    align-items: flex-start;
  }

  .terminal-progress-title {
    font-size: 0.78rem;
    letter-spacing: 0.14em;
  }

  .terminal-progress-track {
    padding: 0.7rem;
  }

  .terminal-progress-blocks {
    grid-template-columns: repeat(16, minmax(0, 1fr));
    gap: 0.22rem;
  }
}

@media (max-width: 520px) {
  .hero-stat-grid {
    grid-template-columns: 1fr;
  }

  .hero-question {
    font-size: 1.95rem;
  }

  .achievement-toast {
    grid-template-columns: 1.35rem 1.35rem minmax(0, 1fr);
    padding: 0.9rem 1rem;
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
    preferred_models = unique_values([LLM_MODEL, *DEFAULT_LLM_MODEL_OPTIONS])
    defaults = [model for model in preferred_models if model in model_options]
    if not defaults:
        defaults = model_options
    return defaults[:MAX_LLM_MODEL_AGENTS]


def clamp_selected_models(model_key, model_options):
    selected = st.session_state.get(model_key)
    if not isinstance(selected, list):
        return

    valid_selection = [model for model in selected if model in model_options]
    if len(valid_selection) > MAX_LLM_MODEL_AGENTS:
        valid_selection = valid_selection[:MAX_LLM_MODEL_AGENTS]
    if valid_selection != selected:
        st.session_state[model_key] = valid_selection


def glossary_markdown(items):
    return "\n".join(f"- **{item}:** {meaning}" for item, meaning in items.items())


@st.cache_data(show_spinner=False)
def cached_baseline_children(population_size, true_high_risk_rate, prediction_noise):
    settings = {
        "population_size": int(population_size),
        "true_high_risk_rate": float(true_high_risk_rate),
        "prediction_noise": float(prediction_noise),
    }
    population_size = settings["population_size"]
    true_high = clamp_count(settings["true_high_risk_rate"] * population_size, population_size)
    false_positives, false_negatives, flagged = risk_signal_counts(true_high, settings)
    true_positive = max(0, true_high - false_negatives)
    low_unflagged = max(0, population_size - true_positive - false_negatives - false_positives)

    children = (
        [{"base": "high", "flagged": True}] * true_positive
        + [{"base": "high", "flagged": False}] * false_negatives
        + [{"base": "low", "flagged": True}] * false_positives
        + [{"base": "low", "flagged": False}] * low_unflagged
    )
    seed = population_size * 17 + true_high * 31 + flagged * 43
    rng = np.random.default_rng(seed)
    rng.shuffle(children)
    return children


def baseline_children(settings):
    return cached_baseline_children(
        int(settings["population_size"]),
        float(settings["true_high_risk_rate"]),
        float(settings["prediction_noise"]),
    )


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
    prevented_value = averages.get("crimes_prevented")
    prevented = 0 if pd.isna(prevented_value) else int(round(float(prevented_value)))
    prevented = max(-population_size, min(population_size, prevented))

    return {
        "prevented": prevented,
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

    seed = stable_seed(seed_text) + len(indices) * 97 + count * 193
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return set(indices[:count])


def stable_seed(seed_text):
    return int.from_bytes(hashlib.blake2s(seed_text.encode("utf-8"), digest_size=8).digest(), "big")


POLICY_BUBBLE_METADATA = {
    "Targeted support for high-risk children": {
        "title": "Targeted Support",
        "description": "Voluntary counselling, mentoring, and practical assistance for flagged children.",
        "border": "var(--primary)",
    },
    "Surveillance of high-risk children": {
        "title": "Surveillance",
        "description": "Flagged children are monitored, reviewed, or recorded more closely.",
        "border": "var(--primary)",
    },
    "Coercive preventive intervention for high-risk children": {
        "title": "Coercive Prevention",
        "description": "Flagged children face mandatory requirements or restrictions before the predicted outcome occurs.",
        "border": "var(--primary)",
    },
}

BUBBLE_LAYOUT_WIDTH = 1500
BUBBLE_LAYOUT_HEIGHT = 620
BUBBLE_LAYOUT_MARGIN = 40
BUBBLE_LAYOUT_GAP = 30
BUBBLE_LAYOUT_CLUSTER_GAP = 78
BUBBLE_LAYOUT_GROUPS = (
    ("safe", "wrong", "diverted"),
    ("missed", "failed", "added"),
)
BUBBLE_CATEGORY_Y = {
    "safe": 310,
    "wrong": 330,
    "diverted": 265,
    "missed": 285,
    "failed": 320,
    "added": 265,
}

BUBBLE_CATEGORY_COLORS = {
    "safe": "#bed9c6",
    "diverted": "#00a757",
    "failed": "#c90024",
    "wrong": "#79c992",
    "missed": "#e3293c",
    "added": "#870016",
}

BUBBLE_FLAGGED_CATEGORIES = {"wrong", "diverted", "failed", "added"}

BUBBLE_CATEGORY_Z_INDEX = {
    "safe": 1,
    "failed": 2,
    "wrong": 3,
    "diverted": 4,
    "missed": 5,
    "added": 6,
}

BUBBLE_FLOAT_OFFSETS = {
    "safe": (0, -13),
    "wrong": (12, 8),
    "diverted": (-10, 9),
    "missed": (9, -11),
    "failed": (-9, 11),
    "added": (12, -8),
}

BUBBLE_FLOAT_DURATIONS = {
    "safe": 8.1,
    "wrong": 6.9,
    "diverted": 7.5,
    "missed": 6.8,
    "failed": 7.9,
    "added": 7.2,
}

def policy_bubble_counts(metrics, settings):
    population_size = int(settings["population_size"])
    baseline = clamp_count(settings["true_high_risk_rate"] * population_size, population_size)
    false_positives, false_negatives, _ = risk_signal_counts(baseline, settings)
    prevented = 0
    has_result = bool(metrics)

    if metrics:
        baseline = clamp_count(metrics.get("baseline"), population_size)
        false_positives = clamp_count(metrics.get("false_positives"), population_size)
        false_negatives = clamp_count(metrics.get("false_negatives"), population_size)
        prevented = int(metrics.get("prevented") or 0)

    flagged_true_positives = max(0, baseline - false_negatives)
    diverted = min(max(0, prevented), flagged_true_positives)
    added = min(max(0, -prevented), false_positives)
    failed = max(0, flagged_true_positives - diverted)
    wrongly_flagged = max(0, false_positives - added)
    missed = max(0, false_negatives)
    safe = max(0, population_size - baseline - false_positives)
    outcomes_after_policy = failed + missed + added

    return {
        "safe": safe,
        "diverted": diverted,
        "failed": failed,
        "wrong": wrongly_flagged,
        "missed": missed,
        "added": added,
        "outcomes_after_policy": outcomes_after_policy,
        "net_prevented": diverted - added,
        "baseline": baseline,
        "has_result": has_result,
    }


def bubble_radius(value, population_size):
    if value <= 0:
        return 0
    ratio = max(0, min(1, value / max(population_size, 1)))
    return max(41, min(261, 35 + 226 * (ratio ** 0.34)))


def bubble_layout(counts, population_size):
    groups = []
    radii = {}
    for group in BUBBLE_LAYOUT_GROUPS:
        visible_group = []
        for category in group:
            value = counts.get(category, 0)
            if value <= 0:
                continue
            radii[category] = bubble_radius(value, population_size)
            visible_group.append(category)
        if visible_group:
            groups.append(visible_group)

    if not groups:
        return {}

    intra_group_gaps = sum(max(0, len(group) - 1) for group in groups)
    cluster_gaps = max(0, len(groups) - 1)
    gap_width = (
        intra_group_gaps * BUBBLE_LAYOUT_GAP
        + cluster_gaps * BUBBLE_LAYOUT_CLUSTER_GAP
    )
    fixed_width = BUBBLE_LAYOUT_MARGIN * 2 + gap_width
    diameter_width = sum(2 * radii[category] for group in groups for category in group)
    if fixed_width + diameter_width > BUBBLE_LAYOUT_WIDTH:
        scale = (BUBBLE_LAYOUT_WIDTH - fixed_width) / max(diameter_width, 1)
        radii = {
            category: max(41, radius * scale)
            for category, radius in radii.items()
        }
        diameter_width = sum(2 * radii[category] for group in groups for category in group)

    layout = {}
    content_width = diameter_width + gap_width
    cursor = max(BUBBLE_LAYOUT_MARGIN, (BUBBLE_LAYOUT_WIDTH - content_width) / 2)
    for group_index, group in enumerate(groups):
        if group_index:
            cursor += BUBBLE_LAYOUT_CLUSTER_GAP
        for item_index, category in enumerate(group):
            if item_index:
                cursor += BUBBLE_LAYOUT_GAP
            radius = radii[category]
            x = cursor + radius
            layout[category] = (x, BUBBLE_CATEGORY_Y[category], radius)
            cursor = x + radius
    return layout


def outcome_delta_text(counts):
    baseline = max(1, counts["baseline"])
    reduction = (counts["net_prevented"] / baseline) * 100
    if abs(reduction) < 0.05:
        return "0%"
    return f"{reduction:+.0f}%".replace("+", "").replace("-", "−")


def bubble_font_size(value, radius):
    digits = len(f"{value:,}")
    if digits <= 2:
        factor = 0.38
    elif digits <= 3:
        factor = 0.30
    else:
        factor = 0.20
    return max(24, min(48, radius * factor))


def bubble_node_html(category, value, layout_entry, index):
    if value <= 0:
        return ""
    x, y, radius = layout_entry
    color = BUBBLE_CATEGORY_COLORS[category]
    font_size = bubble_font_size(value, radius)
    dx, dy = BUBBLE_FLOAT_OFFSETS.get(category, (2, -2))
    duration = BUBBLE_FLOAT_DURATIONS.get(category, 8.0)
    begin = -(index * 0.7)
    soft_dx = dx * 0.24
    soft_dy = -dy * 0.36
    left = (x / BUBBLE_LAYOUT_WIDTH) * 100
    top = (y / BUBBLE_LAYOUT_HEIGHT) * 100
    size = (radius * 2 / BUBBLE_LAYOUT_WIDTH) * 100
    flagged_class = "is-flagged" if category in BUBBLE_FLAGGED_CATEGORIES else "is-unflagged"
    return f"""
<div class="bubble-position bubble-{category}" style="--bubble-left:{left:.2f}%; --bubble-top:{top:.2f}%; --bubble-size:{size:.2f}%; --bubble-color:{color}; --bubble-font:{font_size:.2f}px; --bubble-delay:{index * 90}ms; --float-x:{dx}px; --float-y:{dy}px; --float-x-soft:{soft_dx:.2f}px; --float-y-soft:{soft_dy:.2f}px; --float-x-neg:{-dx}px; --float-y-neg:{-dy}px; --float-duration:{duration:.1f}s; --float-delay:{begin:.1f}s;">
  <div class="bubble-float">
    <div class="bubble-node {flagged_class}">
      <span class="bubble-value">{escape(f"{value:,}")}</span>
    </div>
  </div>
</div>
    """


def policy_bubble_card(policy, metrics, settings, index):
    meta = POLICY_BUBBLE_METADATA[policy]
    counts = policy_bubble_counts(metrics, settings)
    population_size = int(settings["population_size"])
    outcome_delta = outcome_delta_text(counts) if counts["has_result"] else "—"
    delta_class = "is-worse" if counts["net_prevented"] < 0 else "is-better"
    status_class = "is-ready" if counts["has_result"] else "is-waiting"
    layout = bubble_layout(counts, population_size)

    bubble_nodes = [
        bubble_node_html(category, counts[category], layout[category], index)
        for index, category in enumerate(["safe", "wrong", "diverted", "missed", "failed", "added"])
        if category in layout
    ]

    metric_items = [
        ("POST-POLICY OUTCOMES", counts["outcomes_after_policy"]),
        ("PREVENTED OUTCOMES", counts["diverted"]),
        ("FALSE POSITIVES", counts["wrong"] + counts["added"]),
        ("FALSE NEGATIVES", counts["missed"]),
    ]
    metric_html = "".join(
        f"""
<div class="bubble-card-metric">
  <div class="bubble-card-metric-label">{escape(label)}</div>
  <div class="bubble-card-metric-value">{value:,}</div>
</div>
        """
        for label, value in metric_items
    )

    return f"""
<section class="bubble-policy-card {status_class}" style="--policy-border:{meta['border']}; --card-delay:{index * 110}ms">
  <div class="bubble-policy-header">
    <div>
      <h3>{escape(meta["title"])}</h3>
      <p>{escape(meta["description"])}</p>
    </div>
    <div class="bubble-outcome-delta {delta_class}">
      <span>OUTCOME REDUCTION</span>
      <strong>{escape(outcome_delta)}</strong>
    </div>
  </div>
  <div class="bubble-chart-frame">
    <div class="bubble-chart-stage">
      {''.join(bubble_nodes)}
    </div>
  </div>
  <div class="bubble-card-metrics">
    {metric_html}
  </div>
</section>
    """


def bubble_population_animation_html(policy_metrics_by_policy, true_high_risk_rate, prediction_noise, population_size, title, animation_key="", root_id=None):
    settings = {
        "population_size": population_size,
        "true_high_risk_rate": true_high_risk_rate,
        "prediction_noise": prediction_noise,
    }
    if root_id is None:
        seed_text = f"{title}-{animation_key}"
        root_id = f"policyBubble{stable_seed(seed_text) % 100000}"
    metrics_by_policy = policy_metrics_by_policy if isinstance(policy_metrics_by_policy, dict) else {}
    cards = [
        policy_bubble_card(
            policy,
            metrics_by_policy.get(policy),
            settings,
            index,
        )
        for index, policy in enumerate(POLICIES)
    ]

    return f"""
<style>
.bubble-policy-stack {{
  display: flex;
  flex-direction: column;
  gap: 1.15rem;
  margin: 0 0 1.2rem;
}}

.bubble-policy-card {{
  position: relative;
  overflow: hidden;
  border: 1px solid var(--frame-border);
  border-left: 4px solid var(--policy-border);
  border-radius: 8px;
  padding: 1.35rem 1.55rem 1.25rem;
  background: var(--frame-bg);
  box-shadow: none;
  animation: bubbleCardIn 520ms cubic-bezier(.22,.61,.19,1) both;
  animation-delay: var(--card-delay);
}}

.bubble-policy-card.is-waiting {{
  opacity: 0.78;
}}

.bubble-policy-header {{
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.15rem;
}}

.bubble-policy-card h3 {{
  margin: 0;
  color: var(--text);
  font-size: 1.35rem;
  line-height: 1.15;
  letter-spacing: 0;
}}

.bubble-policy-card p {{
  margin: 0.45rem 0 0;
  color: var(--text-muted);
  font-size: 1rem;
  line-height: 1.35;
}}

.bubble-outcome-delta {{
  min-width: 7rem;
  text-align: right;
  font-family: var(--mono);
}}

.bubble-outcome-delta span {{
  display: block;
  color: var(--text-muted);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.18em;
}}

.bubble-outcome-delta strong {{
  display: block;
  margin-top: 0.18rem;
  color: var(--primary);
  font-size: 2rem;
  line-height: 1;
  letter-spacing: 0;
}}

.bubble-outcome-delta.is-worse strong {{
  color: var(--accent);
}}

.bubble-chart-frame {{
  height: clamp(18.4rem, 30.4vw, 24.8rem);
  overflow: hidden;
  border: 1px solid var(--frame-border-soft);
  border-radius: 7px;
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(249,253,251,0.78));
}}

.bubble-chart-stage {{
  position: relative;
  height: 100%;
  width: auto;
  max-width: 100%;
  aspect-ratio: 1500 / 620;
  margin: 0 auto;
  overflow: hidden;
}}

.bubble-position {{
  position: absolute;
  left: var(--bubble-left);
  top: var(--bubble-top);
  width: var(--bubble-size);
  aspect-ratio: 1;
  transform: translate(-50%, -50%);
}}

.bubble-float {{
  width: 100%;
  height: 100%;
  animation: bubbleFloat var(--float-duration) ease-in-out var(--float-delay) infinite;
  will-change: transform;
}}

.bubble-node {{
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border-radius: 999px;
  background: var(--bubble-color);
  filter: drop-shadow(0 1px 2px rgba(7, 24, 35, 0.12));
  color: #ffffff;
  animation: bubbleNodeIn 440ms cubic-bezier(.22,.61,.19,1) both;
  animation-delay: var(--bubble-delay);
}}

.bubble-node.is-flagged::after {{
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: repeating-linear-gradient(
    135deg,
    rgba(0, 0, 0, 0.30) 0,
    rgba(0, 0, 0, 0.30) 0.26rem,
    transparent 0.26rem,
    transparent 0.82rem
  );
  pointer-events: none;
  z-index: 1;
}}

.bubble-value {{
  position: relative;
  z-index: 2;
  display: block;
  font-family: var(--mono);
  font-size: var(--bubble-font);
  font-weight: 850;
  line-height: 1;
  letter-spacing: 0;
  pointer-events: none;
}}

.bubble-card-metrics {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1.1rem;
  margin-top: 1.15rem;
}}

.bubble-card-metric {{
  display: flex;
  flex-direction: column;
}}

.bubble-card-metric-label {{
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.2em;
  line-height: 1.35;
  min-height: 2.2rem;
}}

.bubble-card-metric-value {{
  margin-top: 0.35rem;
  color: var(--text);
  font-family: var(--mono);
  font-size: 1.65rem;
  line-height: 1;
}}

@keyframes bubbleNodeIn {{
  0% {{ opacity: 0; transform: scale(0.92); }}
  100% {{ opacity: 1; transform: scale(1); }}
}}

@keyframes bubbleFloat {{
  0%, 100% {{ transform: translate3d(0, 0, 0); }}
  28% {{ transform: translate3d(var(--float-x), var(--float-y), 0); }}
  52% {{ transform: translate3d(var(--float-x-soft), var(--float-y-soft), 0); }}
  78% {{ transform: translate3d(var(--float-x-neg), var(--float-y-neg), 0); }}
}}

@keyframes bubbleCardIn {{
  0% {{ opacity: 0; transform: translateY(12px); }}
  100% {{ opacity: 1; transform: translateY(0); }}
}}

@media (max-width: 760px) {{
  .bubble-policy-card {{
    padding: 1rem;
  }}

  .bubble-policy-header {{
    flex-direction: column;
  }}

  .bubble-outcome-delta {{
    text-align: left;
  }}

  .bubble-chart-frame {{
    height: 17.6rem;
  }}

  .bubble-card-metrics {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
}}
</style>
<div class="bubble-policy-stack" id="{root_id}">
  {''.join(cards)}
</div>
    """


def _panel_changes(children, policy, metrics):
    high_flagged = [i for i, c in enumerate(children) if c["base"] == "high" and c["flagged"]]
    low_flagged = [i for i, c in enumerate(children) if c["base"] == "low" and c["flagged"]]
    low_flagged_set = set(low_flagged)

    if metrics is None:
        return set(), set(), set(), "Waiting for this policy result. Baseline is shown.", False

    prevented = min(max(0, metrics["prevented"]), len(high_flagged))
    prevented_indices = seeded_subset(high_flagged, prevented, f"{policy}-prevented")
    added = min(max(0, -metrics["prevented"]), len(low_flagged))
    added_indices = seeded_subset(low_flagged, added, f"{policy}-additional-outcomes")
    high_flagged_remaining = [i for i in high_flagged if i not in prevented_indices]
    harmed_pool = low_flagged + high_flagged_remaining
    harmed = min(metrics["harmed"], len(harmed_pool))
    harmed_indices = seeded_subset(harmed_pool, harmed, f"{policy}-harmed")

    harmed_fp = sum(1 for i in harmed_indices if i in low_flagged_set)
    harmed_tp = len(harmed_indices) - harmed_fp
    final_high_indices = {
        i for i, c in enumerate(children)
        if c["base"] == "high" and i not in prevented_indices
    } | added_indices
    final_high = len(final_high_indices)
    harmed_total = harmed_fp + harmed_tp
    added_summary = f"{len(added_indices)} added (green→red); " if added_indices else ""
    summary = (
        f"{len(prevented_indices)} prevented (red→green); "
        f"{added_summary}"
        f"{harmed_total} policy harm count (outlined); "
        f"{final_high} red remain."
    )
    return prevented_indices, added_indices, harmed_indices, summary, True


def policy_panel_html(children, policy, metrics, panel_index):
    prevented_indices, added_indices, harmed_indices, summary, has_result = _panel_changes(children, policy, metrics)

    dots = []
    for index, child in enumerate(children):
        delay = (index % POPULATION_DOT_STAGGER_GROUP) * POPULATION_DOT_STAGGER_SECONDS
        base_class = "base-high" if child["base"] == "high" else "base-low"
        final_class = base_class.replace("base-", "final-")
        change_classes = []
        if index in prevented_indices:
            final_class = "final-low"
            change_classes.append("changed-prevented")
        elif index in added_indices:
            final_class = "final-high"
            change_classes.append("changed-worsened")
        if index in harmed_indices:
            change_classes.append("changed-harmed")
        flagged_class = " flagged-dot" if child["flagged"] else ""
        change_class = f" {' '.join(change_classes)}" if change_classes else ""
        dots.append(
            f'<span class="life-dot {base_class} {final_class}{change_class}{flagged_class}" '
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
    prevented_indices, added_indices, harmed_indices, summary, _ = _panel_changes(children, policy, metrics)
    return f"""<script>
(function() {{
  var root = document.getElementById({json.dumps(root_id)});
  if (!root) return;
  var panel = root.querySelectorAll('.policy-panel')[{panel_index}];
  if (!panel) return;
  var dots = Array.from(panel.querySelectorAll('.life-dot'));
  dots.forEach(function(dot) {{
    dot.classList.remove('final-low','final-high','changed-prevented','changed-worsened','changed-harmed');
    dot.classList.add(dot.classList.contains('base-high') ? 'final-high' : 'final-low');
  }});
  {json.dumps(sorted(prevented_indices))}.forEach(function(i) {{
    if (!dots[i]) return;
    dots[i].classList.remove('final-high');
    dots[i].classList.add('final-low','changed-prevented');
  }});
  {json.dumps(sorted(added_indices))}.forEach(function(i) {{
    if (!dots[i]) return;
    dots[i].classList.remove('final-low');
    dots[i].classList.add('final-high','changed-worsened');
  }});
  {json.dumps(sorted(harmed_indices))}.forEach(function(i) {{
    if (!dots[i]) return;
    dots[i].classList.add('changed-harmed');
  }});
  var el = panel.querySelector('.policy-panel-summary');
  if (el) el.textContent = {json.dumps(summary)};
  panel.dataset.ready = 'true';
  panel.classList.add('show-final');
  var pg = panel.querySelector('.policy-grid');
  if (pg) pg.dispatchEvent(new CustomEvent('resetWaveCache'));
}})();
</script>"""


@st.cache_data(show_spinner=False)
def population_animation_html(policy_metrics_by_policy, true_high_risk_rate, prediction_noise, population_size, title, animation_key="", root_id=None):
    settings = {
        "population_size": population_size,
        "true_high_risk_rate": true_high_risk_rate,
        "prediction_noise": prediction_noise,
    }
    children = baseline_children(settings)
    if root_id is None:
        seed_text = f"{title}-{animation_key}"
        animation_id = (stable_seed(seed_text) + len(children) * 17) % 100000
        root_id = f"policyTransition{animation_id}"
    panels = [
        policy_panel_html(
            children,
            policy,
            policy_metrics_by_policy.get(policy) if isinstance(policy_metrics_by_policy, dict) else None,
            panel_index,
        )
        for panel_index, policy in enumerate(POLICIES)
    ]

    return f"""
<style>
.life-course-card {{
  border: 1px solid var(--frame-border);
  border-radius: 8px;
  padding: var(--frame-padding);
  margin: 0 0 12px;
  background: var(--frame-bg);
  box-shadow: none;
}}
.policy-panels {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.policy-panel {{
  border: 1px solid var(--frame-border-soft);
  border-left: 3px solid var(--primary);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--frame-bg-soft);
  box-shadow: none;
  transition: border-color 200ms ease, box-shadow 200ms ease;
}}
.policy-panel[data-ready="true"] {{
  border-color: var(--primary-border);
  border-left-color: var(--primary);
  border-left-width: 3px;
  box-shadow: none;
}}
.policy-panel-title {{
  color: var(--text);
  font-size: 0.88rem;
  font-weight: 650;
  line-height: 1.3;
  margin-bottom: 4px;
  letter-spacing: -0.01em;
}}
.policy-panel-summary {{
  color: var(--text-muted);
  font-size: 0.75rem;
  line-height: 1.4;
  margin-bottom: 8px;
}}
.policy-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(10px, 1fr));
  grid-auto-rows: 14px;
  gap: 4px;
  align-items: center;
  justify-items: center;
  padding: 10px 9px;
  position: relative;
  overflow: hidden;
  border-radius: 6px;
  background: var(--surface-panel);
  border: 1px solid var(--frame-border-soft);
  contain: layout style paint;
  content-visibility: auto;
  contain-intrinsic-block-size: 200px;
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
  --base-color: var(--risk-safe-soft);
  --base-scale: 0.92;
}}
.base-high {{
  --base-color: var(--accent);
  --base-scale: 1.28;
}}
.final-low {{
  --final-color: var(--risk-safe-soft);
  --final-scale: 0.92;
}}
.final-high {{
  --final-color: var(--accent);
  --final-scale: 1.28;
}}
.flagged-dot {{
  outline: 2px solid var(--amber);
  outline-offset: 0;
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
  background: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-light);
}}
.policy-panel.show-final .life-dot.changed-worsened {{
  background: var(--accent);
  box-shadow: 0 0 0 3px rgba(226, 35, 55, 0.18);
}}
.policy-panel.show-final .life-dot.changed-harmed {{
  box-shadow: 0 0 0 3px var(--accent-border);
}}
@media (max-width: 700px) {{
  .life-course-card {{
    padding: var(--frame-padding);
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
  <div class="policy-panels">
    {''.join(panels)}
  </div>
</div>
<script>
(function() {{
  var root = document.getElementById('{root_id}');
  if (!root) return;

  var grids = root.querySelectorAll('.policy-grid');

  // Start dots paused; IntersectionObserver unpauses them when the grid is visible.
  grids.forEach(function(grid) {{
    grid.querySelectorAll('.life-dot').forEach(function(d) {{
      d.style.animationPlayState = 'paused';
    }});
  }});

  var revealed = new Set();
  var io = ('IntersectionObserver' in window) ? new IntersectionObserver(function(entries) {{
    entries.forEach(function(entry) {{
      if (!entry.isIntersecting || revealed.has(entry.target)) return;
      revealed.add(entry.target);
      io.unobserve(entry.target);
      entry.target.querySelectorAll('.life-dot').forEach(function(d) {{
        d.style.animationPlayState = 'running';
      }});
      // After dots reveal, transition panels with results to their final state.
      setTimeout(function() {{
        var panel = entry.target.closest('.policy-panel');
        if (panel && panel.dataset.ready === 'true') panel.classList.add('show-final');
      }}, 420);
    }});
  }}, {{ threshold: 0.05 }}) : null;

  grids.forEach(function(grid) {{
    if (io) {{
      io.observe(grid);
    }} else {{
      // Fallback for browsers without IntersectionObserver.
      grid.querySelectorAll('.life-dot').forEach(function(d) {{
        d.style.animationPlayState = 'running';
      }});
      setTimeout(function() {{
        root.querySelectorAll('.policy-panel[data-ready="true"]').forEach(function(p) {{
          p.classList.add('show-final');
        }});
      }}, 420);
    }}
  }});
}})();
</script>
"""


def render_population_animation(container, policy_metrics_by_policy, settings, title, animation_key="", root_id=None):
    html = bubble_population_animation_html(
        policy_metrics_by_policy,
        settings["true_high_risk_rate"],
        settings["prediction_noise"],
        settings["population_size"],
        title,
        animation_key,
        root_id=root_id,
    )
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
    rows = []
    for ach in ACHIEVEMENTS:
        is_earned = ach["id"] in earned
        row_class = "achievement-row" if is_earned else "achievement-row is-locked"
        icon = achievement_icon_html(achievement_icon_key(ach["id"], locked=not is_earned))
        description = (
            f'<div class="achievement-desc">{escape(ach["description"])}</div>' if is_earned else ""
        )
        rows.append(
            f"""
<div class="{row_class}">
  {icon}
  <div>
    <div class="achievement-name">{escape(ach['name'])}</div>
    {description}
  </div>
</div>
            """
        )

    st.html(
        f"""
<details class="achievement-shell">
  <summary>
    <div class="achievement-summary-main">
      {achievement_icon_html("trophy")}
      <span class="achievement-title">Achievements</span>
      <span class="achievement-count">{count}<span> / {len(ACHIEVEMENTS)}</span></span>
    </div>
    {achievement_icon_html("chevron", "achievement-toggle")}
  </summary>
  <div class="achievement-list">
    {''.join(rows)}
  </div>
</details>
        """
    )


def render_settings_field_header(label, value=None):
    value_html = ""
    if value is not None:
        value_html = f'<span class="settings-field-value">{escape(str(value))}</span>'
    st.html(
        f"""
<div class="settings-field-head">
  <span>{escape(label)}</span>
  {value_html}
</div>
        """
    )


def format_percent(value):
    value = float(value)
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:.1f}%"


def clamp_percent_session_value(key, default_percent, min_percent, max_percent):
    try:
        value = float(st.session_state.get(key, default_percent))
    except (TypeError, ValueError):
        value = default_percent
    value = min(max(value, min_percent), max_percent)
    st.session_state[key] = value
    return value


def render_settings_field_copy(text):
    st.html(f'<div class="settings-field-copy">{escape(text)}</div>')


def render_settings_stat_rows(population_size, policy_count):
    st.html(
        f"""
<div class="settings-panel-divider"></div>
<div class="settings-kv">
  <div class="settings-kv-row"><span>Population</span><strong>{population_size:,} children</strong></div>
  <div class="settings-kv-row"><span>Policies</span><strong>{policy_count} &times; simulated</strong></div>
</div>
        """
    )


def render_reference_guide():
    if _HOW_THIS_WORKS_EXPANDED_KEY not in st.session_state:
        st.session_state[_HOW_THIS_WORKS_EXPANDED_KEY] = True

    with st.expander(
        "How this works",
        expanded=bool(st.session_state[_HOW_THIS_WORKS_EXPANDED_KEY]),
    ):
        st.markdown(
            f"""
A fictional prediction tool scans {DEFAULT_POPULATION_SIZE:,} children at age 10 and flags those it believes will commit a serious harmful act by age 30. You set how many children would commit that act if no policy were applied, how often the tool is wrong, and how intense the policy response is. The simulation then tests all three policies in parallel — three possible things society could do with those flags.

Python first fixes the full prediction structure: true positives, false positives, false negatives, and the large background group that is neither flagged nor on the predicted-outcome path. The AI then simulates only the relevant groups — flagged children plus false negatives — as weighted life-course profiles rather than 10,000 separate biographies.

For each relevant profile, the model follows life stages from age 10 to 30, tracking how the policy might affect trust, autonomy, relationships, opportunities, stress, support, monitoring, restriction, and the final predicted outcome. The unchanged background group stays in the arithmetic, but it is not individually simulated.

The aggregate results are counted from those simulated lives. They show the baseline predicted outcomes, post-policy predicted outcomes, predicted outcome reduction, policy benefit count, policy harm count, false positives, and false negatives.

The goal is not to find the right answer — it's to see what the trade-offs actually cost.
            """
        )

        st.markdown("### Three policies")
        st.markdown(glossary_markdown(POLICY_DESCRIPTIONS))

        st.markdown("### What you can adjust")
        st.markdown(glossary_markdown(SETTING_DESCRIPTIONS))

        st.markdown("### What the results show")
        st.markdown(glossary_markdown(RESULT_METRIC_DESCRIPTIONS))

        st.markdown("### Key trade-offs")
        st.markdown(glossary_markdown(CHECK_DESCRIPTIONS))


DISPLAY_LABEL_ALIASES = {
    "Would offend without intervention": "Baseline predicted outcomes",
    "Predicted outcomes without policy": "Baseline predicted outcomes",
    "Flagged as high-risk": "Positive predictions",
    "Flagged by prediction": "Positive predictions",
    "Prediction positives": "Positive predictions",
    "Wrongly flagged": "False positives",
    "Missed by prediction": "False negatives",
    "Offenses prevented": "Prevented predicted outcomes",
    "Outcomes prevented": "Prevented predicted outcomes",
    "Net outcome effect": "Predicted outcome reduction (%)",
    "Outcome change": "Predicted outcome reduction (%)",
    "Offense reduction (%)": "Predicted outcome reduction (%)",
    "Harmed (% of flagged)": "Policy harm rate among positive predictions (%)",
    "Harmed by policy (% of flagged)": "Policy harm rate among positive predictions (%)",
    "Policy harm rate among flagged (%)": "Policy harm rate among positive predictions (%)",
    "Would offend without intervention (avg count)": "Baseline predicted outcomes (avg)",
    "Predicted outcomes without policy (avg)": "Baseline predicted outcomes (avg)",
    "Wrongly flagged (avg count)": "False positives (avg)",
    "Wrongly flagged (avg)": "False positives (avg)",
    "Missed by prediction (avg count)": "False negatives (avg)",
    "Missed by prediction (avg)": "False negatives (avg)",
    "Helped by policy (avg count)": "Policy benefit count (avg)",
    "Helped by policy (avg)": "Policy benefit count (avg)",
    "Harmed by policy": "Policy harm count",
    "Harmed by policy (avg count)": "Policy harm count (avg)",
    "Harmed by policy (avg)": "Policy harm count (avg)",
}


def normalize_display_labels(dataframe):
    if dataframe is None:
        return pd.DataFrame()
    if dataframe.empty:
        return dataframe.copy()
    display = dataframe.rename(columns=DISPLAY_LABEL_ALIASES).copy()
    if "Metric" in display.columns:
        display["Metric"] = display["Metric"].replace(DISPLAY_LABEL_ALIASES)
    return display


def display_average_table(average_table):
    st.dataframe(
        formatted_average_results_table(normalize_display_labels(average_table)),
        use_container_width=True,
        hide_index=True,
    )


def dataframe_to_payload(dataframe):
    return {
        "columns": list(dataframe.columns),
        "records": dataframe.to_dict("records"),
    }


@st.cache_data(show_spinner=False)
def dataframe_from_payload(payload):
    if not isinstance(payload, dict):
        return pd.DataFrame()
    return pd.DataFrame(
        payload.get("records", []),
        columns=payload.get("columns", []),
    )


def display_dataframe_payload(payload):
    st.dataframe(
        normalize_display_labels(dataframe_from_payload(payload)),
        use_container_width=True,
        hide_index=True,
    )


def chart_rows_payload(policy_runs):
    chart_columns = [
        column
        for column in ["run", "crimes_prevented", "children_harmed", "llm_model"]
        if column in policy_runs.columns
    ]
    return dataframe_to_payload(policy_runs[chart_columns].copy())


def latest_result_payload(combined_runs, settings):
    policy_results = {}
    population_metrics_by_policy = {}
    for policy in POLICIES:
        policy_runs = combined_runs[combined_runs["policy"] == policy]
        if policy_runs.empty:
            continue

        policy_results[policy] = {
            "average_table": dataframe_to_payload(average_results_table(policy_runs)),
            "chart_rows": chart_rows_payload(policy_runs),
        }
        population_metrics_by_policy[policy] = policy_transition_metrics(combined_runs, policy, settings)

    return {
        "schema_version": 2,
        "comparison_table": dataframe_to_payload(
            combined_policy_totals_table(combined_runs, settings["population_size"])
        ),
        "policy_results": policy_results,
        "population_metrics_by_policy": population_metrics_by_policy,
        "settings_signature": simulation_settings_signature(settings),
    }


def render_charts(run_results):
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.image(
            line_chart_png(
                run_results,
                "run",
                "crimes_prevented",
                "Prevented predicted outcomes by run",
                "Prevented predicted outcomes",
            ),
            use_container_width=True,
        )

    if "children_harmed" in run_results.columns and run_results["children_harmed"].sum() > 0:
        with chart_right:
            st.image(
                line_chart_png(
                    run_results,
                    "run",
                    "children_harmed",
                    "Policy harm count by run",
                    "Policy harm count",
                ),
                use_container_width=True,
            )


def prevented_outcome_phrase(value):
    rounded = abs(value)
    if value < 0:
        return f"adds {rounded:.0f} predicted outcomes per run compared with no policy"
    return f"prevents {rounded:.0f} predicted outcomes per run"


def render_interpretation(policy, average_table):
    average_table = normalize_display_labels(average_table)
    values = dict(zip(average_table["Metric"], average_table["Average per synthetic run"]))
    crimes_prevented = values.get("Prevented predicted outcomes", 0.0)
    false_positives = values.get("False positives", 0.0)
    children_helped = values.get("Policy benefit count", 0.0)
    children_harmed = values.get("Policy harm count", 0.0)
    outcome_phrase = prevented_outcome_phrase(crimes_prevented)

    if policy == "Coercive preventive intervention for high-risk children":
        st.info(
            f"On average, this policy {outcome_phrase}. "
            f"{children_helped:.0f} flagged children show a policy-associated benefit and "
            f"{children_harmed:.0f} show policy-associated harm from mandatory requirements or restrictions. "
            f"{false_positives:.0f} cases are false positives."
        )
    elif policy == "Targeted support for high-risk children":
        st.info(
            f"On average, this policy {outcome_phrase}. "
            f"{children_helped:.0f} flagged children have improved life-course outcomes and "
            f"{children_harmed:.0f} show policy-associated harm or negative side effects. "
            f"{false_positives:.0f} cases are false positives."
        )
    elif policy == "Surveillance of high-risk children":
        st.info(
            f"On average, this policy {outcome_phrase}. "
            f"{children_helped:.0f} flagged children show policy-associated benefit from the monitoring response and "
            f"{children_harmed:.0f} show policy-associated harm from scrutiny, stigma, or trust loss. "
            f"{false_positives:.0f} cases are false positives."
        )


def agent_text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


CASE_STORY_NAMES = [
    "Alex",
    "Maya",
    "Sam",
    "Nina",
    "Leo",
    "Iris",
    "Owen",
    "Rina",
]


def compact_sentence(value, max_length=170):
    text = " ".join(agent_text(value).split())
    if len(text) <= max_length:
        return text

    clipped = text[:max_length].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{clipped}..."


def agent_display_name(agent, index):
    raw_name = agent_text(agent.get("agent_id")).strip()
    if raw_name and not raw_name.lower().startswith(("agent", "case", "id-")):
        return raw_name.split()[0]
    return CASE_STORY_NAMES[(index - 1) % len(CASE_STORY_NAMES)]


def policy_effect_text(effect):
    if not isinstance(effect, dict):
        return agent_text(effect)

    value = "Yes" if effect.get("value") else "No"
    detail = agent_text(effect.get("detail")).strip()
    if not detail:
        return value
    return f"{value}: {detail}"


def life_stage_rows(life_stages):
    if not isinstance(life_stages, list):
        return [{"Stage": "", "Summary": agent_text(life_stages)}]

    rows = []
    for item in life_stages:
        if isinstance(item, dict):
            rows.append(
                {
                    "Stage": agent_text(item.get("stage")),
                    "Summary": agent_text(item.get("summary")),
                }
            )
        else:
            rows.append({"Stage": "", "Summary": agent_text(item)})
    return rows


def representative_agent_dict(agent):
    if isinstance(agent, dict):
        return agent
    return {"description": agent_text(agent)}


def representative_agent_summary(agent):
    return {
        "Policy": agent_text(agent.get("policy")),
        "Model": agent_text(agent.get("llm_model")),
        "Agent": agent_text(agent.get("agent_id")),
        "Prediction status": agent_text(agent.get("prediction_status")),
        "Outcome occurred": agent_text(agent.get("predicted_outcome_occurred")),
        "Policy benefit": policy_effect_text(agent.get("helped_by_policy")),
        "Policy harm": policy_effect_text(agent.get("harmed_by_policy")),
        "Mixed": policy_effect_text(agent.get("mixed_effects")),
    }


def representative_agent_title(index, agent):
    title_parts = [
        agent_text(agent.get("agent_id")) or f"Agent {index}",
        agent_text(agent.get("policy")),
        agent_text(agent.get("prediction_status")),
    ]
    return " | ".join(part for part in title_parts if part)


def representative_agent_case_vignette(agent, index):
    explicit_vignette = compact_sentence(agent.get("case_vignette"), 260)
    if explicit_vignette:
        return explicit_vignette

    name = agent_display_name(agent, index)
    status = agent_text(agent.get("prediction_status")) or "representative case"
    profile = compact_sentence(agent.get("starting_profile"), 95)
    mechanism = compact_sentence(agent.get("mechanism_summary"), 120)
    outcome = "the predicted outcome occurred" if agent.get("predicted_outcome_occurred") else "the predicted outcome did not occur"

    pieces = [f"{name} was a {status}."]
    if profile:
        pieces.append(profile)
    if mechanism:
        pieces.append(mechanism)
    pieces.append(f"By age 30, {outcome}.")
    return " ".join(pieces)


def representative_agent_progress_note(model, policy, policy_agents, max_agents=3):
    agents = [representative_agent_dict(agent) for agent in policy_agents[:max_agents]]
    if not agents:
        return ""

    lines = [f"<div>{escape(model)} | {escape(policy)}:</div>"]
    for index, agent in enumerate(agents, start=1):
        vignette = representative_agent_case_vignette(agent, index)
        name = agent_display_name(agent, index)
        name_is_prefix = vignette.casefold().startswith(name.casefold())
        name_has_boundary = name_is_prefix and (
            len(vignette) == len(name) or not vignette[len(name)].isalnum()
        )
        if name_is_prefix and name_has_boundary:
            name_text = vignette[:len(name)]
            rest = vignette[len(name):]
            line = f"<strong>{escape(name_text)}</strong>{escape(rest)}"
        else:
            line = f"<strong>{escape(name)}</strong>: {escape(vignette)}"
        lines.append(f'<div class="terminal-progress-case">{line}</div>')
    return "".join(lines)


def progress_note_candidate(model, policy, policy_agents, debrief):
    case_note = representative_agent_progress_note(model, policy, policy_agents)
    if case_note:
        return case_note, True
    if debrief:
        return f"{model} | {policy}: {compact_sentence(debrief, 320)}", False
    return "", False


def maybe_update_progress_note(
    current_note,
    current_is_html,
    last_update_at,
    candidate_note,
    candidate_is_html,
    now=None,
):
    if not candidate_note:
        return current_note, current_is_html, last_update_at

    current_time = monotonic() if now is None else float(now)
    can_update = (
        not current_note
        or last_update_at is None
        or current_time - last_update_at >= PROGRESS_NOTE_MIN_SECONDS
    )
    if not can_update:
        return current_note, current_is_html, last_update_at

    return candidate_note, candidate_is_html, current_time


def render_representative_agents(representative_agents):
    agents = [representative_agent_dict(agent) for agent in representative_agents]
    summary_rows = [representative_agent_summary(agent) for agent in agents]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    for index, agent in enumerate(agents, start=1):
        with st.expander(representative_agent_title(index, agent), expanded=False):
            if agent.get("description"):
                st.write(agent_text(agent.get("description")))

            if agent.get("case_vignette"):
                st.markdown("**Case vignette**")
                st.write(agent_text(agent.get("case_vignette")))

            st.markdown("**Starting profile**")
            st.write(agent_text(agent.get("starting_profile")))

            st.markdown("**No-policy counterfactual**")
            st.write(agent_text(agent.get("no_policy_counterfactual")))

            st.markdown("**Life stages**")
            st.dataframe(
                pd.DataFrame(life_stage_rows(agent.get("life_stages"))),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Policy effects**")
            st.write(f"Policy-associated benefit: {policy_effect_text(agent.get('helped_by_policy'))}")
            st.write(f"Policy-associated harm: {policy_effect_text(agent.get('harmed_by_policy'))}")
            st.write(f"Mixed effects: {policy_effect_text(agent.get('mixed_effects'))}")

            st.markdown("**Outcome**")
            st.write(f"Predicted outcome occurred: {agent_text(agent.get('predicted_outcome_occurred'))}")
            st.write(agent_text(agent.get("life_course_outcome")))

            st.markdown("**Mechanism summary**")
            st.write(agent_text(agent.get("mechanism_summary")))


@st.fragment
def render_results_fragment(settings):
    latest_result = st.session_state.get("llm_agent_latest_result")
    if not latest_result or not latest_result_has_current_schema(latest_result, settings):
        return

    st.subheader("Policy comparison")
    st.caption("Average outcomes per run. Use this to compare policies side by side.")

    with st.container(border=False, key="results_content_panel"):
        display_dataframe_payload(latest_result.get("comparison_table"))

        for selected_policy, policy_tab in zip(POLICIES, st.tabs(POLICIES), strict=True):
            with policy_tab:
                policy_result = latest_result.get("policy_results", {}).get(selected_policy)
                if not policy_result:
                    st.caption("No results for this policy in the current session.")
                else:
                    policy_average_table = dataframe_from_payload(policy_result.get("average_table"))
                    chart_rows = dataframe_from_payload(policy_result.get("chart_rows"))
                    st.subheader("Averages")
                    display_average_table(policy_average_table)
                    render_charts(chart_rows)
                    render_interpretation(
                        selected_policy,
                        policy_average_table,
                    )

    st.html('<div class="section-gap section-gap-explanations"></div>')
    st.subheader("Latest LLM-agent explanations")
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
            render_representative_agents(representative_agents)


@st.fragment
def render_llm_run_log(max_entries):
    trim_llm_run_log(max_entries)
    run_log = st.session_state["llm_agent_run_log"]
    if not run_log:
        return

    st.html('<div class="section-gap section-gap-history"></div>')
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


def render_terminal_progress(
    slot,
    completed: int,
    total: int,
    status: str = "",
    title: str | None = None,
    note: str = "",
    note_is_html: bool = False,
):
    total_safe = max(total, 1)
    progress_ratio = completed / total_safe
    pct = int(progress_ratio * 100)
    block_count = 32
    filled = min(block_count, int(progress_ratio * block_count))
    label = title or "RUNNING_SIMULATION"
    status_text = status or "Simulation status live"
    blocks = []
    for index in range(block_count):
        classes = ["terminal-progress-block"]
        if index < filled:
            classes.append("is-filled")
        if index == filled - 1 and filled > 0:
            classes.append("is-active")
        blocks.append(f'<span class="{" ".join(classes)}"></span>')
    if note:
        note_content = note if note_is_html else escape(note)
        note_html = f'<div class="terminal-progress-note">{note_content}</div>'
    else:
        note_html = ""

    slot.html(
        f"""
<div class="terminal-progress-card">
  <div class="terminal-progress-header">
    <div class="terminal-progress-title-wrap">
      <span class="terminal-progress-dot"></span>
      <span class="terminal-progress-title">{escape(label)}</span>
    </div>
    <span class="terminal-progress-pct">{pct}%</span>
  </div>
  <div class="terminal-progress-track">
    <span class="terminal-progress-bracket">[</span>
    <span class="terminal-progress-blocks">{''.join(blocks)}</span>
    <span class="terminal-progress-bracket">]</span>
  </div>
  <div class="terminal-progress-status"><span class="terminal-progress-prompt">&gt;</span> {escape(status_text)}<span class="terminal-progress-cursor"></span></div>
  {note_html}
</div>
        """
    )


def run_llm_policy_task(settings, model, policy):
    policy_settings = {**settings, "policy": policy}
    user_prompt = build_llm_simulation_prompt(policy_settings)
    raw = run_openai_json(DEFAULT_SYSTEM_PROMPT, user_prompt, model=model)
    policy_effects = clean_llm_policy_effects(raw.get("policy_effects", []))
    validate_llm_policy_effects(policy_effects, policy_settings, enforce_bounds=False)
    policy_effects = normalize_llm_policy_effects(policy_effects, policy_settings)
    validate_llm_policy_effects(policy_effects, policy_settings)
    run_results = run_results_from_policy_effects(policy_effects, policy_settings)
    validate_llm_tables(run_results, policy_settings)
    run_results = attach_model_label(run_results, model)
    run_results = attach_policy_label(run_results, policy)
    policy_agents = normalize_representative_agents(
        raw.get("representative_agents", [])[:int(settings["llm_representative_agents"])],
        model,
        policy,
    )
    debrief = str(raw.get("debrief_text", "")).strip()
    aggregate = compact_aggregate_metrics(run_results)

    return {
        "model": model,
        "policy": policy,
        "run_results": run_results,
        "agents": policy_agents,
        "debrief_text": debrief,
        "aggregate_metrics": aggregate,
    }


def _run_simulation(settings, selected_models, progress_slot, update_slot, live_population):
    st.session_state.pop("llm_agent_latest_result", None)
    parameter_summary = compact_parameter_summary(settings)
    total_calls = len(selected_models) * len(POLICIES)
    run_frames, model_summaries = [], []
    agents, debrief_parts, errors = [], [], []
    completed = 0
    progress_note = ""
    progress_note_is_html = False
    progress_note_updated_at = None
    live_policy_metrics = {}

    render_terminal_progress(
        progress_slot, 0, total_calls,
        f"Running {len(POLICIES)} policies × {len(selected_models)} model(s)...",
        note=progress_note,
        note_is_html=progress_note_is_html,
    )
    scroll_to_anchor("simulation-progress-anchor")
    if SHOW_POPULATION_DOT_VIEW and live_population is not None:
        render_population_animation(live_population, live_policy_metrics, settings, "Simulating…", "waiting", root_id=_LIVE_GRID_ID)

    tasks = [
        (task_index, model, policy)
        for task_index, (model, policy) in enumerate(
            (model, policy) for model in selected_models for policy in POLICIES
        )
    ]
    max_workers = max(1, min(MAX_PARALLEL_LLM_CALLS, total_calls))
    render_terminal_progress(
        progress_slot, 0, total_calls,
        f"Running up to {max_workers} LLM call(s) in parallel...",
        note=progress_note,
        note_is_html=progress_note_is_html,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_llm_policy_task, settings, model, policy): (task_index, model, policy)
            for task_index, model, policy in tasks
        }

        for future in as_completed(future_to_task):
            task_index, model, policy = future_to_task[future]
            try:
                result = future.result()
                completed += 1
                policy_agents = result["agents"]
                debrief = result["debrief_text"]
                candidate_note, candidate_is_html = progress_note_candidate(model, policy, policy_agents, debrief)
                progress_note, progress_note_is_html, progress_note_updated_at = maybe_update_progress_note(
                    progress_note,
                    progress_note_is_html,
                    progress_note_updated_at,
                    candidate_note,
                    candidate_is_html,
                )
                render_terminal_progress(
                    progress_slot, completed, total_calls,
                    f"Received {model} / {policy}.",
                    note=progress_note,
                    note_is_html=progress_note_is_html,
                )
                aggregate = result["aggregate_metrics"]
                run_results = result["run_results"]
                run_frames.append(run_results)
                agents.extend(policy_agents)
                if debrief:
                    debrief_parts.append((task_index, f"{model} | {policy}: {debrief}"))
                model_summaries.append(
                    {
                        "task_index": task_index,
                        "llm_model": model,
                        "policy": policy,
                        "aggregate_metrics": aggregate,
                        "debrief_text": debrief,
                    }
                )
                if SHOW_POPULATION_DOT_VIEW and live_population is not None:
                    metrics = policy_transition_metrics(run_results, policy, settings)
                    live_policy_metrics[policy] = metrics
                    render_population_animation(live_population, live_policy_metrics, settings, "Simulating…", "live", root_id=_LIVE_GRID_ID)
            except Exception as error:
                completed += 1
                msg = friendly_llm_error(error)
                progress_note, progress_note_is_html, progress_note_updated_at = maybe_update_progress_note(
                    progress_note,
                    progress_note_is_html,
                    progress_note_updated_at,
                    msg,
                    False,
                )
                render_terminal_progress(
                    progress_slot, completed, total_calls,
                    f"Error: {model} / {policy}.",
                    note=progress_note,
                    note_is_html=progress_note_is_html,
                )
                errors.append((f"{model} | {policy}", msg))

    progress_slot.empty()
    notices = [("error", f"{label}: {msg}") for label, msg in errors]

    if run_frames:
        combined_runs = pd.concat(run_frames, ignore_index=True, copy=False)
        combined_runs = optimize_result_frames(combined_runs)
        model_summaries = sorted(model_summaries, key=lambda item: item["task_index"])
        model_summaries_for_display = [
            {key: value for key, value in summary.items() if key != "task_index"}
            for summary in model_summaries
        ]
        debrief_combined = "\n\n".join(
            text for _, text in sorted(debrief_parts, key=lambda item: item[0])
        )
        aggregate = compact_aggregate_metrics(combined_runs)
        latest_payload = latest_result_payload(combined_runs, settings)
        latest_payload.update(
            {
                "model_results": model_summaries_for_display,
                "representative_agents": agents,
                "debrief_text": debrief_combined,
            }
        )

        st.session_state["llm_agent_latest_result"] = latest_payload

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "policy_summary": "All policies",
            "parameter_summary": parameter_summary,
            "representative_agent_count": len(agents),
            "llm_model": ", ".join(unique_values(s["llm_model"] for s in model_summaries_for_display)),
            "aggregate_metrics": aggregate,
            "debrief_text": debrief_combined,
        }
        add_llm_run_log_entry(entry, MAX_RUN_LOG_SIZE)

        count = st.session_state.get("simulation_run_count", 0) + 1
        st.session_state["simulation_run_count"] = count
        new_achievements = check_achievements(combined_runs, settings, count)
        prev_achievements = st.session_state.get("earned_achievements", set())
        st.session_state["earned_achievements"] = prev_achievements | new_achievements
        unlocked = new_achievements - prev_achievements
        if unlocked:
            st.session_state["achievement_notifications"] = [
                {"id": a["id"], "name": ACHIEVEMENT_INDEX[a["id"]]["name"], "description": ACHIEVEMENT_INDEX[a["id"]]["description"]}
                for a in ACHIEVEMENTS if a["id"] in unlocked
            ]

        if errors:
            notices.append(("warning", "LLM-agent simulation generated for the successful model agents."))
        del combined_runs
        del run_frames
    elif errors:
        notices.append(("warning", "No LLM-agent simulation results were generated."))

    st.session_state["last_run_notices"] = notices
    st.session_state["simulation_running"] = False
    st.rerun()


def render_llm_agent_section(settings, run_info_slot=None, run_button_slot=None):
    initialize_llm_state()
    selected_models = settings["llm_agent_models"]
    total_calls = len(selected_models) * len(POLICIES)
    if SHOW_POPULATION_DOT_VIEW:
        st.subheader("Simulation")
        st.caption(f"{total_calls} LLM call(s) — one per policy × model.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning(
            "LLM-agent simulation is enabled, but OPENAI_API_KEY is not configured. Add it as an environment "
            "variable or Railway secret."
        )

    latest_result = st.session_state.get("llm_agent_latest_result")
    has_valid_result = latest_result and latest_result_has_current_schema(latest_result, settings)
    initial_policy_metrics = latest_result.get("population_metrics_by_policy") if has_valid_result else None
    initial_title = "Latest synthetic population view" if has_valid_result else "Live synthetic population view"

    live_population = None
    update_slot = None
    if SHOW_POPULATION_DOT_VIEW:
        render_population_view_overview(settings)
        live_population = st.empty()
        update_slot = st.empty()
        render_population_animation(live_population, initial_policy_metrics, settings, initial_title, "initial", root_id=_LIVE_GRID_ID)
    st.html('<div id="simulation-progress-anchor" style="height: 1px;"></div>')
    progress_slot = st.empty()

    if run_info_slot is None:
        run_info_slot = st.sidebar.container()
    with run_info_slot:
        st.caption(
            f"Current run: {len(POLICIES)} policies × {len(selected_models)} model agent(s) = "
            f"{total_calls} LLM call(s). Each call generates {int(settings['llm_simulation_runs'])} "
            f"synthetic run(s) over {int(settings['population_size']):,} synthetic children."
        )
    with run_info_slot:
        if not selected_models:
            st.warning("Select at least one LLM model agent.")

    run_disabled = not bool(api_key) or not selected_models
    if run_button_slot is None:
        run_button_slot = st.sidebar.empty()

    if st.session_state.get("simulation_running", False):
        st.html("""
<style>
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]),
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):disabled {
  background: var(--amber) !important;
  border-color: var(--amber) !important;
  box-shadow: none !important;
  opacity: 1 !important;
}
</style>
""")
        with run_button_slot:
            st.button("▶ Run simulation", key="run_simulation_running", disabled=True, type="primary", use_container_width=True)
        run_requested = False
    else:
        with run_button_slot:
            run_requested = st.button("▶ Run simulation", key="run_simulation_start", disabled=run_disabled, type="primary", use_container_width=True)

    if run_requested:
        st.session_state[_HOW_THIS_WORKS_EXPANDED_KEY] = False
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

    if st.session_state.pop("simulation_pending", False):
        with run_button_slot:
            st.button("▶ Run simulation", key="run_simulation_active", disabled=True, type="primary", use_container_width=True)
        _run_simulation(settings, selected_models, progress_slot, update_slot, live_population)

    latest_result = st.session_state.get("llm_agent_latest_result")
    if latest_result and not latest_result_has_current_schema(latest_result, settings):
        st.session_state.pop("llm_agent_latest_result", None)
        st.write("Previous in-session results used older assumptions. Run the simulation again.")
    else:
        st.html('<div class="section-gap section-gap-results"></div>')
        render_results_fragment(settings)
    render_llm_run_log(MAX_RUN_LOG_SIZE)


def sidebar_inputs():
    population_size = DEFAULT_POPULATION_SIZE
    true_rate_key = "true_high_risk_rate_percent"
    prediction_error_key = "prediction_error_percent"
    intensity_key = "policy_intensity_tier"
    model_key = "llm_agent_models"
    true_rate_default = DEFAULT_TRUE_HIGH_RISK_RATE * 100
    prediction_error_default = DEFAULT_PREDICTION_ERROR_RATE * 100
    true_rate_min = TRUE_HIGH_RISK_RATE_MIN * 100
    true_rate_max = TRUE_HIGH_RISK_RATE_MAX * 100
    prediction_error_min = PREDICTION_ERROR_RATE_MIN * 100
    prediction_error_max = PREDICTION_ERROR_RATE_MAX * 100
    true_rate_value = clamp_percent_session_value(
        true_rate_key,
        true_rate_default,
        true_rate_min,
        true_rate_max,
    )
    prediction_error_value = clamp_percent_session_value(
        prediction_error_key,
        prediction_error_default,
        prediction_error_min,
        prediction_error_max,
    )
    intensity_value = st.session_state.get(intensity_key, "Medium")
    is_running = bool(st.session_state.get("simulation_running", False))

    model_options = llm_model_options()
    model_defaults = default_llm_agent_models(model_options)
    clamp_selected_models(model_key, model_options)
    model_default_version_key = f"{model_key}_default_version"
    if (
        st.session_state.get(model_default_version_key) != "four-model-default"
        or model_key not in st.session_state
    ):
        st.session_state[model_key] = model_defaults
        st.session_state[model_default_version_key] = "four-model-default"

    with st.sidebar.container(border=True, key="simulation_params_panel"):
        st.html(
            """
<div class="settings-panel-title">
  <span class="settings-panel-dot"></span>
  <span>Simulation_Params</span>
</div>
            """
        )

        render_settings_field_header("True high-risk rate", format_percent(true_rate_value))
        true_high_risk_rate = st.slider(
            "Percentage of true high-risk children (%)",
            true_rate_min,
            true_rate_max,
            step=0.5,
            format="%.1f%%",
            key=true_rate_key,
            help=SETTING_DESCRIPTIONS["Percentage of true high-risk children (%)"],
            label_visibility="collapsed",
            disabled=is_running,
        ) / 100
        render_settings_field_copy(
            "Share of children who would commit the predicted serious harmful act with no intervention."
        )

        render_settings_field_header("Prediction error", format_percent(prediction_error_value))
        prediction_noise = st.slider(
            "Prediction error rate (%)",
            prediction_error_min,
            prediction_error_max,
            step=0.5,
            format="%.1f%%",
            key=prediction_error_key,
            help=SETTING_DESCRIPTIONS["Prediction error rate (%)"],
            label_visibility="collapsed",
            disabled=is_running,
        ) / 100
        render_settings_field_copy(
            "Symmetric error rate for false negatives among high-risk children and false positives among low-risk children."
        )

        render_settings_field_header("Intervention intensity", intensity_value)
        policy_intensity_tier = st.segmented_control(
            "Intervention intensity",
            options=["Low", "Medium", "High"],
            default="Medium",
            required=True,
            key=intensity_key,
            help=SETTING_DESCRIPTIONS["Intervention intensity"],
            label_visibility="collapsed",
            width="stretch",
            disabled=is_running,
        )

        render_settings_field_header("LLM model agent(s)")
        selected_models = st.multiselect(
            "LLM model agent(s)",
            options=model_options,
            default=None,
            max_selections=MAX_LLM_MODEL_AGENTS,
            key=model_key,
            help=f"Select up to {MAX_LLM_MODEL_AGENTS} model agents at once.",
            label_visibility="collapsed",
            disabled=is_running,
        )

        render_settings_stat_rows(population_size, len(POLICIES))
        run_info_slot = st.container()
        run_button_slot = st.empty()

    settings = {
        "population_size": population_size,
        "true_high_risk_rate": true_high_risk_rate,
        "prediction_noise": prediction_noise,
        "policy_intensity_tier": policy_intensity_tier or "Medium",
        "llm_simulation_runs": 5,
        "llm_representative_agents": 6,
    }
    settings["derived_flagged_rate"] = derived_flagged_count(settings) / population_size
    settings["high_risk_threshold"] = settings["derived_flagged_rate"]
    settings["llm_agent_models"] = selected_models

    return settings, run_info_slot, run_button_slot


def render_app():
    st.set_page_config(page_title="Predictive Justice Thought Experiment", layout="wide")
    render_global_styles()

    achievements_slot = st.sidebar.container()
    settings, run_info_slot, run_button_slot = sidebar_inputs()

    render_app_header()
    render_achievement_notifications(st.session_state.pop("achievement_notifications", []))
    render_hero_statement()
    render_hero_summary(settings)
    render_reference_guide()
    st.html('<div class="section-gap section-gap-lg"></div>')

    render_llm_agent_section(settings, run_info_slot, run_button_slot)
    with achievements_slot:
        render_sidebar_achievements()
