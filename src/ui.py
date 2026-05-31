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
    <span class="badge-chip">N = 1,000</span>
    <span class="badge-chip">{len(POLICIES)} policies</span>
    <span class="badge-chip badge-live">live</span>
  </div>
</header>
        """
    )


def render_hero_badges():
    st.html(
        """
<div class="hero-badges">
  <span class="badge-chip badge-live">Thought Experiment</span>
  <span class="badge-chip">Statistical</span>
  <span class="badge-chip">Agentic AI</span>
</div>
        """
    )


def render_hero_summary(settings):
    population_size = int(settings["population_size"])
    true_high_risk_count = clamp_count(settings["true_high_risk_rate"] * population_size, population_size)
    false_positives, false_negatives, flagged_count = risk_signal_counts(true_high_risk_count, settings)
    cards = [
        ("Synthetic population", f"{population_size:,}", "Children in each run", ""),
        ("True high-risk children", f"{true_high_risk_count:,}", "Without intervention", "primary"),
        ("Prediction errors", f"{false_positives + false_negatives:,}", "False positives + false negatives", "warning"),
        ("Flagged by prediction", f"{flagged_count:,}", "Policy action group", ""),
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
        """
<section class="hero-copy">
  <h1 class="hero-question">
    <span class="hero-nowrap">Suppose we could reliably predict, at age <span class="hero-accent">10</span></span>,<br><span class="hero-nowrap">who will become a violent criminal by age <span class="hero-accent">30</span></span>.
  </h1>
  <p class="hero-subtitle">
    What should we <em>do</em> with that information?<br>Run a synthetic population of 1,000 children through three policies &mdash; and watch what a few percentage points of error actually cost.
  </p>
</section>
        """
    )


def render_population_view_overview(settings):
    population_size = int(settings["population_size"])
    policy_count = len(POLICY_ORDER)
    st.html(
        f"""
<section class="population-overview">
  <div class="population-overview-title">LIVE_SYNTHETIC_POPULATION_VIEW</div>
  <div class="population-overview-heading">{population_size:,} synthetic children &middot; {policy_count} parallel policies</div>
  <div class="population-legend">
    <span class="population-legend-item"><span class="population-legend-dot is-safe"></span>Not flagged &middot; safe</span>
    <span class="population-legend-item"><span class="population-legend-dot is-diverted"></span>Flagged true positive &middot; offense prevented</span>
    <span class="population-legend-item"><span class="population-legend-dot is-violent"></span>Flagged and high-risk</span>
    <span class="population-legend-item"><span class="population-legend-dot is-missed"></span>Missed by prediction (false negative)</span>
    <span class="population-legend-item"><span class="population-legend-dot is-wrong"></span>Wrongly flagged (false positive)</span>
  </div>
</section>
        """
    )


def render_achievement_notifications(notifications):
    if not notifications:
        return

    cards = []
    for index, achievement in enumerate(notifications):
        cards.append(
            f"""
<div class="achievement-toast" style="--toast-index:{index}">
  <span class="achievement-toast-check" aria-hidden="true"></span>
  <span class="achievement-icon achievement-icon-trophy" aria-hidden="true"></span>
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
  --radius-sm: 6px;
  --radius: 8px;
  --mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
  --shadow-xs: 0 1px 0 rgba(7, 24, 35, 0.05);
  --shadow-sm: 0 1px 2px rgba(7, 24, 35, 0.05), 0 12px 30px -22px rgba(7, 24, 35, 0.30);
  --shadow-md: 0 20px 50px -36px rgba(7, 24, 35, 0.38);
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
#MainMenu {
  display: none !important;
}

[data-testid="stToolbar"] {
  background: transparent;
  pointer-events: none;
}

[data-testid="stExpandSidebarButton"] {
  position: fixed;
  top: 0.7rem;
  left: 0.7rem;
  z-index: 999999;
  display: inline-grid !important;
  width: 2.2rem;
  height: 2.2rem;
  place-items: center;
  border: 1px solid rgba(53, 88, 72, 0.16);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.86);
  box-shadow: var(--shadow-xs);
  pointer-events: auto;
}

[data-testid="stExpandSidebarButton"] * {
  pointer-events: auto;
}

/* ── Main content ─────────────────────────── */
[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 1232px;
  padding-top: 1.5rem;
  padding-bottom: 4rem;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
  border-right: 1px solid var(--line-soft);
  box-shadow: 10px 0 36px -28px rgba(7, 24, 35, 0.45);
}

[data-testid="stSidebar"] > div:first-child {
  background:
    linear-gradient(rgba(0, 167, 87, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 167, 87, 0.05) 1px, transparent 1px),
    rgba(255, 255, 255, 0.86);
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
  margin-top: 1.4rem;
  margin-bottom: 1.35rem;
  font-size: 4.1rem;
  line-height: 1.04;
  font-weight: 750;
}

h2 {
  margin-top: 1.65rem;
  font-size: 1.28rem;
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.18em;
}

h3 {
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
  color: #5d737b;
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
  padding: 1rem 0 1.25rem;
  margin: -0.25rem 0 2.1rem;
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
  border: 1px solid rgba(0, 167, 87, 0.44);
  border-radius: 7px;
  background: rgba(0, 167, 87, 0.11);
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
}

.hero-badges {
  margin: 0.5rem 0 -0.4rem;
}

.hero-copy {
  max-width: 980px;
  margin-top: 1.25rem;
}

.hero-question {
  margin: 0;
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
  margin: 2rem 0 0;
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
  background: rgba(255, 255, 255, 0.58);
  color: var(--text-muted);
  font-size: 0.67rem;
  font-weight: 600;
}

.badge-live {
  border-color: rgba(0, 167, 87, 0.48);
  color: var(--primary);
}

.hero-stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
  margin: 2.35rem 0 2.2rem;
}

.hero-stat-card {
  min-height: 7.2rem;
  padding: 1.05rem 1.2rem;
  border: 1px solid rgba(53, 88, 72, 0.16);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.78);
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
  margin: 1.6rem 0 1.05rem;
  padding: 1.35rem 1.55rem 1.45rem;
  border: 1px solid rgba(53, 88, 72, 0.16);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.84);
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
  display: flex;
  flex-wrap: wrap;
  gap: 0.72rem 1.28rem;
  margin-top: 1.35rem;
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 0.86rem;
  line-height: 1.35;
}

.population-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.population-legend-dot {
  display: inline-block;
  width: 0.72rem;
  height: 0.72rem;
  flex: 0 0 auto;
  border-radius: 999px;
  box-sizing: border-box;
}

.population-legend-dot.is-safe {
  background: rgba(157, 190, 168, 0.62);
}

.population-legend-dot.is-diverted {
  background: var(--primary);
  outline: 2px solid #f08a00;
}

.population-legend-dot.is-violent {
  background: #dd2538;
  outline: 2px solid #f08a00;
}

.population-legend-dot.is-missed {
  background: #dd2538;
}

.population-legend-dot.is-wrong {
  background: rgba(157, 190, 168, 0.62);
  outline: 2px solid #f08a00;
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
  border: 1px solid rgba(53, 88, 72, 0.14);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 16px 40px rgba(7, 24, 35, 0.14), 0 2px 6px rgba(7, 24, 35, 0.08);
  animation: achievementToast 7s ease both;
  animation-delay: calc(var(--toast-index) * 130ms);
}

.achievement-toast-check {
  display: grid;
  width: 1.55rem;
  height: 1.55rem;
  place-items: center;
  border-radius: 999px;
  background: #151719;
}

.achievement-toast-check::before {
  content: "";
  width: 0.72rem;
  height: 0.44rem;
  border-left: 2px solid #ffffff;
  border-bottom: 2px solid #ffffff;
  transform: rotate(-45deg) translate(1px, -1px);
}

.achievement-toast .achievement-icon {
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
  color: #3f4e54;
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

.achievement-shell {
  overflow: hidden;
  margin-bottom: 1rem;
  border: 1px solid rgba(53, 88, 72, 0.16);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.82);
  box-shadow: var(--shadow-xs);
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
  border-bottom: 1px solid rgba(53, 88, 72, 0.13);
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

.achievement-toggle {
  display: inline-grid;
  width: 1.35rem;
  height: 1.35rem;
  place-items: center;
  color: var(--text-muted);
  line-height: 1;
  transition: transform 160ms ease;
}

.achievement-toggle svg {
  width: 1rem;
  height: 1rem;
  display: block;
  stroke: currentColor;
}

.achievement-toggle::before {
  content: "";
  display: block;
  width: 1rem;
  height: 1rem;
  background: currentColor;
  -webkit-mask: var(--achievement-mask) center / contain no-repeat;
  mask: var(--achievement-mask) center / contain no-repeat;
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
  border-bottom: 1px solid rgba(53, 88, 72, 0.08);
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

.achievement-icon svg {
  width: 1.25rem;
  height: 1.25rem;
  display: block;
  stroke: currentColor;
}

.achievement-icon::before {
  content: "";
  display: block;
  width: 1.25rem;
  height: 1.25rem;
  background: currentColor;
  -webkit-mask: var(--achievement-mask) center / contain no-repeat;
  mask: var(--achievement-mask) center / contain no-repeat;
}

.achievement-icon-trophy {
  --achievement-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 4h8v5a4 4 0 0 1-8 0V4Z'/%3E%3Cpath d='M8 7H5.8a2.3 2.3 0 0 0 0 4.6H8'/%3E%3Cpath d='M16 7h2.2a2.3 2.3 0 0 1 0 4.6H16'/%3E%3Cpath d='M12 13v5'/%3E%3Cpath d='M9 18h6'/%3E%3Cpath d='M7.5 21h9'/%3E%3C/svg%3E");
}

.achievement-icon-lock {
  --achievement-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='10.5' width='14' height='10' rx='2'/%3E%3Cpath d='M8.5 10.5V7.8a3.5 3.5 0 0 1 7 0v2.7'/%3E%3C/svg%3E");
}

.achievement-icon-chevron {
  --achievement-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 15 6-6 6 6'/%3E%3C/svg%3E");
}

.achievement-row.is-locked .achievement-icon {
  color: #94a2a8;
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
  margin-bottom: 1rem;
  border: 1px solid rgba(53, 88, 72, 0.16);
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.86) 0%, rgba(250, 253, 251, 0.86) 100%);
  box-shadow: var(--shadow-xs);
}

[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.settings-panel-title) > div {
  padding: 1rem;
}

[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.settings-panel-title) [data-testid="stVerticalBlock"] {
  gap: 0.55rem;
}

.settings-panel-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.75rem;
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
  background: #72c58f;
  box-shadow: 0 0 0 4px rgba(0, 167, 87, 0.08);
}

.settings-field-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.75rem;
  margin-top: 0.55rem;
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
  margin: 0.15rem 0 0.45rem;
  color: var(--text-muted);
  font-size: 0.78rem;
  line-height: 1.35;
}

.settings-panel-divider {
  height: 1px;
  margin: 0.7rem 0 0.25rem;
  background: rgba(53, 88, 72, 0.14);
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
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(53, 88, 72, 0.16);
  border-radius: var(--radius);
  box-shadow: var(--shadow-xs);
  transition: box-shadow 180ms ease, border-color 180ms ease;
}

[data-testid="stExpander"] details:hover {
  box-shadow: var(--shadow-sm);
  border-color: var(--line);
}

[data-testid="stExpander"] details[open] {
  border-color: rgba(0, 167, 87, 0.32);
  box-shadow: var(--shadow-sm);
}

[data-testid="stExpander"] summary {
  font-family: var(--mono);
  font-weight: 600;
  font-size: 0.75rem;
  color: var(--text);
  padding: 0.85rem 1rem;
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
  border-color: rgba(0, 167, 87, 0.44);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.stButton > button:active:not(:disabled) {
  transform: translateY(0);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, var(--primary) 0%, var(--primary-strong) 100%);
  border-color: var(--primary-strong);
  color: #ffffff;
  box-shadow: 0 2px 8px rgba(0, 167, 87, 0.22), 0 1px 2px rgba(0, 167, 87, 0.16);
}

.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p {
  color: #ffffff !important;
  font-weight: 600;
}

.stButton > button[kind="primary"]:hover:not(:disabled) {
  background: linear-gradient(180deg, #05b961 0%, #008f4a 100%);
  box-shadow: 0 6px 18px rgba(0, 167, 87, 0.30), 0 2px 6px rgba(0, 167, 87, 0.16);
}

/* ── Form controls ────────────────────────── */
[data-baseweb="select"] > div {
  border-radius: var(--radius-sm) !important;
}

[data-testid="stSlider"] [role="slider"] {
  background-color: var(--primary);
  border-color: #ffffff;
  box-shadow: 0 0 0 3px rgba(0, 167, 87, 0.14), 0 2px 5px rgba(0, 167, 87, 0.22);
}

[data-testid="stSlider"] div[data-testid="stTickBar"] div {
  background: var(--line);
}

[data-testid="stSidebar"] [data-testid="stSlider"] {
  margin-top: -0.3rem;
  margin-bottom: 0.05rem;
}

[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
  background: #ffffff;
  border: 1px solid #72c58f;
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
  background: rgba(255, 255, 255, 0.58);
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
  background: rgba(0, 167, 87, 0.08);
  box-shadow: inset 0 0 0 1.5px var(--primary);
  color: var(--primary);
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
  border-color: var(--line);
  background: rgba(255, 255, 255, 0.72);
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
  background: #00a757;
  border-color: #008846;
  box-shadow: 0 8px 18px rgba(0, 136, 70, 0.22), 0 2px 4px rgba(7, 24, 35, 0.10);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] > div {
  flex: 0 0 auto;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] p {
  font-size: 0.95rem;
  letter-spacing: 0.16em;
  line-height: 1;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--text-muted);
  font-size: 0.74rem;
  line-height: 1.35;
}

/* ── Tabs ─────────────────────────────────── */
[data-baseweb="tab-list"] {
  gap: 0.2rem;
  border-bottom: 2px solid var(--line-soft);
}

[data-baseweb="tab"] {
  border-radius: 999px;
  color: var(--text-muted);
  font-family: var(--mono);
  font-weight: 600;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  padding: 0.45rem 0.9rem;
  transition: color 140ms ease, background 140ms ease;
}

[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  color: var(--text);
  background: var(--surface-muted);
}

[data-baseweb="tab"][aria-selected="true"] {
  background: rgba(0, 167, 87, 0.09);
  color: var(--primary);
  font-weight: 650;
  box-shadow: inset 0 0 0 1px rgba(0, 167, 87, 0.36);
}

/* ── Metric cards ─────────────────────────── */
div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMetric"]) {
  background: rgba(255, 255, 255, 0.80);
  border: 1px solid rgba(53, 88, 72, 0.16);
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
  border: 1px solid rgba(53, 88, 72, 0.16);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow-xs);
}

/* ── Alerts ───────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius);
  border: 1px solid rgba(53, 88, 72, 0.16);
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
    font-size: 2.35rem;
    line-height: 1.08;
  }

  [data-baseweb="tab"] {
    padding-left: 0.65rem;
    padding-right: 0.65rem;
    font-size: 0.82rem;
  }

  .app-topbar {
    align-items: flex-start;
    flex-direction: column;
    margin-bottom: 1.4rem;
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
    margin-top: 1.4rem;
    font-size: 1.05rem;
  }

  .population-overview {
    padding: 1.15rem;
  }

  .population-overview-heading {
    font-size: 1.25rem;
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
  border: 1px solid rgba(53,88,72,0.16);
  border-radius: 8px;
  padding: 18px 20px 16px;
  margin: 0 0 20px;
  background: rgba(255,255,255,0.84);
  box-shadow: 0 1px 2px rgba(7,24,35,0.05), 0 20px 42px -34px rgba(7,24,35,0.38);
}}
.policy-panels {{
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.policy-panel {{
  border: 1px solid rgba(53,88,72,0.14);
  border-left: 3px solid rgba(0,167,87,0.72);
  border-radius: 8px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.78);
  box-shadow: 0 1px 2px rgba(7,24,35,0.04);
  transition: border-color 200ms ease, box-shadow 200ms ease;
}}
.policy-panel[data-ready="true"] {{
  border-color: rgba(0,167,87,0.28);
  border-left-width: 3px;
  box-shadow: 0 2px 10px rgba(0,167,87,0.07);
}}
.policy-panel-title {{
  color: #071823;
  font-size: 0.88rem;
  font-weight: 650;
  line-height: 1.3;
  margin-bottom: 4px;
  letter-spacing: -0.01em;
}}
.policy-panel-summary {{
  color: #48616a;
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
  background: rgba(247,252,249,0.72);
  border: 1px solid rgba(53,88,72,0.12);
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
  --base-color: rgba(157,190,168,0.58);
  --base-scale: 0.92;
}}
.base-high {{
  --base-color: #dd2538;
  --base-scale: 1.28;
}}
.final-low {{
  --final-color: rgba(157,190,168,0.58);
  --final-scale: 0.92;
}}
.final-high {{
  --final-color: #dd2538;
  --final-scale: 1.28;
}}
.flagged-dot {{
  outline: 2px solid #f08a00;
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
  background: #00a757;
  box-shadow: 0 0 0 3px rgba(0,167,87,0.24);
}}
.policy-panel.show-final .life-dot.changed-harmed {{
  box-shadow: 0 0 0 3px rgba(221,37,56,0.30);
}}
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
  <div class="policy-panels">
    {''.join(panels)}
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
    rows = []
    for ach in ACHIEVEMENTS:
        is_earned = ach["id"] in earned
        row_class = "achievement-row" if is_earned else "achievement-row is-locked"
        icon_class = "achievement-icon-trophy" if is_earned else "achievement-icon-lock"
        description = (
            f'<div class="achievement-desc">{escape(ach["description"])}</div>' if is_earned else ""
        )
        rows.append(
            f"""
<div class="{row_class}">
  <div class="achievement-icon {icon_class}" aria-hidden="true"></div>
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
      <span class="achievement-icon achievement-icon-trophy" aria-hidden="true"></span>
      <span class="achievement-title">Achievements</span>
      <span class="achievement-count">{count}<span> / {len(ACHIEVEMENTS)}</span></span>
    </div>
    <span class="achievement-toggle achievement-icon-chevron" aria-hidden="true"></span>
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


def render_llm_agent_section(settings, run_info_slot=None, run_button_slot=None):
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

    render_population_view_overview(settings)
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

    if run_info_slot is None:
        run_info_slot = st.sidebar.container()
    with run_info_slot:
        st.caption(
            f"Current run: {len(POLICIES)} policies × {len(selected_models)} model agent(s) = "
            f"{total_calls} LLM call(s). Each call generates {int(settings['llm_simulation_runs'])} "
            f"synthetic run(s) over {int(settings['population_size']):,} synthetic children."
        )
    settings_invalid = settings["true_high_risk_rate"] == 0
    with run_info_slot:
        if not selected_models:
            st.warning("Select at least one LLM model agent.")
        if settings_invalid:
            st.warning("Set 'Percentage of true high-risk children' above 0% to run a meaningful simulation.")

    run_disabled = not bool(api_key) or not selected_models or settings_invalid
    if run_button_slot is None:
        run_button_slot = st.sidebar.empty()
    if st.session_state.get("simulation_running", False):
        with run_button_slot:
            st.button(
                "▶ Run simulation",
                key="run_simulation_running",
                disabled=True,
                type="primary",
                use_container_width=True,
            )
        run_requested = False
    else:
        with run_button_slot:
            run_requested = st.button(
                "▶ Run simulation",
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
                "▶ Run simulation",
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
            unlocked_ids = newly_earned - previously_earned
            if unlocked_ids:
                st.session_state["achievement_notifications"] = [
                    {
                        "name": ACHIEVEMENT_INDEX[ach["id"]]["name"],
                        "description": ACHIEVEMENT_INDEX[ach["id"]]["description"],
                    }
                    for ach in ACHIEVEMENTS
                    if ach["id"] in unlocked_ids
                ]

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
    population_size = 1000
    true_rate_key = "true_high_risk_rate_percent"
    prediction_error_key = "prediction_error_percent"
    strength_key = "policy_effect_strength"
    model_key = "llm_agent_models"
    true_rate_default = int(DEFAULT_TRUE_HIGH_RISK_RATE * 100)
    prediction_error_default = 5
    true_rate_value = int(st.session_state.get(true_rate_key, true_rate_default))
    prediction_error_value = int(st.session_state.get(prediction_error_key, prediction_error_default))
    strength_value = st.session_state.get(strength_key, "Medium")

    model_options = llm_model_options()

    with st.sidebar.container(border=True, key="simulation_params_panel"):
        st.html(
            """
<div class="settings-panel-title">
  <span class="settings-panel-dot"></span>
  <span>Simulation_Params</span>
</div>
            """
        )

        render_settings_field_header("True high-risk rate", f"{true_rate_value}%")
        true_high_risk_rate = st.slider(
            "Percentage of true high-risk children (%)",
            0,
            100,
            true_rate_default,
            step=1,
            key=true_rate_key,
            help=SETTING_DESCRIPTIONS["Percentage of true high-risk children (%)"],
            label_visibility="collapsed",
        ) / 100
        render_settings_field_copy(
            "Share of children who would commit violence by age 30 with no intervention."
        )

        render_settings_field_header("Prediction error", f"{prediction_error_value}%")
        prediction_noise = st.slider(
            "Prediction error rate (%)",
            0,
            100,
            prediction_error_default,
            step=1,
            key=prediction_error_key,
            help=SETTING_DESCRIPTIONS["Prediction error rate (%)"],
            label_visibility="collapsed",
        ) / 100
        render_settings_field_copy("Rate at which the age-10 prediction misclassifies a child.")

        render_settings_field_header("Intervention strength", strength_value)
        policy_effect_strength = st.segmented_control(
            "Intervention strength",
            options=["Low", "Medium", "High"],
            default="Medium",
            required=True,
            key=strength_key,
            help=SETTING_DESCRIPTIONS["Intervention strength"],
            label_visibility="collapsed",
            width="stretch",
        )

        render_settings_field_header("LLM model agent(s)")
        selected_models = st.multiselect(
            "LLM model agent(s)",
            options=model_options,
            default=default_llm_agent_models(model_options),
            key=model_key,
            label_visibility="collapsed",
        )

        render_settings_stat_rows(population_size, len(POLICIES))
        run_info_slot = st.container()
        run_button_slot = st.empty()

    settings = {
        "population_size": population_size,
        "true_high_risk_rate": true_high_risk_rate,
        "prediction_noise": prediction_noise,
        "bias_against_district_c": 0.0,
        "policy_effect_strength": policy_effect_strength or "Medium",
        "llm_simulation_runs": 5,
        "llm_representative_agents": 2,
    }
    true_high_risk_count = clamp_count(true_high_risk_rate * population_size, population_size)
    fp, fn, flagged_count = risk_signal_counts(true_high_risk_count, settings)
    settings["high_risk_threshold"] = flagged_count / population_size
    settings["llm_agent_models"] = selected_models

    return settings, run_info_slot, run_button_slot


def render_app():
    st.set_page_config(page_title="Predictive Justice Thought Experiment", layout="wide")
    render_global_styles()

    achievements_slot = st.sidebar.container()
    settings, run_info_slot, run_button_slot = sidebar_inputs()

    render_app_header()
    render_achievement_notifications(st.session_state.pop("achievement_notifications", []))
    render_hero_badges()
    render_hero_statement()
    render_hero_summary(settings)
    render_reference_guide()

    render_llm_agent_section(settings, run_info_slot, run_button_slot)
    with achievements_slot:
        render_sidebar_achievements()
