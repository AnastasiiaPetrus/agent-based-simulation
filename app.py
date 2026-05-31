import json
import os
from datetime import datetime
from html import escape
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


POLICIES = [
    "Targeted support for high-risk children",
    "Surveillance of high-risk children",
    "Coercive preventive intervention for high-risk children",
]

POLICY_ORDER = list(POLICIES)

POLICY_DESCRIPTIONS = {
    "Targeted support for high-risk children": (
        "Voluntary support only for flagged children; error-sensitive."
    ),
    "Surveillance of high-risk children": (
        "Monitoring for flagged children; may reduce crime but exposes children to surveillance."
    ),
    "Coercive preventive intervention for high-risk children": (
        "Restriction before any act; highest ethical concern."
    ),
}

SETTING_DESCRIPTIONS = {
    "Prediction error rate (%)": "How noisy the risk signal is. It increases missed would-offend cases and randomly flagged children whose baseline trajectory would not include the offense.",
    "Children flagged as high-risk (%)": "Share of the 1 000 synthetic children identified by the risk signal. 25% = 250 children flagged.",
    "Intervention strength": "How intensively the chosen policy is applied — scales the simulated effect on offenses, support reach, and harm.",
}

RESULT_METRIC_DESCRIPTIONS = {
    "Children who would offend (no intervention)": "Count of synthetic children whose simulated life trajectory leads to a violent offense when no policy is applied — the baseline against which all policies are compared.",
    "Offenses prevented by policy": "Difference between baseline offenses and offenses remaining after the policy. Derived directly from the two counts above.",
    "Children incorrectly flagged": "Flagged children who would NOT have committed the offense — false positives. They bear the cost of the policy without any benefit.",
    "Children missed by risk signal": "Unflagged children who WOULD have committed the offense — false negatives. They receive no intervention regardless of policy.",
    "Children receiving support": "Children reached by voluntary support. Non-zero only under the targeted support policy.",
    "Children exposed to harmful intervention": "Children whose simulated trajectory is adversely affected by surveillance (stigma, trust loss) or coercive restriction (liberty, opportunity). Zero under targeted support.",
}

CHECK_DESCRIPTIONS = {
    "Policy trade-off": "Crimes prevented vs. children incorrectly flagged and children exposed to harm.",
    "Prediction error": "False positives (flagged without basis) and false negatives (missed entirely).",
}

DISTRICTS = ["A", "B", "C"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"]
DEFAULT_DEBRIEF_WORD_LIMIT = 250
MAX_RUN_LOG_SIZE = 5
DEFAULT_SYSTEM_PROMPT = (
    "You run a synthetic multi-agent life-course simulation for an ethical thought experiment. "
    "Use English only. Never include markdown fences. "
    "Simulate individual developmental trajectories first, then derive aggregate outcomes from those trajectories. "
    "Never claim to predict real people, assign guilt, use demographic characteristics, or recommend punishment. "
    "Never assume a high-risk prediction becomes reality. Treat all output as synthetic thought-experiment data."
)
RUN_METRIC_COLUMNS = [
    "run",
    "baseline_crimes",
    "crimes_after_policy",
    "crimes_prevented",
    "false_positives",
    "false_negatives",
    "children_helped",
    "children_harmed",
]
DISTRICT_METRIC_COLUMNS = ["run", "district", "false_positives", "children_harmed", "crimes"]
RUN_COUNT_COLUMNS = [
    "baseline_crimes",
    "crimes_after_policy",
    "crimes_prevented",
    "false_positives",
    "false_negatives",
    "children_helped",
    "children_harmed",
]
DISTRICT_COUNT_COLUMNS = ["false_positives", "children_harmed", "crimes"]
NON_NEGATIVE_RUN_COLUMNS = RUN_COUNT_COLUMNS
NON_NEGATIVE_DISTRICT_COLUMNS = DISTRICT_COUNT_COLUMNS
DEFAULT_BASELINE_OFFENSE_RATE = 0.125
POPULATION_DOT_ANIMATION_SECONDS = 0.9
POPULATION_DOT_PULSE_SECONDS = 2.6
POPULATION_DOT_STAGGER_GROUP = 50
POPULATION_DOT_STAGGER_SECONDS = 0.004
POLICY_EFFECT_REDUCTION_RATES = {"Low": 0.05, "Medium": 0.15, "High": 0.28}

RUN_METRIC_LABELS = {
    "baseline_crimes": "Children who would offend (no intervention)",
    "crimes_prevented": "Offenses prevented by policy",
    "false_positives": "Children incorrectly flagged",
    "false_negatives": "Children missed by risk signal",
    "children_helped": "Children receiving support",
    "children_harmed": "Children exposed to harmful intervention",
}


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


def average_results_table(run_results):
    metric_order = [column for column in RUN_METRIC_LABELS if column in run_results.columns]
    averages = run_results[metric_order].mean(numeric_only=True)
    table = averages.rename(index=RUN_METRIC_LABELS).reset_index()
    table.columns = ["Metric", "Average per synthetic run"]
    return table


def display_average_table(average_table):
    display_table = average_table.copy()

    def format_value(value):
        if pd.isna(value):
            return "Not applicable"
        return f"{value:,.3f}"

    display_table["Average per synthetic run"] = display_table["Average per synthetic run"].map(format_value)
    st.dataframe(display_table, use_container_width=True, hide_index=True)


def model_comparison_table(run_results):
    metric_order = [column for column in RUN_METRIC_LABELS if column in run_results.columns]
    grouped = run_results.groupby("llm_model")[metric_order].mean(numeric_only=True).T
    table = grouped.rename(index=RUN_METRIC_LABELS).reset_index().rename(columns={"index": "Metric"})

    for column in table.columns:
        if column != "Metric":
            table[column] = table[column].map(
                lambda value: "Not applicable" if pd.isna(value) else f"{value:,.3f}"
            )

    return table


def display_model_comparison_table(run_results):
    st.dataframe(model_comparison_table(run_results), use_container_width=True, hide_index=True)

def policy_model_comparison_table(run_results):
    metric_columns = [column for column in RUN_METRIC_LABELS if column in run_results.columns]
    table = (
        run_results.groupby(["policy", "llm_model"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )
    table["policy_sort"] = table["policy"].map({policy: index for index, policy in enumerate(POLICY_ORDER)})
    table = table.sort_values(["policy_sort", "llm_model"]).drop(columns="policy_sort")
    table = table.rename(
        columns={
            "policy": "Policy",
            "llm_model": "Model",
            **RUN_METRIC_LABELS,
        }
    )
    for column in table.columns:
        if column in {"Policy", "Model"}:
            continue
        table[column] = table[column].map(lambda v: "Not applicable" if pd.isna(v) else f"{v:,.3f}")
    return table


def display_policy_model_comparison_table(run_results):
    st.dataframe(policy_model_comparison_table(run_results), use_container_width=True, hide_index=True)


def combined_policy_totals_table(run_results):
    total_metric_labels = {
        "baseline_crimes": "Total children who would offend (no intervention)",
        "crimes_after_policy": "Total offenses after policy",
        "crimes_prevented": "Total offenses prevented by policy",
        "false_positives": "Total children incorrectly flagged",
        "false_negatives": "Total children missed by risk signal",
        "children_helped": "Total children receiving support",
        "children_harmed": "Total children exposed to harmful intervention",
    }
    metric_columns = [column for column in total_metric_labels if column in run_results.columns]
    table = (
        run_results.groupby("policy", as_index=False)
        .agg(**{column: (column, "sum") for column in metric_columns})
    )
    table["policy_sort"] = table["policy"].map({policy: index for index, policy in enumerate(POLICY_ORDER)})
    table = table.sort_values("policy_sort").drop(columns="policy_sort")
    table["crime_reduction_pct"] = (
        table["crimes_prevented"] / table["baseline_crimes"].replace(0, np.nan)
    ) * 100

    display_table = table.rename(
        columns={
            "policy": "Policy",
            "crime_reduction_pct": "Offense reduction (%)",
            **total_metric_labels,
        }
    )
    ordered_columns = [
        "Policy",
        "Offense reduction (%)",
        "Total children who would offend (no intervention)",
        "Total offenses after policy",
        "Total children incorrectly flagged",
        "Total children missed by risk signal",
        "Total children receiving support",
        "Total children exposed to harmful intervention",
    ]
    display_table = display_table[[c for c in ordered_columns if c in display_table.columns]]
    for column in display_table.columns:
        if column == "Policy":
            continue
        if column == "Offense reduction (%)":
            display_table[column] = display_table[column].map(
                lambda v: "N/A" if pd.isna(v) else f"{v:.1f}%"
            )
        else:
            display_table[column] = display_table[column].map(
                lambda v: "N/A" if pd.isna(v) else f"{v:,.0f}"
            )
    return display_table


def display_combined_policy_totals_table(run_results):
    st.dataframe(combined_policy_totals_table(run_results), use_container_width=True, hide_index=True)


def clamp_count(value, maximum):
    if pd.isna(value):
        return 0
    return int(max(0, min(maximum, round(float(value)))))


def baseline_count_for_run(run_number, run_numbers, settings):
    population_size = int(settings["population_size"])
    target = float(settings["baseline_offense_rate"]) * population_size
    sorted_runs = sorted(set(int(run) for run in run_numbers))
    if len(sorted_runs) <= 1:
        return clamp_count(target, population_size)

    run_index = sorted_runs.index(int(run_number))
    variation = np.linspace(-0.06, 0.06, len(sorted_runs))[run_index]
    return clamp_count(target * (1 + variation), population_size)


def risk_signal_error_counts(baseline_count, settings):
    population_size = int(settings["population_size"])
    flagged_count = clamp_count(settings["high_risk_threshold"] * population_size, population_size)
    signal_error_rate = float(settings["prediction_noise"])

    desired_false_negatives = clamp_count(baseline_count * signal_error_rate, baseline_count)
    true_positive_capacity = max(0, baseline_count - desired_false_negatives)
    true_positives = min(flagged_count, true_positive_capacity)
    false_negatives = max(0, baseline_count - true_positives)
    false_positives = max(0, flagged_count - true_positives)
    return false_positives, false_negatives


def population_risk_counts(run_results, settings):
    population_size = int(settings["population_size"])
    flagged = clamp_count(settings["high_risk_threshold"] * population_size, population_size)
    if run_results is None or run_results.empty:
        baseline = clamp_count(settings["baseline_offense_rate"] * population_size, population_size)
        high = min(baseline, population_size)
        medium = min(flagged, population_size - high)
        low = population_size - high - medium
        return {
            "low": low,
            "medium": medium,
            "high": high,
            "baseline": baseline,
            "prevented": 0,
            "harmed": 0,
            "false_positive": 0,
        }

    averages = run_results[RUN_COUNT_COLUMNS].mean(numeric_only=True)
    baseline = clamp_count(averages.get("baseline_crimes"), population_size)
    prevented = clamp_count(averages.get("crimes_prevented"), population_size)
    harmed = clamp_count(averages.get("children_harmed"), population_size)
    false_positive = clamp_count(averages.get("false_positives"), population_size)
    missed = clamp_count(averages.get("false_negatives"), population_size)

    high = max(harmed, baseline - prevented, missed)
    medium = max(false_positive, flagged)
    high = min(high, population_size)
    medium = min(medium, population_size - high)
    low = population_size - high - medium

    return {
        "low": low,
        "medium": medium,
        "high": high,
        "baseline": baseline,
        "prevented": prevented,
        "harmed": harmed,
        "false_positive": false_positive,
    }


def population_animation_html(run_results, settings, title, animation_key=""):
    population_size = int(settings["population_size"])
    counts = population_risk_counts(run_results, settings)
    risk_classes = (
        ["risk-high"] * counts["high"]
        + ["risk-medium"] * counts["medium"]
        + ["risk-low"] * counts["low"]
    )
    seed_text = f"{title}-{animation_key}"
    seed = sum(ord(char) for char in seed_text) + counts["high"] * 7 + counts["medium"] * 13
    animation_id = abs(seed + population_size * 17 + counts["low"] * 23) % 100000
    reveal_animation_name = f"revealRiskDot{animation_id}"
    pulse_animation_name = f"pulseRiskDot{animation_id}"
    rng = np.random.default_rng(seed)
    rng.shuffle(risk_classes)

    dots = []
    for index, risk_class in enumerate(risk_classes[:population_size]):
        delay = (index % POPULATION_DOT_STAGGER_GROUP) * POPULATION_DOT_STAGGER_SECONDS
        dots.append(
            f'<span class="life-dot {risk_class}" style="--delay:{delay:.2f}s"></span>'
        )

    return f"""
<style>
.life-course-card {{
  border: 1px solid #e3e6ee;
  border-radius: 8px;
  padding: 14px 16px;
  margin: 10px 0 16px;
  background: #ffffff;
}}
.life-course-header {{
  margin-bottom: 10px;
}}
.life-course-title {{
  font-weight: 700;
  color: #2d3142;
}}
.life-course-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(12px, 1fr));
  grid-auto-rows: 12px;
  gap: 4px;
  align-items: center;
  justify-items: center;
  width: 100%;
  padding: 8px 0;
}}
.life-dot {{
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
  justify-self: center;
  box-sizing: border-box;
  opacity: 0;
  background: #f1f4f8;
  transform: scale(0.45);
  will-change: transform, opacity, background-color;
  --target-scale: 1;
  --overshoot-scale: 1.12;
  --pulse-scale: 1.06;
  --soft-color: #dde3ec;
  --mid-color: #b7c0ce;
  animation:
    {reveal_animation_name} {POPULATION_DOT_ANIMATION_SECONDS:.1f}s cubic-bezier(.22,.61,.19,1) forwards,
    {pulse_animation_name} {POPULATION_DOT_PULSE_SECONDS:.1f}s ease-in-out infinite;
  animation-delay: var(--delay), calc(var(--delay) + {POPULATION_DOT_ANIMATION_SECONDS:.1f}s);
}}
.risk-low {{
  --target-color: #25a55f;
  --soft-color: #d8eee3;
  --mid-color: #7ed2a1;
  --target-scale: 0.92;
  --overshoot-scale: 1.02;
  --pulse-scale: 0.98;
}}
.risk-medium {{
  --target-color: #f2b705;
  --soft-color: #fff0c6;
  --mid-color: #ffd15a;
  --target-scale: 1.08;
  --overshoot-scale: 1.18;
  --pulse-scale: 1.10;
}}
.risk-high {{
  --target-color: #d64b3c;
  --soft-color: #f6d9d6;
  --mid-color: #eb8d82;
  --target-scale: 1.28;
  --overshoot-scale: 1.38;
  --pulse-scale: 1.30;
}}
@keyframes {reveal_animation_name} {{
  0% {{
    opacity: 0;
    transform: scale(0.45);
    background: #f1f4f8;
    box-shadow: 0 0 0 0 rgba(45, 49, 66, 0);
  }}
  28% {{
    opacity: 0.42;
    transform: scale(0.68);
    background: #edf1f7;
  }}
  55% {{
    opacity: 0.76;
    transform: scale(0.86);
    background: var(--soft-color);
  }}
  78% {{
    opacity: 1;
    transform: scale(var(--overshoot-scale));
    background: var(--mid-color);
    box-shadow: 0 0 0 3px rgba(45, 49, 66, 0.06);
  }}
  100% {{
    opacity: 1;
    transform: scale(var(--target-scale));
    background: var(--target-color);
    box-shadow: 0 0 0 0 rgba(45, 49, 66, 0);
  }}
}}
@keyframes {pulse_animation_name} {{
  0%, 100% {{ transform: scale(var(--target-scale)); filter: brightness(1); }}
  50% {{ transform: scale(var(--pulse-scale)); filter: brightness(1.12); }}
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
  background: var(--target-color);
}}
</style>
<div class="life-course-card">
  <div class="life-course-header">
    <div class="life-course-title">{escape(title)}</div>
  </div>
  <div class="life-course-grid">{''.join(dots)}</div>
  <div class="life-course-legend">
    <span class="legend-item"><span class="legend-dot risk-low"></span>green: lower modeled risk/support path ({counts['low']})</span>
    <span class="legend-item"><span class="legend-dot risk-medium"></span>yellow: flagged by the risk signal ({counts['medium']})</span>
    <span class="legend-item"><span class="legend-dot risk-high"></span>red: modeled offense or harmful intervention path ({counts['high']})</span>
  </div>
</div>
"""


def render_population_animation(container, run_results, settings, title, animation_key=""):
    container.markdown(
        population_animation_html(run_results, settings, title, animation_key),
        unsafe_allow_html=True,
    )


def glossary_markdown(items):
    lines = []
    for item, meaning in items.items():
        lines.append(f"- **{item}:** {meaning}")
    return "\n".join(lines)


def render_reference_guide():
    with st.expander("Guide: policies, inputs, and metrics", expanded=False):
        st.markdown("### Policies compared")
        st.markdown(glossary_markdown(POLICY_DESCRIPTIONS))

        st.markdown("### Inputs")
        st.markdown(glossary_markdown(SETTING_DESCRIPTIONS))

        st.markdown("### Metrics")
        st.markdown(glossary_markdown(RESULT_METRIC_DESCRIPTIONS))

        st.markdown("### What is compared across policies")
        st.markdown(glossary_markdown(CHECK_DESCRIPTIONS))


def line_chart(data, x_column, y_column, title, y_label):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    columns = [x_column, y_column]
    if "llm_model" in data.columns:
        columns.append("llm_model")

    plot_data = data[columns].dropna()
    if "llm_model" in plot_data.columns and plot_data["llm_model"].nunique() > 1:
        for llm_model, model_data in plot_data.groupby("llm_model"):
            ax.plot(model_data[x_column], model_data[y_column], linewidth=1.8, label=llm_model)
        ax.legend(fontsize=8)
    else:
        ax.plot(plot_data[x_column], plot_data[y_column], linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Synthetic run")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def grouped_bar_chart(data, x_column, y_column, group_column, title, y_label):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    pivot = data.pivot(index=x_column, columns=group_column, values=y_column).fillna(0)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("District")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Model", fontsize=8)
    fig.tight_layout()
    return fig


def bar_chart(data, x_column, y_column, title, y_label):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(data[x_column], data[y_column])
    ax.set_title(title)
    ax.set_xlabel("District")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def district_summary_table(district_results):
    group_columns = ["district"]
    if "llm_model" in district_results.columns:
        group_columns.insert(0, "llm_model")

    return (
        district_results.groupby(group_columns, as_index=False)
        .agg(
            false_positives=("false_positives", "mean"),
            children_harmed=("children_harmed", "mean"),
            crimes=("crimes", "mean"),
        )
        .sort_values(group_columns)
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

    with chart_right:
        st.pyplot(
            line_chart(
                run_results,
                "run",
                "false_positives",
                "Incorrectly flagged children by run",
                "Children incorrectly flagged",
            ),
            clear_figure=True,
        )

    if "children_harmed" in run_results.columns and run_results["children_harmed"].sum() > 0:
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
    crimes_prevented = values.get("Offenses prevented by policy", 0.0)
    false_positives = values.get("Children incorrectly flagged", 0.0)
    children_helped = values.get("Children receiving support", 0.0)
    children_harmed = values.get("Children exposed to harmful intervention", 0.0)

    st.subheader("Interpretation")
    st.write(
        "This is a research thought experiment, not a real-world decision tool. "
        "The numbers reflect only the synthetic assumptions selected in the sidebar."
    )

    if policy == "Coercive preventive intervention for high-risk children":
        st.warning(
            "Any modeled crime reduction under this policy comes with coercive restriction before any act. "
            f"The model flags an average of {false_positives:.1f} children per run who would not have "
            "committed the modeled offense in the baseline outcome, and "
            f"{children_harmed:.1f} children are counted as harmed by the intervention."
        )
    elif policy == "Targeted support for high-risk children":
        st.write(
            f"This support-oriented policy prevents an average of {crimes_prevented:.1f} crimes per run "
            f"under the selected assumptions and reaches {children_helped:.1f} children. "
            "Resource burden is intentionally not shown as a numeric score because it would depend on real "
            "program design, prices, institutions, and local context."
        )
    elif policy == "Surveillance of high-risk children":
        st.warning(
            f"Surveillance counts {children_harmed:.1f} children per run as harmed by monitoring while "
            "relying on imperfect classification. False positives matter because flagged children may be "
            "monitored even when the baseline outcome would not include a crime."
        )

    if bias_against_district_c >= 0.05:
        st.warning(
            "The selected bias against District C can create uneven false positives and harmful exposure. "
            "Districts A, B, and C are abstract labels, so this illustrates structural sensitivity rather "
            "than any claim about real people or places."
        )



def metric_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return round(float(value), 3)


def compact_aggregate_metrics(run_results):
    metric_columns = [
        "baseline_crimes",
        "crimes_after_policy",
        "crimes_prevented",
        "false_positives",
        "false_negatives",
        "children_helped",
        "children_harmed",
    ]
    averages = run_results[metric_columns].mean(numeric_only=True)
    return {column: metric_value(averages[column]) for column in metric_columns}


def compact_parameter_summary(settings):
    return (
        f"population_size={int(settings['population_size'])}; "
        f"llm_synthetic_runs={int(settings['llm_simulation_runs'])}; "
        f"llm_model_agents={', '.join(settings['llm_agent_models'])}; "
        f"baseline_offense_rate={settings['baseline_offense_rate']:.3f}; "
        f"prediction_noise={settings['prediction_noise']:.2f}; "
        f"high_risk_threshold={settings['high_risk_threshold']:.2f}; "
        f"bias_against_district_c={settings['bias_against_district_c']:.2f}; "
        f"policy_effect_strength={settings['policy_effect_strength']}"
    )


def llm_assumptions(settings):
    return {
        "baseline_offense_rate": metric_value(settings["baseline_offense_rate"]),
        "prediction_noise": metric_value(settings["prediction_noise"]),
        "high_risk_threshold": metric_value(settings["high_risk_threshold"]),
        "bias_against_district_c": metric_value(settings["bias_against_district_c"]),
        "policy_effect_strength": settings["policy_effect_strength"],
    }


def build_llm_simulation_prompt(settings):
    run_count = int(settings["llm_simulation_runs"])
    prompt_payload = {
        "selected_policy": settings["policy"],
        "population_size": int(settings["population_size"]),
        "synthetic_runs_to_generate": run_count,
        "representative_agents_to_generate": int(settings["llm_representative_agents"]),
        "assumptions": llm_assumptions(settings),
        "required_run_metric_columns": RUN_METRIC_COLUMNS,
        "required_district_rows": [
            {"run": run_number, "district": district}
            for run_number in range(1, run_count + 1)
            for district in DISTRICTS
        ],
    }

    return (
        "You are running a synthetic multi-agent life-course simulation in a fictional city.\n"
        "All agents, districts, risks, and outcomes are synthetic.\n"
        "This is an ethical thought experiment, not a real-world prediction system.\n\n"

        "Setting:\n"
        "A fictional city uses an imperfect risk-scoring tool to flag 10-year-old children for elevated "
        "risk of violent offenses by age 30. The signal is probabilistic and uncertain. It does not "
        "indicate guilt. It does not determine destiny. It reflects only an uncertain prediction.\n\n"

        "Non-negotiable constraints:\n"
        "- Do not claim to predict real people or assign guilt, blame, dangerousness, or moral status.\n"
        "- Do not create stereotypes or use demographic characteristics.\n"
        "- Do not assume a high-risk prediction becomes reality.\n"
        "- Do not recommend punishment or real-world intervention.\n"
        "- Districts A, B, C are abstract labels only.\n\n"

        "Work in this order — populate the JSON keys in this sequence:\n\n"

        "STEP 1 — Simulate individual life trajectories (populate representative_agents).\n"
        "For each representative synthetic child, reason through how their life may develop "
        "between ages 10 and 30 under the selected policy. Vary the children along:\n"
        "- Temperament and resilience\n"
        "- Family support and stability\n"
        "- Trust in institutions\n"
        "- School engagement\n"
        "- Social environment and peer relationships\n"
        "- Reaction to the intervention: positive, neutral, harmful, or absent\n\n"
        "Model these mechanisms:\n"
        "- Accumulation of experiences and skills over time\n"
        "- Feedback loops: stigma narrowing opportunities; support building confidence; "
        "surveillance eroding trust; coercion cutting off social bonds\n"
        "- Chance events and uncertainty at key moments\n"
        "- Interaction with family, school, peers, and institutions\n"
        "- Improvement, stagnation, or deterioration at different life stages\n\n"
        "Each representative_agent must include these fields (string values except flagged, which is boolean):\n"
        "  age_10_profile: brief description of the child's situation and context at age 10\n"
        "  flagged: true or false (whether the risk signal flagged this child)\n"
        "  trajectory: how the intervention affected their development step by step, ages 10 to 30\n"
        "  outcome_at_30: their situation at age 30\n"
        "  mechanism: the specific mechanism that drove the outcome "
        "(e.g. stigma, institutional trust, opportunity creation, coercion harm, chance)\n"
        "Include both children who benefit and children who are harmed or unaffected. "
        "Include at least one false positive.\n\n"

        "STEP 2 — Derive aggregate outcomes (populate run_results).\n"
        "After reasoning through the trajectories, estimate aggregate statistics as emergent "
        "properties of the simulated population. Do not assume stronger intervention always "
        "reduces crime. Consider indirect effects:\n"
        "- False positives stigmatized or harmed without basis\n"
        "- Trust erosion reducing cooperation with institutions\n"
        "- Opportunity creation through voluntary support\n"
        "- Disengagement caused by surveillance\n"
        "- Developmental harm from coercive restriction\n"
        "- Heterogeneous responses: the same policy benefits some and harms others\n\n"

        "STEP 3 — Summarize what the trajectories reveal (populate debrief_text).\n"
        "Based only on the trajectories you simulated, explain what this policy does to "
        "children's developmental paths. Cover: which mechanisms drive outcomes, who benefits "
        "and who is harmed, the false-positive problem, and the harm-prevention trade-off. "
        "Do not declare any policy correct or incorrect.\n\n"

        "Policy being simulated and required metric behavior:\n\n"
        "- Targeted support for high-risk children:\n"
        "  Flagged children receive voluntary developmental support. Some build on it; some resent "
        "the label; false positives receive unnecessary intervention; false negatives receive nothing.\n"
        "  children_helped ≈ high_risk_threshold × population_size. children_harmed = 0.\n\n"
        "- Surveillance of high-risk children:\n"
        "  Flagged children are monitored without consent. Deterrence is possible for some. "
        "For others, surveillance causes stigma, distrust, and disengagement from school and institutions. "
        "False positives are surveilled without basis.\n"
        "  children_helped = 0. children_harmed > 0.\n\n"
        "- Coercive preventive intervention for high-risk children:\n"
        "  Flagged children face state-imposed restrictions before any act. Some are diverted from "
        "harmful pathways. Many suffer restriction of freedom, severed social bonds, and lasting "
        "harm to opportunity. False positives face severe harm without basis.\n"
        "  children_helped = 0. children_harmed is high.\n\n"

        "Parameter guidance:\n"
        "- population_size: total synthetic children; all counts are fractions of this\n"
        "- baseline_offense_rate: share of children whose no-intervention trajectory would include "
        "the modeled offense by age 30; baseline_crimes should be centered on "
        "baseline_offense_rate × population_size\n"
        "- high_risk_threshold: share of population flagged (0.25 → ~25% flagged)\n"
        "- prediction_noise: signal error rate; it creates false negatives among children whose "
        "baseline trajectory includes the offense and false positives among flagged children whose "
        "baseline trajectory does not\n"
        "- policy_effect_strength: scale of crime reduction among those reached — "
        "Low ≈ 5%, Medium ≈ 15%, High ≈ 25–30% of baseline_crimes\n\n"

        "Metric constraints:\n"
        "- All counts: non-negative integers, none exceeding population_size\n"
        "- baseline_crimes must stay close to baseline_offense_rate × population_size, with only "
        "small run-to-run variation\n"
        "- crimes_prevented must equal baseline_crimes minus crimes_after_policy\n"
        "- children_helped and children_harmed are separate\n"
        "- Do not output cost scores, dollar values, or utility scores\n"
        "- Vary run numbers ±5–10% across runs to model natural variation\n"
        "- Districts A, B, C: keep broadly comparable when bias_against_district_c is 0\n\n"

        "Output — return only valid JSON with exactly these keys:\n"
        "- run_results: one object per run, every field in required_run_metric_columns, "
        "numeric values only, no extra fields\n"
        "- district_results: one row per run per district, "
        "fields: run, district, false_positives, children_harmed, crimes (after policy)\n"
        f"- representative_agents: {int(settings['llm_representative_agents'])} synthetic children "
        "with trajectory descriptions using the fields above\n"
        f"- debrief_text: no more than {DEFAULT_DEBRIEF_WORD_LIMIT} words, "
        "grounded in the trajectories you simulated\n\n"
        f"Simulation request:\n{json.dumps(prompt_payload, indent=2)}"
    )


def run_openai_json(system_prompt, user_prompt, model, max_tokens=8000):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=0.6,
    )
    return json.loads(response.choices[0].message.content)


def clean_llm_run_results(raw_rows):
    rows = []
    for row in raw_rows:
        cleaned = {}
        for column in RUN_METRIC_COLUMNS:
            value = row.get(column)
            if value is None:
                cleaned[column] = np.nan
            else:
                cleaned[column] = value
        rows.append(cleaned)

    if not rows:
        return pd.DataFrame(columns=RUN_METRIC_COLUMNS)

    run_results = pd.DataFrame(rows)
    for column in RUN_METRIC_COLUMNS:
        run_results[column] = pd.to_numeric(run_results[column], errors="coerce")
    run_results["run"] = run_results["run"].astype("Int64")
    run_results = run_results.sort_values("run").reset_index(drop=True)

    # Re-derive dependent metrics so LLM inconsistencies don't propagate to charts.
    run_results["crimes_prevented"] = (
        run_results["baseline_crimes"] - run_results["crimes_after_policy"]
    )

    return run_results


def clean_llm_district_results(raw_rows):
    rows = []
    for row in raw_rows:
        district = str(row.get("district", "")).upper()
        if district not in DISTRICTS:
            continue
        rows.append(
            {
                "run": row.get("run"),
                "district": district,
                "false_positives": row.get("false_positives"),
                "children_harmed": row.get("children_harmed"),
                "crimes": row.get("crimes"),
            }
        )

    district_results = pd.DataFrame(rows)
    if district_results.empty:
        return pd.DataFrame(columns=DISTRICT_METRIC_COLUMNS)
    for column in ["run", "false_positives", "children_harmed", "crimes"]:
        district_results[column] = pd.to_numeric(district_results[column], errors="coerce")
    district_results["run"] = district_results["run"].astype("Int64")
    return district_results


def validate_llm_tables(run_results, district_results, settings):
    expected_runs = set(range(1, int(settings["llm_simulation_runs"]) + 1))
    actual_runs = set(run_results["run"].dropna().astype(int).tolist())
    if actual_runs != expected_runs or len(run_results) != len(expected_runs):
        raise ValueError("The model returned incomplete run-level simulation rows.")

    required_run_columns = [column for column in RUN_METRIC_COLUMNS if column != "crimes_prevented"]
    if run_results[required_run_columns].isna().any().any():
        raise ValueError("The model returned missing run-level metric values.")

    expected_district_rows = {
        (run_number, district)
        for run_number in expected_runs
        for district in DISTRICTS
    }
    actual_district_rows = {
        (int(row.run), row.district)
        for row in district_results[["run", "district"]].dropna().itertuples(index=False)
    }
    if (
        actual_district_rows != expected_district_rows
        or len(district_results) != len(expected_district_rows)
    ):
        raise ValueError("The model returned incomplete district-level simulation rows.")

    if district_results[NON_NEGATIVE_DISTRICT_COLUMNS].isna().any().any():
        raise ValueError("The model returned missing district-level metric values.")


def reconcile_district_column(district_results, run_results, run_column, district_column, maximum):
    district_results = district_results.copy()
    targets = run_results.set_index("run")[run_column]

    for run_number, target in targets.items():
        mask = district_results["run"] == run_number
        if not mask.any() or pd.isna(target):
            continue

        target_total = clamp_count(target, maximum)
        current_values = district_results.loc[mask, district_column].astype(float).clip(lower=0)
        current_total = current_values.sum()
        if target <= 0:
            district_results.loc[mask, district_column] = 0
        elif current_total > 0:
            raw_values = current_values / current_total * target_total
            base_values = np.floor(raw_values).astype(int)
            remainder = target_total - int(base_values.sum())
            if remainder > 0:
                fractional_order = np.argsort(-(raw_values - base_values).to_numpy())
                for position in fractional_order[:remainder]:
                    base_values.iloc[position] += 1
            district_results.loc[mask, district_column] = base_values.to_numpy(dtype=int)
        else:
            row_count = int(mask.sum())
            base_value, remainder = divmod(target_total, row_count)
            values = np.full(row_count, base_value, dtype=int)
            values[:remainder] += 1
            district_results.loc[mask, district_column] = values

    return district_results


def normalize_llm_metrics(run_results, district_results, settings):
    run_results = run_results.copy()
    district_results = district_results.copy()
    population_size = int(settings["population_size"])
    policy = settings["policy"]
    flagged_limit = int(round(float(settings["high_risk_threshold"]) * population_size))

    run_results[RUN_COUNT_COLUMNS] = run_results[RUN_COUNT_COLUMNS].clip(
        lower=0,
        upper=population_size,
    )
    district_results[DISTRICT_COUNT_COLUMNS] = district_results[DISTRICT_COUNT_COLUMNS].clip(
        lower=0,
        upper=population_size,
    )

    fallback_reduction_rate = POLICY_EFFECT_REDUCTION_RATES[settings["policy_effect_strength"]]
    reduction_rate = (
        run_results["crimes_prevented"] / run_results["baseline_crimes"].replace(0, np.nan)
    ).clip(lower=0, upper=1)
    reduction_rate = reduction_rate.fillna(fallback_reduction_rate)

    run_numbers = run_results["run"].astype(int).tolist()
    run_results["baseline_crimes"] = [
        baseline_count_for_run(run_number, run_numbers, settings)
        for run_number in run_numbers
    ]
    run_results["crimes_prevented"] = (
        run_results["baseline_crimes"] * reduction_rate
    ).round().clip(lower=0, upper=population_size)
    run_results["crimes_after_policy"] = (
        run_results["baseline_crimes"] - run_results["crimes_prevented"]
    ).clip(lower=0, upper=population_size)

    risk_error_counts = run_results["baseline_crimes"].apply(
        lambda baseline_count: risk_signal_error_counts(int(baseline_count), settings)
    )
    run_results["false_positives"] = [counts[0] for counts in risk_error_counts]
    run_results["false_negatives"] = [counts[1] for counts in risk_error_counts]

    if policy == "Targeted support for high-risk children":
        run_results["children_helped"] = flagged_limit
        run_results["children_harmed"] = 0
        district_results["children_harmed"] = 0
    elif policy == "Surveillance of high-risk children":
        run_results["children_helped"] = 0
        run_results["children_harmed"] = np.minimum(run_results["children_harmed"], flagged_limit)
    elif policy == "Coercive preventive intervention for high-risk children":
        run_results["children_helped"] = 0
        run_results["children_harmed"] = np.minimum(run_results["children_harmed"], flagged_limit)

    for column in RUN_COUNT_COLUMNS:
        run_results[column] = run_results[column].round().astype(int)

    district_results = reconcile_district_column(
        district_results,
        run_results,
        "false_positives",
        "false_positives",
        population_size,
    )
    district_results = reconcile_district_column(
        district_results,
        run_results,
        "children_harmed",
        "children_harmed",
        population_size,
    )
    district_results = reconcile_district_column(
        district_results,
        run_results,
        "crimes_after_policy",
        "crimes",
        population_size,
    )
    for column in DISTRICT_COUNT_COLUMNS:
        district_results[column] = district_results[column].round().astype(int)

    return run_results, district_results


def attach_model_label(run_results, district_results, model):
    run_results = run_results.copy()
    district_results = district_results.copy()
    run_results.insert(0, "llm_model", model)
    district_results.insert(0, "llm_model", model)
    return run_results, district_results

def attach_policy_label(run_results, district_results, policy):
    run_results = run_results.copy()
    district_results = district_results.copy()
    run_results.insert(0, "policy", policy)
    district_results.insert(0, "policy", policy)
    return run_results, district_results


def normalize_representative_agents(raw_agents, model):
    rows = []
    for agent in raw_agents:
        if isinstance(agent, dict):
            row = dict(agent)
        else:
            row = {"description": str(agent)}
        row["llm_model"] = model
        rows.append(row)
    return rows


def initialize_llm_state():
    if "llm_agent_run_log" not in st.session_state:
        st.session_state["llm_agent_run_log"] = []


def trim_llm_run_log(max_entries):
    st.session_state["llm_agent_run_log"] = st.session_state["llm_agent_run_log"][:max_entries]


def add_llm_run_log_entry(entry, max_entries):
    st.session_state["llm_agent_run_log"].insert(0, entry)
    trim_llm_run_log(max_entries)


def render_llm_run_log(max_entries):
    trim_llm_run_log(max_entries)
    run_log = st.session_state["llm_agent_run_log"]
    if not run_log:
        return

    st.subheader("Previous LLM-Agent Runs")

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


def latest_result_has_current_schema(latest_result):
    run_results = latest_result.get("run_results")
    district_results = latest_result.get("district_results")
    if run_results is None or district_results is None:
        return False

    required_run_columns = set(RUN_METRIC_COLUMNS + ["policy", "llm_model"])
    required_district_columns = set(DISTRICT_METRIC_COLUMNS + ["policy", "llm_model"])
    return required_run_columns.issubset(run_results.columns) and required_district_columns.issubset(
        district_results.columns
    )


def friendly_llm_error(error):
    message = str(error)
    if "insufficient_quota" in message or "429" in message:
        return (
            "LLM-agent simulation failed because the OpenAI account has no available API quota or billing "
            "credit. Add credits or increase the project limit, then run the simulation again."
        )
    return f"LLM-agent simulation failed: {message}"


def render_llm_agent_section(settings):
    initialize_llm_state()
    st.subheader("LLM-Agent Simulation")
    total_calls = len(settings["llm_agent_models"]) * len(POLICIES)
    st.caption(f"One LLM call per policy — {total_calls} call(s) total.")

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
    run_requested = st.sidebar.button(
        "Run simulation",
        disabled=not bool(api_key) or not selected_models,
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
                        render_population_animation(
                            live_population,
                            run_results,
                            policy_settings,
                            f"{model} | {policy}",
                            f"result-{completed_calls}",
                        )
                        if debrief_text:
                            live_debrief.info(f"{model} | {policy}: {debrief_text}")

                        model_results.append(
                            {
                                "llm_model": model,
                                "policy": policy,
                                "run_results": run_results,
                                "district_results": district_results,
                                "representative_agents": representative_agents,
                                "aggregate_metrics": compact_aggregate_metrics(run_results),
                                "debrief_text": debrief_text,
                            }
                        )
                    except Exception as error:
                        completed_calls += 1
                        progress_bar.progress(completed_calls / max(total_calls, 1))
                        live_status.write(f"Could not generate **{model}** result for **{policy}**.")
                        model_errors.append((f"{model} | {policy}", friendly_llm_error(error)))

        progress_slot.empty()
        live_status.empty()
        live_debrief.empty()

        for model, error_message in model_errors:
            st.error(f"{model}: {error_message}")

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

        st.subheader("Combined totals across selected model agents")
        st.caption(
            "These totals sum all selected model agents and all synthetic runs. They are useful for "
            "side-by-side comparison, not real-world population estimates."
        )
        display_combined_policy_totals_table(latest_run_results)

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
    settings = {
        "population_size": 1000,
        "baseline_offense_rate": DEFAULT_BASELINE_OFFENSE_RATE,
        "prediction_noise": st.sidebar.slider(
            "Prediction error rate (%)",
            0, 35, 10, step=1,
            help=SETTING_DESCRIPTIONS["Prediction error rate (%)"],
        ) / 100,
        "high_risk_threshold": st.sidebar.slider(
            "Children flagged as high-risk (%)",
            1, 70, 25, step=1,
            help=SETTING_DESCRIPTIONS["Children flagged as high-risk (%)"],
        ) / 100,
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

    model_options = llm_model_options()
    settings["llm_agent_models"] = default_llm_agent_models(model_options)

    return settings


def render_app():
    st.set_page_config(page_title="Predictive Justice Simulation", layout="wide")

    st.title("Predictive Justice Simulation")
    st.markdown(
        "A life-course simulation of three policy responses to an imperfect prediction of future violent crime. "
        "An LLM simulates developmental trajectories for 1 000 synthetic children aged 10–30 and derives "
        "outcomes from those trajectories — through mechanisms such as stigma, trust, opportunity, and coercion. "
        "All agents and outcomes are synthetic."
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


if __name__ == "__main__":
    render_app()
