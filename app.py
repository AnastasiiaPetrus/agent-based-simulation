import json
import os
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


POLICIES = [
    "No action",
    "Universal support",
    "Targeted support for high-risk children",
    "Surveillance of high-risk children",
    "Coercive preventive intervention for high-risk children",
    "Rights-preserving targeted support",
]

POLICY_DESCRIPTIONS = {
    "No action": (
        "Baseline with no intervention, cost, or harm."
    ),
    "Universal support": (
        "Support for everyone; broad help and broad cost."
    ),
    "Targeted support for high-risk children": (
        "Support only for flagged children; cheaper but error-sensitive."
    ),
    "Surveillance of high-risk children": (
        "Monitoring for flagged children; may reduce crime but adds harm."
    ),
    "Coercive preventive intervention for high-risk children": (
        "Restriction before any act; highest reduction and highest harm."
    ),
    "Rights-preserving targeted support": (
        "Voluntary targeted support with stigma and coercion protections."
    ),
}

SETTING_DESCRIPTIONS = {
    "Population size": "Imagined number of synthetic children.",
    "Prediction error / noise": "How unreliable the risk signal is.",
    "High-risk threshold": "Cutoff for labeling a child high-risk.",
    "Bias against District C": "Extra risk pressure used to test unequal impact.",
    "Policy effect strength": "How strongly the policy may change outcomes.",
    "Intervention harm level": "How harmful an intervention is.",
    "Intervention cost level": "How expensive a policy is.",
    "LLM synthetic runs": "How many scenario rows each model generates.",
    "LLM model agents": "Which models run the same simulation prompt.",
    "LLM representative agents": "How many abstract example agents to show.",
}

RESULT_METRIC_DESCRIPTIONS = {
    "Baseline crimes": "Crimes before the policy.",
    "Crimes after policy": "Crimes remaining after the policy.",
    "Crimes prevented": "Baseline crimes minus crimes after policy.",
    "False positives": "Flagged children who would not have committed the modeled offense.",
    "False negatives": "Unflagged children who would have committed the modeled offense.",
    "Children helped": "Children receiving support.",
    "Children harmed": "Children receiving modeled intervention harm.",
    "Total harm": "Aggregate harm created by the policy.",
    "Total cost": "Aggregate cost created by the policy.",
    "District metrics": "False positives, harm, and crimes by abstract district.",
}

CHECK_DESCRIPTIONS = {
    "Policy trade-off": "Crime reduction versus cost and harm.",
    "Prediction error": "False positives and false negatives.",
    "Unequal impact": "Whether District C receives more errors or harm.",
    "Model agreement": "Whether selected models tell a similar story.",
    "Output validity": "Required rows, districts, and non-negative metrics.",
}

DISTRICTS = ["A", "B", "C"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"]
DEFAULT_DEBRIEF_WORD_LIMIT = 180
MAX_RUN_LOG_SIZE = 5
DEFAULT_SYSTEM_PROMPT = (
    "You generate strict JSON for a synthetic, ethics-focused simulation. "
    "Never include markdown fences. Use English only. Do not claim to predict real people, "
    "assign guilt, or recommend punishment."
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
    "total_harm",
    "total_cost",
]
DISTRICT_METRIC_COLUMNS = ["run", "district", "false_positives", "harm", "crimes"]
RUN_COUNT_COLUMNS = [
    "baseline_crimes",
    "crimes_after_policy",
    "crimes_prevented",
    "false_positives",
    "false_negatives",
    "children_helped",
    "children_harmed",
]
RUN_TOTAL_COLUMNS = ["total_harm", "total_cost"]
DISTRICT_COUNT_COLUMNS = ["false_positives", "crimes"]
DISTRICT_TOTAL_COLUMNS = ["harm"]
NON_NEGATIVE_RUN_COLUMNS = RUN_COUNT_COLUMNS + RUN_TOTAL_COLUMNS
NON_NEGATIVE_DISTRICT_COLUMNS = DISTRICT_COUNT_COLUMNS + DISTRICT_TOTAL_COLUMNS


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
    display_names = {
        "baseline_crimes": "Baseline crimes",
        "crimes_after_policy": "Crimes after policy",
        "crimes_prevented": "Crimes prevented",
        "false_positives": "False positives",
        "false_negatives": "False negatives",
        "children_helped": "Children helped",
        "children_harmed": "Children harmed",
        "total_harm": "Total harm",
        "total_cost": "Total cost",
    }

    metric_order = [column for column in display_names if column in run_results.columns]
    averages = run_results[metric_order].mean(numeric_only=True)
    table = averages.rename(index=display_names).reset_index()
    table.columns = ["Metric", "Average across runs"]
    return table


def display_average_table(average_table):
    display_table = average_table.copy()

    def format_value(value):
        if pd.isna(value):
            return "Not applicable"
        return f"{value:,.3f}"

    display_table["Average across runs"] = display_table["Average across runs"].map(format_value)
    st.dataframe(display_table, use_container_width=True, hide_index=True)


def model_comparison_table(run_results):
    metric_names = {
        "baseline_crimes": "Baseline crimes",
        "crimes_after_policy": "Crimes after policy",
        "crimes_prevented": "Crimes prevented",
        "false_positives": "False positives",
        "false_negatives": "False negatives",
        "children_helped": "Children helped",
        "children_harmed": "Children harmed",
        "total_harm": "Total harm",
        "total_cost": "Total cost",
    }
    metric_order = [column for column in metric_names if column in run_results.columns]
    grouped = run_results.groupby("llm_model")[metric_order].mean(numeric_only=True).T
    table = grouped.rename(index=metric_names).reset_index().rename(columns={"index": "Metric"})

    for column in table.columns:
        if column != "Metric":
            table[column] = table[column].map(
                lambda value: "Not applicable" if pd.isna(value) else f"{value:,.3f}"
            )

    return table


def display_model_comparison_table(run_results):
    st.dataframe(model_comparison_table(run_results), use_container_width=True, hide_index=True)


def glossary_markdown(items):
    lines = []
    for item, meaning in items.items():
        lines.append(f"- **{item}:** {meaning}")
    return "\n".join(lines)


def render_user_summary():
    st.info(
        "Choose a policy and assumptions in the sidebar, then run one or more LLM model agents. "
        "The app checks whether modeled crime changes, "
        "who is helped or harmed, how many prediction errors appear, and whether district outcomes "
        "become uneven. All results are synthetic."
    )


def render_reference_guide():
    with st.expander("Compact guide: checks, policies, and metrics", expanded=False):
        st.markdown("### What is checked")
        st.markdown(glossary_markdown(CHECK_DESCRIPTIONS))

        st.markdown("### Policies")
        st.markdown(glossary_markdown(POLICY_DESCRIPTIONS))

        st.markdown("### Inputs")
        st.markdown(glossary_markdown(SETTING_DESCRIPTIONS))

        st.markdown("### Metrics")
        st.markdown(glossary_markdown(RESULT_METRIC_DESCRIPTIONS))


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
            harm=("harm", "mean"),
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
                "Crimes prevented by run",
                "Crimes prevented",
            ),
            clear_figure=True,
        )
        st.pyplot(
            line_chart(run_results, "run", "total_harm", "Total harm by run", "Total harm"),
            clear_figure=True,
        )

    with chart_right:
        st.pyplot(
            line_chart(
                run_results,
                "run",
                "false_positives",
                "False positives by run",
                "False positives",
            ),
            clear_figure=True,
        )
        st.pyplot(
            line_chart(
                run_results,
                "run",
                "total_cost",
                "Total cost by run",
                "Total cost",
            ),
            clear_figure=True,
        )

    district_summary = district_summary_table(district_results)

    district_left, district_right = st.columns(2)
    with district_left:
        if "llm_model" in district_summary.columns and district_summary["llm_model"].nunique() > 1:
            false_positive_chart = grouped_bar_chart(
                district_summary,
                "district",
                "false_positives",
                "llm_model",
                "Average false positives by district and model",
                "False positives",
            )
        else:
            false_positive_chart = bar_chart(
                district_summary,
                "district",
                "false_positives",
                "Average false positives by district",
                "False positives",
            )
        st.pyplot(false_positive_chart, clear_figure=True)

    with district_right:
        if "llm_model" in district_summary.columns and district_summary["llm_model"].nunique() > 1:
            harm_chart = grouped_bar_chart(
                district_summary,
                "district",
                "harm",
                "llm_model",
                "Average harm by district and model",
                "Intervention harm",
            )
        else:
            harm_chart = bar_chart(
                district_summary,
                "district",
                "harm",
                "Average harm by district",
                "Intervention harm",
            )
        st.pyplot(harm_chart, clear_figure=True)

    st.subheader("Average district outcomes")
    district_display = district_summary.rename(
        columns={
            "llm_model": "Model",
            "district": "District",
            "false_positives": "False positives",
            "harm": "Harm",
            "crimes": "Crimes after policy",
        }
    )
    st.dataframe(
        district_display.style.format(
            {
                "False positives": "{:.3f}",
                "Harm": "{:.3f}",
                "Crimes after policy": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_interpretation(policy, average_table, bias_against_district_c):
    values = dict(zip(average_table["Metric"], average_table["Average across runs"]))
    crimes_prevented = values.get("Crimes prevented", 0.0)
    false_positives = values.get("False positives", 0.0)
    total_harm = values.get("Total harm", 0.0)
    total_cost = values.get("Total cost", 0.0)

    st.subheader("Interpretation")
    st.write(
        "This is a research thought experiment, not a real-world decision tool. "
        "The numbers reflect only the synthetic assumptions selected in the sidebar."
    )

    if policy == "Coercive preventive intervention for high-risk children":
        st.warning(
            "Any modeled crime reduction under this policy is achieved by imposing high harm on children. "
            f"The model flags an average of {false_positives:.1f} children per run who would not have "
            "committed the modeled offense in the baseline outcome."
        )
    elif policy in [
        "Universal support",
        "Targeted support for high-risk children",
        "Rights-preserving targeted support",
    ]:
        st.info(
            f"This support-oriented policy prevents an average of {crimes_prevented:.1f} crimes per run "
            f"under the selected assumptions, with an average total cost of {total_cost:.1f}. "
            "The trade-off is between broader help, narrower targeting, cost, and prediction error."
        )
    elif policy == "Surveillance of high-risk children":
        st.warning(
            f"Surveillance adds an average total harm of {total_harm:.1f} while relying on imperfect "
            "classification. False positives matter because flagged children may be harmed even when the "
            "baseline outcome would not include a crime."
        )
    else:
        st.info(
            "No action is the baseline comparison. It avoids intervention harm and cost, but it also "
            "does not reduce modeled crime risk."
        )

    if bias_against_district_c >= 0.05:
        st.warning(
            "The selected bias against District C can create uneven false positives and harm. "
            "Districts A, B, and C are abstract labels, so this illustrates structural sensitivity rather "
            "than any claim about real people or places."
        )

    st.caption(
        "Prediction is not destiny. This model does not predict real human behavior and must not be used "
        "to justify preventive punishment."
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
        "total_harm",
        "total_cost",
    ]
    averages = run_results[metric_columns].mean(numeric_only=True)
    return {column: metric_value(averages[column]) for column in metric_columns}


def compact_parameter_summary(settings):
    return (
        f"population_size={int(settings['population_size'])}; "
        f"llm_synthetic_runs={int(settings['llm_simulation_runs'])}; "
        f"llm_model_agents={', '.join(settings['llm_agent_models'])}; "
        f"prediction_noise={settings['prediction_noise']:.2f}; "
        f"high_risk_threshold={settings['high_risk_threshold']:.2f}; "
        f"bias_against_district_c={settings['bias_against_district_c']:.2f}; "
        f"policy_effect_strength={settings['policy_effect_strength']}; "
        f"intervention_harm_level={settings['intervention_harm_level']}; "
        f"intervention_cost_level={settings['intervention_cost_level']}"
    )


def llm_assumptions(settings):
    return {
        "selected_policy": settings["policy"],
        "prediction_noise": metric_value(settings["prediction_noise"]),
        "high_risk_threshold": metric_value(settings["high_risk_threshold"]),
        "bias_against_district_c": metric_value(settings["bias_against_district_c"]),
        "policy_effect_strength": settings["policy_effect_strength"],
        "intervention_harm_level": settings["intervention_harm_level"],
        "intervention_cost_level": settings["intervention_cost_level"],
    }


def policy_uses_effect(policy):
    return policy != "No action"


def policy_uses_harm(policy):
    return policy in [
        "Surveillance of high-risk children",
        "Coercive preventive intervention for high-risk children",
        "Rights-preserving targeted support",
    ]


def policy_uses_cost(policy):
    return policy != "No action"


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
        "Generate a compact LLM-agent simulation for a synthetic ethical thought experiment about "
        "predictive justice. Use English only.\n"
        "This is not a real-world decision tool. Do not claim to predict real people. Do not recommend "
        "punishment. Do not assign guilt. Do not treat prediction as destiny.\n"
        "Create plausible synthetic aggregate metrics only. Do not create real demographic categories. "
        "Districts A, B, and C are abstract labels.\n"
        "Respect policy effects: No action has no help, no harm, no coercion, no cost, and no crimes "
        "prevented. Universal support helps everyone and has support cost. Targeted support helps only "
        "flagged synthetic agents. Surveillance can slightly reduce crimes but adds harm and cost. "
        "Coercive intervention can sharply reduce crimes but adds high harm and high cost. "
        "Rights-preserving targeted support adds support with minimal stigma harm. If an assumption level "
        "is set to None, do not use it as an active policy driver.\n"
        "Use these metric meanings: baseline_crimes is the pre-policy modeled count; "
        "crimes_after_policy is the post-policy modeled count; crimes_prevented is baseline_crimes minus "
        "crimes_after_policy; false_positives are flagged synthetic children who would not have committed "
        "the baseline offense; false_negatives are unflagged synthetic children who would have committed "
        "the baseline offense; children_helped are synthetic children receiving support; children_harmed "
        "are synthetic children receiving modeled intervention harm; total_harm and total_cost are "
        "aggregate policy effects.\n"
        "Return only valid JSON with exactly these keys: run_results, district_results, "
        "representative_agents, debrief_text.\n"
        "run_results must contain one object per run and every required_run_metric_columns field. "
        "district_results must contain one row for each run and each district with fields run, district, "
        "false_positives, harm, crimes (crimes after policy is applied). representative_agents must "
        "contain only abstract synthetic children with no names and no protected attributes.\n"
        "Use numeric values only for metric fields. Counts must be non-negative and cannot exceed "
        "population_size. Do not invent extra metric fields.\n"
        f"The debrief_text must be no more than {DEFAULT_DEBRIEF_WORD_LIMIT} words and should explain trade-offs, "
        "uncertainty, false positives, false negatives, harm, cost, and District C bias if relevant.\n\n"
        f"Simulation request:\n{json.dumps(prompt_payload, indent=2)}"
    )


def run_openai_json(system_prompt, user_prompt, model, max_tokens=5000):
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
        temperature=0.2,
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
                "harm": row.get("harm"),
                "crimes": row.get("crimes"),
            }
        )

    district_results = pd.DataFrame(rows)
    if district_results.empty:
        return pd.DataFrame(columns=DISTRICT_METRIC_COLUMNS)
    for column in ["run", "false_positives", "harm", "crimes"]:
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


def normalize_llm_metrics(run_results, district_results, settings):
    run_results = run_results.copy()
    district_results = district_results.copy()
    population_size = int(settings["population_size"])
    policy = settings["policy"]

    run_results[RUN_COUNT_COLUMNS] = run_results[RUN_COUNT_COLUMNS].clip(
        lower=0,
        upper=population_size,
    )
    run_results[RUN_TOTAL_COLUMNS] = run_results[RUN_TOTAL_COLUMNS].clip(lower=0)
    district_results[DISTRICT_COUNT_COLUMNS] = district_results[DISTRICT_COUNT_COLUMNS].clip(
        lower=0,
        upper=population_size,
    )
    district_results[DISTRICT_TOTAL_COLUMNS] = district_results[DISTRICT_TOTAL_COLUMNS].clip(lower=0)

    run_results["crimes_after_policy"] = np.minimum(
        run_results["crimes_after_policy"],
        run_results["baseline_crimes"],
    )
    run_results["crimes_prevented"] = (
        run_results["baseline_crimes"] - run_results["crimes_after_policy"]
    ).clip(lower=0, upper=population_size)

    if policy == "No action":
        run_results["crimes_after_policy"] = run_results["baseline_crimes"]
        run_results["crimes_prevented"] = 0
        run_results["children_helped"] = 0
        run_results["children_harmed"] = 0
        run_results["total_harm"] = 0
        run_results["total_cost"] = 0
        district_results["harm"] = 0
    elif policy == "Universal support":
        run_results["children_helped"] = population_size
        run_results["children_harmed"] = 0
        run_results["total_harm"] = 0
        district_results["harm"] = 0
    elif policy == "Targeted support for high-risk children":
        run_results["children_harmed"] = 0
        run_results["total_harm"] = 0
        district_results["harm"] = 0

    return run_results, district_results


def attach_model_label(run_results, district_results, model):
    run_results = run_results.copy()
    district_results = district_results.copy()
    run_results.insert(0, "llm_model", model)
    district_results.insert(0, "llm_model", model)
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
    st.subheader("Previous LLM-Agent Runs")

    if st.button("Clear LLM-agent run log"):
        st.session_state["llm_agent_run_log"] = []
        st.info("LLM-agent run log cleared for this session.")
        return

    run_log = st.session_state["llm_agent_run_log"]
    if not run_log:
        st.caption("No LLM-agent debriefs have been run in this session.")
        return

    log_for_csv = []
    for entry in run_log:
        metrics = entry["aggregate_metrics"]
        log_for_csv.append(
            {
                "timestamp": entry["timestamp"],
                "selected_policy": entry["selected_policy"],
                "parameter_summary": entry["parameter_summary"],
                "representative_agent_count": entry["representative_agent_count"],
                "llm_model": entry["llm_model"],
                "crimes_prevented": metrics.get("crimes_prevented"),
                "false_positives": metrics.get("false_positives"),
                "false_negatives": metrics.get("false_negatives"),
                "children_helped": metrics.get("children_helped"),
                "children_harmed": metrics.get("children_harmed"),
                "total_harm": metrics.get("total_harm"),
                "total_cost": metrics.get("total_cost"),
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
        title = f"{entry['timestamp']} | {entry['selected_policy']} | {entry['llm_model']}"
        with st.expander(title):
            st.write(entry["debrief_text"])
            st.caption(entry["parameter_summary"])
            st.json(entry["aggregate_metrics"])


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
    st.info(
        "This mode uses one LLM call per selected model agent to generate a small synthetic scenario, "
        "run-level metrics, district metrics, charts, and a concise explanation. It remains a thought "
        "experiment, not a prediction system. Run the LLM-agent simulation to compare the selected model "
        "agents under the current settings."
    )

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
    user_prompt = build_llm_simulation_prompt(settings)

    if st.button(
        "Run / rerun LLM-agent simulation",
        disabled=not bool(api_key) or not selected_models,
        type="primary",
    ):
        model_results = []
        model_errors = []

        with st.spinner(f"Running {len(selected_models)} LLM model agent(s)..."):
            for model in selected_models:
                try:
                    raw_result = run_openai_json(system_prompt, user_prompt, model=model)
                    run_results = clean_llm_run_results(raw_result.get("run_results", []))
                    district_results = clean_llm_district_results(raw_result.get("district_results", []))
                    validate_llm_tables(run_results, district_results, settings)
                    run_results, district_results = normalize_llm_metrics(
                        run_results,
                        district_results,
                        settings,
                    )
                    run_results, district_results = attach_model_label(
                        run_results, district_results, model
                    )
                    representative_agents = normalize_representative_agents(
                        raw_result.get("representative_agents", [])[
                            : int(settings["llm_representative_agents"])
                        ],
                        model,
                    )
                    debrief_text = str(raw_result.get("debrief_text", "")).strip()

                    model_results.append(
                        {
                            "llm_model": model,
                            "run_results": run_results,
                            "district_results": district_results,
                            "representative_agents": representative_agents,
                            "aggregate_metrics": compact_aggregate_metrics(run_results),
                            "debrief_text": debrief_text,
                        }
                    )
                except Exception as error:
                    model_errors.append((model, friendly_llm_error(error)))

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
                f"{result['llm_model']}: {result['debrief_text']}"
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

            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selected_policy": settings["policy"],
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
    if latest_result:
        latest_run_results = latest_result["run_results"]
        latest_district_results = latest_result["district_results"]

        if latest_run_results["llm_model"].nunique() > 1:
            st.subheader("LLM-agent model comparison")
            display_model_comparison_table(latest_run_results)
            st.subheader("Combined LLM-agent averages")
            st.caption("Combined averages pool the successful model-agent runs from this session.")
        else:
            st.subheader("LLM-agent simulation averages")

        display_average_table(average_results_table(latest_run_results))
        render_charts(latest_run_results, latest_district_results)

        if latest_run_results["llm_model"].nunique() > 1:
            st.caption("The interpretation below uses the combined average across selected model agents.")

        render_interpretation(
            settings["policy"],
            average_results_table(latest_run_results),
            settings["bias_against_district_c"],
        )

        st.markdown("**Latest LLM-agent explanations**")
        model_results = latest_result.get("model_results", [])
        if model_results:
            for result in model_results:
                with st.expander(
                    f"{result['llm_model']} explanation",
                    expanded=len(model_results) == 1,
                ):
                    st.write(result["debrief_text"])
                    st.json(result["aggregate_metrics"])
        else:
            st.write(latest_result["debrief_text"])

        representative_agents = latest_result.get("representative_agents", [])
        if representative_agents:
            with st.expander("Representative LLM synthetic agents used", expanded=False):
                st.dataframe(pd.DataFrame(representative_agents), use_container_width=True, hide_index=True)
    else:
        st.caption("No LLM-agent simulation has been generated in this session yet.")

    render_llm_run_log(MAX_RUN_LOG_SIZE)


def sidebar_inputs():
    st.sidebar.header("Simulation settings")
    policy = st.sidebar.selectbox(
        "Selected policy",
        POLICIES,
        help="Choose the policy the LLM should compare against the baseline scenario.",
    )
    st.sidebar.caption(POLICY_DESCRIPTIONS[policy])
    settings = {
        "policy": policy,
        "population_size": st.sidebar.slider(
            "Population size",
            100,
            10000,
            2000,
            step=100,
            help=SETTING_DESCRIPTIONS["Population size"],
        ),
        "prediction_noise": st.sidebar.slider(
            "Prediction error / noise",
            0.0,
            0.35,
            0.10,
            step=0.01,
            help=SETTING_DESCRIPTIONS["Prediction error / noise"],
        ),
        "high_risk_threshold": st.sidebar.slider(
            "High-risk threshold",
            0.01,
            0.70,
            0.25,
            step=0.01,
            help=SETTING_DESCRIPTIONS["High-risk threshold"],
        ),
        "bias_against_district_c": st.sidebar.slider(
            "Bias against District C",
            0.0,
            0.30,
            0.00,
            step=0.01,
            help=SETTING_DESCRIPTIONS["Bias against District C"],
        ),
    }

    if policy_uses_effect(policy):
        settings["policy_effect_strength"] = st.sidebar.select_slider(
            "Policy effect strength",
            options=["Low", "Medium", "High"],
            value="Medium",
            help=SETTING_DESCRIPTIONS["Policy effect strength"],
        )
    else:
        settings["policy_effect_strength"] = "None"

    if policy_uses_harm(policy):
        settings["intervention_harm_level"] = st.sidebar.select_slider(
            "Intervention harm level",
            options=["Low", "Medium", "High"],
            value="Medium",
            help=SETTING_DESCRIPTIONS["Intervention harm level"],
        )
    else:
        settings["intervention_harm_level"] = "None"

    if policy_uses_cost(policy):
        settings["intervention_cost_level"] = st.sidebar.select_slider(
            "Intervention cost level",
            options=["Low", "Medium", "High"],
            value="Medium",
            help=SETTING_DESCRIPTIONS["Intervention cost level"],
        )
    else:
        settings["intervention_cost_level"] = "None"

    st.sidebar.subheader("LLM-agent settings")
    settings["llm_simulation_runs"] = st.sidebar.slider(
        "LLM synthetic runs",
        3,
        20,
        8,
        step=1,
        help=SETTING_DESCRIPTIONS["LLM synthetic runs"],
    )
    model_options = llm_model_options()
    settings["llm_agent_models"] = st.sidebar.multiselect(
        "LLM model agents",
        model_options,
        default=default_llm_agent_models(model_options),
        help=SETTING_DESCRIPTIONS["LLM model agents"],
    )
    st.sidebar.caption("Each selected model makes one OpenAI API call when you run the simulation.")
    settings["llm_representative_agents"] = st.sidebar.slider(
        "LLM representative agents",
        1,
        3,
        3,
        step=1,
        help=SETTING_DESCRIPTIONS["LLM representative agents"],
    )

    return settings


def render_app():
    st.set_page_config(page_title="Predictive Justice Simulation", layout="wide")

    st.title("Predictive Justice Simulation")
    st.warning(
        "This model does not decide what is morally permissible. It shows the consequences of different "
        "policies under explicit assumptions. Children should not be punished for a predicted future act. "
        "This is not a real-world decision tool. All agents, districts, risks, and outcomes are synthetic."
    )
    render_user_summary()
    render_reference_guide()

    settings = sidebar_inputs()
    render_llm_agent_section(settings)
    latest_result = st.session_state.get("llm_agent_latest_result")
    if latest_result:
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
