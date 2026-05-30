import json
import os
from datetime import datetime
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
        "Restriction before any act; highest harm and cost."
    ),
}

SETTING_DESCRIPTIONS = {
    "Population size": "Imagined number of synthetic children.",
    "Prediction error / noise": "How unreliable the risk signal is.",
    "High-risk threshold": "Cutoff for labeling a child high-risk.",
    "Bias against District C": "Extra risk pressure used to test unequal impact.",
    "Policy effect strength": "Scenario assumption for how strongly a policy may change modeled crime.",
    "LLM synthetic runs": "How many scenario rows each model generates.",
    "LLM model agents": "Which models run the same simulation prompt.",
    "LLM representative agents": "How many abstract example agents to show.",
}

RESULT_METRIC_DESCRIPTIONS = {
    "Modeled crimes before policy": "Synthetic baseline crimes before a policy is applied.",
    "Modeled crimes after policy": "Synthetic crimes remaining after the policy.",
    "Modeled crimes prevented": "Before-policy crimes minus after-policy crimes.",
    "Modeled crime reduction (%)": "Prevented crimes as a percentage of before-policy crimes.",
    "Children incorrectly flagged": "Flagged children who would not have committed the modeled offense.",
    "Children missed by risk signal": "Unflagged children who would have committed the modeled offense.",
    "Children receiving support": "Children receiving voluntary support.",
    "Children exposed to harmful intervention": (
        "Children exposed to surveillance, coercion, or residual stigma in the scenario."
    ),
    "District metrics": "Incorrect flags, harmful exposure, and modeled crimes by abstract district.",
}

CHECK_DESCRIPTIONS = {
    "Policy trade-off": "Crime reduction, support reach, and children exposed to harmful intervention.",
    "Prediction error": "False positives and false negatives.",
    "Unequal impact": "Whether District C receives more errors or harmful exposure.",
    "Model agreement": "Whether selected models tell a similar story.",
    "Output validity": "Required rows, districts, and non-negative metrics.",
}

DISTRICTS = ["A", "B", "C"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"]
DEFAULT_DEBRIEF_WORD_LIMIT = 180
MAX_RUN_LOG_SIZE = 5
DEFAULT_SYSTEM_PROMPT = (
    "You generate strict JSON for a synthetic, ethics-focused policy simulation. "
    "Use English only. Never include markdown fences. Never claim to predict real people, "
    "assign guilt, rank children morally, or recommend punishment. Treat all outputs as aggregate "
    "thought-experiment data under explicit assumptions."
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

RUN_METRIC_LABELS = {
    "baseline_crimes": "Modeled crimes before policy",
    "crimes_after_policy": "Modeled crimes after policy",
    "crimes_prevented": "Modeled crimes prevented",
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
    metric_columns = [column for column in RUN_METRIC_LABELS if column in run_results.columns]
    total_metric_labels = {
        "baseline_crimes": "Total modeled crimes before policy",
        "crimes_after_policy": "Total modeled crimes after policy",
        "crimes_prevented": "Total modeled crimes prevented",
        "false_positives": "Total children incorrectly flagged",
        "false_negatives": "Total children missed by risk signal",
        "children_helped": "Total children receiving support",
        "children_harmed": "Total children exposed to harmful intervention",
    }
    table = (
        run_results.groupby("policy", as_index=False)
        .agg(
            synthetic_rows=("run", "count"),
            model_agents=("llm_model", "nunique"),
            **{column: (column, "sum") for column in metric_columns},
        )
    )
    table["policy_sort"] = table["policy"].map({policy: index for index, policy in enumerate(POLICY_ORDER)})
    table = table.sort_values("policy_sort").drop(columns="policy_sort")
    baseline = table["baseline_crimes"].replace(0, np.nan)
    table["crime_reduction_percent"] = (table["crimes_prevented"] / baseline) * 100

    display_table = table.rename(
        columns={
            "policy": "Policy",
            "synthetic_rows": "Synthetic model-runs included",
            "model_agents": "Model agents included",
            **total_metric_labels,
            "crime_reduction_percent": "Modeled crime reduction (%)",
        }
    )
    ordered_columns = [
        "Policy",
        "Model agents included",
        "Synthetic model-runs included",
        "Total modeled crimes before policy",
        "Total modeled crimes after policy",
        "Total modeled crimes prevented",
        "Modeled crime reduction (%)",
        "Total children incorrectly flagged",
        "Total children missed by risk signal",
        "Total children receiving support",
        "Total children exposed to harmful intervention",
    ]
    display_table = display_table[[column for column in ordered_columns if column in display_table.columns]]

    count_columns = [
        "Total modeled crimes before policy",
        "Total modeled crimes after policy",
        "Total modeled crimes prevented",
        "Total children incorrectly flagged",
        "Total children missed by risk signal",
        "Total children receiving support",
        "Total children exposed to harmful intervention",
    ]
    for column in count_columns:
        if column in display_table.columns:
            display_table[column] = display_table[column].map(
                lambda value: "Not applicable" if pd.isna(value) else f"{value:,.0f}"
            )
    if "Modeled crime reduction (%)" in display_table.columns:
        display_table["Modeled crime reduction (%)"] = display_table[
            "Modeled crime reduction (%)"
        ].map(lambda value: "Not applicable" if pd.isna(value) else f"{value:,.1f}%")

    return display_table


def display_combined_policy_totals_table(run_results):
    st.dataframe(combined_policy_totals_table(run_results), use_container_width=True, hide_index=True)


def glossary_markdown(items):
    lines = []
    for item, meaning in items.items():
        lines.append(f"- **{item}:** {meaning}")
    return "\n".join(lines)


def render_user_summary():
    st.info(
        "Choose assumptions in the sidebar, then run one or more LLM model agents across all policies. "
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
                "Modeled crimes prevented by run",
                "Modeled crimes prevented",
            ),
            clear_figure=True,
        )
        st.pyplot(
            line_chart(
                run_results,
                "run",
                "children_helped",
                "Children receiving support by run",
                "Children receiving support",
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

    district_summary = district_summary_table(district_results)

    district_left, district_right = st.columns(2)
    with district_left:
        if "llm_model" in district_summary.columns and district_summary["llm_model"].nunique() > 1:
            false_positive_chart = grouped_bar_chart(
                district_summary,
                "district",
                "false_positives",
                "llm_model",
                "Average incorrectly flagged children by district and model",
                "Children incorrectly flagged",
            )
        else:
            false_positive_chart = bar_chart(
                district_summary,
                "district",
                "false_positives",
                "Average incorrectly flagged children by district",
                "Children incorrectly flagged",
            )
        st.pyplot(false_positive_chart, clear_figure=True)

    with district_right:
        if "llm_model" in district_summary.columns and district_summary["llm_model"].nunique() > 1:
            harm_chart = grouped_bar_chart(
                district_summary,
                "district",
                "children_harmed",
                "llm_model",
                "Average children exposed by district and model",
                "Children exposed",
            )
        else:
            harm_chart = bar_chart(
                district_summary,
                "district",
                "children_harmed",
                "Average children exposed by district",
                "Children exposed",
            )
        st.pyplot(harm_chart, clear_figure=True)

    st.subheader("Average district outcomes")
    district_display = district_summary.rename(
        columns={
            "llm_model": "Model",
            "district": "District",
            "false_positives": "Children incorrectly flagged",
            "children_harmed": "Children exposed to harmful intervention",
            "crimes": "Modeled crimes after policy",
        }
    )
    st.dataframe(
        district_display.style.format(
            {
                "Children incorrectly flagged": "{:.3f}",
                "Children exposed to harmful intervention": "{:.3f}",
                "Modeled crimes after policy": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_interpretation(policy, average_table, bias_against_district_c):
    values = dict(zip(average_table["Metric"], average_table["Average per synthetic run"]))
    crimes_prevented = values.get("Modeled crimes prevented", 0.0)
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
        st.info(
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
        f"policy_effect_strength={settings['policy_effect_strength']}"
    )


def llm_assumptions(settings):
    return {
        "prediction_noise": metric_value(settings["prediction_noise"]),
        "high_risk_threshold": metric_value(settings["high_risk_threshold"]),
        "bias_against_district_c": metric_value(settings["bias_against_district_c"]),
        "policy_effect_strength": settings["policy_effect_strength"],
    }


def policy_uses_effect(policy):
    return True


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
        "Generate a synthetic-agent simulation for an ethical thought experiment about predictive justice. "
        "Your task is to produce plausible aggregate numbers that illustrate consequences under the stated "
        "assumptions. Do not evaluate which policy is better or worse.\n\n"
        "Thought experiment premise:\n"
        "A society possesses a risk-scoring tool that predicts, with stated accuracy, which children at "
        "age 10 are likely to commit a violent offense by age 30. The tool is imperfect: it produces both "
        "false positives (children flagged who would not have committed the offense) and false negatives "
        "(children missed who would have). The simulation asks: what happens if the society acts on that "
        "prediction under each policy choice? The prediction is a synthetic signal only. It does not "
        "describe real children, real communities, or real predictability.\n\n"
        "Non-negotiable framing:\n"
        "- This is not a real-world decision tool.\n"
        "- Do not claim to predict real people or assign guilt, blame, dangerousness, or moral status.\n"
        "- Do not recommend punishment, incapacitation, or real-world intervention.\n"
        "- Districts A, B, and C are abstract labels; they are not demographic groups, neighborhoods, "
        "protected classes, or proxies for real places.\n\n"
        "Policy definitions and required metric behavior:\n"
        "- Targeted support for high-risk children: Only flagged synthetic children receive voluntary "
        "support. children_helped is approximately high_risk_threshold × population_size. "
        "children_harmed is 0. Crime reduction depends on flagging accuracy: false positives receive "
        "unnecessary support, false negatives receive none. Do not output a numeric cost score.\n\n"
        "- Surveillance of high-risk children: Flagged synthetic children are monitored without consent. "
        "This is monitoring, not support. children_helped is 0. children_harmed is greater than 0 because "
        "surveillance causes stigma and privacy harm. Crime reduction is modest: surveillance may deter "
        "or detect but does not address underlying causes. False positives are flagged children subjected "
        "to surveillance without basis. Do not output a numeric harm score.\n\n"
        "- Coercive preventive intervention for high-risk children: Flagged synthetic children face "
        "state-imposed restrictions before committing any act (e.g., mandatory programs, movement limits, "
        "or institutional placement). This is preventive restriction of liberty. children_helped is 0. "
        "children_harmed is high because each coerced child suffers restriction of freedom. Crime reduction "
        "may be the strongest among the three policies. False positives are children coerced without basis. "
        "Do not output a numeric harm or cost score.\n\n"
        "How to translate parameters into numbers:\n"
        "- population_size: total synthetic children; all counts are fractions of this.\n"
        "- high_risk_threshold (0.01–0.70): share of population flagged. At 0.25, roughly 25% are flagged "
        "(true positives + false positives combined).\n"
        "- prediction_noise (0.0–0.35): controls error rate. At 0.0, the signal is accurate; at 0.35, "
        "false positives and false negatives are substantially elevated relative to the flagged group.\n"
        "- policy_effect_strength (Low/Medium/High/None): scales crimes_prevented among those reached. "
        "Approximate share of baseline_crimes prevented — Low ≈ 5%, Medium ≈ 15%, High ≈ 25–30%. "
        "None means 0 crimes prevented.\n"
        "- bias_against_district_c (0.0–0.30): inflates false_positives and children_harmed in District C by "
        "approximately this fraction above the baseline rate. Districts A and B remain comparable.\n\n"
        "Metric constraints:\n"
        "- All counts must be non-negative integers; no count may exceed population_size.\n"
        "- crimes_prevented must equal baseline_crimes minus crimes_after_policy.\n"
        "- children_helped and children_harmed are separate; a child cannot be both in the same scenario.\n"
        "- Do not include total_harm, total_cost, dollar values, utility scores, welfare scores, or any other "
        "aggregate harm/cost scale. These are deliberately omitted because they would be falsely precise.\n"
        "- Across runs, vary numbers by small random amounts (roughly ±5–10%) to model natural variation. "
        "Do not change the qualitative pattern run-to-run.\n\n"
        "Fairness:\n"
        "- Do not invent demographic explanations for District C differences. Unequal outcomes must reflect "
        "only the bias_against_district_c parameter.\n"
        "- If bias_against_district_c is 0, all three districts must be broadly comparable.\n\n"
        "Output requirements:\n"
        "- Return only valid JSON with exactly these keys: run_results, district_results, "
        "representative_agents, debrief_text.\n"
        "- run_results: one object per run, every field in required_run_metric_columns, numeric values only, "
        "no extra fields.\n"
        "- district_results: one row per run per district with fields run, district, false_positives, "
        "children_harmed, crimes (crimes = crimes after policy is applied in that district).\n"
        "- representative_agents: abstract synthetic children only; no names, no protected attributes, "
        "no diagnoses, no family details, no real-world identifiers.\n"
        f"- debrief_text: no more than {DEFAULT_DEBRIEF_WORD_LIMIT} words covering exactly: "
        "(1) average crimes prevented, (2) false positives and what they mean for the flagged children, "
        "(3) children helped and children harmed, (4) the trade-off between crime reduction and exposing "
        "children to support, surveillance, or coercion, and (5) District C differences if "
        "bias_against_district_c is greater than 0. "
        "Do not declare any policy morally correct or incorrect.\n\n"
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


def normalize_llm_metrics(run_results, district_results, settings):
    run_results = run_results.copy()
    district_results = district_results.copy()
    population_size = int(settings["population_size"])
    policy = settings["policy"]

    run_results[RUN_COUNT_COLUMNS] = run_results[RUN_COUNT_COLUMNS].clip(
        lower=0,
        upper=population_size,
    )
    district_results[DISTRICT_COUNT_COLUMNS] = district_results[DISTRICT_COUNT_COLUMNS].clip(
        lower=0,
        upper=population_size,
    )

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
        district_results["children_harmed"] = 0
    elif policy == "Universal support":
        run_results["children_helped"] = population_size
        run_results["children_harmed"] = 0
        district_results["children_harmed"] = 0
    elif policy == "Targeted support for high-risk children":
        run_results["children_harmed"] = 0
        district_results["children_harmed"] = 0

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
    st.info(
        "This mode runs one LLM call per selected model agent and per policy to generate run-level metrics, "
        "district metrics, charts, and explanations. It remains a thought experiment, not a prediction system."
        f" This run will make {total_calls} call(s)."
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

    if st.button(
        "Run / rerun LLM-agent simulation",
        disabled=not bool(api_key) or not selected_models,
        type="primary",
    ):
        model_results = []
        model_errors = []

        with st.spinner(f"Running {len(POLICIES)} policy(ies) × {len(selected_models)} model agent(s)..."):
            for model in selected_models:
                for policy in POLICY_ORDER:
                    policy_settings = dict(settings)
                    policy_settings["policy"] = policy
                    if not policy_uses_effect(policy):
                        policy_settings["policy_effect_strength"] = "None"

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
                        model_errors.append((f"{model} | {policy}", friendly_llm_error(error)))

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
        st.info("Previous in-session results used an older metric schema. Run the simulation again.")

    if latest_result:
        latest_run_results = latest_result["run_results"]
        latest_district_results = latest_result["district_results"]

        st.subheader("Combined totals across selected model agents")
        st.caption(
            "These totals sum all selected model agents and all synthetic runs. They are useful for "
            "side-by-side comparison, not real-world population estimates."
        )
        display_combined_policy_totals_table(latest_run_results)

        st.subheader("Policy comparison (average per model agent)")
        display_policy_model_comparison_table(latest_run_results)

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

                # Interpretation uses combined averages across selected models for this policy.
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
    else:
        st.caption("No LLM-agent simulation has been generated in this session yet.")

    render_llm_run_log(MAX_RUN_LOG_SIZE)


def sidebar_inputs():
    st.sidebar.header("Simulation settings")
    settings = {
        "population_size": 1000,
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
        "bias_against_district_c": 0.0,
        "policy_effect_strength": st.sidebar.select_slider(
            "Policy effect strength",
            options=["Low", "Medium", "High"],
            value="Medium",
            help=SETTING_DESCRIPTIONS["Policy effect strength"],
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
