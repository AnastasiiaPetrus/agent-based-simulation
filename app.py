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
        "Baseline comparison. No support, surveillance, coercion, cost, or intervention harm is applied; "
        "modeled baseline crimes remain unchanged."
    ),
    "Universal support": (
        "Every synthetic child receives non-punitive support. This can reduce risk without targeting, but it "
        "creates broad support cost."
    ),
    "Targeted support for high-risk children": (
        "Only synthetic children flagged as high-risk receive support. It costs less than universal support, "
        "but depends on an imperfect risk signal."
    ),
    "Surveillance of high-risk children": (
        "Flagged synthetic children are monitored. It may reduce some modeled crimes, but it adds privacy, "
        "stigma, and error-related harm."
    ),
    "Coercive preventive intervention for high-risk children": (
        "Flagged synthetic children face a restrictive intervention before any real act. It can reduce modeled "
        "crime most strongly, but imposes the highest harm and is ethically dangerous."
    ),
    "Rights-preserving targeted support": (
        "Flagged synthetic children receive voluntary, non-punitive support with extra protections against "
        "stigma and coercion."
    ),
}

SETTING_DESCRIPTIONS = {
    "Population size": "How many synthetic children the LLM should use as the imagined population.",
    "Prediction error / noise": "How unreliable the synthetic risk signal is. Higher values mean more mistakes.",
    "High-risk threshold": "The cutoff for labeling a synthetic child as high-risk. Lower values flag more children.",
    "Bias against District C": "Extra synthetic risk pressure applied to District C to test uneven false positives and harm.",
    "Policy effect strength": "How strongly the selected policy is allowed to change modeled outcomes.",
    "Intervention harm level": "How much harm the selected intervention adds to affected synthetic children.",
    "Intervention cost level": "How expensive the selected policy is in the synthetic scenario.",
    "LLM synthetic runs": "How many scenario runs the LLM should generate for charts and averages.",
    "LLM representative agents": "How many abstract synthetic example agents the LLM may include.",
    "LLM output word limit": "Maximum length of the written LLM explanation.",
    "LLM run log size": "How many previous LLM runs are kept in this browser session.",
}

RESULT_METRIC_DESCRIPTIONS = {
    "Baseline crimes": "Modeled crimes before any policy is applied.",
    "Crimes after policy": "Modeled crimes remaining after the selected policy is applied.",
    "Crimes prevented": "Baseline crimes minus crimes after policy.",
    "High-risk flagged": "Synthetic children labeled high-risk by the risk signal.",
    "True positives": "Flagged synthetic children who would have committed the modeled offense in the baseline.",
    "False positives": "Flagged synthetic children who would not have committed the modeled offense in the baseline.",
    "False negatives": "Unflagged synthetic children who would have committed the modeled offense in the baseline.",
    "Precision": "Share of flagged children who are true positives.",
    "Recall": "Share of baseline crimes captured by the high-risk flag.",
    "Children helped": "Synthetic children receiving support.",
    "Children harmed": "Synthetic children receiving modeled intervention harm.",
    "Coerced children": "Synthetic children affected by coercive restriction.",
    "Total harm": "Aggregate modeled harm created by the selected policy.",
    "Total cost": "Aggregate modeled cost created by the selected policy.",
    "Harm per crime prevented": "Total harm divided by crimes prevented; not applicable if no crimes are prevented.",
    "Cost per crime prevented": "Total cost divided by crimes prevented; not applicable if no crimes are prevented.",
    "District metrics": "District-level false positives, harm, and crimes for abstract Districts A, B, and C.",
}

DISTRICTS = ["A", "B", "C"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
RUN_METRIC_COLUMNS = [
    "run",
    "baseline_crimes",
    "crimes_after_policy",
    "crimes_prevented",
    "high_risk_flagged",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "children_helped",
    "children_harmed",
    "coerced_children",
    "total_harm",
    "total_cost",
    "harm_per_crime_prevented",
    "cost_per_crime_prevented",
    "false_positives_district_a",
    "false_positives_district_b",
    "false_positives_district_c",
    "harm_district_a",
    "harm_district_b",
    "harm_district_c",
    "crimes_district_a",
    "crimes_district_b",
    "crimes_district_c",
]


def average_results_table(run_results):
    display_names = {
        "baseline_crimes": "Baseline crimes",
        "crimes_after_policy": "Crimes after policy",
        "crimes_prevented": "Crimes prevented",
        "high_risk_flagged": "High-risk flagged",
        "true_positives": "True positives",
        "false_positives": "False positives",
        "false_negatives": "False negatives",
        "precision": "Precision",
        "recall": "Recall",
        "children_helped": "Children helped",
        "children_harmed": "Children harmed",
        "coerced_children": "Coerced children",
        "total_harm": "Total harm",
        "total_cost": "Total cost",
        "harm_per_crime_prevented": "Harm per crime prevented",
        "cost_per_crime_prevented": "Cost per crime prevented",
        "false_positives_district_a": "False positives in District A",
        "false_positives_district_b": "False positives in District B",
        "false_positives_district_c": "False positives in District C",
        "harm_district_a": "Harm in District A",
        "harm_district_b": "Harm in District B",
        "harm_district_c": "Harm in District C",
        "crimes_district_a": "Crimes in District A",
        "crimes_district_b": "Crimes in District B",
        "crimes_district_c": "Crimes in District C",
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


def glossary_table(items):
    return pd.DataFrame(
        [{"Item": item, "Meaning": meaning} for item, meaning in items.items()]
    )


def render_reference_guide():
    with st.expander("Policy and metric guide", expanded=False):
        st.markdown("**Policy options**")
        st.dataframe(
            glossary_table(POLICY_DESCRIPTIONS),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Sidebar settings**")
        st.dataframe(
            glossary_table(SETTING_DESCRIPTIONS),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Result metrics**")
        st.dataframe(
            glossary_table(RESULT_METRIC_DESCRIPTIONS),
            use_container_width=True,
            hide_index=True,
        )


def line_chart(data, x_column, y_column, title, y_label):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    plot_data = data[[x_column, y_column]].dropna()
    ax.plot(plot_data[x_column], plot_data[y_column], linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Synthetic run")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
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
    return (
        district_results.groupby("district", as_index=False)
        .agg(
            false_positives=("false_positives", "mean"),
            harm=("harm", "mean"),
            crimes=("crimes", "mean"),
        )
        .sort_values("district")
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
                "harm_per_crime_prevented",
                "Harm per crime prevented by run",
                "Harm per crime prevented",
            ),
            clear_figure=True,
        )

    district_summary = district_summary_table(district_results)

    district_left, district_right = st.columns(2)
    with district_left:
        st.pyplot(
            bar_chart(
                district_summary,
                "district",
                "false_positives",
                "Average false positives by district",
                "False positives",
            ),
            clear_figure=True,
        )

    with district_right:
        st.pyplot(
            bar_chart(
                district_summary,
                "district",
                "harm",
                "Average harm by district",
                "Intervention harm",
            ),
            clear_figure=True,
        )

    st.subheader("Average district outcomes")
    district_display = district_summary.rename(
        columns={
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
        "crimes_prevented",
        "false_positives",
        "false_negatives",
        "children_harmed",
        "total_harm",
        "total_cost",
        "harm_per_crime_prevented",
    ]
    averages = run_results[metric_columns].mean(numeric_only=True)
    return {column: metric_value(averages[column]) for column in metric_columns}


def compact_parameter_summary(settings):
    return (
        f"population_size={int(settings['population_size'])}; "
        f"llm_synthetic_runs={int(settings['llm_simulation_runs'])}; "
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
    debrief_word_limit = int(settings["llm_output_word_limit"])
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
        "Return only valid JSON with exactly these keys: run_results, district_results, "
        "representative_agents, debrief_text.\n"
        "run_results must contain one object per run and every required_run_metric_columns field. "
        "district_results must contain one row for each run and each district with fields run, district, "
        "false_positives, harm, crimes (crimes after policy is applied). representative_agents must contain up to 3 abstract synthetic "
        "children with no names and no protected attributes.\n"
        "Use numeric values only for metric fields. If crimes_prevented is 0 or negative, set "
        "harm_per_crime_prevented and cost_per_crime_prevented to null.\n"
        f"The debrief_text must be no more than {debrief_word_limit} words and should explain trade-offs, "
        "uncertainty, false positives, false negatives, harm, cost, and District C bias if relevant.\n\n"
        f"Simulation request:\n{json.dumps(prompt_payload, indent=2)}"
    )


def run_openai_json(prompt, max_tokens=8000):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate strict JSON for a synthetic, ethics-focused simulation. "
                    "Never include markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
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

    run_results = pd.DataFrame(rows)
    for column in RUN_METRIC_COLUMNS:
        run_results[column] = pd.to_numeric(run_results[column], errors="coerce")
    run_results["run"] = run_results["run"].astype("Int64")
    run_results = run_results.sort_values("run").reset_index(drop=True)

    # Re-derive dependent metrics so LLM inconsistencies don't propagate to charts.
    run_results["crimes_prevented"] = (
        run_results["baseline_crimes"] - run_results["crimes_after_policy"]
    )
    flagged = run_results["true_positives"] + run_results["false_positives"]
    run_results["high_risk_flagged"] = flagged
    run_results["precision"] = run_results["true_positives"] / flagged
    denominator_recall = run_results["true_positives"] + run_results["false_negatives"]
    run_results["recall"] = run_results["true_positives"] / denominator_recall
    safe_cp = run_results["crimes_prevented"].replace(0, np.nan)
    run_results["harm_per_crime_prevented"] = run_results["total_harm"] / safe_cp
    run_results["cost_per_crime_prevented"] = run_results["total_cost"] / safe_cp

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
        return pd.DataFrame(columns=["run", "district", "false_positives", "harm", "crimes"])
    for column in ["run", "false_positives", "harm", "crimes"]:
        district_results[column] = pd.to_numeric(district_results[column], errors="coerce")
    district_results["run"] = district_results["run"].astype("Int64")
    return district_results


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
                "crimes_prevented": metrics["crimes_prevented"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "children_harmed": metrics["children_harmed"],
                "total_harm": metrics["total_harm"],
                "total_cost": metrics["total_cost"],
                "harm_per_crime_prevented": metrics["harm_per_crime_prevented"],
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
        "This mode uses one LLM call to generate a small synthetic scenario, run-level metrics, district "
        "metrics, charts, and a concise explanation. It remains a thought experiment, not a prediction "
        "system. Run the LLM-agent simulation to generate metrics and charts for the current settings."
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning(
            "LLM-agent simulation is enabled, but OPENAI_API_KEY is not configured. Add it as an environment "
            "variable or Railway secret."
        )

    parameter_summary = compact_parameter_summary(settings)

    if st.button("Run LLM-agent simulation", disabled=not bool(api_key), type="primary"):
        try:
            raw_result = run_openai_json(build_llm_simulation_prompt(settings))
            run_results = clean_llm_run_results(raw_result.get("run_results", []))
            district_results = clean_llm_district_results(raw_result.get("district_results", []))
            representative_agents = raw_result.get("representative_agents", [])[:3]
            debrief_text = str(raw_result.get("debrief_text", "")).strip()

            if run_results.empty or district_results.empty:
                raise ValueError("The model returned incomplete simulation tables.")

            aggregate_metrics = compact_aggregate_metrics(run_results)
            st.session_state["llm_agent_latest_result"] = {
                "run_results": run_results,
                "district_results": district_results,
                "representative_agents": representative_agents,
                "debrief_text": debrief_text,
            }
        except Exception as error:
            st.error(friendly_llm_error(error))
            debrief_text = None

        if debrief_text:
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selected_policy": settings["policy"],
                "parameter_summary": parameter_summary,
                "representative_agent_count": len(representative_agents),
                "llm_model": LLM_MODEL,
                "aggregate_metrics": aggregate_metrics,
                "debrief_text": debrief_text,
            }
            add_llm_run_log_entry(entry, int(settings["llm_run_log_size"]))
            st.success("LLM-agent simulation generated.")

    latest_result = st.session_state.get("llm_agent_latest_result")
    if latest_result:
        st.subheader("LLM-agent simulation averages")
        latest_run_results = latest_result["run_results"]
        latest_district_results = latest_result["district_results"]
        display_average_table(average_results_table(latest_run_results))
        render_charts(latest_run_results, latest_district_results)

        render_interpretation(
            settings["policy"],
            average_results_table(latest_run_results),
            settings["bias_against_district_c"],
        )

        st.markdown("**Latest LLM-agent explanation**")
        st.write(latest_result["debrief_text"])

        representative_agents = latest_result.get("representative_agents", [])
        if representative_agents:
            with st.expander("Representative LLM synthetic agents used", expanded=False):
                st.dataframe(pd.DataFrame(representative_agents), use_container_width=True, hide_index=True)
    else:
        st.caption("No LLM-agent simulation has been generated in this session yet.")

    render_llm_run_log(int(settings["llm_run_log_size"]))


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
    settings["llm_representative_agents"] = st.sidebar.slider(
        "LLM representative agents",
        1,
        3,
        3,
        step=1,
        help=SETTING_DESCRIPTIONS["LLM representative agents"],
    )
    settings["llm_output_word_limit"] = st.sidebar.slider(
        "LLM output word limit",
        100,
        400,
        200,
        step=25,
        help=SETTING_DESCRIPTIONS["LLM output word limit"],
    )
    settings["llm_run_log_size"] = st.sidebar.slider(
        "LLM run log size",
        1,
        10,
        5,
        step=1,
        help=SETTING_DESCRIPTIONS["LLM run log size"],
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
