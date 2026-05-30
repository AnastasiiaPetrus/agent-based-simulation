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

DISTRICTS = ["A", "B", "C"]
RIGHTS_PRESERVING_STIGMA_HARM = 0.08
COERCIVE_RISK_MULTIPLIER = 0.18
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


def clamp(values, lower=0.0, upper=1.0):
    return np.clip(values, lower, upper)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def per_crime_prevented(value, crimes_prevented):
    if crimes_prevented <= 0:
        return np.nan
    return value / crimes_prevented


def generate_population(population_size, prediction_noise, bias_against_district_c, rng):
    """Generate a fully synthetic population with hidden risk and observed predicted risk."""
    population = pd.DataFrame({"id": np.arange(1, population_size + 1)})
    population["district"] = rng.choice(DISTRICTS, size=population_size, p=[0.34, 0.33, 0.33])

    district_stress_shift = population["district"].map({"A": -0.04, "B": 0.0, "C": 0.04}).to_numpy()
    district_support_shift = population["district"].map({"A": 0.04, "B": 0.0, "C": -0.03}).to_numpy()

    population["socioeconomic_stress"] = clamp(
        rng.beta(2.2, 3.2, population_size) + district_stress_shift
    )
    population["family_stress"] = clamp(
        rng.beta(2.0, 3.0, population_size) + 0.5 * district_stress_shift
    )
    population["school_support"] = clamp(
        rng.beta(3.0, 2.2, population_size) + district_support_shift
    )

    risk_score = (
        -3.35
        + 1.30 * population["socioeconomic_stress"]
        + 1.20 * population["family_stress"]
        - 1.15 * population["school_support"]
        + rng.normal(0.0, 0.22, population_size)
    )
    population["true_risk"] = clamp(1 / (1 + np.exp(-risk_score)), 0.01, 0.65)

    district_bias = np.where(population["district"] == "C", bias_against_district_c, 0.0)
    population["predicted_risk"] = clamp(
        population["true_risk"] + rng.normal(0.0, prediction_noise, population_size) + district_bias
    )

    return population


def add_prediction_flags(population, high_risk_threshold, prediction_noise, rng):
    population["high_risk_flag"] = population["predicted_risk"] >= high_risk_threshold
    population["reassessed_predicted_risk"] = clamp(
        0.7 * population["predicted_risk"]
        + 0.3 * population["true_risk"]
        + rng.normal(0.0, prediction_noise * 0.5, len(population))
    )
    population["reassessment_flag"] = population["reassessed_predicted_risk"] >= high_risk_threshold
    return population


def apply_policy(
    population,
    policy,
    support_effectiveness,
    surveillance_effectiveness,
    surveillance_harm,
    coercive_harm,
    support_cost,
    surveillance_cost,
    coercive_cost,
):
    """Apply the selected thought-experiment policy to synthetic agents."""
    adjusted = population.copy()
    adjusted["applied_policy"] = policy
    adjusted["received_help"] = False
    adjusted["coercive_intervention"] = False
    adjusted["intervention_harm"] = 0.0
    adjusted["intervention_cost"] = 0.0
    adjusted["adjusted_risk"] = adjusted["true_risk"]

    high_risk = adjusted["high_risk_flag"]
    reassessed_high_risk = high_risk & adjusted["reassessment_flag"]

    if policy == "Universal support":
        adjusted["received_help"] = True
        adjusted["adjusted_risk"] = adjusted["true_risk"] * (1 - support_effectiveness)
        adjusted["intervention_cost"] = support_cost

    elif policy == "Targeted support for high-risk children":
        adjusted.loc[high_risk, "received_help"] = True
        adjusted.loc[high_risk, "adjusted_risk"] = (
            adjusted.loc[high_risk, "true_risk"] * (1 - support_effectiveness)
        )
        adjusted.loc[high_risk, "intervention_cost"] = support_cost

    elif policy == "Surveillance of high-risk children":
        adjusted.loc[high_risk, "adjusted_risk"] = (
            adjusted.loc[high_risk, "true_risk"] * (1 - surveillance_effectiveness)
        )
        adjusted.loc[high_risk, "intervention_harm"] = surveillance_harm
        adjusted.loc[high_risk, "intervention_cost"] = surveillance_cost

    elif policy == "Coercive preventive intervention for high-risk children":
        adjusted.loc[high_risk, "coercive_intervention"] = True
        adjusted.loc[high_risk, "adjusted_risk"] = (
            adjusted.loc[high_risk, "true_risk"] * COERCIVE_RISK_MULTIPLIER
        )
        adjusted.loc[high_risk, "intervention_harm"] = coercive_harm
        adjusted.loc[high_risk, "intervention_cost"] = coercive_cost

    elif policy == "Rights-preserving targeted support":
        adjusted.loc[reassessed_high_risk, "received_help"] = True
        adjusted.loc[reassessed_high_risk, "adjusted_risk"] = (
            adjusted.loc[reassessed_high_risk, "true_risk"] * (1 - support_effectiveness)
        )
        adjusted.loc[reassessed_high_risk, "intervention_harm"] = RIGHTS_PRESERVING_STIGMA_HARM
        adjusted.loc[reassessed_high_risk, "intervention_cost"] = support_cost * 1.15

    adjusted["adjusted_risk"] = clamp(adjusted["adjusted_risk"])
    return adjusted


def aggregate_district_metrics(population, run_id):
    rows = []

    for district in DISTRICTS:
        district_data = population[population["district"] == district]
        false_positive_mask = district_data["high_risk_flag"] & ~district_data["baseline_committed_crime"]

        rows.append(
            {
                "run": run_id,
                "district": district,
                "false_positives": int(false_positive_mask.sum()),
                "harm": float(district_data["intervention_harm"].sum()),
                "crimes": int(district_data["committed_crime"].sum()),
            }
        )

    return rows


def calculate_metrics(population, district_rows, run_id):
    flagged = population["high_risk_flag"]
    baseline_crime = population["baseline_committed_crime"]

    baseline_crimes = int(baseline_crime.sum())
    crimes_after_policy = int(population["committed_crime"].sum())
    crimes_prevented = baseline_crimes - crimes_after_policy
    true_positives = int((flagged & baseline_crime).sum())
    false_positives = int((flagged & ~baseline_crime).sum())
    false_negatives = int((~flagged & baseline_crime).sum())
    total_harm = float(population["intervention_harm"].sum())
    total_cost = float(population["intervention_cost"].sum())

    metrics = {
        "run": run_id,
        "baseline_crimes": baseline_crimes,
        "crimes_after_policy": crimes_after_policy,
        "crimes_prevented": crimes_prevented,
        "high_risk_flagged": int(flagged.sum()),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": safe_divide(true_positives, true_positives + false_positives),
        "recall": safe_divide(true_positives, true_positives + false_negatives),
        "children_helped": int(population["received_help"].sum()),
        "children_harmed": int((population["intervention_harm"] > 0).sum()),
        "coerced_children": int(population["coercive_intervention"].sum()),
        "total_harm": total_harm,
        "total_cost": total_cost,
        "harm_per_crime_prevented": per_crime_prevented(total_harm, crimes_prevented),
        "cost_per_crime_prevented": per_crime_prevented(total_cost, crimes_prevented),
    }

    for row in district_rows:
        district_key = row["district"].lower()
        metrics[f"false_positives_district_{district_key}"] = row["false_positives"]
        metrics[f"harm_district_{district_key}"] = row["harm"]
        metrics[f"crimes_district_{district_key}"] = row["crimes"]

    return metrics


def simulate_once(
    run_id,
    population_size,
    seed,
    prediction_noise,
    high_risk_threshold,
    bias_against_district_c,
    policy,
    support_effectiveness,
    surveillance_effectiveness,
    surveillance_harm,
    coercive_harm,
    support_cost,
    surveillance_cost,
    coercive_cost,
    return_population=False,
):
    rng = np.random.default_rng(seed + run_id)
    population = generate_population(
        population_size, prediction_noise, bias_against_district_c, rng
    )
    population = add_prediction_flags(population, high_risk_threshold, prediction_noise, rng)
    outcome_draw = rng.random(population_size)
    population["baseline_committed_crime"] = outcome_draw < population["true_risk"]
    population["baseline_outcome"] = np.where(
        population["baseline_committed_crime"], "Committed crime", "Did not commit crime"
    )

    population = apply_policy(
        population,
        policy,
        support_effectiveness,
        surveillance_effectiveness,
        surveillance_harm,
        coercive_harm,
        support_cost,
        surveillance_cost,
        coercive_cost,
    )
    population["committed_crime"] = outcome_draw < population["adjusted_risk"]
    population["outcome"] = np.where(
        population["committed_crime"], "Committed crime", "Did not commit crime"
    )

    district_rows = aggregate_district_metrics(population, run_id)
    run_metrics = calculate_metrics(population, district_rows, run_id)
    if return_population:
        return run_metrics, district_rows, population
    return run_metrics, district_rows


def run_monte_carlo(
    population_size,
    monte_carlo_runs,
    seed,
    prediction_noise,
    high_risk_threshold,
    bias_against_district_c,
    policy,
    support_effectiveness,
    surveillance_effectiveness,
    surveillance_harm,
    coercive_harm,
    support_cost,
    surveillance_cost,
    coercive_cost,
):
    run_rows = []
    district_rows = []

    for run_id in range(1, monte_carlo_runs + 1):
        run_metrics, run_district_rows = simulate_once(
            run_id,
            population_size,
            seed,
            prediction_noise,
            high_risk_threshold,
            bias_against_district_c,
            policy,
            support_effectiveness,
            surveillance_effectiveness,
            surveillance_harm,
            coercive_harm,
            support_cost,
            surveillance_cost,
            coercive_cost,
        )
        run_rows.append(run_metrics)
        district_rows.extend(run_district_rows)

    return pd.DataFrame(run_rows), pd.DataFrame(district_rows)


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


def line_chart(data, x_column, y_column, title, y_label):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(data[x_column], data[y_column], linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Monte Carlo run")
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
        f"monte_carlo_runs={int(settings['monte_carlo_runs'])}; "
        f"seed={int(settings['seed'])}; "
        f"prediction_noise={settings['prediction_noise']:.2f}; "
        f"high_risk_threshold={settings['high_risk_threshold']:.2f}; "
        f"bias_against_district_c={settings['bias_against_district_c']:.2f}; "
        f"support_effectiveness={settings['support_effectiveness']:.2f}; "
        f"surveillance_effectiveness={settings['surveillance_effectiveness']:.2f}; "
        f"surveillance_harm={settings['surveillance_harm']:.1f}; "
        f"coercive_harm={settings['coercive_harm']:.1f}"
    )


def llm_assumptions(settings):
    return {
        "selected_policy": settings["policy"],
        "prediction_noise": metric_value(settings["prediction_noise"]),
        "high_risk_threshold": metric_value(settings["high_risk_threshold"]),
        "bias_against_district_c": metric_value(settings["bias_against_district_c"]),
        "support_effectiveness": metric_value(settings["support_effectiveness"]),
        "surveillance_effectiveness": metric_value(settings["surveillance_effectiveness"]),
        "surveillance_harm": metric_value(settings["surveillance_harm"]),
        "coercive_intervention_harm": metric_value(settings["coercive_harm"]),
        "support_cost": metric_value(settings["support_cost"]),
        "surveillance_cost": metric_value(settings["surveillance_cost"]),
        "coercive_intervention_cost": metric_value(settings["coercive_cost"]),
    }


def sample_representative_agents(population_df, max_agents, seed):
    category_masks = [
        population_df["high_risk_flag"] & ~population_df["baseline_committed_crime"],
        population_df["high_risk_flag"] & population_df["baseline_committed_crime"],
        ~population_df["high_risk_flag"] & population_df["baseline_committed_crime"],
        population_df["received_help"],
        population_df["intervention_harm"] > 0,
    ]
    selected_indices = []
    rng = np.random.default_rng(seed)

    for mask in category_masks:
        candidates = population_df.loc[mask & ~population_df.index.isin(selected_indices)]
        if not candidates.empty and len(selected_indices) < max_agents:
            selected_indices.append(rng.choice(candidates.index.to_numpy()))

    if len(selected_indices) < max_agents:
        remaining = population_df.loc[~population_df.index.isin(selected_indices)].sort_values(
            ["intervention_harm", "predicted_risk"], ascending=False
        )
        selected_indices.extend(remaining.index[: max_agents - len(selected_indices)].to_list())

    labels = ["Representative child A", "Representative child B", "Representative child C"]
    representatives = []

    for label, (_, child) in zip(labels, population_df.loc[selected_indices].iterrows()):
        representatives.append(
            {
                "label": label,
                "district": child["district"],
                "predicted_risk": round(float(child["predicted_risk"]), 3),
                "true_risk": round(float(child["true_risk"]), 3),
                "high_risk": bool(child["high_risk_flag"]),
                "received_help": bool(child["received_help"]),
                "experienced_harm": bool(child["intervention_harm"] > 0),
                "coercive_intervention": bool(child["coercive_intervention"]),
                "baseline_outcome": child["baseline_outcome"],
                "post_policy_outcome": child["outcome"],
            }
        )

    return representatives


def build_llm_display_population(settings):
    _, _, population = simulate_once(
        10_001,
        int(settings["llm_display_population_size"]),
        int(settings["seed"]),
        settings["prediction_noise"],
        settings["high_risk_threshold"],
        settings["bias_against_district_c"],
        settings["policy"],
        settings["support_effectiveness"],
        settings["surveillance_effectiveness"],
        settings["surveillance_harm"],
        settings["coercive_harm"],
        settings["support_cost"],
        settings["surveillance_cost"],
        settings["coercive_cost"],
        return_population=True,
    )
    return population


def build_llm_debrief_prompt(policy_name, aggregate_metrics, representative_agents, assumptions, max_words):
    prompt_payload = {
        "selected_policy": policy_name,
        "aggregate_metrics": aggregate_metrics,
        "assumptions": assumptions,
        "representative_synthetic_agents": representative_agents,
    }

    return (
        "You are writing a concise debrief for a synthetic ethical thought experiment about predictive "
        "justice.\n"
        "Use English only.\n"
        f"Do not exceed {max_words} words.\n"
        "Do not recommend real-world punishment.\n"
        "Do not claim the simulation predicts real people.\n"
        "Do not treat prediction as destiny.\n"
        "Do not identify anyone.\n"
        "Do not make up numbers not provided.\n"
        "Explain trade-offs and uncertainty.\n"
        "Highlight false positives, false negatives, harm, cost, and District C bias if relevant.\n"
        "Keep the output concise and grounded in the provided values.\n\n"
        f"Simulation context:\n{json.dumps(prompt_payload, indent=2)}"
    )


def build_llm_simulation_prompt(settings):
    run_count = int(settings["llm_simulation_runs"])
    population_size = int(settings["llm_display_population_size"])
    debrief_word_limit = int(settings["llm_output_word_limit"])
    prompt_payload = {
        "selected_policy": settings["policy"],
        "population_size": population_size,
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
        "Rights-preserving targeted support adds support with minimal stigma harm.\n"
        "Return only valid JSON with exactly these keys: run_results, district_results, "
        "representative_agents, debrief_text.\n"
        "run_results must contain one object per run and every required_run_metric_columns field. "
        "district_results must contain one row for each run and each district with fields run, district, "
        "false_positives, harm, crimes. representative_agents must contain up to 3 abstract synthetic "
        "children with no names and no protected attributes.\n"
        "Use numeric values only for metric fields. If crimes_prevented is 0 or negative, set "
        "harm_per_crime_prevented and cost_per_crime_prevented to null.\n"
        f"The debrief_text must be no more than {debrief_word_limit} words and should explain trade-offs, "
        "uncertainty, false positives, false negatives, harm, cost, and District C bias if relevant.\n\n"
        f"Simulation request:\n{json.dumps(prompt_payload, indent=2)}"
    )


def run_openai_debrief(prompt, max_words):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You provide cautious ethical analysis for synthetic simulations.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max(250, min(900, max_words * 3)),
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def run_openai_json(prompt, max_tokens=4000):
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
    if not settings["enable_llm_agent_mode"]:
        return

    initialize_llm_state()
    st.subheader("LLM-Agent Simulation")
    st.write(
        "This mode uses one LLM call to generate a small synthetic scenario, run-level metrics, district "
        "metrics, charts, and a concise explanation. It remains a thought experiment, not a prediction "
        "system."
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning(
            "LLM-agent simulation is enabled, but OPENAI_API_KEY is not configured. Add it as an environment "
            "variable or Railway secret."
        )

    parameter_summary = compact_parameter_summary(settings)

    if st.button("Run LLM-agent simulation", disabled=not bool(api_key)):
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

        st.markdown("**Latest LLM-agent explanation**")
        st.write(latest_result["debrief_text"])

        representative_agents = latest_result.get("representative_agents", [])
        if representative_agents:
            with st.expander("Representative LLM synthetic agents used", expanded=False):
                st.dataframe(pd.DataFrame(representative_agents), use_container_width=True, hide_index=True)
    else:
        st.info("Run the LLM-agent simulation to generate metrics and charts for the current settings.")

    render_llm_run_log(int(settings["llm_run_log_size"]))


def sidebar_inputs():
    st.sidebar.header("Simulation settings")
    settings = {
        "population_size": st.sidebar.slider("Population size", 100, 10000, 2000, step=100),
        "monte_carlo_runs": st.sidebar.slider("Number of Monte Carlo runs", 5, 300, 50, step=5),
        "seed": st.sidebar.number_input(
            "Random seed", min_value=0, max_value=1_000_000, value=42, step=1
        ),
        "prediction_noise": st.sidebar.slider("Prediction error / noise", 0.0, 0.35, 0.10, step=0.01),
        "high_risk_threshold": st.sidebar.slider("High-risk threshold", 0.01, 0.70, 0.25, step=0.01),
        "support_effectiveness": st.sidebar.slider(
            "Support effectiveness", 0.0, 0.80, 0.30, step=0.01
        ),
        "surveillance_effectiveness": st.sidebar.slider(
            "Surveillance effectiveness", 0.0, 0.50, 0.12, step=0.01
        ),
        "surveillance_harm": st.sidebar.slider("Surveillance harm", 0.0, 10.0, 1.5, step=0.1),
        "coercive_harm": st.sidebar.slider(
            "Coercive intervention harm", 0.0, 50.0, 15.0, step=0.5
        ),
        "bias_against_district_c": st.sidebar.slider(
            "Bias against District C", 0.0, 0.30, 0.00, step=0.01
        ),
        "support_cost": st.sidebar.slider("Support cost", 0.0, 10000.0, 1200.0, step=100.0),
        "surveillance_cost": st.sidebar.slider(
            "Surveillance cost", 0.0, 10000.0, 1800.0, step=100.0
        ),
        "coercive_cost": st.sidebar.slider(
            "Coercive intervention cost", 0.0, 50000.0, 15000.0, step=500.0
        ),
        "policy": st.sidebar.selectbox("Selected policy", POLICIES),
    }

    settings["enable_llm_agent_mode"] = st.sidebar.toggle(
        "Enable lightweight LLM-agent mode", value=False
    )
    if settings["enable_llm_agent_mode"]:
        st.sidebar.subheader("LLM-agent settings")
        settings["llm_display_population_size"] = st.sidebar.slider(
            "LLM display population size", 20, 300, 100, step=10
        )
        settings["llm_simulation_runs"] = st.sidebar.slider(
            "LLM synthetic runs", 3, 20, 8, step=1
        )
        settings["llm_representative_agents"] = st.sidebar.slider(
            "LLM representative agents", 1, 3, 3, step=1
        )
        settings["llm_output_word_limit"] = st.sidebar.slider(
            "LLM output word limit", 100, 400, 200, step=25
        )
        settings["llm_run_log_size"] = st.sidebar.slider("LLM run log size", 1, 10, 5, step=1)
    else:
        settings["llm_display_population_size"] = 100
        settings["llm_simulation_runs"] = 8
        settings["llm_representative_agents"] = 3
        settings["llm_output_word_limit"] = 200
        settings["llm_run_log_size"] = 5

    return settings


def render_app():
    st.set_page_config(page_title="Predictive Justice Simulation", layout="wide")

    st.title("Predictive Justice Simulation")
    st.warning(
        "This model does not decide what is morally permissible. It shows the consequences of different "
        "policies under explicit assumptions. Children should not be punished for a predicted future act."
    )
    st.caption(
        "This is not a real-world decision tool. All agents, districts, risks, and outcomes are synthetic."
    )

    settings = sidebar_inputs()
    if settings["enable_llm_agent_mode"]:
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
        return

    run_results, district_results = run_monte_carlo(
        int(settings["population_size"]),
        int(settings["monte_carlo_runs"]),
        int(settings["seed"]),
        settings["prediction_noise"],
        settings["high_risk_threshold"],
        settings["bias_against_district_c"],
        settings["policy"],
        settings["support_effectiveness"],
        settings["surveillance_effectiveness"],
        settings["surveillance_harm"],
        settings["coercive_harm"],
        settings["support_cost"],
        settings["surveillance_cost"],
        settings["coercive_cost"],
    )

    st.subheader("Average results across Monte Carlo runs")
    average_table = average_results_table(run_results)
    display_average_table(average_table)

    render_charts(run_results, district_results)
    render_interpretation(
        settings["policy"],
        average_table,
        settings["bias_against_district_c"],
    )

    csv_buffer = StringIO()
    run_results.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download results as CSV",
        data=csv_buffer.getvalue(),
        file_name="predictive_justice_simulation_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    render_app()
