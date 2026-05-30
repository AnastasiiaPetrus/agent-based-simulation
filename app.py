import math
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


def clamp_series(values, lower=0.0, upper=1.0):
    return np.clip(values, lower, upper)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def generate_population(population_size, prediction_noise, bias_against_district_c, rng):
    """Create a synthetic population with hidden risk and observed predicted risk."""
    population = pd.DataFrame({"id": np.arange(1, population_size + 1)})
    population["district"] = rng.choice(DISTRICTS, size=population_size, p=[0.34, 0.33, 0.33])

    district_stress_shift = population["district"].map({"A": -0.04, "B": 0.0, "C": 0.04}).to_numpy()
    district_support_shift = population["district"].map({"A": 0.04, "B": 0.0, "C": -0.03}).to_numpy()

    population["socioeconomic_stress"] = clamp_series(
        rng.beta(2.2, 3.2, population_size) + district_stress_shift
    )
    population["family_stress"] = clamp_series(
        rng.beta(2.0, 3.0, population_size) + 0.5 * district_stress_shift
    )
    population["school_support"] = clamp_series(
        rng.beta(3.0, 2.2, population_size) + district_support_shift
    )

    raw_risk = (
        -3.4
        + 1.35 * population["socioeconomic_stress"]
        + 1.25 * population["family_stress"]
        - 1.15 * population["school_support"]
        + rng.normal(0.0, 0.25, population_size)
    )
    population["true_risk"] = clamp_series(1 / (1 + np.exp(-raw_risk)), 0.01, 0.65)

    district_bias = np.where(population["district"] == "C", bias_against_district_c, 0.0)
    population["predicted_risk"] = clamp_series(
        population["true_risk"] + rng.normal(0.0, prediction_noise, population_size) + district_bias
    )

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
    """Apply one policy to the population and return adjusted risks and intervention values."""
    adjusted = population.copy()
    adjusted["applied_policy"] = policy
    adjusted["received_help"] = False
    adjusted["coercive_intervention"] = False
    adjusted["intervention_harm"] = 0.0
    adjusted["intervention_cost"] = 0.0
    adjusted["adjusted_risk"] = adjusted["true_risk"]

    high_risk = adjusted["high_risk_flag"]

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
        adjusted.loc[high_risk, "adjusted_risk"] = adjusted.loc[high_risk, "true_risk"] * 0.18
        adjusted.loc[high_risk, "intervention_harm"] = coercive_harm
        adjusted.loc[high_risk, "intervention_cost"] = coercive_cost
    elif policy == "Rights-preserving targeted support":
        adjusted.loc[high_risk, "received_help"] = True
        adjusted.loc[high_risk, "adjusted_risk"] = (
            adjusted.loc[high_risk, "true_risk"] * (1 - support_effectiveness * 0.9)
        )
        adjusted.loc[high_risk, "intervention_harm"] = 0.08
        adjusted.loc[high_risk, "intervention_cost"] = support_cost * 1.15

    adjusted["adjusted_risk"] = clamp_series(adjusted["adjusted_risk"], 0.0, 1.0)
    return adjusted


def calculate_metrics(population, run_id):
    actual_risk_group = population["baseline_committed_crime"]
    predicted_risk_group = population["high_risk_flag"]

    baseline_crimes = int(population["baseline_committed_crime"].sum())
    crimes_after_policy = int(population["committed_crime"].sum())
    crimes_prevented = baseline_crimes - crimes_after_policy
    high_risk_flagged = int(predicted_risk_group.sum())
    true_positives = int((predicted_risk_group & actual_risk_group).sum())
    false_positives = int((predicted_risk_group & ~actual_risk_group).sum())
    false_negatives = int((~predicted_risk_group & actual_risk_group).sum())
    children_helped = int(population["received_help"].sum())
    children_harmed = int((population["intervention_harm"] > 0).sum())
    coerced_children = int(population["coercive_intervention"].sum())
    total_harm = float(population["intervention_harm"].sum())
    total_cost = float(population["intervention_cost"].sum())

    return {
        "run": run_id,
        "baseline_crimes": baseline_crimes,
        "crimes_after_policy": crimes_after_policy,
        "crimes_prevented": crimes_prevented,
        "high_risk_flagged": high_risk_flagged,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": safe_divide(true_positives, true_positives + false_positives),
        "recall": safe_divide(true_positives, true_positives + false_negatives),
        "children_helped": children_helped,
        "children_harmed": children_harmed,
        "coerced_children": coerced_children,
        "total_harm": total_harm,
        "total_cost": total_cost,
        "harm_per_crime_prevented": safe_divide(total_harm, crimes_prevented)
        if crimes_prevented > 0
        else math.nan,
        "cost_per_crime_prevented": safe_divide(total_cost, crimes_prevented)
        if crimes_prevented > 0
        else math.nan,
    }


def aggregate_district_metrics(population, run_id):
    rows = []
    for district in DISTRICTS:
        district_data = population[population["district"] == district]
        baseline_actual = district_data["baseline_committed_crime"]
        false_positives = int((district_data["high_risk_flag"] & ~baseline_actual).sum())

        rows.append(
            {
                "run": run_id,
                "district": district,
                "false_positives": false_positives,
                "harm": float(district_data["intervention_harm"].sum()),
                "crimes_after_policy": int(district_data["committed_crime"].sum()),
            }
        )
    return rows


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
):
    rng = np.random.default_rng(seed + run_id)
    population = generate_population(
        population_size, prediction_noise, bias_against_district_c, rng
    )
    population["high_risk_flag"] = population["predicted_risk"] >= high_risk_threshold
    population["baseline_committed_crime"] = rng.random(population_size) < population["true_risk"]

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
    population["committed_crime"] = rng.random(population_size) < population["adjusted_risk"]

    run_metrics = calculate_metrics(population, run_id)
    district_metrics = aggregate_district_metrics(population, run_id)
    return run_metrics, district_metrics


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
        run_metrics, district_metrics = simulate_once(
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
        district_rows.extend(district_metrics)

    return pd.DataFrame(run_rows), pd.DataFrame(district_rows)


def format_average_results(run_results):
    average_results = run_results.mean(numeric_only=True).drop(labels=["run"])
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
    }
    table = average_results.rename(index=display_names).reset_index()
    table.columns = ["Metric", "Average across runs"]
    return table


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


def render_charts(run_results, district_results):
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.pyplot(
            line_chart(
                run_results, "run", "crimes_prevented", "Crimes prevented by run", "Crimes prevented"
            )
        )
        st.pyplot(
            line_chart(
                run_results, "run", "total_harm", "Total harm by run", "Total harm"
            )
        )

    with chart_right:
        st.pyplot(
            line_chart(
                run_results, "run", "false_positives", "False positives by run", "False positives"
            )
        )
        st.pyplot(
            line_chart(
                run_results,
                "run",
                "harm_per_crime_prevented",
                "Harm per crime prevented by run",
                "Harm per crime prevented",
            )
        )

    district_summary = (
        district_results.groupby("district", as_index=False)
        .agg(
            false_positives=("false_positives", "mean"),
            harm=("harm", "mean"),
            crimes_after_policy=("crimes_after_policy", "mean"),
        )
        .sort_values("district")
    )

    district_left, district_right = st.columns(2)
    with district_left:
        st.pyplot(
            bar_chart(
                district_summary,
                "district",
                "false_positives",
                "Average false positives by district",
                "False positives",
            )
        )
    with district_right:
        st.pyplot(
            bar_chart(
                district_summary,
                "district",
                "harm",
                "Average harm by district",
                "Intervention harm",
            )
        )

    st.subheader("Average district outcomes")
    district_display = district_summary.rename(
        columns={
            "district": "District",
            "false_positives": "False positives",
            "harm": "Harm",
            "crimes_after_policy": "Crimes after policy",
        }
    )
    st.dataframe(district_display, use_container_width=True, hide_index=True)


def render_interpretation(policy, average_table, bias_against_district_c):
    values = {
        row["Metric"]: row["Average across runs"] for _, row in average_table.iterrows()
    }
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
            "In this run set, any reduction in crime is achieved by imposing high harm on children, "
            f"including an average of {false_positives:.1f} falsely flagged children per run who would "
            "not have committed the modeled offense in the baseline outcome."
        )
    elif policy in [
        "Universal support",
        "Targeted support for high-risk children",
        "Rights-preserving targeted support",
    ]:
        st.info(
            f"This support-oriented policy prevents an average of {crimes_prevented:.1f} crimes per run "
            f"under the selected assumptions, with an average total cost of {total_cost:.1f}. "
            "The main trade-off is between broader help, narrower targeting, cost, and prediction error."
        )
    elif policy == "Surveillance of high-risk children":
        st.warning(
            f"Surveillance adds an average total harm of {total_harm:.1f} while relying on imperfect "
            "classification. False positives matter because flagged children may be harmed even when the "
            "baseline outcome would not include a crime."
        )
    else:
        st.info(
            "No action provides a baseline comparison. It avoids intervention harm and cost, but it also "
            "does not reduce modeled crime risk."
        )

    if bias_against_district_c >= 0.05:
        st.warning(
            "The selected bias against District C is high enough to create a visible risk of unequal "
            "false positives and harm. Because the districts are abstract, this illustrates structural "
            "sensitivity rather than any real population claim."
        )

    st.caption(
        "Prediction is not destiny. The model does not claim to predict real human behavior, and it must "
        "not be used to justify preventive punishment."
    )


def main():
    st.set_page_config(page_title="Predictive Justice Simulation", layout="wide")

    st.title("Predictive Justice Simulation")
    st.warning(
        "This model does not decide what is morally permissible. It shows the consequences of different "
        "policies under explicit assumptions. Children should not be punished for a predicted future act."
    )
    st.caption(
        "This is not a real-world decision tool. All agents, districts, risks, and outcomes are synthetic."
    )

    st.sidebar.header("Simulation settings")
    population_size = st.sidebar.slider("Population size", 100, 10000, 2000, step=100)
    monte_carlo_runs = st.sidebar.slider("Number of Monte Carlo runs", 5, 300, 50, step=5)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
    prediction_noise = st.sidebar.slider("Prediction error / noise", 0.0, 0.35, 0.10, step=0.01)
    high_risk_threshold = st.sidebar.slider("High-risk threshold", 0.01, 0.70, 0.25, step=0.01)
    support_effectiveness = st.sidebar.slider("Support effectiveness", 0.0, 0.80, 0.30, step=0.01)
    surveillance_effectiveness = st.sidebar.slider(
        "Surveillance effectiveness", 0.0, 0.50, 0.12, step=0.01
    )
    surveillance_harm = st.sidebar.slider("Surveillance harm", 0.0, 10.0, 1.5, step=0.1)
    coercive_harm = st.sidebar.slider("Coercive intervention harm", 0.0, 50.0, 15.0, step=0.5)
    bias_against_district_c = st.sidebar.slider(
        "Bias against District C", 0.0, 0.30, 0.00, step=0.01
    )
    support_cost = st.sidebar.slider("Support cost", 0.0, 10000.0, 1200.0, step=100.0)
    surveillance_cost = st.sidebar.slider("Surveillance cost", 0.0, 10000.0, 1800.0, step=100.0)
    coercive_cost = st.sidebar.slider(
        "Coercive intervention cost", 0.0, 50000.0, 15000.0, step=500.0
    )
    policy = st.sidebar.selectbox("Selected policy", POLICIES)

    run_results, district_results = run_monte_carlo(
        population_size,
        monte_carlo_runs,
        int(seed),
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

    st.subheader("Average results across Monte Carlo runs")
    average_table = format_average_results(run_results)
    st.dataframe(
        average_table.style.format({"Average across runs": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    render_charts(run_results, district_results)
    render_interpretation(policy, average_table, bias_against_district_c)

    csv_buffer = StringIO()
    run_results.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download results as CSV",
        data=csv_buffer.getvalue(),
        file_name="predictive_justice_simulation_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
