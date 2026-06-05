import numpy as np
import pandas as pd
import streamlit as st

from src.constants import POLICIES, RUN_METRIC_LABELS


POLICY_SORT_INDEX = {policy: index for index, policy in enumerate(POLICIES)}
POLICY_TOTAL_AVERAGE_LABELS = {
    "baseline_crimes": "Baseline predicted outcomes (avg)",
    "false_positives": "False positives (avg)",
    "false_negatives": "False negatives (avg)",
    "children_helped": "Policy benefit count (avg)",
    "children_harmed": "Policy harm count (avg)",
}
OUTCOME_EFFECT_COLUMN = "Predicted outcome reduction (%)"
POLICY_HARM_RATE_COLUMN = "Policy harm rate among positive predictions (%)"
PERCENT_COLUMNS = {OUTCOME_EFFECT_COLUMN, POLICY_HARM_RATE_COLUMN}
ORDERED_POLICY_TOTAL_COLUMNS = [
    "Policy",
    OUTCOME_EFFECT_COLUMN,
    POLICY_HARM_RATE_COLUMN,
    "Baseline predicted outcomes (avg)",
    "False positives (avg)",
    "False negatives (avg)",
    "Policy benefit count (avg)",
    "Policy harm count (avg)",
]


def format_outcome_effect(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


@st.cache_data(show_spinner=False)
def average_results_table(run_results):
    metric_order = [column for column in RUN_METRIC_LABELS if column in run_results.columns]
    averages = run_results[metric_order].mean(numeric_only=True)
    table = averages.rename(index=RUN_METRIC_LABELS).reset_index()
    table.columns = ["Metric", "Average per synthetic run"]
    return table


@st.cache_data(show_spinner=False)
def formatted_average_results_table(average_table):
    display_table = average_table.copy()

    def format_value(value):
        if pd.isna(value):
            return "Not applicable"
        return f"{value:,.3f}"

    display_table["Average per synthetic run"] = display_table["Average per synthetic run"].map(format_value)
    return display_table


@st.cache_data(show_spinner=False)
def combined_policy_totals_table(run_results, population_size):
    required_metric_columns = [*POLICY_TOTAL_AVERAGE_LABELS, "crimes_prevented", "children_flagged"]
    metric_columns = [column for column in required_metric_columns if column in run_results.columns]
    table = (
        run_results.groupby("policy", as_index=False, observed=True)[metric_columns].mean(numeric_only=True)
    )
    table["policy_sort"] = table["policy"].map(POLICY_SORT_INDEX)
    table = table.sort_values("policy_sort").drop(columns="policy_sort")
    table["outcome_reduction_pct"] = (
        table["crimes_prevented"] / table["baseline_crimes"].replace(0, np.nan)
    ) * 100
    flagged_denominator = (
        table["children_flagged"].replace(0, np.nan)
        if "children_flagged" in table.columns
        else np.nan
    )
    table["harmed_pct"] = (table["children_harmed"] / flagged_denominator) * 100

    display_table = table.rename(
        columns={
            "policy": "Policy",
            "outcome_reduction_pct": OUTCOME_EFFECT_COLUMN,
            "harmed_pct": POLICY_HARM_RATE_COLUMN,
            **POLICY_TOTAL_AVERAGE_LABELS,
        }
    )
    if "children_flagged" in display_table.columns:
        display_table = display_table.drop(columns="children_flagged")
    display_table = display_table[[c for c in ORDERED_POLICY_TOTAL_COLUMNS if c in display_table.columns]]
    for column in display_table.columns:
        if column == "Policy":
            continue
        if column == OUTCOME_EFFECT_COLUMN:
            display_table[column] = display_table[column].map(format_outcome_effect)
            continue
        if column in PERCENT_COLUMNS:
            display_table[column] = display_table[column].map(
                lambda v: "N/A" if pd.isna(v) else f"{v:.1f}%"
            )
        else:
            display_table[column] = display_table[column].map(
                lambda v: "N/A" if pd.isna(v) else f"{v:.1f}"
            )
    return display_table
