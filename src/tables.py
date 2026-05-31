import numpy as np
import pandas as pd

from src.constants import POLICY_ORDER, RUN_METRIC_LABELS


def average_results_table(run_results):
    metric_order = [column for column in RUN_METRIC_LABELS if column in run_results.columns]
    averages = run_results[metric_order].mean(numeric_only=True)
    table = averages.rename(index=RUN_METRIC_LABELS).reset_index()
    table.columns = ["Metric", "Average per synthetic run"]
    return table


def combined_policy_totals_table(run_results, population_size):
    avg_metric_labels = {
        "baseline_crimes": "Would offend without policy",
        "false_positives": "Wrongly flagged",
        "false_negatives": "Missed by prediction",
        "children_helped": "Receiving support",
        "children_harmed": "Harmed by intervention",
    }
    required_metric_columns = [*avg_metric_labels, "crimes_prevented", "children_flagged"]
    metric_columns = [column for column in required_metric_columns if column in run_results.columns]
    table = (
        run_results.groupby("policy", as_index=False)[metric_columns].mean(numeric_only=True)
    )
    table["policy_sort"] = table["policy"].map({policy: index for index, policy in enumerate(POLICY_ORDER)})
    table = table.sort_values("policy_sort").drop(columns="policy_sort")
    table["crime_reduction_pct"] = (
        table["crimes_prevented"] / table["baseline_crimes"].replace(0, np.nan)
    ) * 100
    flagged_denominator = (
        table["children_flagged"].replace(0, np.nan)
        if "children_flagged" in table.columns
        else max(int(population_size), 1)
    )
    table["harmed_pct"] = (table["children_harmed"] / flagged_denominator) * 100

    display_table = table.rename(
        columns={
            "policy": "Policy",
            "crime_reduction_pct": "Offense reduction (%)",
            "harmed_pct": "Harmed (% of flagged)",
            **avg_metric_labels,
        }
    )
    if "children_flagged" in display_table.columns:
        display_table = display_table.drop(columns="children_flagged")
    ordered_columns = [
        "Policy",
        "Offense reduction (%)",
        "Harmed (% of flagged)",
        "Would offend without policy",
        "Wrongly flagged",
        "Missed by prediction",
        "Receiving support",
        "Harmed by intervention",
    ]
    display_table = display_table[[c for c in ordered_columns if c in display_table.columns]]
    for column in display_table.columns:
        if column == "Policy":
            continue
        if column in {"Offense reduction (%)", "Harmed (% of flagged)"}:
            display_table[column] = display_table[column].map(
                lambda v: "N/A" if pd.isna(v) else f"{v:.1f}%"
            )
        else:
            display_table[column] = display_table[column].map(
                lambda v: "N/A" if pd.isna(v) else f"{v:.1f}"
            )
    return display_table
