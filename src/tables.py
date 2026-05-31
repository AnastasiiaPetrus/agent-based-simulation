import numpy as np
import pandas as pd

from src.constants import POLICY_ORDER, RUN_METRIC_LABELS


def average_results_table(run_results):
    metric_order = [column for column in RUN_METRIC_LABELS if column in run_results.columns]
    averages = run_results[metric_order].mean(numeric_only=True)
    table = averages.rename(index=RUN_METRIC_LABELS).reset_index()
    table.columns = ["Metric", "Average per synthetic run"]
    return table


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
