import numpy as np
import pandas as pd

from src.constants import RUN_COUNT_COLUMNS, RUN_METRIC_COLUMNS


def clamp_count(value, maximum):
    if pd.isna(value):
        return 0
    return int(max(0, min(maximum, round(float(value)))))


def risk_signal_counts(true_high_risk_count, settings):
    population_size = int(settings["population_size"])
    # signal_error_rate is used symmetrically as both FNR and FPR.
    # FNR = P(not flagged | truly high-risk) = error_rate
    # FPR = P(flagged | truly low-risk)      = error_rate
    # This is a deliberate simplification: one "prediction error" parameter
    # controls both miss rate and false alarm rate equally.
    signal_error_rate = float(settings["prediction_noise"])
    true_high_risk_count = clamp_count(true_high_risk_count, population_size)
    not_true_high_risk_count = population_size - true_high_risk_count

    false_negatives = clamp_count(true_high_risk_count * signal_error_rate, true_high_risk_count)
    false_positives = clamp_count(not_true_high_risk_count * signal_error_rate, not_true_high_risk_count)
    true_positives = true_high_risk_count - false_negatives
    flagged_count = true_positives + false_positives
    return false_positives, false_negatives, flagged_count


def derived_flagged_count(settings):
    population_size = int(settings["population_size"])
    true_high_risk_count = clamp_count(settings["true_high_risk_rate"] * population_size, population_size)
    return risk_signal_counts(true_high_risk_count, settings)[2]


def metric_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return round(float(value), 3)


def compact_aggregate_metrics(run_results):
    averages = run_results[RUN_COUNT_COLUMNS].mean(numeric_only=True)
    return {column: metric_value(averages[column]) for column in RUN_COUNT_COLUMNS}


def optimize_result_frames(run_results):
    # Downcast integer columns to reduce Streamlit session memory.
    run_results = run_results.copy()

    run_integer_columns = [
        "run",
        *RUN_COUNT_COLUMNS,
        "children_flagged",
    ]

    for column in run_integer_columns:
        if column in run_results.columns:
            run_results[column] = pd.to_numeric(run_results[column], downcast="integer")

    for column in ["policy", "llm_model"]:
        if column in run_results.columns:
            run_results[column] = run_results[column].astype("category")

    return run_results


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


def validate_llm_tables(run_results, settings, enforce_bounds=True):
    expected_runs = set(range(1, int(settings["llm_simulation_runs"]) + 1))
    actual_runs = set(run_results["run"].dropna().astype(int).tolist())
    if actual_runs != expected_runs or len(run_results) != len(expected_runs):
        raise ValueError("The model returned incomplete run-level simulation rows.")

    required_run_columns = [column for column in RUN_METRIC_COLUMNS if column != "crimes_prevented"]
    if run_results[required_run_columns].isna().any().any():
        raise ValueError("The model returned missing run-level metric values.")

    if not enforce_bounds:
        return

    population_size = int(settings["population_size"])
    count_columns = [column for column in RUN_COUNT_COLUMNS if column != "crimes_prevented"]
    if (run_results[count_columns] < 0).any().any():
        raise ValueError("The model returned negative run-level counts.")
    if (run_results[count_columns] > population_size).any().any():
        raise ValueError("The model returned run-level counts above population size.")

    expected_prevented = run_results["baseline_crimes"] - run_results["crimes_after_policy"]
    if not expected_prevented.equals(run_results["crimes_prevented"]):
        raise ValueError("The model returned inconsistent prevented-outcome counts.")

    low_risk_count = population_size - run_results["baseline_crimes"]
    if (run_results["false_negatives"] > run_results["baseline_crimes"]).any():
        raise ValueError("The model returned more false negatives than baseline outcomes.")
    if (run_results["false_positives"] > low_risk_count).any():
        raise ValueError("The model returned more false positives than no-outcome agents.")
    if (run_results["crimes_after_policy"] < run_results["false_negatives"]).any():
        raise ValueError("The model returned fewer post-policy outcomes than unreached false negatives.")
    if "children_flagged" in run_results.columns:
        if (run_results["children_helped"] > run_results["children_flagged"]).any():
            raise ValueError("The model returned more helped agents than flagged agents.")
        if (run_results["children_harmed"] > run_results["children_flagged"]).any():
            raise ValueError("The model returned more harmed agents than flagged agents.")


def normalize_llm_metrics(run_results, settings):
    run_results = run_results.copy()
    population_size = int(settings["population_size"])
    count_columns = [column for column in RUN_COUNT_COLUMNS if column != "crimes_prevented"]
    run_results[count_columns] = run_results[count_columns].clip(lower=0, upper=population_size)

    for column in count_columns:
        run_results[column] = run_results[column].round().astype(int)

    run_results["false_negatives"] = np.minimum(
        run_results["false_negatives"],
        run_results["baseline_crimes"],
    ).astype(int)
    low_risk_count = (population_size - run_results["baseline_crimes"]).clip(lower=0)
    run_results["false_positives"] = np.minimum(
        run_results["false_positives"],
        low_risk_count,
    ).astype(int)

    run_results["crimes_after_policy"] = np.maximum(
        run_results["crimes_after_policy"],
        run_results["false_negatives"],
    ).clip(lower=0, upper=population_size).astype(int)

    true_positives = run_results["baseline_crimes"] - run_results["false_negatives"]
    run_results["children_flagged"] = (true_positives + run_results["false_positives"]).clip(
        lower=0,
        upper=population_size,
    ).round().astype(int)
    run_results["children_helped"] = np.minimum(
        run_results["children_helped"],
        run_results["children_flagged"],
    ).astype(int)
    run_results["children_harmed"] = np.minimum(
        run_results["children_harmed"],
        run_results["children_flagged"],
    ).astype(int)

    run_results["crimes_prevented"] = (
        run_results["baseline_crimes"] - run_results["crimes_after_policy"]
    ).round().astype(int)

    return run_results
