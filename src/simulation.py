import numpy as np
import pandas as pd

from src.constants import (
    DISTRICTS,
    DISTRICT_COUNT_COLUMNS,
    DISTRICT_METRIC_COLUMNS,
    NON_NEGATIVE_DISTRICT_COLUMNS,
    POLICY_EFFECT_REDUCTION_RATES,
    RUN_COUNT_COLUMNS,
    RUN_METRIC_COLUMNS,
)


def clamp_count(value, maximum):
    if pd.isna(value):
        return 0
    return int(max(0, min(maximum, round(float(value)))))


def baseline_count_for_run(run_number, run_numbers, settings):
    population_size = int(settings["population_size"])
    target = float(settings["true_high_risk_rate"]) * population_size
    sorted_runs = sorted(set(int(run) for run in run_numbers))
    if len(sorted_runs) <= 1:
        return clamp_count(target, population_size)

    # Include settings in the seed so different scenarios produce independent variation patterns.
    settings_fingerprint = int(settings["true_high_risk_rate"] * 1000) * 7 + int(float(settings["prediction_noise"]) * 100) * 13
    rng = np.random.default_rng(seed=int(run_number) + settings_fingerprint)
    variation = rng.uniform(-0.06, 0.06)
    return clamp_count(target * (1 + variation), population_size)


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

    # baseline_crimes = children who would offend without intervention = true_high_risk_count.
    risk_signal_results = run_results["baseline_crimes"].apply(
        lambda baseline_count: risk_signal_counts(int(baseline_count), settings)
    )
    run_results["false_positives"] = [counts[0] for counts in risk_signal_results]
    run_results["false_negatives"] = [counts[1] for counts in risk_signal_results]
    flagged_counts = pd.Series(
        [counts[2] for counts in risk_signal_results],
        index=run_results.index,
    )

    run_results["children_flagged"] = flagged_counts.round().astype(int)

    if policy == "Targeted support for high-risk children":
        run_results["children_helped"] = flagged_counts
        run_results["children_harmed"] = 0
        district_results["children_harmed"] = 0
    elif policy == "Surveillance of high-risk children":
        run_results["children_helped"] = 0
        run_results["children_harmed"] = np.minimum(run_results["children_harmed"], flagged_counts)
    elif policy == "Coercive preventive intervention for high-risk children":
        run_results["children_helped"] = 0
        run_results["children_harmed"] = np.minimum(run_results["children_harmed"], flagged_counts)

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
