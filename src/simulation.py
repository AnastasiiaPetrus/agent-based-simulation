import numpy as np
import pandas as pd

from src.constants import POLICY_EFFECT_COLUMNS, RUN_COUNT_COLUMNS, RUN_METRIC_COLUMNS


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


def prediction_base_rows(settings):
    population_size = int(settings["population_size"])
    run_count = int(settings["llm_simulation_runs"])
    baseline_crimes = clamp_count(settings["true_high_risk_rate"] * population_size, population_size)
    false_positives, false_negatives, children_flagged = risk_signal_counts(baseline_crimes, settings)
    true_positives = baseline_crimes - false_negatives
    true_negatives = population_size - baseline_crimes - false_positives
    unflagged_agents = population_size - children_flagged
    relevant_agents = children_flagged + false_negatives

    return [
        {
            "run": run,
            "baseline_crimes": baseline_crimes,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "children_flagged": children_flagged,
            "unflagged_agents": unflagged_agents,
            "relevant_agents": relevant_agents,
        }
        for run in range(1, run_count + 1)
    ]


def prediction_base_frame(settings):
    return pd.DataFrame(prediction_base_rows(settings))


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


def clean_llm_policy_effects(raw_rows):
    rows = []
    for row in raw_rows:
        cleaned = {}
        for column in POLICY_EFFECT_COLUMNS:
            value = row.get(column)
            cleaned[column] = np.nan if value is None else value
        rows.append(cleaned)

    if not rows:
        return pd.DataFrame(columns=POLICY_EFFECT_COLUMNS)

    policy_effects = pd.DataFrame(rows)
    for column in POLICY_EFFECT_COLUMNS:
        policy_effects[column] = pd.to_numeric(policy_effects[column], errors="coerce")
    policy_effects["run"] = policy_effects["run"].astype("Int64")
    return policy_effects.sort_values("run").reset_index(drop=True)


def validate_llm_policy_effects(policy_effects, settings, enforce_bounds=True):
    expected_runs = set(range(1, int(settings["llm_simulation_runs"]) + 1))
    actual_runs = set(policy_effects["run"].dropna().astype(int).tolist())
    if actual_runs != expected_runs or len(policy_effects) != len(expected_runs):
        raise ValueError("The model returned incomplete policy-effect rows.")

    if policy_effects[POLICY_EFFECT_COLUMNS].isna().any().any():
        raise ValueError("The model returned missing policy-effect values.")

    if not enforce_bounds:
        return

    effect_columns = [column for column in POLICY_EFFECT_COLUMNS if column != "run"]
    if (policy_effects[effect_columns] < 0).any().any():
        raise ValueError("The model returned negative policy-effect counts.")

    base = prediction_base_frame(settings)
    checked = policy_effects.merge(base, on="run", how="left", validate="one_to_one")
    if (checked["prevented_outcomes"] > checked["true_positives"]).any():
        raise ValueError("The model returned more prevented outcomes than flagged true positives.")
    if (checked["policy_caused_outcomes"] > checked["false_positives"]).any():
        raise ValueError("The model returned more policy-caused outcomes than flagged false positives.")
    if (checked["children_helped"] > checked["children_flagged"]).any():
        raise ValueError("The model returned more helped agents than flagged agents.")
    if (checked["children_harmed"] > checked["children_flagged"]).any():
        raise ValueError("The model returned more harmed agents than flagged agents.")


def normalize_llm_policy_effects(policy_effects, settings):
    policy_effects = policy_effects.copy()
    effect_columns = [column for column in POLICY_EFFECT_COLUMNS if column != "run"]
    policy_effects[effect_columns] = policy_effects[effect_columns].clip(lower=0)

    for column in effect_columns:
        policy_effects[column] = policy_effects[column].round().astype(int)

    base = prediction_base_frame(settings)
    merged = policy_effects.merge(base, on="run", how="left", validate="one_to_one")
    policy_effects["prevented_outcomes"] = np.minimum(
        merged["prevented_outcomes"],
        merged["true_positives"],
    ).astype(int)
    policy_effects["policy_caused_outcomes"] = np.minimum(
        merged["policy_caused_outcomes"],
        merged["false_positives"],
    ).astype(int)
    policy_effects["children_helped"] = np.minimum(
        merged["children_helped"],
        merged["children_flagged"],
    ).astype(int)
    policy_effects["children_harmed"] = np.minimum(
        merged["children_harmed"],
        merged["children_flagged"],
    ).astype(int)
    policy_effects["run"] = policy_effects["run"].astype(int)
    return policy_effects


def run_results_from_policy_effects(policy_effects, settings):
    base = prediction_base_frame(settings)
    merged = base.merge(policy_effects, on="run", how="left", validate="one_to_one")
    run_results = pd.DataFrame(
        {
            "run": merged["run"].astype(int),
            "baseline_crimes": merged["baseline_crimes"].astype(int),
            "crimes_after_policy": (
                merged["baseline_crimes"]
                - merged["prevented_outcomes"]
                + merged["policy_caused_outcomes"]
            ).astype(int),
            "false_positives": merged["false_positives"].astype(int),
            "false_negatives": merged["false_negatives"].astype(int),
            "children_helped": merged["children_helped"].astype(int),
            "children_harmed": merged["children_harmed"].astype(int),
            "children_flagged": merged["children_flagged"].astype(int),
        }
    )
    run_results["crimes_prevented"] = (
        run_results["baseline_crimes"] - run_results["crimes_after_policy"]
    ).astype(int)
    return run_results[
        [
            "run",
            "baseline_crimes",
            "crimes_after_policy",
            "crimes_prevented",
            "false_positives",
            "false_negatives",
            "children_helped",
            "children_harmed",
            "children_flagged",
        ]
    ]


def validate_llm_tables(run_results, settings, enforce_bounds=True):
    expected_runs = set(range(1, int(settings["llm_simulation_runs"]) + 1))
    actual_runs = set(run_results["run"].dropna().astype(int).tolist())
    if actual_runs != expected_runs or len(run_results) != len(expected_runs):
        raise ValueError("The computed run-level result table is incomplete.")

    required_run_columns = [column for column in RUN_METRIC_COLUMNS if column != "crimes_prevented"]
    if run_results[required_run_columns].isna().any().any():
        raise ValueError("The computed run-level result table has missing metric values.")

    if not enforce_bounds:
        return

    population_size = int(settings["population_size"])
    count_columns = [column for column in RUN_COUNT_COLUMNS if column != "crimes_prevented"]
    if (run_results[count_columns] < 0).any().any():
        raise ValueError("The computed run-level result table has negative counts.")
    if (run_results[count_columns] > population_size).any().any():
        raise ValueError("The computed run-level result table has counts above population size.")

    expected_prevented = run_results["baseline_crimes"] - run_results["crimes_after_policy"]
    if not expected_prevented.equals(run_results["crimes_prevented"]):
        raise ValueError("The computed run-level result table has inconsistent prevented-outcome counts.")

    low_risk_count = population_size - run_results["baseline_crimes"]
    if (run_results["false_negatives"] > run_results["baseline_crimes"]).any():
        raise ValueError("The computed run-level result table has more false negatives than baseline outcomes.")
    if (run_results["false_positives"] > low_risk_count).any():
        raise ValueError("The computed run-level result table has more false positives than no-outcome agents.")
    if (run_results["crimes_after_policy"] < run_results["false_negatives"]).any():
        raise ValueError("The computed run-level result table has fewer post-policy outcomes than unreached false negatives.")
    if "children_flagged" in run_results.columns:
        if (run_results["children_helped"] > run_results["children_flagged"]).any():
            raise ValueError("The computed run-level result table has more helped agents than flagged agents.")
        if (run_results["children_harmed"] > run_results["children_flagged"]).any():
            raise ValueError("The computed run-level result table has more harmed agents than flagged agents.")
