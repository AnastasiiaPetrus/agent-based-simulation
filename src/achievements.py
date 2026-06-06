from datetime import datetime

from src.constants import NO_POLICY_OUTCOME_RATE_MAX, POLICIES

ACHIEVEMENTS = [
    {
        "id": "first_run",
        "icon": "🚀",
        "name": "First Simulation",
        "description": "Complete your first simulation run.",
    },
    {
        "id": "do_no_harm",
        "icon": "💚",
        "name": "Do No Harm",
        "description": "Achieve a policy harm count of zero under Targeted Support.",
    },
    {
        "id": "full_comparison",
        "icon": "📊",
        "name": "Full Picture",
        "description": "Compare all three policies in a single session.",
    },
    {
        "id": "false_alarm",
        "icon": "🚨",
        "name": "False Alarm",
        "description": "Mean false positives exceed mean net target outcomes prevented.",
    },
    {
        "id": "tinkerer",
        "icon": "🔬",
        "name": "Tinkerer",
        "description": "Run 7 or more simulations in a single session.",
    },
    {
        "id": "outcome_preventer",
        "icon": "🛡️",
        "name": "Outcome Detour",
        "description": "Reach a mean of 20+ net target outcomes prevented per run under any policy.",
    },
    {
        "id": "outcome_reducer",
        "icon": "💥",
        "name": "Risk Signal Wrangler",
        "description": "Achieve a 25%+ mean target-outcome reduction under any policy.",
    },
    {
        "id": "base_rate_trap",
        "icon": "🪤",
        "name": "Base Rate Trap",
        "description": "Observe more false positives than true positives.",
    },
    {
        "id": "schrodinger",
        "icon": "🧩",
        "name": "Base Rate Warning",
        "description": "False positives outnumber false negatives by at least 10 to 1.",
    },
    {
        "id": "helping_hundreds",
        "icon": "🤝",
        "name": "Helping Hundreds",
        "description": "Reach a mean policy benefit count of 200+ per run under Targeted Support.",
    },
    {
        "id": "overreaction",
        "icon": "😱",
        "name": "Overreaction",
        "description": "Flag at least five times as many children as would have the target outcome.",
    },
    {
        "id": "sharp_signal",
        "icon": "🎯",
        "name": "Sharp Signal",
        "description": "Run a simulation with symmetric misclassification rate at most 2%.",
    },
    {
        "id": "high_risk_world",
        "icon": "⚠️",
        "name": "Base Rate Bonanza",
        "description": "Simulate a population at the maximum no-policy outcome rate.",
    },
    {
        "id": "night_owl",
        "icon": "🦉",
        "name": "It's not an owl",
        "description": "Run a simulation at exactly the right moment.",
    },
]

ACHIEVEMENT_INDEX = {a["id"]: a for a in ACHIEVEMENTS}


def check_achievements(combined_run_results, settings, simulation_count=1):
    """Return set of achievement IDs earned in this simulation result."""
    earned = {"first_run"}

    has_policy_col = "policy" in combined_run_results.columns
    policies_present = (
        set(combined_run_results["policy"].dropna().unique()) if has_policy_col else set()
    )
    targeted = (
        combined_run_results[combined_run_results["policy"] == "Targeted support for flagged children"]
        if has_policy_col else combined_run_results.iloc[0:0]
    )

    avg_baseline = combined_run_results["baseline_outcomes"].mean()
    avg_fp = combined_run_results["false_positives"].mean()
    avg_fn = combined_run_results["false_negatives"].mean()
    avg_prevented = combined_run_results["net_outcomes_prevented"].mean()
    avg_tp = avg_baseline - avg_fn

    if set(POLICIES).issubset(policies_present):
        earned.add("full_comparison")

    if not targeted.empty and targeted["children_harmed"].mean() == 0:
        earned.add("do_no_harm")

    if avg_fp > avg_prevented:
        earned.add("false_alarm")

    if simulation_count >= 7:
        earned.add("tinkerer")

    if has_policy_col:
        policy_avg_prevented = combined_run_results.groupby("policy", observed=True)[
            "net_outcomes_prevented"
        ].mean()
        max_prevented = policy_avg_prevented.max()
    else:
        policy_avg_prevented = None
        max_prevented = avg_prevented

    if max_prevented >= 20:
        earned.add("outcome_preventer")

    if has_policy_col:
        policy_avg_baseline = combined_run_results.groupby("policy", observed=True)[
            "baseline_outcomes"
        ].mean()
        reduction_rates = policy_avg_prevented / policy_avg_baseline.replace(0, float("nan"))
        if reduction_rates.max() >= 0.25:
            earned.add("outcome_reducer")
    else:
        if avg_baseline > 0 and max_prevented / avg_baseline >= 0.25:
            earned.add("outcome_reducer")

    if avg_fp > avg_tp:
        earned.add("base_rate_trap")

    if avg_fp > 0 and (avg_fn == 0 or avg_fp >= (10 * avg_fn)):
        earned.add("schrodinger")

    if not targeted.empty and targeted["children_helped"].mean() >= 200:
        earned.add("helping_hundreds")

    if "children_flagged" in combined_run_results.columns:
        flagged_to_baseline_ratio = (
            combined_run_results["children_flagged"]
            / combined_run_results["baseline_outcomes"].replace(0, float("nan"))
        )
        if flagged_to_baseline_ratio.max() >= 5:
            earned.add("overreaction")

    if float(settings["symmetric_error_rate"]) <= 0.02:
        earned.add("sharp_signal")

    if float(settings["no_policy_outcome_rate"]) >= NO_POLICY_OUTCOME_RATE_MAX:
        earned.add("high_risk_world")

    now = datetime.now()
    if now.hour == 3 and 30 <= now.minute <= 36:
        earned.add("night_owl")

    return earned
