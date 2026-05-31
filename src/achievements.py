from datetime import datetime

from src.constants import POLICIES

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
        "description": "Achieve zero children harmed under Targeted Support.",
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
        "description": "More children wrongly flagged on average than offenses prevented.",
    },
    {
        "id": "tinkerer",
        "icon": "🔬",
        "name": "Tinkerer",
        "description": "Run 3 or more simulations in a single session.",
    },
    {
        "id": "crime_preventer",
        "icon": "🛡️",
        "name": "Crime Fighter",
        "description": "Prevent an average of 20+ offenses per run under any policy.",
    },
    {
        "id": "crime_crusher",
        "icon": "💥",
        "name": "Crime Crusher",
        "description": "Achieve 25%+ average crime reduction under any policy.",
    },
    {
        "id": "base_rate_trap",
        "icon": "🪤",
        "name": "Base Rate Trap",
        "description": "Observe more wrongly flagged children than correctly flagged ones.",
    },
    {
        "id": "schrodinger",
        "icon": "🐱",
        "name": "Schrödinger",
        "description": "Total prediction errors (FP + FN) exceed the number of children who would actually offend.",
    },
    {
        "id": "helping_hundreds",
        "icon": "🤝",
        "name": "Helping Hundreds",
        "description": "Average 200+ children helped per run under Targeted Support.",
    },
    {
        "id": "overreaction",
        "icon": "😱",
        "name": "Overreaction",
        "description": "More than half the population (500+) flagged as high-risk in a single run.",
    },
    {
        "id": "sharp_signal",
        "icon": "🎯",
        "name": "Sharp Signal",
        "description": "Run a simulation with prediction error rate at most 2%.",
    },
    {
        "id": "high_risk_world",
        "icon": "⚠️",
        "name": "High-Risk World",
        "description": "Simulate a population where 30%+ of children are truly high-risk.",
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
        combined_run_results[combined_run_results["policy"] == "Targeted support for high-risk children"]
        if has_policy_col else combined_run_results.iloc[0:0]
    )

    if set(POLICIES).issubset(policies_present):
        earned.add("full_comparison")

    if not targeted.empty and targeted["children_harmed"].mean() == 0:
        earned.add("do_no_harm")

    avg_fp = combined_run_results["false_positives"].mean()
    avg_prevented = combined_run_results["crimes_prevented"].mean()
    if avg_fp > avg_prevented:
        earned.add("false_alarm")

    if simulation_count >= 3:
        earned.add("tinkerer")

    policy_avg_prevented = (
        combined_run_results.groupby("policy", observed=True)["crimes_prevented"].mean()
        if has_policy_col
        else combined_run_results["crimes_prevented"].mean().__class__(
            [combined_run_results["crimes_prevented"].mean()]
        )
    )
    if hasattr(policy_avg_prevented, "max") and policy_avg_prevented.max() >= 20:
        earned.add("crime_preventer")

    policy_avg_baseline = (
        combined_run_results.groupby("policy", observed=True)["baseline_crimes"].mean()
        if has_policy_col else None
    )
    if policy_avg_baseline is not None and (policy_avg_baseline > 0).any():
        reduction_rates = policy_avg_prevented / policy_avg_baseline.replace(0, float("nan"))
        if reduction_rates.max() >= 0.25:
            earned.add("crime_crusher")

    avg_fn = combined_run_results["false_negatives"].mean()
    avg_tp = combined_run_results["baseline_crimes"].mean() - avg_fn
    if avg_fp > avg_tp:
        earned.add("base_rate_trap")

    if (avg_fp + avg_fn) > combined_run_results["baseline_crimes"].mean():
        earned.add("schrodinger")

    if not targeted.empty and targeted["children_helped"].mean() >= 200:
        earned.add("helping_hundreds")

    if "children_flagged" in combined_run_results.columns:
        if combined_run_results["children_flagged"].max() > 500:
            earned.add("overreaction")

    if float(settings["prediction_noise"]) <= 0.02:
        earned.add("sharp_signal")

    if float(settings["true_high_risk_rate"]) >= 0.30:
        earned.add("high_risk_world")

    now = datetime.now()
    if now.hour == 3 and 30 <= now.minute <= 36:
        earned.add("night_owl")

    return earned
