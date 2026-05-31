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
        "id": "heavy_hand",
        "icon": "⛓️",
        "name": "Heavy Hand",
        "description": "100+ children harmed on average under Coercive intervention.",
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
        "id": "helping_hundreds",
        "icon": "🤝",
        "name": "Helping Hundreds",
        "description": "Average 200+ children helped per run under Targeted Support.",
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
]

ACHIEVEMENT_INDEX = {a["id"]: a for a in ACHIEVEMENTS}


def check_achievements(combined_run_results, settings):
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
    coercive = (
        combined_run_results[combined_run_results["policy"] == "Coercive preventive intervention for high-risk children"]
        if has_policy_col else combined_run_results.iloc[0:0]
    )

    if set(POLICIES).issubset(policies_present):
        earned.add("full_comparison")

    if not targeted.empty and targeted["children_harmed"].mean() == 0:
        earned.add("do_no_harm")

    if not coercive.empty and coercive["children_harmed"].mean() >= 100:
        earned.add("heavy_hand")

    policy_avg_prevented = (
        combined_run_results.groupby("policy")["crimes_prevented"].mean()
        if has_policy_col
        else combined_run_results["crimes_prevented"].mean().__class__([combined_run_results["crimes_prevented"].mean()])
    )
    if (
        hasattr(policy_avg_prevented, "max")
        and policy_avg_prevented.max() >= 20
    ):
        earned.add("crime_preventer")

    policy_avg_baseline = (
        combined_run_results.groupby("policy")["baseline_crimes"].mean()
        if has_policy_col
        else None
    )
    if policy_avg_baseline is not None and (policy_avg_baseline > 0).any():
        reduction_rates = policy_avg_prevented / policy_avg_baseline.replace(0, float("nan"))
        if reduction_rates.max() >= 0.25:
            earned.add("crime_crusher")

    if "false_positives" in combined_run_results.columns and "false_negatives" in combined_run_results.columns:
        avg_fp = combined_run_results["false_positives"].mean()
        avg_fn = combined_run_results["false_negatives"].mean()
        avg_tp = combined_run_results["baseline_crimes"].mean() - avg_fn
        if avg_fp > avg_tp:
            earned.add("base_rate_trap")

    if not targeted.empty and targeted["children_helped"].mean() >= 200:
        earned.add("helping_hundreds")

    if float(settings["prediction_noise"]) <= 0.02:
        earned.add("sharp_signal")

    if float(settings["true_high_risk_rate"]) >= 0.30:
        earned.add("high_risk_world")

    return earned
