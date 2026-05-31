import os

POLICIES = [
    "Targeted support for high-risk children",
    "Surveillance of high-risk children",
    "Coercive preventive intervention for high-risk children",
]

POLICY_ORDER = list(POLICIES)

POLICY_DESCRIPTIONS = {
    "Targeted support for high-risk children": (
        "Flagged children receive voluntary help — counselling, mentoring, or social support. "
        "Children wrongly flagged get unnecessary help; children the system missed get nothing."
    ),
    "Surveillance of high-risk children": (
        "Flagged children are monitored without consent. "
        "May deter some offenses, but creates stigma and erodes trust — even for those flagged by mistake."
    ),
    "Coercive preventive intervention for high-risk children": (
        "Flagged children are restricted before committing any offense. "
        "Strongest crime reduction — and the highest harm, especially for children wrongly flagged."
    ),
}

SETTING_DESCRIPTIONS = {
    "Percentage of true high-risk children (%)": (
        "How many of the 1 000 children would actually go on to commit a violent offense by age 30 "
        "if no policy were applied. This is the ground truth the prediction tool is trying to identify."
    ),
    "Prediction error rate (%)": (
        "How often the prediction is wrong. This % of truly high-risk children are missed, "
        "and the same % of low-risk children are wrongly flagged. "
        "Even a small error rate creates many wrong flags, because low-risk children far outnumber high-risk ones."
    ),
    "Intervention strength": (
        "How strongly the policy is applied. "
        "Low = mild effect on crime, help, and harm; High = strong effect on all three."
    ),
}

RESULT_METRIC_DESCRIPTIONS = {
    "Would offend without intervention": (
        "How many children would commit a violent offense if no policy were applied. "
        "Every other metric compares against this number."
    ),
    "Flagged as high-risk": (
        "Total children identified as high-risk by the prediction tool — "
        "both correctly and incorrectly identified. This is who the policy acts on."
    ),
    "Wrongly flagged": (
        "Children flagged as high-risk who would not have offended. "
        "They bear the full cost of the policy for no reason."
    ),
    "Missed by prediction": (
        "Children who would have offended but were not flagged. "
        "They receive no intervention under any targeted policy."
    ),
    "Offenses prevented": (
        "How many fewer offenses occur compared to doing nothing — "
        "the direct benefit of the policy."
    ),
    "Received support": (
        "Children who received help under the targeted support policy. "
        "Includes both correctly and wrongly flagged children."
    ),
    "Harmed by intervention": (
        "Children harmed by the policy itself — through surveillance stigma or coercive restriction. "
        "Zero under targeted support."
    ),
}

CHECK_DESCRIPTIONS = {
    "Policy trade-off": (
        "How many offenses each policy prevents versus how many children it wrongly flags or harms."
    ),
    "Prediction error": (
        "How many children are wrongly flagged (flagged despite not being at risk) "
        "and how many are missed (at risk but not flagged)."
    ),
}

DISTRICTS = ["A", "B", "C"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"]
DEFAULT_DEBRIEF_WORD_LIMIT = 250
MAX_RUN_LOG_SIZE = 5

RUN_METRIC_COLUMNS = [
    "run",
    "baseline_crimes",
    "crimes_after_policy",
    "crimes_prevented",
    "false_positives",
    "false_negatives",
    "children_helped",
    "children_harmed",
]
DISTRICT_METRIC_COLUMNS = ["run", "district", "false_positives", "children_harmed", "crimes"]
RUN_COUNT_COLUMNS = [
    "baseline_crimes",
    "crimes_after_policy",
    "crimes_prevented",
    "false_positives",
    "false_negatives",
    "children_helped",
    "children_harmed",
]
DISTRICT_COUNT_COLUMNS = ["false_positives", "children_harmed", "crimes"]
NON_NEGATIVE_RUN_COLUMNS = RUN_COUNT_COLUMNS
NON_NEGATIVE_DISTRICT_COLUMNS = DISTRICT_COUNT_COLUMNS
DEFAULT_TRUE_HIGH_RISK_RATE = 0.125
POPULATION_DOT_ANIMATION_SECONDS = 0.18
POPULATION_DOT_PULSE_SECONDS = 2.6
POPULATION_DOT_STAGGER_GROUP = 12
POPULATION_DOT_STAGGER_SECONDS = 0.0005
POLICY_EFFECT_REDUCTION_RATES = {"Low": 0.05, "Medium": 0.15, "High": 0.28}

RUN_METRIC_LABELS = {
    "baseline_crimes": "Would offend without intervention",
    "children_flagged": "Flagged as high-risk",
    "false_positives": "Wrongly flagged",
    "false_negatives": "Missed by prediction",
    "crimes_prevented": "Offenses prevented",
    "children_helped": "Received support",
    "children_harmed": "Harmed by intervention",
}
