import os

POLICIES = [
    "Targeted support for high-risk children",
    "Surveillance of high-risk children",
    "Coercive preventive intervention for high-risk children",
]

POLICY_DESCRIPTIONS = {
    "Targeted support for high-risk children": (
        "Flagged children are offered voluntary support such as counselling, mentoring, skills, "
        "or practical assistance. The simulated outcome can be helpful, neutral, mixed, or harmful."
    ),
    "Surveillance of high-risk children": (
        "Flagged children are monitored, reviewed, or recorded more closely. The simulated outcome can "
        "include deterrence, stigma, trust loss, no change, or mixed effects."
    ),
    "Coercive preventive intervention for high-risk children": (
        "Flagged children face mandatory requirements or restrictions before the predicted outcome occurs. "
        "The simulation may show prevention, harm, backfire effects, or no meaningful change."
    ),
}

SETTING_DESCRIPTIONS = {
    "Percentage of true high-risk children (%)": (
        "How many of the 1 000 children would have the predicted serious harmful outcome by age 30 "
        "if no policy were applied. This is the ground truth the prediction tool is trying to identify."
    ),
    "Prediction error rate (%)": (
        "How often the prediction is wrong. This % of truly high-risk children are missed, "
        "and the same % of low-risk children are wrongly flagged. "
        "Even a small error rate creates many wrong flags, because low-risk children far outnumber high-risk ones."
    ),
    "Intervention intensity": (
        "Which intensity tier is used when choosing the concrete intervention measure. "
        "Low = light-touch; Medium = structured or recurring; High = intensive, broad, or restrictive."
    ),
}

RESULT_METRIC_DESCRIPTIONS = {
    "Would offend without intervention": (
        "How many children would have the predicted serious harmful outcome if no policy were applied. "
        "In average tables, this is an average count per synthetic run."
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
        "How many fewer predicted outcomes occur compared to doing nothing. "
        "This can be negative if a policy worsens outcomes."
    ),
    "Helped by policy": (
        "Flagged children whose simulated life-course outcome improves because of the policy. "
        "This is not the same as merely receiving a service, and it can overlap with harm in mixed cases."
    ),
    "Harmed by policy": (
        "Flagged children whose simulated life-course outcome worsens because of the policy. "
        "This can occur under any policy type and can overlap with help in mixed cases."
    ),
}

CHECK_DESCRIPTIONS = {
    "Policy trade-off": (
        "How many predicted outcomes each policy prevents versus how many children it wrongly flags, helps, or harms."
    ),
    "Prediction error": (
        "How many children are wrongly flagged (flagged despite not being at risk) "
        "and how many are missed (at risk but not flagged)."
    ),
}

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"]
MAX_LLM_MODEL_AGENTS = 2
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
POLICY_EFFECT_COLUMNS = [
    "run",
    "prevented_outcomes",
    "policy_caused_outcomes",
    "children_helped",
    "children_harmed",
]
RUN_COUNT_COLUMNS = [
    "baseline_crimes",
    "crimes_after_policy",
    "crimes_prevented",
    "false_positives",
    "false_negatives",
    "children_helped",
    "children_harmed",
]
DEFAULT_TRUE_HIGH_RISK_RATE = 0.125
POPULATION_DOT_ANIMATION_SECONDS = 0.18
POPULATION_DOT_STAGGER_GROUP = 12
POPULATION_DOT_STAGGER_SECONDS = 0.0005

RUN_METRIC_LABELS = {
    "baseline_crimes": "Would offend without intervention",
    "children_flagged": "Flagged as high-risk",
    "false_positives": "Wrongly flagged",
    "false_negatives": "Missed by prediction",
    "crimes_prevented": "Offenses prevented",
    "children_helped": "Helped by policy",
    "children_harmed": "Harmed by policy",
}
