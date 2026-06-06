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
        "How many of the 10,000 children would have the predicted serious harmful outcome by age 30 "
        "if no policy were applied. This is the ground truth the prediction tool is trying to identify. "
        "The simulator keeps this between 1% and 10%."
    ),
    "Prediction error rate (%)": (
        "A simplified symmetric error setting: this % of children on the predicted-outcome path are missed, "
        "and the same % of children not on that path become false positives. "
        "Because most children are not on the predicted-outcome path, even a small rate can create many false positives."
    ),
    "Intervention intensity": (
        "Which intensity tier is used when choosing the concrete intervention measure. "
        "Low = light-touch; Medium = structured or recurring; High = intensive, broad, or restrictive."
    ),
}

RESULT_METRIC_DESCRIPTIONS = {
    "Baseline predicted outcomes": (
        "How many children would have the predicted serious harmful outcome if no policy were applied. "
        "In tables, this is the mean count per synthetic run across successful model-agent results."
    ),
    "Positive predictions": (
        "Total children identified as high-risk by the prediction tool — "
        "both correctly and incorrectly identified. This is who the policy acts on."
    ),
    "False positives": (
        "Children flagged as high-risk who would not have had the predicted outcome. "
        "They are exposed to the policy despite not being on the predicted-outcome path."
    ),
    "False negatives": (
        "Children who would have had the predicted outcome but were not flagged. "
        "They receive no intervention under any targeted policy."
    ),
    "Prevented predicted outcomes": (
        "How many fewer predicted outcomes occur compared to doing nothing. "
        "This can be negative if a policy worsens outcomes."
    ),
    "Policy benefit count": (
        "Flagged children whose simulated life-course outcome improves because of the policy. "
        "This is not the same as merely receiving a service, and it can overlap with harm in mixed cases."
    ),
    "Policy harm count": (
        "Flagged children whose simulated life-course outcome worsens because of the policy. "
        "This can occur under any policy type and can overlap with help in mixed cases."
    ),
    "Model-averaged cohort view": (
        "The bubble visualization converts each selected model agent's result into a full synthetic cohort, "
        "takes the mean of those cohort counts across model agents with equal weight, and rounds the categories back to "
        "the configured population size."
    ),
}

CHECK_DESCRIPTIONS = {
    "Policy trade-off": (
        "How many predicted outcomes each policy prevents versus false positives, policy benefits, and policy harms."
    ),
    "Prediction error": (
        "How many false positives occur among children not on the predicted-outcome path "
        "and how many false negatives occur among children on that path."
    ),
}

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"]
MAX_LLM_MODEL_AGENTS = 4
MAX_PARALLEL_LLM_CALLS = 3
MAX_RUN_LOG_SIZE = 5
PROGRESS_NOTE_MIN_SECONDS = 6
DEFAULT_POPULATION_SIZE = 10_000
DEFAULT_TRUE_HIGH_RISK_RATE = 0.01
DEFAULT_PREDICTION_ERROR_RATE = 0.01
TRUE_HIGH_RISK_RATE_MIN = 0.01
TRUE_HIGH_RISK_RATE_MAX = 0.10
PREDICTION_ERROR_RATE_MIN = 0.0
PREDICTION_ERROR_RATE_MAX = 0.10

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
POPULATION_DOT_ANIMATION_SECONDS = 0.18
POPULATION_DOT_STAGGER_GROUP = 12
POPULATION_DOT_STAGGER_SECONDS = 0.0005

RUN_METRIC_LABELS = {
    "baseline_crimes": "Baseline predicted outcomes",
    "children_flagged": "Positive predictions",
    "false_positives": "False positives",
    "false_negatives": "False negatives",
    "crimes_prevented": "Prevented predicted outcomes",
    "children_helped": "Policy benefit count",
    "children_harmed": "Policy harm count",
}
