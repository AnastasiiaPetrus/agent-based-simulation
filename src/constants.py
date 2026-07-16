import os

POLICIES = [
    "Targeted support for flagged children",
    "Surveillance of flagged children",
    "Coercive prevention for flagged children",
]

POLICY_DESCRIPTIONS = {
    "Targeted support for flagged children": (
        "Flagged children are offered voluntary support such as counselling, mentoring, skills, "
        "or practical assistance. Support can improve stability, trust, or opportunity, but poor fit "
        "or unwanted attention can still leave effects neutral, mixed, or harmful."
    ),
    "Surveillance of flagged children": (
        "Flagged children are monitored, reviewed, or recorded more closely. Scrutiny may deter some "
        "target harmful outcomes, but stigma, trust loss, or disengagement can also create mixed or harmful effects."
    ),
    "Coercive prevention for flagged children": (
        "Flagged children face mandatory requirements or restrictions before the target harmful outcome occurs. "
        "Restriction may interrupt some pathways, but loss of autonomy, escalation, or stigma can produce "
        "harm, backfire effects, or no meaningful change if restriction does not change underlying pathways."
    ),
}

SETTING_DESCRIPTIONS = {
    "No-policy outcome rate (%)": (
        "The percentage of children whose path includes the target harmful outcome by age 30 if no policy "
        "is applied. This prevalence, or base rate, is the simulation's assumed no-policy truth that the "
        "prediction tool is trying to detect. Low base rates can make false positives dominate the flagged group."
    ),
    "Symmetric misclassification rate (%)": (
        "A simplified shared-error setting where the same percentage applies to both prediction error types: "
        "children on the target-outcome path who are missed, and children not on that path who are wrongly flagged. "
        "Equal error rates can still produce very unequal error counts when most children are not on the target-outcome path."
    ),
    "Intervention intensity": (
        "Which intensity tier is used when choosing the concrete intervention measure. Higher tiers make the "
        "selected measure more structured, frequent, broad, or restrictive, giving the AI simulation stronger "
        "policy mechanisms to weigh for both benefit and harm."
    ),
}

RESULT_METRIC_DESCRIPTIONS = {
    "No-policy target outcomes": (
        "How many children would have the target harmful outcome if no policy were applied. "
        "This count is computed from the selected no-policy outcome rate."
    ),
    "Flagged by prediction": (
        "Total children flagged by the prediction tool, including both correct and incorrect flags. "
        "This is who the policy acts on."
    ),
    "False positives": (
        "Children flagged by prediction who would not have had the target harmful outcome in the simulation's "
        "assumed no-policy truth. They are exposed to the policy despite not being on the target-outcome path."
    ),
    "False negatives": (
        "Children who would have had the target harmful outcome in the simulation's assumed no-policy truth "
        "but were not flagged. They receive no intervention under any targeted policy."
    ),
    "Net target outcomes prevented": (
        "How many fewer target harmful outcomes occur compared to the no-policy-action baseline. Python derives "
        "this from AI stage scores using fixed scenario coefficients; it can be negative if a policy worsens outcomes."
    ),
    "Precision among flagged children": (
        "Among children flagged by prediction, the share who are in the no-policy outcome group. "
        "This is also called positive predictive value."
    ),
    "Policy benefit count": (
        "The weighted number of flagged children represented by profiles with beneficial AI stage-score points, "
        "as converted by Python's fixed score-to-share rule. This is not the same as merely receiving a service. "
        "The same child can be represented in both benefit and harm counts."
    ),
    "Policy harm count": (
        "The weighted number of flagged children represented by profiles with harmful AI stage-score points, "
        "as converted by Python's fixed score-to-share rule. This can occur under any policy type. "
        "The same child can be represented in both harm and benefit counts."
    ),
    "AI-averaged cohort view": (
        "Different selected AI models may estimate policy effects differently. This view combines the estimates "
        "from successful model responses into one population-sized picture rather than showing a separate panel for every model."
    ),
}

CHECK_DESCRIPTIONS = {
    "Policy trade-off": (
        "Whether a policy's estimated target-outcome reduction is large enough to justify the children wrongly "
        "subjected to the policy, policy-associated benefits, and policy-associated harms it creates."
    ),
    "Misclassification": (
        "How the prediction tool splits its errors: children wrongly flagged despite not being on the "
        "target-outcome path, and children missed despite being on that path."
    ),
}

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"]
MAX_LLM_MODEL_AGENTS = 4
MAX_PARALLEL_LLM_CALLS = 3
MAX_RUN_LOG_SIZE = 5
PROGRESS_NOTE_MIN_SECONDS = 6
DEFAULT_POPULATION_SIZE = 10_000
DEFAULT_NO_POLICY_OUTCOME_RATE = 0.01
DEFAULT_SYMMETRIC_MISCLASSIFICATION_RATE = 0.01
NO_POLICY_OUTCOME_RATE_MIN = 0.01
NO_POLICY_OUTCOME_RATE_MAX = 0.10
SYMMETRIC_MISCLASSIFICATION_RATE_MIN = 0.0
SYMMETRIC_MISCLASSIFICATION_RATE_MAX = 0.10

RUN_METRIC_COLUMNS = [
    "run",
    "baseline_outcomes",
    "outcomes_after_policy",
    "net_outcomes_prevented",
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
    "baseline_outcomes",
    "outcomes_after_policy",
    "net_outcomes_prevented",
    "false_positives",
    "false_negatives",
    "children_helped",
    "children_harmed",
]
POPULATION_DOT_ANIMATION_SECONDS = 0.18
POPULATION_DOT_STAGGER_GROUP = 12
POPULATION_DOT_STAGGER_SECONDS = 0.0005

RUN_METRIC_LABELS = {
    "baseline_outcomes": "No-policy target outcomes",
    "children_flagged": "Flagged by prediction",
    "false_positives": "False positives",
    "false_negatives": "False negatives",
    "net_outcomes_prevented": "Net target outcomes prevented",
    "children_helped": "Policy benefit count",
    "children_harmed": "Policy harm count",
}
