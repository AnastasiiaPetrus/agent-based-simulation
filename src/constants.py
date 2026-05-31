import os

POLICIES = [
    "Targeted support for high-risk children",
    "Surveillance of high-risk children",
    "Coercive preventive intervention for high-risk children",
]

POLICY_ORDER = list(POLICIES)

POLICY_DESCRIPTIONS = {
    "Targeted support for high-risk children": (
        "Voluntary support only for flagged children; error-sensitive."
    ),
    "Surveillance of high-risk children": (
        "Monitoring for flagged children; may reduce crime but exposes children to surveillance."
    ),
    "Coercive preventive intervention for high-risk children": (
        "Restriction before any act; highest ethical concern."
    ),
}

SETTING_DESCRIPTIONS = {
    "Percentage of true high-risk children (%)": "Share of the 1 000 synthetic children whose no-intervention trajectory would include the modeled offense.",
    "Prediction error rate (%)": "How noisy the risk signal is. It creates missed true high-risk children and incorrectly flagged children outside the true high-risk group.",
    "Intervention strength": "How intensively the chosen policy is applied — scales the simulated effect on offenses, support reach, and harm.",
}

RESULT_METRIC_DESCRIPTIONS = {
    "Children who would offend (no intervention)": "Baseline count — synthetic children whose trajectory leads to a violent offense with no policy applied. All other metrics are relative to this.",
    "Children flagged by risk signal": "Total children identified as high-risk by the signal (true positives + false positives). This is the denominator for interpreting the two error metrics below.",
    "Children incorrectly flagged (false positives)": "Subset of flagged children who would NOT have offended. They bear the full cost of the policy with no corresponding benefit.",
    "Children missed by risk signal (false negatives)": "Children who WOULD have offended but were not flagged. They receive no intervention under any targeted policy.",
    "Offenses prevented by policy": "Baseline offenses minus offenses remaining after the policy. Derived from the LLM's simulated reduction rate applied to the baseline.",
    "Children receiving support": "Children offered voluntary support under the targeted support policy. Equals the total flagged count — includes both true and false positives.",
    "Children exposed to harmful intervention": "Children whose trajectory is adversely affected by surveillance (stigma, eroded trust) or coercive restriction (loss of liberty and opportunity). Always zero under targeted support.",
}

CHECK_DESCRIPTIONS = {
    "Policy trade-off": "Crimes prevented vs. children incorrectly flagged and children exposed to harm.",
    "Prediction error": "False positives (flagged without basis) and false negatives (missed entirely).",
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
    "baseline_crimes": "Children who would offend (no intervention)",
    "children_flagged": "Children flagged by risk signal",
    "false_positives": "Children incorrectly flagged (false positives)",
    "false_negatives": "Children missed by risk signal (false negatives)",
    "crimes_prevented": "Offenses prevented by policy",
    "children_helped": "Children receiving support",
    "children_harmed": "Children exposed to harmful intervention",
}
