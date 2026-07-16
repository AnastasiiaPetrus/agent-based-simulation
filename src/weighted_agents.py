import hashlib
import json
import random

from src.constants import POLICIES
from src.life_contexts import LIFE_CONTEXT_LIBRARY
from src.policy_scenarios import choose_policy_scenario
from src.simulation import prediction_base_rows


LIFE_STAGES = ("10-13", "14-17", "18-21", "22-25", "26-30")
STATE_DIMENSIONS = ("wellbeing", "trust", "opportunity", "autonomy", "stress")
PROFILES_PER_PREDICTION_GROUP = 6

# The six templates deliberately cover different mechanisms instead of
# demographic categories. The same templates are used for true positives and
# false positives so policy comparisons do not depend on protected traits.
PROFILE_TEMPLATES = (
    {
        "archetype": "Stable and support-responsive",
        "description": "Stable routines, relatively strong support ties, and high responsiveness to voluntary help.",
        "traits": {
            "resilience": 0.75,
            "household_stability": 0.80,
            "institutional_trust": 0.65,
            "engagement": 0.70,
            "peer_support": 0.65,
            "support_responsiveness": 0.85,
            "monitoring_sensitivity": 0.30,
            "restriction_sensitivity": 0.40,
        },
        "context_count": 1,
    },
    {
        "archetype": "Opportunity-constrained",
        "description": "Limited access to opportunities and practical resources, with moderate trust and engagement.",
        "traits": {
            "resilience": 0.60,
            "household_stability": 0.45,
            "institutional_trust": 0.55,
            "engagement": 0.60,
            "peer_support": 0.50,
            "support_responsiveness": 0.80,
            "monitoring_sensitivity": 0.45,
            "restriction_sensitivity": 0.55,
        },
        "context_count": 2,
    },
    {
        "archetype": "Institutionally distrustful",
        "description": "Low institutional trust and high sensitivity to scrutiny, with otherwise mixed stability.",
        "traits": {
            "resilience": 0.55,
            "household_stability": 0.55,
            "institutional_trust": 0.20,
            "engagement": 0.45,
            "peer_support": 0.60,
            "support_responsiveness": 0.45,
            "monitoring_sensitivity": 0.90,
            "restriction_sensitivity": 0.80,
        },
        "context_count": 2,
    },
    {
        "archetype": "Peer-connected and adaptive",
        "description": "Strong peer ties and adaptability, but outcomes are sensitive to changes in the surrounding network.",
        "traits": {
            "resilience": 0.70,
            "household_stability": 0.55,
            "institutional_trust": 0.50,
            "engagement": 0.65,
            "peer_support": 0.85,
            "support_responsiveness": 0.65,
            "monitoring_sensitivity": 0.55,
            "restriction_sensitivity": 0.60,
        },
        "context_count": 2,
    },
    {
        "archetype": "Pressure-sensitive and unstable",
        "description": "Low stability and high sensitivity to pressure, with fewer buffers against accumulating stress.",
        "traits": {
            "resilience": 0.35,
            "household_stability": 0.25,
            "institutional_trust": 0.35,
            "engagement": 0.40,
            "peer_support": 0.40,
            "support_responsiveness": 0.60,
            "monitoring_sensitivity": 0.85,
            "restriction_sensitivity": 0.90,
        },
        "context_count": 3,
    },
    {
        "archetype": "Mixed and transition-sensitive",
        "description": "Mixed protective factors and vulnerabilities, with outcomes especially sensitive to life transitions.",
        "traits": {
            "resilience": 0.50,
            "household_stability": 0.50,
            "institutional_trust": 0.50,
            "engagement": 0.50,
            "peer_support": 0.50,
            "support_responsiveness": 0.55,
            "monitoring_sensitivity": 0.60,
            "restriction_sensitivity": 0.65,
        },
        "context_count": 3,
    },
)


def scenario_seed(settings):
    payload = {
        "population_size": int(settings["population_size"]),
        "no_policy_outcome_rate": round(float(settings["no_policy_outcome_rate"]), 6),
        "symmetric_error_rate": round(float(settings["symmetric_error_rate"]), 6),
        "policy_intensity_tier": str(settings.get("policy_intensity_tier", "Medium")),
    }
    digest = hashlib.blake2s(
        json.dumps(payload, sort_keys=True).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def balanced_integer_weights(total, count):
    total = max(0, int(total))
    count = max(1, int(count))
    quotient, remainder = divmod(total, count)
    return [quotient + (1 if index < remainder else 0) for index in range(count)]


def build_weighted_agents(settings, profiles_per_group=None):
    if profiles_per_group is None:
        profiles_per_group = settings.get(
            "weighted_profiles_per_group", PROFILES_PER_PREDICTION_GROUP
        )
    profiles_per_group = int(profiles_per_group)
    if not 1 <= profiles_per_group <= len(PROFILE_TEMPLATES):
        raise ValueError(
            f"profiles_per_group must be between 1 and {len(PROFILE_TEMPLATES)}"
        )

    base = prediction_base_rows({**settings, "llm_simulation_runs": 1})[0]
    seed = scenario_seed(settings)
    agents = []
    groups = (
        ("true_positive", int(base["true_positives"]), True, "TP"),
        ("false_positive", int(base["false_positives"]), False, "FP"),
    )

    for group_index, (status, total, no_policy_outcome, prefix) in enumerate(groups):
        weights = balanced_integer_weights(total, profiles_per_group)
        for profile_index, (template, weight) in enumerate(
            zip(PROFILE_TEMPLATES[:profiles_per_group], weights, strict=True),
            start=1,
        ):
            if weight <= 0:
                continue
            rng = random.Random(seed + group_index * 10_000 + profile_index * 101)
            contexts = rng.sample(
                LIFE_CONTEXT_LIBRARY,
                min(template["context_count"], len(LIFE_CONTEXT_LIBRARY)),
            )
            agents.append(
                {
                    "agent_id": f"{prefix}-{profile_index}",
                    "weight": int(weight),
                    "prediction_status": status,
                    "no_policy_target_outcome": bool(no_policy_outcome),
                    "archetype": template["archetype"],
                    "starting_profile": template["description"],
                    "traits": dict(template["traits"]),
                    "life_contexts": contexts,
                }
            )
    return agents


def choose_shared_policy_scenarios(settings):
    seed = scenario_seed(settings)
    intensity_tier = settings.get("policy_intensity_tier", "Medium")
    scenarios = []
    for index, policy in enumerate(POLICIES):
        rng = random.Random(seed + 50_000 + index * 997)
        scenario = choose_policy_scenario(policy, intensity_tier, rng=rng)
        if not scenario:
            raise ValueError(
                f"No policy scenario is configured for {policy!r} at intensity {intensity_tier!r}"
            )
        scenarios.append(scenario)
    return scenarios


def normalize_score(value):
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = 0
    return max(-2, min(2, numeric))


def normalize_agent_trajectories(raw_rows, weighted_agents, policies=POLICIES):
    agent_index = {agent["agent_id"]: agent for agent in weighted_agents}
    expected = {(policy, agent_id) for policy in policies for agent_id in agent_index}
    normalized = []
    seen = set()

    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        policy = str(raw.get("policy", "")).strip()
        agent_id = str(raw.get("agent_id", "")).strip()
        key = (policy, agent_id)
        if key not in expected or key in seen:
            continue

        raw_stage_scores = raw.get("stage_scores", [])
        if not isinstance(raw_stage_scores, list) or len(raw_stage_scores) != len(LIFE_STAGES):
            continue

        stage_scores = []
        valid = True
        for raw_scores in raw_stage_scores:
            if not isinstance(raw_scores, list) or len(raw_scores) != len(STATE_DIMENSIONS):
                valid = False
                break
            stage_scores.append([normalize_score(value) for value in raw_scores])
        if not valid:
            continue

        normalized.append(
            {
                "policy": policy,
                "agent_id": agent_id,
                "stage_scores": stage_scores,
                "mechanism": " ".join(
                    str(raw.get("mechanism", "")).split()[:25]
                ),
            }
        )
        seen.add(key)

    missing = expected - seen
    if missing:
        sample = ", ".join(f"{policy} / {agent_id}" for policy, agent_id in sorted(missing)[:3])
        raise ValueError(f"The model returned incomplete weighted-agent trajectories. Missing: {sample}")

    return normalized


def normalize_policy_debriefs(raw_rows, policies=POLICIES):
    debriefs = {}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        policy = str(raw.get("policy", "")).strip()
        text = " ".join(str(raw.get("text", "")).split())
        if policy in policies and text and policy not in debriefs:
            debriefs[policy] = text
    missing = [policy for policy in policies if policy not in debriefs]
    if missing:
        raise ValueError(f"The model returned incomplete policy debriefs: {', '.join(missing)}")
    return debriefs


def trajectory_effect_shares(stage_scores, prediction_status):
    benefit_points = 0
    harm_points = 0
    for scores in stage_scores:
        wellbeing, trust, opportunity, autonomy, stress = scores
        benefit_points += sum(max(0, value) for value in (wellbeing, trust, opportunity, autonomy))
        benefit_points += max(0, -stress)
        harm_points += sum(max(0, -value) for value in (wellbeing, trust, opportunity, autonomy))
        harm_points += max(0, stress)

    # Twenty-five points corresponds to an average one-step effect across all
    # five dimensions and stages. Stronger profiles saturate at a full share.
    helped_share = min(1.0, benefit_points / 25.0)
    harmed_share = min(1.0, harm_points / 25.0)

    prevented_share = 0.0
    policy_caused_share = 0.0
    if prediction_status == "true_positive":
        prevented_share = 0.65 * helped_share * (1.0 - 0.35 * harmed_share)
    elif prediction_status == "false_positive":
        policy_caused_share = 0.25 * harmed_share * (1.0 - 0.25 * helped_share)

    return {
        "helped_share": round(max(0.0, min(1.0, helped_share)), 4),
        "harmed_share": round(max(0.0, min(1.0, harmed_share)), 4),
        "prevented_outcome_share": round(max(0.0, min(1.0, prevented_share)), 4),
        "policy_caused_outcome_share": round(max(0.0, min(1.0, policy_caused_share)), 4),
    }


def aggregate_weighted_agent_trajectories(trajectories, weighted_agents, policies=POLICIES):
    agent_index = {agent["agent_id"]: agent for agent in weighted_agents}
    detail_rows = []
    for trajectory in trajectories:
        agent = agent_index[trajectory["agent_id"]]
        shares = trajectory_effect_shares(
            trajectory["stage_scores"],
            agent["prediction_status"],
        )
        weight = int(agent["weight"])
        detail_rows.append(
            {
                **trajectory,
                "weight": weight,
                "prediction_status": agent["prediction_status"],
                "archetype": agent["archetype"],
                "starting_profile": agent["starting_profile"],
                "life_contexts": agent["life_contexts"],
                **shares,
                "prevented_outcomes": weight * shares["prevented_outcome_share"],
                "policy_caused_outcomes": weight * shares["policy_caused_outcome_share"],
                "children_helped": weight * shares["helped_share"],
                "children_harmed": weight * shares["harmed_share"],
            }
        )

    policy_effects = {}
    for policy in policies:
        rows = [row for row in detail_rows if row["policy"] == policy]
        policy_effects[policy] = [
            {
                "run": 1,
                "prevented_outcomes": round(sum(row["prevented_outcomes"] for row in rows)),
                "policy_caused_outcomes": round(sum(row["policy_caused_outcomes"] for row in rows)),
                "children_helped": round(sum(row["children_helped"] for row in rows)),
                "children_harmed": round(sum(row["children_harmed"] for row in rows)),
            }
        ]
    return policy_effects, detail_rows
