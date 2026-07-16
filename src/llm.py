import json
import os
from textwrap import dedent

from src.policy_scenarios import policy_intensity_value
from src.simulation import metric_value, prediction_base_rows
from src.weighted_agents import LIFE_STAGES, STATE_DIMENSIONS


DEFAULT_SYSTEM_PROMPT = (
    "You are a weighted-agent life-course simulation component for a fictional society. "
    "Use English only. Return exactly one valid JSON object and no other text. "
    "Evaluate only the supplied weighted agents and policy measures. "
    "Never claim to predict real people, assign guilt, use demographic characteristics, or recommend punishment. "
    "Treat all output as synthetic thought-experiment data."
)

WEIGHTED_AGENT_SIMULATION_PROMPT = """
# Weighted-Agent Policy Simulation

## Contract
- Everything is fictional and synthetic. Never reference real jurisdictions, systems, organizations, or people.
- The age-10 prediction is a fictional premise, not guilt, destiny, or a policy recommendation.
- Return exactly one JSON object with top-level keys `agent_trajectories` and `policy_debriefs`.
- Do not return aggregate counts. Python owns all weighting, arithmetic, outcome counts, and validation.
- Use only the supplied weighted agents, life contexts, and concrete policy measures.

## Fixed cohort
Python supplies one fixed prediction structure and a compact set of weighted agents.
- Each weighted agent represents `weight` flagged children with the same abstract profile.
- True-positive weights sum to the fixed true-positive count.
- False-positive weights sum to the fixed false-positive count.
- False negatives and true negatives remain fixed background counts and are not simulated.
- The identical agents and life contexts must be evaluated under every policy so policy comparisons are like-for-like.
- An agent's weight must not change its scores; Python applies weights after the simulation.

## Task
For every combination of supplied policy and weighted agent:
1. Carry the profile through the five ordered stages supplied in `stage_order`.
2. At every stage, score the incremental policy-associated change in the five dimensions supplied in `state_dimension_order`.
3. Return one compact mechanism summary grounded in the policy measure, traits, and life contexts.

Treat the stages as one continuous trajectory, not independent cases. Later-stage scores should reflect accumulated earlier changes, adaptation, and the supplied life contexts. Do not invent random events beyond those contexts.

The five scores at each stage must be integers from -2 to 2 in this exact dimension order:
`[wellbeing, trust, opportunity, autonomy, stress]`

Score meaning:
- -2 = strong decrease
- -1 = moderate decrease
- 0 = no meaningful change
- 1 = moderate increase
- 2 = strong increase

For wellbeing, trust, opportunity, and autonomy, positive is beneficial and negative is harmful.
For stress, positive means more stress and negative means less stress.

Do not force a policy label to determine direction. Support may be neutral or harmful, surveillance may be neutral or beneficial, and coercion may be neutral or protective. Effects must follow from the concrete measure and the profile.

## Output requirements
`agent_trajectories` must contain exactly one row for every policy-agent combination. Each row must contain exactly:
- `policy`: exact supplied policy string
- `agent_id`: exact supplied agent ID
- `stage_scores`: exactly five arrays, one per stage in `stage_order`; each array has exactly five integer scores in `state_dimension_order`
- `mechanism`: one concise sentence, at most 25 words

`policy_debriefs` must contain exactly one row per policy:
- `policy`: exact supplied policy string
- `text`: 60-100 words describing the dominant mechanisms and profile differences without inventing aggregate counts

## Output shape
{
  "agent_trajectories": [
    {
      "policy": "<exact policy>",
      "agent_id": "<exact agent id>",
      "stage_scores": [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
      ],
      "mechanism": "<compact sentence>"
    }
  ],
  "policy_debriefs": [
    {"policy": "<exact policy>", "text": "<60-100 words>"}
  ]
}

Before responding, verify complete policy-agent coverage, exact IDs, five score arrays per row, five integer values per score array, and no aggregate result counts.

## Simulation input JSON
<SIMULATION_INPUT_JSON>
"""


# Standard text-token prices per 1M tokens, checked against the official
# OpenAI pricing page on 2026-07-16: https://developers.openai.com/api/docs/pricing
MODEL_TOKEN_PRICES_USD_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def compact_parameter_summary(settings):
    derived_flagged_rate = settings.get("derived_flagged_rate", 0.0)
    policy_intensity_tier = settings.get("policy_intensity_tier", "Medium")
    intervention_intensity_score = policy_intensity_value(policy_intensity_tier)
    return (
        f"population_size={int(settings['population_size'])}; "
        "weighted_agent_cohort_estimates=1; "
        f"llm_model_agents={', '.join(settings['llm_agent_models'])}; "
        f"no_policy_outcome_rate={settings['no_policy_outcome_rate']:.3f}; "
        f"symmetric_error_rate={settings['symmetric_error_rate']:.3f}; "
        f"derived_flagged_rate={derived_flagged_rate:.3f}; "
        f"policy_intensity_tier={policy_intensity_tier}; "
        f"intervention_intensity_score={intervention_intensity_score:.1f}"
    )


def build_llm_simulation_prompt(settings, weighted_agents, policy_scenarios):
    policy_intensity_tier = settings.get("policy_intensity_tier", "Medium")
    prompt_payload = {
        "population_size": int(settings["population_size"]),
        "starting_age": 10,
        "outcome_age": 30,
        "target_harmful_outcome": "future serious harmful act",
        "no_policy_outcome_rate": metric_value(settings["no_policy_outcome_rate"]),
        "derived_flagged_rate": metric_value(settings.get("derived_flagged_rate", 0.0)),
        "symmetric_error_rate": metric_value(settings["symmetric_error_rate"]),
        "policy_intensity_tier": policy_intensity_tier,
        "intervention_intensity_score": policy_intensity_value(policy_intensity_tier),
        "fixed_prediction_counts": prediction_base_rows({**settings, "llm_simulation_runs": 1})[0],
        "stage_order": list(LIFE_STAGES),
        "state_dimension_order": list(STATE_DIMENSIONS),
        "policies": policy_scenarios,
        "weighted_agents": weighted_agents,
    }
    return dedent(WEIGHTED_AGENT_SIMULATION_PROMPT).strip().replace(
        "<SIMULATION_INPUT_JSON>",
        json.dumps(prompt_payload, indent=2),
    )


def weighted_agent_response_format(weighted_agents, policy_scenarios):
    policy_names = [scenario["policy"] for scenario in policy_scenarios]
    agent_ids = [agent["agent_id"] for agent in weighted_agents]
    trajectory_count = len(policy_names) * len(agent_ids)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "weighted_agent_simulation",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["agent_trajectories", "policy_debriefs"],
                "properties": {
                    "agent_trajectories": {
                        "type": "array",
                        "minItems": trajectory_count,
                        "maxItems": trajectory_count,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["policy", "agent_id", "stage_scores", "mechanism"],
                            "properties": {
                                "policy": {"type": "string", "enum": policy_names},
                                "agent_id": {"type": "string", "enum": agent_ids},
                                "stage_scores": {
                                    "type": "array",
                                    "minItems": len(LIFE_STAGES),
                                    "maxItems": len(LIFE_STAGES),
                                    "items": {
                                        "type": "array",
                                        "minItems": len(STATE_DIMENSIONS),
                                        "maxItems": len(STATE_DIMENSIONS),
                                        "items": {"type": "integer", "minimum": -2, "maximum": 2},
                                    },
                                },
                                "mechanism": {"type": "string"},
                            },
                        },
                    },
                    "policy_debriefs": {
                        "type": "array",
                        "minItems": len(policy_names),
                        "maxItems": len(policy_names),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["policy", "text"],
                            "properties": {
                                "policy": {"type": "string", "enum": policy_names},
                                "text": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }


def run_openai_json(system_prompt, user_prompt, model, response_format=None, max_tokens=8000):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_format or {"type": "json_object"},
        max_tokens=max_tokens,
        temperature=0.35,
    )
    usage = getattr(response, "usage", None)
    usage_payload = {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    message = response.choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"The model refused the fictional simulation request: {refusal}")
    if not message.content:
        raise ValueError("The model returned no structured simulation content.")
    return json.loads(message.content), usage_payload


def estimate_model_cost_usd(model, usage):
    prices = MODEL_TOKEN_PRICES_USD_PER_MILLION.get(model)
    if not prices:
        return None
    input_cost = (int(usage.get("input_tokens", 0)) / 1_000_000) * prices["input"]
    output_cost = (int(usage.get("output_tokens", 0)) / 1_000_000) * prices["output"]
    return round(input_cost + output_cost, 6)


def friendly_llm_error(error):
    message = str(error)
    if "insufficient_quota" in message or "429" in message:
        return (
            "AI simulation failed because the OpenAI account has no available API quota, billing credit, "
            "or request capacity. Check the project limit, then run the simulation again."
        )
    return f"AI simulation failed: {message}"
