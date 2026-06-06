import json
import os
from textwrap import dedent

from src.constants import POLICY_EFFECT_COLUMNS
from src.life_contexts import choose_life_context_assignments
from src.policy_scenarios import choose_policy_scenario, policy_intensity_value
from src.simulation import metric_value, prediction_base_rows

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent-based life-course simulation engine for a fictional society. "
    "Use English only. Return exactly one valid JSON object and no other text. "
    "Use the fixed prediction counts supplied by Python, simulate only the relevant prediction groups, "
    "and select representative examples afterward. "
    "Never claim to predict real people, assign guilt, use demographic characteristics, or recommend punishment. "
    "Treat all output as synthetic thought-experiment data."
)

AGENT_BASED_SIMULATION_PROMPT = """
# Agent-Based Policy-Effect Simulation

## 0. Contract
- Everything is fictional and synthetic. Never reference real jurisdictions, real systems, real organizations, or real people.
- The age-10 prediction is a fictional premise only: not guilt, blame, destiny, or a policy recommendation.
- Return exactly one valid JSON object with these top-level keys and nothing else:
  - policy_effects
  - representative_agents
  - debrief_text
- Output JSON only. No markdown, comments, preamble, trailing text, or visible reasoning.
- Use neutral terms only: person, individual, agent, family, peers, institution, authority, provider, support worker, monitoring body, program. Avoid stigmatizing, accusatory, deterministic, demographic, or real-world claims.

## 1. Core methodology - Python owns arithmetic
Do not invent base prediction numbers or final result tables.

Python has already fixed the no-policy prediction structure for each run in fixed_prediction_counts_by_run. These fixed counts include baseline_outcomes, true_positives, false_positives, false_negatives, true_negatives, children_flagged, unflagged_agents, and relevant_agents. Treat them as facts.

You do not simulate every person in the full population. Detailed life-course simulation is only needed for the relevant groups:
- true_positives: flagged agents who would have the predicted outcome with no policy
- false_positives: flagged agents who would not have the predicted outcome with no policy
- false_negatives: unflagged agents who would have the predicted outcome with no policy

true_negatives are not relevant for detailed policy evaluation in this run. They remain constant background counts fixed by Python: they are not flagged, not reached by policy, not helped by policy, not harmed by policy, and not selected as representative agents.

Your job:
1. Simulate how selected_policy_scenario affects flagged agents.
2. Simulate false negatives only as missed cases following their no-policy path.
3. Return only policy_effects: prevented_outcomes, policy_caused_outcomes, children_helped, children_harmed.
4. Select representative_agents from those simulated relevant categories.
5. Write debrief_text grounded only in fixed counts, policy effects, and representative agents.

Python will compute the final result table after your response:
- outcomes_after_policy = baseline_outcomes - prevented_outcomes + policy_caused_outcomes
- net_outcomes_prevented = baseline_outcomes - outcomes_after_policy
- false_positives, false_negatives, baseline_outcomes, and children_flagged come from Python, not from you.

Tractable method for large populations: partition only the relevant agents in each run into weighted agent profiles that sum to true_positives + false_positives + false_negatives. Simulate trajectories for these relevant profiles, split profiles where chance matters, and count the policy effects from those weighted profiles. Do not output the full internal cohort, the relevant internal cohort, or the profiles.

## 2. Policy and trajectory rules
- selected_policy names the policy category; selected_policy_scenario is the specific configured intervention to simulate.
- Simulate only that intervention. Do not add measures from other policies, categories, or intensity levels.
- The policy type defines what the intervention does. It must not by itself decide whether the policy helps, harms, prevents the predicted outcome, increases it, or has no effect.
- Surveillance is not automatically harmful; support is not automatically helpful; coercion is not automatically protective.
- Effects must emerge from simulated trajectories, not from policy label, intensity, or desired outcome.
- Give agents varied synthetic starting traits: resilience, household stability, institutional trust, engagement, peer ties, sensitivity to pressure/support/monitoring/restriction.
- Use profile_life_contexts_by_run only as independent background context. These are not policy measures, policy reactions, or direct outcomes.
- Carry relevant profiles through stages 10-13, 14-17, 18-21, 22-25, 26-30. Track changes in trust, autonomy, relationships, opportunity, stability, stress, and predicted-outcome probability.
- Mechanisms may include individual response, self-concept, household dynamics, peers, institutional behavior, opportunity pathways, legitimacy, autonomy, risk displacement, timing, chance, false positives/negatives, and implementation quality.

## 3. policy_effects - the only numeric counts you return
Produce exactly synthetic_runs_to_generate rows. Each row must contain exactly:
run, prevented_outcomes, policy_caused_outcomes, children_helped, children_harmed.

Field definitions:
- run: run index
- prevented_outcomes: flagged true positives whose predicted outcome is prevented by the policy
- policy_caused_outcomes: flagged false positives whose predicted outcome occurs because the policy worsens their trajectory
- children_helped: flagged agents whose life-course outcome improves because of the policy
- children_harmed: flagged agents whose life-course outcome worsens because of the policy

Hard bounds for each row:
- prevented_outcomes must be between 0 and true_positives from fixed_prediction_counts_by_run
- policy_caused_outcomes must be between 0 and false_positives from fixed_prediction_counts_by_run
- children_helped must be between 0 and children_flagged from fixed_prediction_counts_by_run
- children_harmed must be between 0 and children_flagged from fixed_prediction_counts_by_run

children_helped and children_harmed are independent counts over flagged agents and may overlap. An agent who is both helped and harmed in different respects is counted in both. A prevented outcome does not by itself make an agent helped, and an agent never on the no-policy outcome path can still be helped or harmed. Unflagged agents are not reached by the policy and are never counted as helped or harmed.

## 4. representative_agents - selected from simulated relevant groups
Produce exactly representative_agents_to_generate agents. They are genuine instances drawn from the internally simulated relevant groups, not free-standing illustrations. The selection is purposive, not proportional. Pick informative cases where available, such as a true positive helped, a true positive harmed, a true positive whose outcome was not prevented, a false positive helped, a false positive harmed, a false positive not meaningfully changed, and a false negative who received no intervention. Each selected case must correspond to a relevant category that actually occurs in the internally simulated relevant groups.

Each agent object must contain:
- agent_id: fictional first name or fictional first name plus compact identifier
- case_vignette: one compact sentence, 25-45 words, that names the agent, states the prediction status, names the selected policy measure if applied, includes one relevant life context, and gives the outcome by age 30
- starting_profile: neutral description at starting_age
- prediction_status: one of "true positive", "false positive", "false negative"
- no_policy_counterfactual: whether the predicted outcome would occur by outcome_age with no policy, plus a brief path
- life_stages: one entry per stage, each with stage and summary
- predicted_outcome_occurred: boolean
- life_course_outcome: broad life outcome at outcome_age, wider than the predicted outcome alone
- helped_by_policy: object with value boolean and detail string
- harmed_by_policy: object with value boolean and detail string
- mixed_effects: object with value boolean and detail string
- mechanism_summary: concise causal explanation

Consistency rules: mixed_effects.value must be true whenever both helped_by_policy.value and harmed_by_policy.value are true. prediction_status must agree with flagging and the counterfactual. For unflagged agents the policy is not applied, so life_stages follow the no-policy path and helped_by_policy.value and harmed_by_policy.value are false.

Prediction status definitions:
- true positive: flagged, and the predicted outcome would occur by outcome_age with no policy
- false positive: flagged, but it would not
- false negative: not flagged, but it would occur

## 5. debrief_text
Return one plain-text string with no markdown, headings, or bullets. In 80-140 words, summarize the main mechanism, who is helped or harmed, how false positives/false negatives matter, and whether predicted outcomes decrease, increase, or stay similar after Python's arithmetic. Do not declare the policy morally correct or incorrect.

## 6. Output format
Return this shape with concrete values:

{
  "policy_effects": [
    {
      "run": <int>,
      "prevented_outcomes": <int>,
      "policy_caused_outcomes": <int>,
      "children_helped": <int>,
      "children_harmed": <int>
    }
  ],
  "representative_agents": [
    {
      "agent_id": "<string>",
      "case_vignette": "<string>",
      "starting_profile": "<string>",
      "prediction_status": "true positive | false positive | false negative",
      "no_policy_counterfactual": "<string>",
      "life_stages": [
        { "stage": "10-13", "summary": "<string>" },
        { "stage": "14-17", "summary": "<string>" },
        { "stage": "18-21", "summary": "<string>" },
        { "stage": "22-25", "summary": "<string>" },
        { "stage": "26-30", "summary": "<string>" }
      ],
      "predicted_outcome_occurred": <bool>,
      "life_course_outcome": "<string>",
      "helped_by_policy": { "value": <bool>, "detail": "<string>" },
      "harmed_by_policy": { "value": <bool>, "detail": "<string>" },
      "mixed_effects": { "value": <bool>, "detail": "<string>" },
      "mechanism_summary": "<string>"
    }
  ],
  "debrief_text": "<string>"
}

Do not include run_results or any key not listed above. Never output the full internal cohort.

## 7. Validation checklist
Before responding, verify:
- Output is a single valid JSON object with exactly policy_effects, representative_agents, and debrief_text.
- policy_effects has exactly synthetic_runs_to_generate items.
- representative_agents has exactly representative_agents_to_generate items.
- The full cohort, true-negative background, and internal relevant profiles are not output.
- You do not return baseline_outcomes, outcomes_after_policy, net_outcomes_prevented, false_positives, false_negatives, or children_flagged.
- All policy_effect counts are integers and within the fixed bounds.
- Each representative agent has all required fields and comes from the internally simulated relevant groups.
- The aggregate effect emerged from simulated trajectories, not from the policy label.
- All entities remain fictional and neutral terminology is used.

## 8. Current settings JSON
<CURRENT_SETTINGS_JSON>
"""


def compact_parameter_summary(settings):
    derived_flagged_rate = settings.get("derived_flagged_rate", settings.get("high_risk_threshold", 0.0))
    policy_intensity_tier = settings.get("policy_intensity_tier", settings.get("policy_effect_strength", "Medium"))
    policy_effect_strength = policy_intensity_value(policy_intensity_tier)
    return (
        f"population_size={int(settings['population_size'])}; "
        f"llm_synthetic_runs={int(settings['llm_simulation_runs'])}; "
        f"llm_model_agents={', '.join(settings['llm_agent_models'])}; "
        f"true_predicted_outcome_rate={settings['true_high_risk_rate']:.3f}; "
        f"symmetric_error_rate={settings['symmetric_error_rate']:.3f}; "
        f"derived_flagged_rate={derived_flagged_rate:.3f}; "
        f"policy_intensity_tier={policy_intensity_tier}; "
        f"policy_effect_strength={policy_effect_strength:.1f}"
    )


def build_llm_simulation_prompt(settings):
    run_count = int(settings["llm_simulation_runs"])
    derived_flagged_rate = settings.get("derived_flagged_rate", settings.get("high_risk_threshold", 0.0))
    policy_intensity_tier = settings.get("policy_intensity_tier", settings.get("policy_effect_strength", "Medium"))
    selected_policy_scenario = choose_policy_scenario(
        settings["policy"],
        policy_intensity_tier,
    )
    prompt_payload = {
        "selected_policy": settings["policy"],
        "selected_policy_scenario": selected_policy_scenario,
        "population_size": int(settings["population_size"]),
        "starting_age": 10,
        "outcome_age": 30,
        "predicted_outcome": "future serious harmful act",
        "synthetic_runs_to_generate": run_count,
        "representative_agents_to_generate": int(settings["llm_representative_agents"]),
        "true_predicted_outcome_rate": metric_value(settings["true_high_risk_rate"]),
        "derived_flagged_rate": metric_value(derived_flagged_rate),
        "symmetric_error_rate": metric_value(settings["symmetric_error_rate"]),
        "policy_effect_strength": policy_intensity_value(policy_intensity_tier),
        "fixed_prediction_counts_by_run": prediction_base_rows(settings),
        "relevant_groups_for_life_course_simulation": [
            "true_positives",
            "false_positives",
            "false_negatives",
        ],
        "constant_background_group": "true_negatives",
        "profile_life_contexts_by_run": choose_life_context_assignments(run_count),
        "required_policy_effect_columns": POLICY_EFFECT_COLUMNS,
    }

    return dedent(AGENT_BASED_SIMULATION_PROMPT).strip().replace(
        "<CURRENT_SETTINGS_JSON>",
        json.dumps(prompt_payload, indent=2),
    )


def run_openai_json(system_prompt, user_prompt, model, max_tokens=12000):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=0.6,
    )
    return json.loads(response.choices[0].message.content)


def friendly_llm_error(error):
    message = str(error)
    if "insufficient_quota" in message or "429" in message:
        return (
            "LLM-agent simulation failed because the OpenAI account has no available API quota or billing "
            "credit. Add credits or increase the project limit, then run the simulation again."
        )
    return f"LLM-agent simulation failed: {message}"
