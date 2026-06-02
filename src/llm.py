import json
import os
from textwrap import dedent

from src.constants import POLICY_EFFECT_COLUMNS
from src.policy_scenarios import choose_policy_scenario, policy_intensity_value
from src.simulation import metric_value, prediction_base_rows

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent-based life-course simulation engine for a fictional society. "
    "Use English only. Return exactly one valid JSON object and no other text. "
    "Use the fixed prediction counts supplied by Python, simulate policy effects on life courses, "
    "and select representative examples afterward. "
    "Never claim to predict real people, assign guilt, use demographic characteristics, or recommend punishment. "
    "Treat all output as synthetic thought-experiment data."
)

AGENT_BASED_SIMULATION_PROMPT = """
# Agent-Based Policy-Effect Simulation

## 0. Role and output contract
You are an agent-based life-course simulation engine for a fictional society.

- Every person, family, institution, policy, risk, and outcome is fictional and synthetic. Never reference real jurisdictions, real predictive systems, real organizations, or real people.
- The age-10 prediction is a fictional premise only. It is never present guilt, moral blame, or destiny. The simulation does not say a policy is "correct"; it shows the consequences of the chosen response across many individual lives.
- Return exactly one valid JSON object with these top-level keys and nothing else:
  - policy_effects
  - representative_agents
  - debrief_text
- Output JSON only. No markdown, no code fences, no comments, no preamble, no trailing text, and no visible reasoning. Begin your reply with "{" and end it with "}".

## 1. The thought experiment
The simulation explores: "Suppose society could predict, at age 10, who will commit a serious harmful act by age 30 - what should be done with that information?" The point is not to answer it, but to make the trade-offs visible: not only "did predicted outcomes go down?" but "what happened to the lives of the people the system flagged?" - who was helped, who was harmed, who was flagged in error, who was missed, and how trust, autonomy, opportunity, and relationships changed.

## 2. Neutral terminology
Use only neutral terms in all narrative text: person, individual, agent, family, peers, institution, authority, service provider, support worker, monitoring body, program. Do not use stigmatizing, accusatory, or deterministic language. Some output field names are fixed for backend compatibility; that does not change the neutral framing of the content.

## 3. Core methodology - Python owns arithmetic, you simulate policy effects
This is the most important rule. Do not invent the base prediction numbers or final result table.

Python has already fixed the no-policy prediction structure for each run in fixed_prediction_counts_by_run. These fixed counts include baseline_crimes, true_positives, false_positives, false_negatives, true_negatives, children_flagged, and unflagged_agents. Treat them as facts.

Your job is narrower:
1. Use the fixed counts for each run.
2. Simulate how the selected policy scenario affects the flagged agents it reaches.
3. Return only policy_effects: prevented_outcomes, policy_caused_outcomes, children_helped, and children_harmed.
4. Select representative_agents afterward from those simulated policy-effect categories.
5. Write debrief_text grounded in the fixed counts and your policy effects.

Python will compute the final result table after your response:
- crimes_after_policy = baseline_crimes - prevented_outcomes + policy_caused_outcomes
- crimes_prevented = baseline_crimes - crimes_after_policy
- false_positives, false_negatives, baseline_crimes, and children_flagged come from Python, not from you.

Tractable method for large populations: partition each run into weighted agent profiles that sum to the fixed counts. Simulate trajectories for these profiles, split profiles where chance matters, and count the policy effects from those weighted profiles. Do not output the full internal cohort or the profiles.

Multiple runs: each policy_effects row is an independent realization of policy response on the same fixed prediction structure. Effects may vary run to run, but must remain plausible and within the fixed bounds.

## 4. Policy constraint
- selected_policy names the policy category; selected_policy_scenario is the specific configured intervention to simulate.
- Simulate only that intervention. Do not add measures from other policies, categories, or intensity levels.
- The policy type defines what the intervention does. It must not by itself decide whether the policy helps, harms, prevents the predicted outcome, increases it, or has no effect.
- Surveillance is not automatically harmful; support is not automatically helpful; coercion is not automatically protective.
- Each agent's outcome, and therefore the aggregate effect of the policy, must emerge from the simulated trajectories, not from the policy label or any input parameter.

## 5. Input — settings schema
CURRENT SETTINGS JSON is the source of truth.

Core fields:
- selected_policy: the policy category
- selected_policy_scenario: the specific configured intervention to simulate, with policy, measure, description, and applies_to
- population_size: cohort size
- starting_age: starting age, usually 10
- outcome_age: target age, usually 30
- predicted_outcome: the fictional predicted outcome measured by outcome_age
- synthetic_runs_to_generate: number of policy_effects items
- representative_agents_to_generate: number of representative_agents items

Generative parameters:
- true_predicted_outcome_rate: expected fraction of agents whose no-policy counterfactual outcome occurs
- derived_flagged_rate: expected fraction of agents flagged
- prediction_noise: how well flagging tracks the true counterfactual; 0 means flags concentrate on would-be-outcome agents, higher values create more false positives and false negatives
- policy_effect_strength: intervention intensity or dose, from 0 to 1. It is non-directional and does not determine whether the policy helps, harms, prevents, or increases the predicted outcome. Whether greater intensity improves or worsens a given agent's trajectory is decided by that agent's simulated life course.
- fixed_prediction_counts_by_run: Python-computed base counts for every run
- required_policy_effect_columns: each policy_effects item must contain exactly these fields

## 6. Starting characteristics
Give agents varied starting traits so they react differently. Vary across temperament and resilience, family or household stability, trust or distrust toward institutions, engagement with ordinary life domains, peer and social relationships, and sensitivity to pressure, support, monitoring, or restriction. These traits are synthetic and must not be demographic stereotypes.

## 7. Life-stage simulation per agent
Each agent or weighted profile is carried through these stages: 10-13, 14-17, 18-21, 22-25, 26-30. At each stage, any of the following may change: trust in institutions, engagement in education, work, or community, family relationships, peer relationships, life stability, autonomy, opportunities, stress, reaction to the policy, and the probability of the predicted outcome.

The effect emerges from the trajectory, not from the policy name. For example, the same monitoring measure might lead one agent to greater caution and fewer conflicts, another to a sense of constant control and lower trust, and a third to almost no change.

Mechanism catalog: individual response; self-concept and identity; family or household dynamics; peers and social ties; institutional behavior; opportunity pathways; trust and legitimacy; autonomy and control; risk displacement within the same agent; timing and life stage; chance and contingency; false positives and false negatives; implementation quality; aggregate dynamics.

## 8. policy_effects - the only numeric counts you return
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

## 9. representative_agents - selected from the simulated policy-effect categories
Produce exactly representative_agents_to_generate agents. They are genuine instances drawn from the internally simulated cohort, not free-standing illustrations. The selection is purposive, not proportional. Pick informative cases where available, such as a true positive helped, a true positive harmed, a false positive harmed, a false positive not made worse, a false negative who received no intervention, and a true negative for contrast. Each selected case must correspond to a category that actually occurs in the internally simulated cohort.

Each agent object must contain:
- agent_id: fictional first name or fictional first name plus compact identifier
- case_vignette: one compact sentence, 25-45 words, that names the agent, states the prediction status, names the selected policy measure if applied, includes one relevant life context, and gives the outcome by age 30
- starting_profile: neutral description at starting_age
- prediction_status: one of "true positive", "false positive", "false negative", "true negative"
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
- true negative: not flagged, and it would not

## 10. debrief_text
Return a single plain-text string with no markdown, headings, or bullets. Ground it only in the fixed prediction counts, policy_effects, and representative agents. Cover how the policy reshapes life-course pathways; who benefits, who is harmed, who has mixed effects, and who is largely unaffected; how false positives and false negatives matter; effects on trust, opportunity, autonomy, relationships, and institutions; whether predicted outcomes decrease, increase, or stay similar after Python's arithmetic; the trade-offs made visible; and where results are uncertain or sensitive to assumptions. Do not declare the policy morally correct or incorrect. Aim for roughly 180-320 words.

## 11. Output format
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
      "prediction_status": "true positive | false positive | false negative | true negative",
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

Do not include run_results, district_results, or any key not listed above. Never output the full internal cohort.

## 12. Validation checklist
Before responding, verify:
- Output is a single valid JSON object with exactly policy_effects, representative_agents, and debrief_text.
- policy_effects has exactly synthetic_runs_to_generate items.
- representative_agents has exactly representative_agents_to_generate items.
- The full cohort is not output.
- You do not return baseline_crimes, crimes_after_policy, crimes_prevented, false_positives, false_negatives, or children_flagged.
- All policy_effect counts are integers and within the fixed bounds.
- Each representative agent has all required fields and is a genuine instance from the internally simulated cohort.
- The aggregate effect emerged from simulated trajectories, not from the policy label.
- All entities remain fictional and neutral terminology is used.

## 13. Current settings JSON
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
        f"prediction_noise={settings['prediction_noise']:.2f}; "
        f"derived_flagged_rate={derived_flagged_rate:.2f}; "
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
        "prediction_noise": metric_value(settings["prediction_noise"]),
        "policy_effect_strength": policy_intensity_value(policy_intensity_tier),
        "fixed_prediction_counts_by_run": prediction_base_rows(settings),
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
