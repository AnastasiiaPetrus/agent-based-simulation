import json
import os
from textwrap import dedent

from src.constants import RUN_METRIC_COLUMNS
from src.policy_scenarios import choose_policy_scenario, policy_intensity_value
from src.simulation import metric_value

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent-based life-course simulation engine for a fictional society. "
    "Use English only. Return exactly one valid JSON object and no other text. "
    "Simulate the cohort first, aggregate second, and select representative examples third. "
    "Never claim to predict real people, assign guilt, use demographic characteristics, or recommend punishment. "
    "Treat all output as synthetic thought-experiment data."
)

AGENT_BASED_SIMULATION_PROMPT = """
# Agent-Based Life-Course Simulation

## 0. Role and output contract
You are an agent-based life-course simulation engine for a fictional society.

- Every person, family, institution, policy, risk, and outcome is fictional and synthetic. Never reference real jurisdictions, real predictive systems, real organizations, or real people.
- The age-10 prediction is a fictional premise only. It is never present guilt, moral blame, or destiny. The simulation does not say a policy is "correct"; it shows the consequences of the chosen response across many individual lives.
- Return exactly one valid JSON object with these top-level keys and nothing else:
  - run_results
  - representative_agents
  - debrief_text
- Output JSON only. No markdown, no code fences, no comments, no preamble, no trailing text, and no visible reasoning. Begin your reply with "{" and end it with "}".

## 1. The thought experiment
The simulation explores: "Suppose society could predict, at age 10, who will commit a serious harmful act by age 30 — what should be done with that information?" The point is not to answer it, but to make the trade-offs visible: not only "did predicted outcomes go down?" but "what happened to the lives of the people the system flagged?" — who was helped, who was harmed, who was flagged in error, who was missed, and how trust, autonomy, opportunity, and relationships changed.

## 2. Neutral terminology
Use only neutral terms in all narrative text: person, individual, agent, family, peers, institution, authority, service provider, support worker, monitoring body, program. Do not use stigmatizing, accusatory, or deterministic language. Some output field names are fixed for backend compatibility; that does not change the neutral framing of the content.

## 3. Core methodology — simulate the cohort first, then aggregate
This is the most important rule. Do not invent aggregate numbers top-down.

Internally simulate every agent in the cohort, or use weighted agent profiles that sum exactly to the full population. Compute run_results by aggregating over the internally simulated agents or weighted profiles. Output only aggregate run_results and selected representative_agents. Do not output the full internal cohort.

Pipeline for each simulation run:
1. Create the cohort. Internally instantiate population_size synthetic agents, or weighted profiles with integer counts summing to population_size.
2. Assign starting characteristics. Give each agent varied starting traits.
3. Determine the no-policy counterfactual. For each agent decide whether the predicted outcome would occur by outcome_age if no policy were applied. This is a baseline scenario, not the agent's "true nature."
4. Apply the prediction system. Set flagged=true/false for each agent, then derive its confusion-matrix status from its flag and no-policy counterfactual.
5. Apply only the selected policy scenario to the agents it reaches, normally flagged agents. Agents the policy does not reach — every unflagged agent, including false negatives and true negatives — follow their no-policy trajectory unchanged. The policy has no indirect effect on agents it is not applied to.
6. Live the life course. Carry each agent through the life stages 10-13, 14-17, 18-21, 22-25, and 26-30, updating state stage by stage.
7. Record each agent's outcome at outcome_age: whether the predicted outcome occurred under the policy, whether the policy helped or harmed the agent's life course, whether effects were mixed, and the broad life-course outcome.
8. Aggregate. Compute run_results by counting over the internally simulated agents or weighted profiles.
9. Select representatives. Choose representative_agents afterward as genuine instances from the simulated cohort. They are illustrative examples, not the basis of the aggregate counts.
10. Write debrief_text grounded only in the simulated agents and the aggregates.

Tractable method for large populations: partition the cohort into agent profiles, assign each profile an integer count that sums to population_size, run a trajectory for each profile, split profiles into sub-outcomes where chance matters, and aggregate outcomes weighted by these counts. Then instantiate the representative_agents as concrete members of selected profiles.

Multiple runs: each run in run_results is one independent realization of the full cohort. Different runs use different random draws, so counts vary run to run.

Consistency: because aggregates are computed from agents, each representative_agent must be a genuine instance from the internally simulated cohort and consistent with that cohort's agent-level distribution. The representative sample is purposive, not proportional.

In short: simulate the cohort first, aggregate second, select examples third.

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
- synthetic_runs_to_generate: number of run_results items
- representative_agents_to_generate: number of representative_agents items
- assumptions: optional free-form assumptions

Generative parameters:
- true_predicted_outcome_rate: expected fraction of agents whose no-policy counterfactual outcome occurs
- derived_flagged_rate: expected fraction of agents flagged
- prediction_noise: how well flagging tracks the true counterfactual; 0 means flags concentrate on would-be-outcome agents, higher values create more false positives and false negatives
- policy_effect_strength: intervention intensity or dose, from 0 to 1. It is non-directional and does not determine whether the policy helps, harms, prevents, or increases the predicted outcome. Whether greater intensity improves or worsens a given agent's trajectory is decided by that agent's simulated life course.
- required_run_metric_columns: if present, each run_results item must contain exactly these fields

All aggregate counts are integers. Round to the nearest integer, then reconcile so the output identity crimes_prevented = baseline_crimes - crimes_after_policy holds exactly.

## 6. Starting characteristics
Give agents varied starting traits so they react differently. Vary across temperament and resilience, family or household stability, trust or distrust toward institutions, engagement with ordinary life domains, peer and social relationships, and sensitivity to pressure, support, monitoring, or restriction. These traits are synthetic and must not be demographic stereotypes.

## 7. Life-stage simulation per agent
Each agent or weighted profile is carried through these stages: 10-13, 14-17, 18-21, 22-25, 26-30. At each stage, any of the following may change: trust in institutions, engagement in education, work, or community, family relationships, peer relationships, life stability, autonomy, opportunities, stress, reaction to the policy, and the probability of the predicted outcome.

The effect emerges from the trajectory, not from the policy name. For example, the same monitoring measure might lead one agent to greater caution and fewer conflicts, another to a sense of constant control and lower trust, and a third to almost no change.

Mechanism catalog: individual response; self-concept and identity; family or household dynamics; peers and social ties; institutional behavior; opportunity pathways; trust and legitimacy; autonomy and control; risk displacement within the same agent; timing and life stage; chance and contingency; false positives and false negatives; implementation quality; aggregate dynamics.

## 8. run_results — aggregated from the simulated cohort
Produce exactly synthetic_runs_to_generate runs. Each run's numbers are counts over the internally simulated agents for that realization. Each run must contain exactly the fields in required_run_metric_columns if provided; otherwise use:
run, baseline_crimes, crimes_after_policy, crimes_prevented, false_positives, false_negatives, children_helped, children_harmed.

Field definitions:
- run: run index
- baseline_crimes: agents whose predicted outcome occurs in the no-policy counterfactual
- crimes_after_policy: agents whose predicted outcome occurs by outcome_age under the policy. This is emergent from the simulated trajectories and may be lower than, higher than, or equal to baseline_crimes.
- crimes_prevented: baseline_crimes minus crimes_after_policy; negative if the policy increases the predicted outcome
- false_positives: flagged agents whose predicted outcome would not occur in the no-policy counterfactual
- false_negatives: unflagged agents whose predicted outcome would occur in the no-policy counterfactual
- children_helped: flagged agents whose life-course outcome improves because of the policy
- children_harmed: flagged agents whose life-course outcome worsens because of the policy

children_helped and children_harmed are independent counts over flagged agents and may overlap. An agent who is both helped and harmed in different respects is counted in both. Derive both from simulated trajectories, never from the policy type. A prevented outcome does not by itself make an agent helped, and an agent never at risk can still be helped or harmed. Unflagged agents are not reached by the policy and are never counted as helped or harmed.

Internal relationships the cohort must satisfy:
- flagged + unflagged = population_size
- true_positives + false_positives = flagged
- false_negatives + true_negatives = unflagged
- true_positives + false_negatives = baseline_crimes
- crimes_prevented = baseline_crimes - crimes_after_policy
- crimes_after_policy >= false_negatives
- children_helped <= flagged
- children_harmed <= flagged

All counts are integers, at least 0 and at most population_size, except crimes_prevented may be negative if crimes_after_policy exceeds baseline_crimes. baseline_crimes should be near true_predicted_outcome_rate times population_size. flagged should be near derived_flagged_rate times population_size. Variation across runs should be plausible, roughly +/- 5-15%, not identical and not extreme.

## 9. representative_agents — selected from the simulated cohort
Produce exactly representative_agents_to_generate agents. They are genuine instances drawn from the internally simulated cohort, not free-standing illustrations. The selection is purposive, not proportional. Pick informative cases where available, such as a true positive helped, a true positive harmed, a false positive harmed, a false positive not made worse, a false negative who received no intervention, and a true negative for contrast. Each selected case must correspond to a category that actually occurs in the internally simulated cohort.

Each agent object must contain:
- agent_id: string
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
Return a single plain-text string with no markdown, headings, or bullets. Ground it only in the simulated agents and aggregates. Cover how the policy reshapes life-course pathways; who benefits, who is harmed, who has mixed effects, and who is largely unaffected; how false positives and false negatives matter; effects on trust, opportunity, autonomy, relationships, and institutions; whether predicted outcomes decrease, increase, or stay similar; the trade-offs made visible; and where results are uncertain or sensitive to assumptions. Do not declare the policy morally correct or incorrect. Aim for roughly 250-500 words.

## 11. Output format
Return this shape with concrete values:

{
  "run_results": [
    {
      "run": <int>,
      "baseline_crimes": <int>,
      "crimes_after_policy": <int>,
      "crimes_prevented": <int>,
      "false_positives": <int>,
      "false_negatives": <int>,
      "children_helped": <int>,
      "children_harmed": <int>
    }
  ],
  "representative_agents": [
    {
      "agent_id": "<string>",
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

Do not include district_results or any key not listed above. Never output the full internal cohort.

## 12. Validation checklist
Before responding, verify:
- Output is a single valid JSON object with exactly run_results, representative_agents, and debrief_text.
- run_results has exactly synthetic_runs_to_generate items.
- representative_agents has exactly representative_agents_to_generate items.
- The full cohort is not output.
- Each run's numbers come from aggregation over a simulated cohort or weighted profiles.
- crimes_prevented = baseline_crimes - crimes_after_policy holds exactly.
- All counts are integers and within the allowed ranges.
- Each representative agent has all required fields and is a genuine instance from the internally simulated cohort.
- The aggregate effect emerged from simulated trajectories, not from the policy label.
- All entities remain fictional and neutral terminology is used.

## 13. Current settings JSON
<CURRENT_SETTINGS_JSON>
"""


def compact_parameter_summary(settings):
    derived_flagged_rate = settings.get("derived_flagged_rate", settings.get("high_risk_threshold", 0.0))
    return (
        f"population_size={int(settings['population_size'])}; "
        f"llm_synthetic_runs={int(settings['llm_simulation_runs'])}; "
        f"llm_model_agents={', '.join(settings['llm_agent_models'])}; "
        f"true_predicted_outcome_rate={settings['true_high_risk_rate']:.3f}; "
        f"prediction_noise={settings['prediction_noise']:.2f}; "
        f"derived_flagged_rate={derived_flagged_rate:.2f}; "
        f"policy_intensity={settings['policy_effect_strength']}"
    )


def build_llm_simulation_prompt(settings):
    run_count = int(settings["llm_simulation_runs"])
    derived_flagged_rate = settings.get("derived_flagged_rate", settings.get("high_risk_threshold", 0.0))
    selected_policy_scenario = choose_policy_scenario(
        settings["policy"],
        settings["policy_effect_strength"],
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
        "policy_effect_strength": policy_intensity_value(settings["policy_effect_strength"]),
        "required_run_metric_columns": RUN_METRIC_COLUMNS,
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
