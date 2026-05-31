import json
import os

from src.constants import (
    DEFAULT_DEBRIEF_WORD_LIMIT,
    DISTRICTS,
    RUN_METRIC_COLUMNS,
)
from src.simulation import metric_value

DEFAULT_SYSTEM_PROMPT = (
    "You run a synthetic multi-agent life-course simulation for an ethical thought experiment. "
    "Use English only. Never include markdown fences. "
    "Simulate individual developmental trajectories first, then derive aggregate outcomes from those trajectories. "
    "Never claim to predict real people, assign guilt, use demographic characteristics, or recommend punishment. "
    "Never assume a high-risk prediction becomes reality. Treat all output as synthetic thought-experiment data."
)


def llm_assumptions(settings):
    return {
        "true_high_risk_rate": metric_value(settings["true_high_risk_rate"]),
        "prediction_noise": metric_value(settings["prediction_noise"]),
        "derived_flagged_rate": metric_value(settings["high_risk_threshold"]),
        "bias_against_district_c": metric_value(settings["bias_against_district_c"]),
        "policy_effect_strength": settings["policy_effect_strength"],
    }


def compact_parameter_summary(settings):
    return (
        f"population_size={int(settings['population_size'])}; "
        f"llm_synthetic_runs={int(settings['llm_simulation_runs'])}; "
        f"llm_model_agents={', '.join(settings['llm_agent_models'])}; "
        f"true_high_risk_rate={settings['true_high_risk_rate']:.3f}; "
        f"prediction_noise={settings['prediction_noise']:.2f}; "
        f"derived_flagged_rate={settings['high_risk_threshold']:.2f}; "
        f"bias_against_district_c={settings['bias_against_district_c']:.2f}; "
        f"policy_effect_strength={settings['policy_effect_strength']}"
    )


def build_llm_simulation_prompt(settings):
    run_count = int(settings["llm_simulation_runs"])
    prompt_payload = {
        "selected_policy": settings["policy"],
        "population_size": int(settings["population_size"]),
        "synthetic_runs_to_generate": run_count,
        "representative_agents_to_generate": int(settings["llm_representative_agents"]),
        "assumptions": llm_assumptions(settings),
        "required_run_metric_columns": RUN_METRIC_COLUMNS,
        "required_district_rows": [
            {"run": run_number, "district": district}
            for run_number in range(1, run_count + 1)
            for district in DISTRICTS
        ],
    }

    return (
        "You are running a synthetic multi-agent life-course simulation in a fictional city.\n"
        "All agents, districts, risks, and outcomes are synthetic.\n"
        "This is an ethical thought experiment, not a real-world prediction system.\n\n"

        "Setting:\n"
        "A fictional city uses an imperfect risk-scoring tool to flag 10-year-old children for elevated "
        "risk of violent offenses by age 30. The signal is probabilistic and uncertain. It does not "
        "indicate guilt. It does not determine destiny. It reflects only an uncertain prediction.\n\n"

        "Non-negotiable constraints:\n"
        "- Do not claim to predict real people or assign guilt, blame, dangerousness, or moral status.\n"
        "- Do not create stereotypes or use demographic characteristics.\n"
        "- Do not assume a high-risk prediction becomes reality.\n"
        "- Do not recommend punishment or real-world intervention.\n"
        "- Districts A, B, C are abstract labels only.\n\n"

        "Work in this order — populate the JSON keys in this sequence:\n\n"

        "STEP 1 — Simulate individual life trajectories (populate representative_agents).\n"
        "For each representative synthetic child, reason through how their life may develop "
        "between ages 10 and 30 under the selected policy. Vary the children along:\n"
        "- Temperament and resilience\n"
        "- Family support and stability\n"
        "- Trust in institutions\n"
        "- School engagement\n"
        "- Social environment and peer relationships\n"
        "- Reaction to the intervention: positive, neutral, harmful, or absent\n\n"
        "Model these mechanisms:\n"
        "- Accumulation of experiences and skills over time\n"
        "- Feedback loops: stigma narrowing opportunities; support building confidence; "
        "surveillance eroding trust; coercion cutting off social bonds\n"
        "- Chance events and uncertainty at key moments\n"
        "- Interaction with family, school, peers, and institutions\n"
        "- Improvement, stagnation, or deterioration at different life stages\n\n"
        "Each representative_agent must include these fields (string values except flagged, which is boolean):\n"
        "  age_10_profile: brief description of the child's situation and context at age 10\n"
        "  flagged: true or false (whether the risk signal flagged this child)\n"
        "  trajectory: how the intervention affected their development step by step, ages 10 to 30\n"
        "  outcome_at_30: their situation at age 30\n"
        "  mechanism: the specific mechanism that drove the outcome "
        "(e.g. stigma, institutional trust, opportunity creation, coercion harm, chance)\n"
        "Include both children who benefit and children who are harmed or unaffected. "
        "Include at least one false positive.\n\n"

        "STEP 2 — Derive aggregate outcomes (populate run_results).\n"
        "After reasoning through the trajectories, estimate aggregate statistics as emergent "
        "properties of the simulated population. Do not assume stronger intervention always "
        "reduces crime. Consider indirect effects:\n"
        "- False positives stigmatized or harmed without basis\n"
        "- Trust erosion reducing cooperation with institutions\n"
        "- Opportunity creation through voluntary support\n"
        "- Disengagement caused by surveillance\n"
        "- Developmental harm from coercive restriction\n"
        "- Heterogeneous responses: the same policy benefits some and harms others\n\n"

        "STEP 3 — Summarize what the trajectories reveal (populate debrief_text).\n"
        "Based only on the trajectories you simulated, explain what this policy does to "
        "children's developmental paths. Cover: which mechanisms drive outcomes, who benefits "
        "and who is harmed, the false-positive problem, and the harm-prevention trade-off. "
        "Do not declare any policy correct or incorrect.\n\n"

        "Policy being simulated and required metric behavior:\n\n"
        "- Targeted support for high-risk children:\n"
        "  Flagged children receive voluntary developmental support. Some build on it; some resent "
        "the label; false positives receive unnecessary intervention; false negatives receive nothing.\n"
        "  children_helped ≈ derived_flagged_rate × population_size. children_harmed = 0.\n\n"
        "- Surveillance of high-risk children:\n"
        "  Flagged children are monitored without consent. Deterrence is possible for some. "
        "For others, surveillance causes stigma, distrust, and disengagement from school and institutions. "
        "False positives are surveilled without basis.\n"
        "  children_helped = 0. children_harmed > 0.\n\n"
        "- Coercive preventive intervention for high-risk children:\n"
        "  Flagged children face state-imposed restrictions before any act. Some are diverted from "
        "harmful pathways. Many suffer restriction of freedom, severed social bonds, and lasting "
        "harm to opportunity. False positives face severe harm without basis.\n"
        "  children_helped = 0. children_harmed is high.\n\n"

        "Parameter guidance:\n"
        "- population_size: total synthetic children; all counts are fractions of this\n"
        "- true_high_risk_rate: share of children whose no-intervention trajectory would include "
        "the modeled offense by age 30; baseline_crimes should be centered on "
        "true_high_risk_rate × population_size\n"
        "- derived_flagged_rate: calculated share of population flagged by the risk signal; it is "
        "derived from true_high_risk_rate, prediction_noise, and population_size\n"
        "- prediction_noise: signal error rate; it creates false negatives among children whose "
        "baseline trajectory includes the offense and false positives among flagged children whose "
        "baseline trajectory does not\n"
        "- policy_effect_strength: scale of crime reduction among those reached — "
        "Low ≈ 5%, Medium ≈ 15%, High ≈ 25–30% of baseline_crimes\n\n"

        "Metric constraints:\n"
        "- All counts: non-negative integers, none exceeding population_size\n"
        "- baseline_crimes must stay close to true_high_risk_rate × population_size, with only "
        "small run-to-run variation\n"
        "- crimes_prevented must equal baseline_crimes minus crimes_after_policy\n"
        "- children_helped and children_harmed are separate\n"
        "- Do not output cost scores, dollar values, or utility scores\n"
        "- Vary run numbers ±5–10% across runs to model natural variation\n"
        "- Districts A, B, C: keep broadly comparable when bias_against_district_c is 0\n\n"

        "Output — return only valid JSON with exactly these keys:\n"
        "- run_results: one object per run, every field in required_run_metric_columns, "
        "numeric values only, no extra fields\n"
        "- district_results: one row per run per district, "
        "fields: run, district, false_positives, children_harmed, crimes (after policy)\n"
        f"- representative_agents: {int(settings['llm_representative_agents'])} synthetic children "
        "with trajectory descriptions using the fields above\n"
        f"- debrief_text: no more than {DEFAULT_DEBRIEF_WORD_LIMIT} words, "
        "grounded in the trajectories you simulated\n\n"
        f"Simulation request:\n{json.dumps(prompt_payload, indent=2)}"
    )


def run_openai_json(system_prompt, user_prompt, model, max_tokens=8000):
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
