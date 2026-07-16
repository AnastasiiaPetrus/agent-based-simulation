import unittest
from unittest.mock import patch

from src.constants import POLICIES
from src.ui import run_llm_model_task
from src.weighted_agents import LIFE_STAGES, build_weighted_agents, choose_shared_policy_scenarios


SETTINGS = {
    "population_size": 10_000,
    "no_policy_outcome_rate": 0.01,
    "symmetric_error_rate": 0.01,
    "derived_flagged_rate": 0.0198,
    "policy_intensity_tier": "Medium",
    "llm_simulation_runs": 1,
    "weighted_profiles_per_group": 6,
    "llm_agent_models": ["gpt-4o-mini"],
}


class BatchedModelPipelineTests(unittest.TestCase):
    def test_one_model_response_produces_all_three_policy_results(self):
        agents = build_weighted_agents(SETTINGS)
        scenarios = choose_shared_policy_scenarios(SETTINGS)
        raw = {
            "agent_trajectories": [
                {
                    "policy": policy,
                    "agent_id": agent["agent_id"],
                    "stage_scores": [[0, 0, 0, 0, 0] for _ in LIFE_STAGES],
                    "mechanism": "No material policy-associated change.",
                }
                for policy in POLICIES
                for agent in agents
            ],
            "policy_debriefs": [
                {"policy": policy, "text": "A complete synthetic policy explanation."}
                for policy in POLICIES
            ],
        }
        usage = {"input_tokens": 5_000, "output_tokens": 2_000, "total_tokens": 7_000}

        with patch("src.ui.run_openai_json", return_value=(raw, usage)) as api_call:
            result = run_llm_model_task(SETTINGS, "gpt-4o-mini", agents, scenarios)

        api_call.assert_called_once()
        self.assertEqual(len(result["run_results"]), len(POLICIES))
        self.assertEqual(set(result["run_results"]["policy"].astype(str)), set(POLICIES))
        self.assertEqual(len(result["weighted_agent_results"]), len(agents) * len(POLICIES))
        self.assertEqual(len(result["policy_summaries"]), len(POLICIES))
        self.assertEqual(result["usage"], usage)
        self.assertEqual(result["estimated_cost_usd"], 0.00195)


if __name__ == "__main__":
    unittest.main()
