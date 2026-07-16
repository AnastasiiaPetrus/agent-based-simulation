import unittest

from src.constants import POLICIES
from src.llm import (
    build_llm_simulation_prompt,
    estimate_model_cost_usd,
    weighted_agent_response_format,
)
from src.weighted_agents import (
    LIFE_STAGES,
    STATE_DIMENSIONS,
    aggregate_weighted_agent_trajectories,
    build_weighted_agents,
    choose_shared_policy_scenarios,
    normalize_agent_trajectories,
    normalize_policy_debriefs,
    scenario_seed,
)


def settings(outcome_rate=0.01, error_rate=0.01, intensity="Medium", profiles_per_group=6):
    return {
        "population_size": 10_000,
        "no_policy_outcome_rate": outcome_rate,
        "symmetric_error_rate": error_rate,
        "derived_flagged_rate": 0.0198,
        "policy_intensity_tier": intensity,
        "llm_simulation_runs": 1,
        "weighted_profiles_per_group": profiles_per_group,
        "llm_agent_models": ["gpt-4o-mini"],
    }


def trajectory_rows(agents, score_for_agent=None):
    rows = []
    for policy in POLICIES:
        for agent in agents:
            score = [0, 0, 0, 0, 0]
            if score_for_agent is not None:
                score = score_for_agent(agent)
            rows.append(
                {
                    "policy": policy,
                    "agent_id": agent["agent_id"],
                    "stage_scores": [list(score) for _ in LIFE_STAGES],
                    "mechanism": "A compact synthetic mechanism.",
                }
            )
    return rows


class WeightedAgentConstructionTests(unittest.TestCase):
    def test_default_weights_match_prediction_groups(self):
        agents = build_weighted_agents(settings())
        self.assertEqual(len(agents), 12)
        true_positive_weight = sum(
            agent["weight"] for agent in agents if agent["prediction_status"] == "true_positive"
        )
        false_positive_weight = sum(
            agent["weight"] for agent in agents if agent["prediction_status"] == "false_positive"
        )
        self.assertEqual(true_positive_weight, 99)
        self.assertEqual(false_positive_weight, 99)

    def test_maximum_sidebar_settings_represent_1800_flagged_children(self):
        agents = build_weighted_agents(settings(0.10, 0.10, "High"))
        self.assertEqual(len(agents), 12)
        self.assertEqual(sum(agent["weight"] for agent in agents), 1_800)
        self.assertEqual(
            sum(agent["weight"] for agent in agents if agent["prediction_status"] == "true_positive"),
            900,
        )
        self.assertEqual(
            sum(agent["weight"] for agent in agents if agent["prediction_status"] == "false_positive"),
            900,
        )

    def test_empty_false_positive_group_is_omitted(self):
        agents = build_weighted_agents(settings(0.01, 0.0))
        self.assertEqual(len(agents), 6)
        self.assertEqual(sum(agent["weight"] for agent in agents), 100)
        self.assertTrue(all(agent["prediction_status"] == "true_positive" for agent in agents))

    def test_profile_count_comes_from_settings(self):
        agents = build_weighted_agents(settings(profiles_per_group=4))
        self.assertEqual(len(agents), 8)
        self.assertEqual(
            sum(agent["weight"] for agent in agents if agent["prediction_status"] == "true_positive"),
            99,
        )
        self.assertEqual(
            sum(agent["weight"] for agent in agents if agent["prediction_status"] == "false_positive"),
            99,
        )

    def test_invalid_profile_count_is_rejected(self):
        for profile_count in (0, 7):
            with self.subTest(profile_count=profile_count):
                with self.assertRaisesRegex(ValueError, "profiles_per_group"):
                    build_weighted_agents(settings(profiles_per_group=profile_count))

    def test_scenario_is_deterministic_and_shared(self):
        current_settings = settings(0.055, 0.025, "High")
        self.assertEqual(build_weighted_agents(current_settings), build_weighted_agents(current_settings))
        self.assertEqual(
            choose_shared_policy_scenarios(current_settings),
            choose_shared_policy_scenarios(current_settings),
        )
        self.assertEqual(len(choose_shared_policy_scenarios(current_settings)), len(POLICIES))
        self.assertEqual(
            [scenario["policy"] for scenario in choose_shared_policy_scenarios(current_settings)],
            POLICIES,
        )
        self.assertEqual(scenario_seed(current_settings), scenario_seed(current_settings))


class WeightedAgentContractTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings()
        self.agents = build_weighted_agents(self.settings)

    def test_complete_trajectory_contract(self):
        normalized = normalize_agent_trajectories(trajectory_rows(self.agents), self.agents)
        self.assertEqual(len(normalized), len(self.agents) * len(POLICIES))
        self.assertTrue(
            all(
                len(row["stage_scores"]) == len(LIFE_STAGES)
                and all(len(stage) == len(STATE_DIMENSIONS) for stage in row["stage_scores"])
                for row in normalized
            )
        )

    def test_missing_trajectory_is_rejected(self):
        rows = trajectory_rows(self.agents)
        with self.assertRaisesRegex(ValueError, "incomplete weighted-agent trajectories"):
            normalize_agent_trajectories(rows[:-1], self.agents)

    def test_mechanism_is_limited_to_25_words(self):
        rows = trajectory_rows(self.agents)
        rows[0]["mechanism"] = " ".join(f"word-{index}" for index in range(30))
        normalized = normalize_agent_trajectories(rows, self.agents)
        self.assertEqual(len(normalized[0]["mechanism"].split()), 25)

    def test_missing_debrief_is_rejected(self):
        debriefs = [{"policy": policy, "text": "Synthetic explanation."} for policy in POLICIES[:-1]]
        with self.assertRaisesRegex(ValueError, "incomplete policy debriefs"):
            normalize_policy_debriefs(debriefs)

    def test_zero_scores_produce_zero_policy_effects(self):
        trajectories = normalize_agent_trajectories(trajectory_rows(self.agents), self.agents)
        effects, details = aggregate_weighted_agent_trajectories(trajectories, self.agents)
        self.assertEqual(len(details), len(self.agents) * len(POLICIES))
        for policy in POLICIES:
            self.assertEqual(
                effects[policy][0],
                {
                    "run": 1,
                    "prevented_outcomes": 0,
                    "policy_caused_outcomes": 0,
                    "children_helped": 0,
                    "children_harmed": 0,
                },
            )

    def test_scores_are_weighted_into_counts(self):
        def score_for_agent(agent):
            if agent["prediction_status"] == "true_positive":
                return [1, 1, 1, 1, -1]
            return [-1, -1, -1, -1, 1]

        trajectories = normalize_agent_trajectories(
            trajectory_rows(self.agents, score_for_agent),
            self.agents,
        )
        effects, _ = aggregate_weighted_agent_trajectories(trajectories, self.agents)
        for policy in POLICIES:
            row = effects[policy][0]
            self.assertEqual(row["prevented_outcomes"], 64)
            self.assertEqual(row["policy_caused_outcomes"], 25)
            self.assertEqual(row["children_helped"], 99)
            self.assertEqual(row["children_harmed"], 99)

    def test_prompt_contains_every_agent_and_policy(self):
        scenarios = choose_shared_policy_scenarios(self.settings)
        prompt = build_llm_simulation_prompt(self.settings, self.agents, scenarios)
        for policy in POLICIES:
            self.assertIn(policy, prompt)
        for agent in self.agents:
            self.assertIn(agent["agent_id"], prompt)
        self.assertIn('"stage_order"', prompt)
        self.assertIn('"state_dimension_order"', prompt)

    def test_structured_output_schema_has_exact_dynamic_size(self):
        scenarios = choose_shared_policy_scenarios(self.settings)
        response_format = weighted_agent_response_format(self.agents, scenarios)
        schema = response_format["json_schema"]["schema"]
        trajectories = schema["properties"]["agent_trajectories"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(trajectories["minItems"], len(self.agents) * len(POLICIES))
        self.assertEqual(trajectories["maxItems"], len(self.agents) * len(POLICIES))

    def test_standard_token_cost_uses_model_prices(self):
        cost = estimate_model_cost_usd(
            "gpt-4o-mini",
            {"input_tokens": 10_000, "output_tokens": 2_000},
        )
        self.assertEqual(cost, 0.0027)
        self.assertIsNone(estimate_model_cost_usd("unknown-model", {}))


if __name__ == "__main__":
    unittest.main()
