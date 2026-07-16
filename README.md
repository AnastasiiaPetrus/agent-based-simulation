# Predictive Justice Thought Experiment

Suppose a fictional system predicts at age 10 who will commit a serious harmful act by age 30. What should society do with that information? This Streamlit app compares three policy responses through a compact weighted-agent simulation.

The app is a thought experiment, not a real-world decision tool. It uses no real personal data, justice-system data, protected traits, demographic proxies, or real locations. No real children are predicted or evaluated.

## Simulation Architecture

One click creates a single shared scenario:

1. Python calculates the fixed prediction groups for the 10,000-child cohort: true positives, false positives, false negatives, and true negatives.
2. Python creates six abstract true-positive profiles and six abstract false-positive profiles. Every profile has numeric traits, life contexts, and an integer weight. The weights add up exactly to the number of flagged children in the corresponding group.
3. Python chooses one concrete measure for each of the three policies. The same profiles, contexts, weights, and measures are sent to every selected AI model.
4. Each AI model evaluates all 12 weighted profiles under all three policies in one batched API call.
5. For every profile-policy combination, the model returns five life-stage score vectors covering ages 10-30. It does not return cohort totals.
6. Python validates complete coverage, converts the stage scores into benefit, harm, prevented-outcome, and policy-caused-outcome shares, multiplies those shares by profile weights, and calculates the final cohort metrics.
7. Results are averaged across the selected AI models. The interface also shows the range between their estimates.

This is a hybrid weighted-agent simulation: the AI models simulate the changing states of representative agents, while Python owns sampling, weights, validation, arithmetic, and population-level aggregation. It does not create 10,000 separate AI biographies.

## What One Agent Represents

A weighted agent is an abstract profile, not one child. For example, an agent with weight 17 represents 17 flagged children with the same simulation profile.

Profiles are deliberately mechanism-based rather than demographic. They vary in resilience, stability, institutional trust, engagement, peer support, responsiveness to help, and sensitivity to monitoring or restriction. Each profile also receives deterministic fictional life contexts, such as a change in routine or a missed opportunity.

The six profile templates are reused for true positives and false positives. Profiles with zero weight are omitted, so a run may contain fewer than 12 agents when a prediction group is empty.

## Why There Are Five Life Stages

The stages are `10-13`, `14-17`, `18-21`, `22-25`, and `26-30`. They are not five separate runs or five groups of children. They are five consecutive periods in the same agent's simulated path. This lets an effect appear early, accumulate, reverse, or remain neutral instead of forcing one score for the entire 20-year period.

At every stage, the model scores policy-associated change in:

- wellbeing;
- trust;
- opportunity;
- autonomy;
- stress.

Scores are integers from -2 to 2. Positive values are beneficial for the first four dimensions; for stress, a positive value means more stress and is treated as harmful.

## Policies Compared

The app automatically evaluates all three policies under the same scenario:

- **Targeted support for flagged children** — voluntary developmental or practical support.
- **Surveillance of flagged children** — monitoring, recording, or institutional scrutiny.
- **Coercive prevention for flagged children** — mandatory requirements or restrictions imposed before any harmful act.

## Inputs

The sidebar controls:

- **Synthetic population** — fixed at 10,000 children.
- **No-policy outcome rate** — share whose no-policy path includes the target harmful outcome; 1-10%.
- **Symmetric misclassification rate** — the same rate is applied to false negatives and false positives; 0-10%.
- **Intervention intensity** — low, medium, or high; determines the concrete policy-measure tier.
- **AI models** — up to four selected OpenAI models.

The weighted-agent structure is fixed at up to six true-positive and six false-positive profiles. There is one simulation scenario per click; there are no five synthetic runs.

## API Calls and Cost

Each selected model makes one batched OpenAI API call containing all profiles and all policies. With four selected models, a full simulation makes four calls, not twelve.

The app reads the actual input and output token counts returned by the API and displays an estimated token cost using the configured standard per-token prices. The estimate covers model tokens only and should be updated if provider pricing changes.

Optional model configuration:

```bash
LLM_MODEL=gpt-4o-mini
LLM_AGENT_MODELS=gpt-4o-mini,gpt-4.1-mini
```

`LLM_MODEL` sets the first preferred model. `LLM_AGENT_MODELS` controls the available model list. By default, the app selects up to four configured models.

## Metrics

Python computes directly from the settings:

- **No-policy target outcomes** — children whose path contains the target outcome without policy action.
- **Flagged by prediction** — true positives plus false positives; the group exposed to policy.
- **False positives** — flagged children who would not have had the target outcome.
- **False negatives** — unflagged children who would have had the target outcome.
- **Precision among flagged children** — true positives divided by all flagged children.

Python derives from AI-generated stage scores and agent weights:

- **Net target outcomes prevented** — prevented outcomes minus policy-caused outcomes.
- **Policy benefit count** — weighted number of flagged children showing beneficial changes.
- **Policy harm count** — weighted number showing harmful changes.

A profile can show both benefit and harm across different dimensions or stages. These counts are scenario estimates, not empirical predictions.

## Outputs

After a successful simulation, the app shows:

- live progress by completed AI-model call;
- an aggregate 10,000-child policy comparison view;
- AI-model-average metrics for every policy;
- ranges across selected models;
- per-model explanations;
- weighted profile effects and weights;
- actual API token use and estimated token cost;
- an in-session log of the latest five runs.

## Install and Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

On Railway, configure `OPENAI_API_KEY` as a service variable.

## Important Limitations

The conversion from stage scores to outcome shares is an explicit modelling assumption, not an empirically calibrated causal model. Results depend on the selected assumptions, generated profiles, prompt, model version, and provider behavior. The design makes those steps inspectable and holds the scenario constant across policies and models, but it does not establish real-world validity.

Prediction is not destiny. Children should not be punished for a predicted future act.
