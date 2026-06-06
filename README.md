# Predictive Justice Thought Experiment

Suppose we could reliably predict, at age 10, who will commit a serious harmful act by age 30. What should we do with that information? This Streamlit app compares three policy responses to that question using synthetic AI-generated life-course scenarios.

The app is not a real-world decision tool. It does not use real justice-system data, personal data, protected-class data, demographic proxies, or real locations. No real children are predicted or evaluated. All agents, flags, trajectories, and outcomes are synthetic.

## What The App Does

The app uses selected AI models to generate policy-effect estimates from synthetic life-course scenarios. Python fixes the full prediction structure first, then the AI model estimates effects only for the relevant groups: flagged children plus missed children. The large true-negative background remains in the arithmetic but is not individually simulated.

Each LLM response must return JSON with:

- `policy_effects`: only the estimated policy-effect counts Python cannot know directly.
- `representative_agents`: a few abstract synthetic child trajectories.
- `debrief_text`: a concise explanation of the mechanisms and trade-offs.

Before showing results, the app validates and normalizes the LLM output. It checks required rows, rejects missing metric values, removes negative counts, caps policy-effect counts to Python-computed prediction groups, and computes the final run-level metrics in Python.

During a run, the app updates a live policy comparison view after each model-policy response arrives. It shows three panels, one per policy, with the same 10,000 synthetic children in the same positions. Dots start from the shared no-policy-action baseline and then transition to the outcome implied by each policy:

- Green: no target harmful outcome or target outcome prevented.
- Red: target-outcome path / outcome remains.
- Orange: wrongly flagged and harmed by an intervention.
- Yellow outline: flagged by the prediction.

This animation is an aggregate visualization, not 10,000 individually returned LLM records.

## Policies Compared

The app automatically runs all three policies:

- **Targeted support for flagged children**: flagged children receive voluntary developmental support. Some may benefit; false positives receive unnecessary intervention; false negatives receive no support.
- **Surveillance of flagged children**: flagged children are monitored without consent. It may deter some target outcomes, but can also create stigma, distrust, and disengagement.
- **Coercive prevention for flagged children**: flagged children face state-imposed restrictions before any act. It may reduce some target harmful outcomes, but has the highest ethical danger and harmful exposure.

There is no user-selected policy dropdown. The point is comparison across all policies under the same assumptions.

## Current Inputs

The sidebar intentionally keeps only the main assumptions:

- **Synthetic population**: fixed at 10,000 children.
- **No-policy outcome rate (%)**: share of the population whose no-policy trajectory would include the target harmful outcome; limited to 1-10%.
- **Symmetric misclassification rate (%)**: shared false-negative and false-positive rate; limited to 0-10%.
- **Intervention intensity**: low, medium, or high scenario assumption for how structured, frequent, broad, or restrictive the selected policy measure is.

The number of children flagged by the prediction is calculated from population size, no-policy outcome rate, and symmetric misclassification rate.

The app currently uses fixed simulation settings:

- `llm_simulation_runs = 5`
- `llm_representative_agents = 6`

## Models

AI models are configured through environment variables rather than the visible sidebar.

Optional variables:

```bash
LLM_MODEL=gpt-4o-mini
LLM_AGENT_MODELS=gpt-4o-mini,gpt-4.1-mini
```

`LLM_MODEL` sets the first preferred model. `LLM_AGENT_MODELS` controls the available model list. By default, the app selects up to four configured AI models, starting with `LLM_MODEL` and then the built-in defaults.

Each selected model makes one OpenAI API call per policy. With four default AI models and three policies, one full run makes twelve OpenAI API calls.

## Metrics

The app separates computed prediction counts from AI-estimated policy effects:

- **No-policy target outcomes**: computed count before any policy action.
- **Flagged by prediction**: children exposed to the selected policy response.
- **False positives**: flagged children who would not have had the target harmful outcome in the simulation's assumed no-policy truth.
- **False negatives**: unflagged children who would have had the target harmful outcome in the simulation's assumed no-policy truth.
- **Precision among flagged children**: true positives divided by all flagged children, also called positive predictive value.
- **Net target outcomes prevented**: AI-estimated target outcomes prevented or added compared with no policy action.
- **Policy benefit count**: flagged children whose AI-generated life-course scenario improves because of the policy.
- **Policy harm count**: flagged children whose AI-generated life-course scenario worsens because of the policy.

The app deliberately does not calculate total cost, total harm, dollar values, utility scores, recall, or per-outcome ratios.

## Outputs

After a successful run, the app shows:

- Live progress while model-policy calls complete.
- Animated 10,000-child population view based on the latest received aggregate result.
- Combined totals across all selected AI models and synthetic runs.
- Net target-outcome effect for each policy.
- Per-policy average tables.
- Line charts across synthetic runs.
- Per-policy interpretation text.
- Per-model AI explanations.
- Representative synthetic trajectories.
- CSV download for run-level results.
- In-session run log.

The run log is stored only in the current Streamlit session and is limited to the latest five entries.

## Install

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## Railway Setup

On Railway, add this service variable:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Important Limitations

This app is a thought experiment. It does not predict real behavior, estimate real-world harmful-outcome risk, or recommend policy. The outputs are synthetic and depend on selected assumptions and LLM-generated scenario data.

Prediction is not destiny. Children should not be punished for a predicted future act.
