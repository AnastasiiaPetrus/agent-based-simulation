# Predictive Justice Thought Experiment

Suppose we could reliably predict, at age 10, who will become a violent criminal by age 30. What should we do with that information? This Streamlit app compares three policy responses to that question using synthetic LLM-generated life-course trajectories.

The app is not a real-world decision tool. It does not use real crime data, personal data, protected-class data, demographic proxies, or real locations. All agents, districts, risks, trajectories, and outcomes are synthetic.

## What The App Does

The app uses LLM model agents to generate a small life-course simulation. For each policy and model, the LLM simulates synthetic developmental trajectories first, then derives aggregate metrics from those trajectories.

Each LLM response must return JSON with:

- `run_results`: run-level outcome metrics.
- `district_results`: District A/B/C outcome rows for each run.
- `representative_agents`: a few abstract synthetic child trajectories.
- `debrief_text`: a concise explanation of the mechanisms and trade-offs.

Before showing results, the app validates and normalizes the LLM output. It checks required rows, rejects missing metric values, removes negative counts, recomputes `crimes_prevented`, applies policy constraints, and reconciles district-level totals with run-level totals.

During a run, the app updates a live policy comparison view after each model-policy response arrives. It shows three panels, one per policy, with the same 1,000 synthetic children in the same positions. Dots start from the shared baseline and then transition to the outcome implied by each policy:

- Green: no modeled offense or offense prevented.
- Red: true high-risk / offense remains.
- Orange: wrongly flagged and harmed by an intervention.
- Yellow outline: flagged by the prediction.

This animation is an aggregate visualization, not 1,000 individually returned LLM records.

## Policies Compared

The app automatically runs all three policies:

- **Targeted support for high-risk children**: flagged children receive voluntary developmental support. Some may benefit; false positives receive unnecessary intervention; false negatives receive no support.
- **Surveillance of high-risk children**: flagged children are monitored without consent. It may deter some offenses, but can also create stigma, distrust, and disengagement.
- **Coercive preventive intervention for high-risk children**: flagged children face state-imposed restrictions before any act. It may reduce some modeled offenses, but has the highest ethical danger and harmful exposure.

There is no user-selected policy dropdown. The point is comparison across all policies under the same assumptions.

## Current Inputs

The sidebar intentionally keeps only the main assumptions:

- **Synthetic population**: fixed at 1,000 children.
- **Percentage of true high-risk children (%)**: share of the population whose no-policy trajectory would include the modeled offense.
- **Prediction error rate (%)**: how noisy the risk signal is, creating false positives and false negatives.
- **Intervention strength**: low, medium, or high scenario assumption for how strongly the policy may affect modeled offenses.

The number of children flagged by the prediction is calculated from population size, true high-risk percentage, and prediction error rate.

The app currently uses fixed simulation settings:

- `llm_simulation_runs = 5`
- `llm_representative_agents = 2`
- `bias_against_district_c = 0.0`

## Models

Models are configured through environment variables rather than the visible sidebar.

Optional variables:

```bash
LLM_MODEL=gpt-4o-mini
LLM_AGENT_MODELS=gpt-4o-mini,gpt-4.1-mini
```

`LLM_MODEL` sets the first preferred model. `LLM_AGENT_MODELS` controls the available model list. By default, the app runs the preferred model and `gpt-4.1-mini` when available.

Each selected model makes one OpenAI API call per policy. With two default model agents and three policies, one full run makes six OpenAI API calls.

## Metrics

The app keeps the metrics count-based to avoid false precision:

- **Children who would offend (no intervention)**: baseline count before the policy.
- **Offenses prevented by policy**: baseline offenses minus offenses after policy.
- **Crime reduction (%)**: offenses prevented as a percentage of baseline offenses.
- **Children incorrectly flagged**: false positives, meaning flagged children who would not have committed the modeled offense.
- **Children missed by risk signal**: false negatives, meaning unflagged children who would have committed the modeled offense.
- **Children receiving support**: children reached by voluntary support.
- **Children exposed to harmful intervention**: children exposed to surveillance, coercion, or residual stigma.

The app deliberately does not calculate total cost, total harm, dollar values, utility scores, precision, recall, or per-crime ratios.

## Outputs

After a successful run, the app shows:

- Live progress while model-policy calls complete.
- Animated 1,000-child population view based on the latest received aggregate result.
- Combined totals across all selected model agents and synthetic runs.
- Crime reduction percentage for each policy.
- Per-policy average tables.
- Line charts across synthetic runs.
- Per-policy interpretation text.
- Per-model LLM explanations.
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

This app is a thought experiment. It does not predict real behavior, estimate real crime risk, or recommend policy. The outputs are synthetic and depend on selected assumptions and LLM-generated scenario data.

Prediction is not destiny. Children should not be punished for a predicted future act.
