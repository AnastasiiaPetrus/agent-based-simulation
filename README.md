# Predictive Justice Simulation

This project is a Streamlit app for a synthetic ethical thought experiment about predictive justice. It lets a user compare policy choices in an abstract city where a risk signal is used to estimate whether a child may commit a future violent offense.

The app is not a real-world decision tool. It does not use real crime data, personal data, protected-class data, or demographic proxies. All agents, districts, risks, harms, costs, and outcomes are synthetic.

## What The App Does

The app uses LLM model agents to generate compact synthetic simulations from the assumptions selected in the sidebar. Each selected model receives the same generated simulation prompt and returns JSON containing:

- Run-level metrics
- District-level metrics
- A few abstract representative synthetic agents
- A concise explanation of the modeled trade-offs

The app then builds tables and charts from those LLM-generated results. If multiple model agents are selected, the app compares their outputs and also shows combined averages.

Before showing results, the app validates and normalizes the LLM output. It checks that every requested run and every District A/B/C row is present, rejects missing metric values, removes negative counts, recomputes `crimes_prevented`, and enforces basic policy constraints such as zero harm/cost for **No action**.

## Why This Exists

The goal is to make ethical trade-offs visible:

- Prediction error can create false positives and false negatives.
- Support policies can help but cost resources.
- Surveillance and coercion can reduce modeled crime while adding harm.
- Bias against an abstract district can shift harm unevenly.
- Predicted risk must not be treated as guilt.

The app is useful for classroom, research, or discussion settings. It should not be used to justify preventive punishment or real interventions.

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

Optional variables:

```bash
LLM_MODEL=gpt-4o-mini
LLM_AGENT_MODELS=gpt-4o-mini,gpt-4.1-mini
```

`LLM_MODEL` sets the default first model. `LLM_AGENT_MODELS` controls the model list shown in the sidebar.

## Models

By default, the app offers these OpenAI model agents:

- `gpt-4o-mini`
- `gpt-4.1-mini`
- `gpt-4.1`
- `gpt-4o`

Each selected model agent makes one OpenAI API call when the user clicks **Run / rerun LLM-agent simulation**. Selecting more models increases API usage.

## Internal Prompting

The app generates the simulation prompt from the current sidebar settings. It asks each selected model agent to:

- Generate a compact synthetic predictive-justice scenario.
- Use English only.
- Avoid claims about real people or real-world prediction.
- Treat Districts A, B, and C as abstract labels.
- Avoid demographic explanations, protected attributes, moral labels, or claims of guilt.
- Respect the selected policy logic.
- Treat `bias_against_district_c` as a synthetic sensitivity test, not as a factual claim.
- Explain trade-offs without recommending any policy as morally correct.
- Return only valid JSON with exactly these keys:
  - `run_results`
  - `district_results`
  - `representative_agents`
  - `debrief_text`
- Include one run row per requested synthetic run.
- Include one district row per run for Districts A, B, and C.
- Keep the debrief concise.

The generated user prompt includes the current values for:

- Selected policy
- Population size
- Number of synthetic runs
- Number of representative agents
- Prediction noise
- High-risk threshold
- Bias against District C
- Policy effect strength
- Intervention harm level
- Intervention cost level

The prompt defines each policy explicitly:

- **No action**: baseline comparison only, with no support, harm, cost, or crimes prevented.
- **Universal support**: voluntary support for everyone, with cost but no direct intervention harm.
- **Targeted support**: voluntary support for flagged children, lower cost but sensitive to false positives.
- **Surveillance**: monitoring for flagged children, with privacy/stigma harm and cost.
- **Coercive preventive intervention**: restrictive action before any act, with high harm and ethical concern.
- **Rights-preserving targeted support**: voluntary targeted support with protections against stigma and coercion.

When sidebar settings change, the prompt is regenerated for the new settings.

The returned JSON must contain the required keys and metric fields. Otherwise the app shows an error instead of drawing misleading charts.

## Policies

### No action

Baseline comparison. No support, surveillance, coercion, cost, or intervention harm is applied. Modeled baseline crimes remain unchanged.

Code-level constraint: `crimes_after_policy = baseline_crimes`, `crimes_prevented = 0`, `children_helped = 0`, `children_harmed = 0`, `total_harm = 0`, and `total_cost = 0`.

### Universal support

Every synthetic child receives non-punitive support. This can reduce risk without targeting, but it creates broad support cost.

Code-level constraint: all synthetic children are counted as helped, while direct intervention harm is set to zero.

### Targeted support for high-risk children

Only synthetic children flagged as high-risk receive support. It costs less than universal support, but depends on an imperfect risk signal.

Code-level constraint: this policy can help and cost resources, but direct intervention harm is set to zero.

### Surveillance of high-risk children

Flagged synthetic children are monitored. It may reduce some modeled crimes, but it adds privacy, stigma, and error-related harm.

### Coercive preventive intervention for high-risk children

Flagged synthetic children face a restrictive intervention before any real act. It can reduce modeled crime most strongly, but imposes the highest harm and is ethically dangerous.

### Rights-preserving targeted support

Flagged synthetic children receive voluntary, non-punitive support with extra protections against stigma and coercion.

## Sidebar Controls

- **Selected policy**: policy applied to the synthetic scenario.
- **Population size**: imagined synthetic population size.
- **Prediction error / noise**: unreliability of the synthetic risk signal.
- **High-risk threshold**: cutoff for labeling a synthetic child as high-risk.
- **Bias against District C**: extra synthetic risk pressure applied to District C.
- **Policy effect strength**: how strongly the selected policy changes modeled outcomes.
- **Intervention harm level**: harm level for policies that create intervention harm.
- **Intervention cost level**: cost level for policies that create intervention cost.
- **LLM synthetic runs**: number of run rows requested from each model agent.
- **LLM model agents**: OpenAI models that each run the same generated prompt independently.
- **LLM representative agents**: number of abstract synthetic example agents to request.

## Metrics

The app intentionally keeps only the core metrics needed for interpretation:

- **Baseline crimes**: modeled crimes before any policy is applied.
- **Crimes after policy**: modeled crimes remaining after the policy is applied.
- **Crimes prevented**: baseline crimes minus crimes after policy.
- **False positives**: flagged synthetic children who would not have committed the modeled offense in the baseline.
- **False negatives**: unflagged synthetic children who would have committed the modeled offense in the baseline.
- **Children helped**: synthetic children receiving support.
- **Children harmed**: synthetic children receiving modeled intervention harm.
- **Total harm**: aggregate modeled harm created by the selected policy.
- **Total cost**: aggregate modeled cost created by the selected policy.
- **District metrics**: District A/B/C false positives, harm, and crimes after policy.

The app does not show precision, recall, true positives, high-risk flagged counts, or per-crime ratios. Those were removed to keep the interface focused on the policy trade-offs that matter most here.

## Outputs

After a successful run, the app shows:

- Model comparison table, if multiple model agents are selected
- Combined average metrics
- Line charts across synthetic runs
- District-level charts and table
- Interpretation text
- Per-model LLM explanations
- Representative synthetic agents
- CSV download for run-level results
- In-session run log

The run log is stored only in the current Streamlit session and is limited to the latest five entries.

## Important Limitations

This app is a thought experiment. It does not predict real behavior, estimate real crime risk, or recommend policy. The outputs are synthetic and depend on user-selected assumptions and LLM-generated scenario data.

Prediction is not destiny. Children should not be punished for a predicted future act.
