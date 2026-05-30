# Predictive Justice Simulation

This project is a small Streamlit app for exploring a synthetic ethical thought experiment about predictive justice. It models an abstract city where a prediction system estimates, at age 10, the risk that a child will commit a violent crime by age 30.

The simulation does not recommend real policy. It shows how assumptions about prediction error, thresholds, intervention effects, costs, harm, and district-level bias can change aggregate outcomes.

All data is synthetic. The model does not use real crime data, race, ethnicity, protected-class data, or personal data.

## Install dependencies

```bash
pip install -r requirements.txt
```

The app uses an LLM-agent simulation workflow. An OpenAI API key is required to generate new LLM-agent simulation runs.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## How to use the app

1. Choose assumptions in the sidebar (population size, noise, threshold, policy, costs, harms, and bias).
2. Configure the LLM-agent settings (synthetic runs, representative agents, output word limit, and run log size).
3. Click "Run LLM-agent simulation" to generate synthetic run metrics, district metrics, representative synthetic agents, and an explanation.
4. Review:
   - "LLM-agent simulation averages" for the main aggregated metrics.
   - The charts for variability across LLM-generated synthetic runs and district-level differences.
   - The "Latest LLM-agent explanation" block for a concise interpretation.
5. Click "Download LLM-agent results as CSV" to export run-level metrics to:
   `llm_agent_simulation_results.csv`

## LLM-agent simulation mode

The visible app flow is the LLM-agent simulation mode.

The LLM is called only when the user clicks "Run LLM-agent simulation". It is not called automatically on Streamlit reruns, inside loops, or once per child. This keeps token use limited and predictable.

The LLM-agent simulation asks the model to generate a small synthetic set of:

- Run-level metrics
- District-level metrics
- Up to 3 representative synthetic agents
- A concise explanation of trade-offs

It can summarize uncertainty, false positives, false negatives, harm, cost, and District C bias. It is still an ethical thought experiment, not a real-world prediction system.

To run it locally or on Railway, set:

```bash
OPENAI_API_KEY=your_api_key
```

Optional model override:

```bash
LLM_MODEL=gpt-4o-mini
```

On Railway, add these as service variables or secrets.

The LLM-agent run log and latest LLM-agent simulation results are stored only in the current Streamlit session. They may disappear after refresh, restart, redeploy, or server sleep. This is intentional to keep the app simple and privacy-preserving.

## Compared policies

- No action
- Universal support
- Targeted support for high-risk children
- Surveillance of high-risk children
- Coercive preventive intervention for high-risk children
- Rights-preserving targeted support with periodic reassessment

## What the simulation shows

The app asks the LLM for compact synthetic run metrics and district metrics, then builds tables and charts from those LLM-generated metrics.

The exported CSV contains run-level metrics plus district-level columns for false positives, harm, and crimes after policy in Districts A, B, and C.

## Sidebar parameters (what each control means)

The sidebar lets you change:

- Population size
- Prediction error
- High-risk threshold
- Bias against District C
- Selected policy
- Policy effect strength (shown only when the selected policy changes outcomes)
- Intervention harm level (shown only for policies that can create intervention harm)
- Intervention cost level (shown only for policies with intervention cost)
- LLM-agent controls

Details:

- Population size: Main scenario population reference shown in the sidebar.
- Prediction error / noise: Assumption passed to the LLM about how noisy the prediction system is.
- High-risk threshold: A cutoff applied to predicted risk. Children with `predicted_risk >= threshold` are flagged as high-risk.
- Bias against District C: Additive upward shift applied to predicted risk for District C only, illustrating how small bias can increase false positives and unequal harm.
- Selected policy: Which policy is applied in the LLM-agent simulation.
- Policy effect strength: Low, medium, or high assumption about how strongly the selected policy changes modeled outcomes. Hidden for "No action".
- Intervention harm level: Low, medium, or high assumption about intervention harm. Shown for surveillance, coercive intervention, and rights-preserving targeted support.
- Intervention cost level: Low, medium, or high assumption about intervention cost. Hidden for "No action".
- LLM synthetic runs: Number of run-level metric rows requested from the LLM, capped at 20.
- LLM representative agents: Number of representative synthetic agents requested from the LLM, capped at 3.
- LLM output word limit: Maximum requested length of the generated explanation.
- LLM run log size: Number of previous in-session debriefs to keep.

Notes on interpretation:

- "True risk" is a hidden probability used only by the simulation.
- "Predicted risk" is the imperfect estimate used for high-risk flags.
- Rights-preserving targeted support uses a second synthetic reassessment before support is applied.
- False positives are flagged children whose baseline outcome would not include a crime.
- The districts A, B, and C are abstract labels and are not proxies for real demographic groups.

## Model limitations

This is an ethical thought experiment, not a predictive system. It uses LLM-generated synthetic metrics to illustrate trade-offs. It does not claim to predict real human behavior.

Prediction is not destiny. The app should not be used to justify preventive punishment, coercion, or real-world classification of children.

Districts A, B, and C are abstract labels only. They are not proxies for real demographic groups or real places.

## Why this is not a real-world decision tool

The model omits the legal, social, psychological, historical, and institutional realities that would matter in any real justice context. It also depends heavily on user-selected assumptions. For that reason, it is useful only as a classroom or research discussion aid about ethical trade-offs, not as evidence for real-world intervention decisions.
