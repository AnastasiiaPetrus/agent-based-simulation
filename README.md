# Predictive Justice Simulation

This project is a small Streamlit app for exploring a synthetic ethical thought experiment about predictive justice. It models an abstract city where a prediction system estimates, at age 10, the risk that a child will commit a violent crime by age 30.

The simulation does not recommend real policy. It shows how assumptions about prediction error, thresholds, intervention effects, costs, harm, and district-level bias can change aggregate outcomes.

All data is synthetic. The model does not use real crime data, race, ethnicity, protected-class data, or personal data.

## Install dependencies

```bash
pip install -r requirements.txt
```

The OpenAI package is included only for the optional LLM-agent debrief mode. The default simulation does not require an API key and does not call any LLM API.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## How to use the app

1. Choose assumptions in the sidebar (population size, noise, threshold, policy, costs, harms, and bias).
2. The app runs repeated Monte Carlo simulations (many independent random runs) using the selected random seed.
3. Review:
   - "Average results across Monte Carlo runs" for the main aggregated metrics.
   - The charts for variability across runs and district-level differences.
   - The "Interpretation" block for a short, automatic summary of trade-offs under the current assumptions.
4. Click "Download results as CSV" to export run-level metrics to:
   `predictive_justice_simulation_results.csv`

## Optional LLM-agent debrief mode

The app includes an optional lightweight LLM-agent debrief mode. It is disabled by default.

When disabled:

- No API key is needed.
- No LLM requests are made.
- No tokens are spent.
- The Monte Carlo simulation remains fully NumPy/Pandas-based.
- Synthetic agents are still rows in a DataFrame, not AI agents.

When enabled in the sidebar, the app shows a "Lightweight LLM-Agent Debrief" section. The LLM is called only when the user clicks "Run one LLM-agent debrief". It is not called automatically on Streamlit reruns, inside Monte Carlo loops, or once per child.

The LLM debrief can summarize trade-offs, uncertainty, false positives, false negatives, harm, cost, and District C bias. It does not change risks, high-risk flags, outcomes, metrics, or charts.

To enable it locally or on Railway, set:

```bash
OPENAI_API_KEY=your_api_key
```

Optional model override:

```bash
LLM_MODEL=gpt-4o-mini
```

On Railway, add these as service variables or secrets.

The LLM-agent run log is stored only in the current Streamlit session. It may disappear after refresh, restart, redeploy, or server sleep. This is intentional to keep the app simple and privacy-preserving.

## Compared policies

- No action
- Universal support
- Targeted support for high-risk children
- Surveillance of high-risk children
- Coercive preventive intervention for high-risk children
- Rights-preserving targeted support with periodic reassessment

## What the simulation shows

The app runs repeated Monte Carlo simulations and reports average outcomes such as baseline crimes, crimes after policy, crimes prevented, false positives, false negatives, children helped, children harmed, total harm, total cost, and district-level differences.

The exported CSV contains run-level metrics plus district-level columns for false positives, harm, and crimes after policy in Districts A, B, and C.

## Sidebar parameters (what each control means)

The sidebar lets you change:

- Population size
- Number of Monte Carlo runs
- Random seed
- Prediction error
- High-risk threshold
- Support effectiveness
- Surveillance effectiveness
- Intervention harms
- Bias against District C
- Intervention costs
- Selected policy
- Optional lightweight LLM-agent mode controls

Details:

- Population size: Number of synthetic child agents generated per Monte Carlo run.
- Number of Monte Carlo runs: How many independent runs to average over. More runs reduces randomness but takes longer.
- Random seed: Reproducibility control. With the same seed and settings, results should repeat.
- Prediction error / noise: How much random error is added when converting hidden true risk into the observed predicted risk.
- High-risk threshold: A cutoff applied to predicted risk. Children with `predicted_risk >= threshold` are flagged as high-risk.
- Support effectiveness: Fractional reduction applied to true risk for children receiving support (for example, 0.30 means a 30% reduction).
- Surveillance effectiveness: Smaller fractional reduction applied to true risk under surveillance (for high-risk children in that policy).
- Surveillance harm: Harm units assigned to each child placed under surveillance (in the surveillance policy).
- Coercive intervention harm: Harm units assigned to each child receiving coercive intervention (in the coercive policy).
- Bias against District C: Additive upward shift applied to predicted risk for District C only, illustrating how small bias can increase false positives and unequal harm.
- Support cost: Cost units assigned per child receiving support (universal support; targeted support; rights-preserving targeted support).
- Surveillance cost: Cost units assigned per child placed under surveillance (surveillance policy).
- Coercive intervention cost: Cost units assigned per coerced child (coercive policy).
- Selected policy: Which policy is applied in each Monte Carlo run.
- Enable lightweight LLM-agent mode: Shows the optional LLM debrief section. It does not call the LLM by itself.
- LLM display population size: Size of the small single-run synthetic sample used only for representative debrief context.
- LLM representative agents: Number of representative synthetic agents sent to the LLM, capped at 3.
- LLM output word limit: Maximum requested length of the generated debrief.
- LLM run log size: Number of previous in-session debriefs to keep.

Notes on interpretation:

- "True risk" is a hidden probability used only by the simulation.
- "Predicted risk" is the imperfect estimate used for high-risk flags.
- Rights-preserving targeted support uses a second synthetic reassessment before support is applied.
- False positives are flagged children whose baseline outcome would not include a crime.
- The districts A, B, and C are abstract labels and are not proxies for real demographic groups.

## Model limitations

This is an ethical thought experiment, not a predictive system. It uses simple synthetic formulas and random draws to illustrate trade-offs. It does not claim to predict real human behavior.

Prediction is not destiny. The app should not be used to justify preventive punishment, coercion, or real-world classification of children.

Districts A, B, and C are abstract labels only. They are not proxies for real demographic groups or real places.

## Why this is not a real-world decision tool

The model omits the legal, social, psychological, historical, and institutional realities that would matter in any real justice context. It also depends heavily on user-selected assumptions. For that reason, it is useful only as a classroom or research discussion aid about ethical trade-offs, not as evidence for real-world intervention decisions.
