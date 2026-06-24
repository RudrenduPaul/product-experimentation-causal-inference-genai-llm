# Article 13 — Causal Inference in Production: Airbnb, Netflix, Lyft, and Uber

How four engineering teams measure the causal impact of AI feature changes in production. This folder implements all four case-study reference pipelines — Airbnb's future value framework, Netflix's quasi-experiment taxonomy, Lyft's doubly robust diagnostics, and Uber's causal forecasting pipeline — on the shared 50,000-user synthetic LLM product dataset.

## What this teaches

- Why short-term A/B test metrics (thumbs-up rates, 30-day task completion) miss long-term user behavior changes
- How to build a future-value proxy model and use it as the DiD outcome instead of immediate engagement signals
- How to match deployment structure to causal method: staged rollout → DiD, threshold routing → RDD, full-population upgrade → Synthetic Control, opt-in feature → IPW/AIPW
- Why doubly robust (AIPW) estimation gives you one free model misspecification — and when it still fails
- How to run Lyft's four production diagnostics: weight distribution check, trim threshold, covariate balance plots, placebo outcome test
- How to embed a causal estimate into forward-looking volume scenarios for routing and infrastructure decisions
- Bootstrap confidence intervals for three observational estimators (DiD, IPW ATE, RDD local effect)

## Run

```bash
# From repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
jupyter notebook 13_case_studies/case_studies_demo.ipynb
```

## What you should see

All values below are from executed runs against `synthetic_llm_logs.csv` (50,000 users, seed=42).

### Case study 1 — Airbnb DiD on future-value score

| Metric | Value |
|--------|-------|
| Future-value model R² | 0.024 |
| Mean future-value score (wave 1) | 0.6325 |
| Mean future-value score (wave 2) | 0.6271 |
| DiD effect on future-value score | +0.0059 |
| DiD 95% CI (500 bootstrap replicates, seed=7) | [+0.0023, +0.0093] |

### Case study 3 — Lyft IPW ATE with diagnostics

| Metric | Value |
|--------|-------|
| IPW weight 50th pct | 1.52 |
| IPW weight 75th pct | 1.57 |
| IPW weight 90th pct | 2.88 |
| IPW weight 95th pct | 8.14 |
| IPW weight 99th pct (trim threshold) | 8.58 |
| ATE untrimmed | +0.0851 |
| ATE trimmed | +0.0852 |
| Trim shift | 0.0001 (extreme weights negligible) |
| IPW ATE 95% CI (500 bootstrap replicates, seed=7) | [+0.0727, +0.0966] |

### Case study 4 — Uber RDD routing threshold + causal forecast

| Metric | Value |
|--------|-------|
| Estimated quality effect of premium routing | +0.0613 |
| Estimated cost effect of premium routing | +0.0080 |
| Queries shifting at threshold 0.85 → 0.90 | 5,415 |
| Quality change at 500k monthly volume | +0.0066 |
| Cost saving at 500k monthly volume | -$436/mo |
| Quality change at 1M monthly volume | +0.0066 |
| Cost saving at 1M monthly volume | -$871/mo |
| Quality change at 2M monthly volume | +0.0066 |
| Cost saving at 2M monthly volume | -$1,742/mo |
| RDD quality effect 95% CI (500 bootstrap replicates, seed=7) | [+0.0490, +0.0748] |

## Files

| File | Description |
|------|-------------|
| `case_studies_demo.ipynb` | Executed Jupyter notebook — reproduces all four case-study implementations and bootstrap CIs |
| `generate_case_studies_charts.py` | Generates Figure 1 (method-selection map) and Figure 2 (IPW weight distribution) |
| `case_studies_method_map.png` | Figure 1 — deployment structure → identifying assumption → causal method, by team |
| `case_studies_weight_distribution.png` | Figure 2 — IPW weight distribution diagnostic with trim threshold |
| `README.md` | This file |

## Source material

The four case studies draw directly from each team's engineering blog:

- **Airbnb** — Jenny Chen et al., ["How Airbnb Measures Future Value to Standardize Tradeoffs"](https://medium.com/airbnb-engineering/how-airbnb-measures-future-value-to-standardize-tradeoffs-3aa99a941ba5), Airbnb Tech Blog
- **Netflix** — ["Key Challenges with Quasi Experiments at Netflix"](https://netflixtechblog.com/key-challenges-with-quasi-experiments-at-netflix-89b4f234b852), Netflix Technology Blog
- **Lyft** — Shima Nassiri and Ross Chu, ["Trusting the Untestable: Validation and Diagnostics for Doubly Robust Models"](https://eng.lyft.com/trusting-the-untestable-validation-and-diagnostics-for-the-doubly-robust-models-00853df009df), Lyft Engineering Blog
- **Uber** — Totte Harinen and Bonnie Li, ["Using Causal Inference to Improve the Uber User Experience"](https://www.uber.com/blog/causal-inference-at-uber/), Uber Engineering Blog

