# Article 11 — Instrumental Variables for Confounded LLM Routing

Companion code for the freeCodeCamp article:
**"Unconfounding the AI product experiment: instrumental variable analysis for multi-model LLM gateways in Python"**

## What this teaches

- Why naive OLS is biased when routing decisions correlate with unobserved query characteristics
- The three IV assumptions: relevance, exclusion restriction, and monotonicity
- How to implement two-stage least squares (2SLS) from scratch with two chained OLS regressions
- Diagnosing instrument strength with the first-stage F-statistic (rule of thumb: F > 10)
- Interpreting the LATE (Local Average Treatment Effect) for compliers only
- Bootstrap 95% CIs to quantify uncertainty around both the OLS and 2SLS estimates
- Using `linearmodels` IV2SLS for production-grade correct standard errors

## Run

```bash
# From repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 11_instrumental_variables/iv_demo.py
```

Or open the notebook:

```bash
jupyter notebook 11_instrumental_variables/iv_demo.ipynb
```

## What you should see

After running `iv_demo.py` (500 bootstrap replicates take ~2 minutes):

| Metric | Value |
|---|---|
| Rate-limit fallback rate | 0.151 |
| Premium routing rate (actual) | 0.271 |
| Mean confidence — fallback=1 | 0.716 |
| Mean confidence — fallback=0 | 0.715 |
| Naive OLS estimate (biased) | +0.0327 |
| Stage 1 instrument coefficient | −0.3190 |
| First-stage F-statistic | 3780.94 |
| 2SLS estimate (LATE) | +0.0599 |
| Ground truth premium effect | +0.0600 |
| OLS bias | −0.0273 |
| OLS 95% CI | [+0.0227, +0.0426] |
| 2SLS 95% CI | [+0.0247, +0.0969] |
| OLS CI covers ground truth | False |
| 2SLS CI covers ground truth | True |
| Complier population | 7,575 (15.2%) |

The OLS CI does not cover the true +0.06 effect. The 2SLS CI does.

## Files

| File | Description |
|---|---|
| `iv_demo.py` | Main analysis script — loads data, runs OLS, 2SLS, diagnostics, bootstrap CIs |
| `iv_demo.ipynb` | Executed notebook with pre-saved outputs |
| `build_notebook.py` | Script used to generate iv_demo.ipynb programmatically |
| `generate_iv_charts.py` | Generates Figure 1 (IV causal DAG) and Figure 2 (first-stage + estimates) |
| `iv_causal_dag.png` | Figure 1 — conceptual IV causal structure (DAG) |
| `iv_first_stage_estimates.png` | Figure 2 — first-stage relationship and OLS vs 2SLS comparison |
| `README.md` | This file |
