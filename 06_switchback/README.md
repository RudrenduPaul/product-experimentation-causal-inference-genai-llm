# Article 6 — Switchback Experiments for LLM Platforms

Run a valid experiment on a shared-resource AI platform when user-level randomization breaks market equilibrium. This folder implements the full switchback pipeline using time-slot randomization, carryover-adjusted OLS regression, HAC standard errors, and slot-level bootstrap CIs on the shared 50,000-session synthetic LLM product dataset.

## What this teaches

- Why user-level A/B testing fails when all users share the same model capacity pool (SUTVA violation)
- How switchback design restores a clean treatment-vs-control comparison by randomizing time slots
- How carryover bias arises and how to remove it with a one-period lag term
- Why OLS standard errors understate uncertainty in time-series data and how Newey-West HAC SEs fix it
- How to construct slot-level bootstrap confidence intervals for switchback estimates
- When switchback fails: minimum slot length, heterogeneous user populations, non-stationary baselines

## Run

```bash
# From repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 06_switchback/switchback_demo.py
```

## What you should see

All values below are from executed runs against `synthetic_llm_logs.csv` (50,000 sessions, seed=42).

| Metric | Value |
|--------|-------|
| Total sessions | 50,000 |
| Slots | 48 (30-min equivalent, 3-on / 3-off schedule) |
| Sessions per slot | 1,042 |
| AI-on slots | 24 |
| AI-off slots | 24 |
| True direct effect (data generator) | 0.0600 (6pp) |
| True carryover effect (data generator) | 0.0300 (3pp) |
| Naive ATE estimate (no carryover control) | 0.0688 |
| Naive ATE bias | +0.0088 |
| Naive OLS SE | 0.0048 |
| Naive 95% CI | [0.0595, 0.0782] |
| Adjusted ATE (`ai_on` coefficient) | 0.0607 |
| Adjusted ATE residual bias | +0.0007 |
| Carryover coefficient (`ai_on_lag1`) | 0.0244 |
| OLS SE on `ai_on` (adjusted model) | 0.0036 |
| HAC SE on `ai_on` (Newey-West, nlags=3) | 0.0037 |
| OLS t-stat | 16.83 |
| HAC t-stat | 16.41 |
| HAC 95% CI | [0.0535, 0.0680] |
| True 0.06 inside HAC CI | yes |
| Durbin-Watson statistic | 1.9628 |
| Bootstrap naive ATE 95% CI (B=500, seed=7) | [0.0596, 0.0783] |
| Bootstrap adjusted ATE 95% CI (B=500, seed=7) | [0.0541, 0.0683] |
| Bootstrap carryover 95% CI (B=500, seed=7) | [0.0175, 0.0320] |
| True 0.06 inside bootstrap adj CI | yes |
| True 0.03 inside bootstrap carryover CI | yes |

## Files

| File | Description |
|------|-------------|
| `switchback_demo.py` | Main analysis script — all five steps including bootstrap CIs |
| `generate_switchback_charts.py` | Produces Figure 1 (conceptual design) and Figure 2 (estimates comparison) |
| `switchback_design_conceptual.png` | Figure 1 — treatment schedule and carryover mechanics |
| `switchback_estimates_comparison.png` | Figure 2 — naive vs adjusted ATE with bootstrap CIs |
| `README.md` | This file |
