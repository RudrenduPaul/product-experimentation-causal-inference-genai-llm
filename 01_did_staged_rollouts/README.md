# 01 — Difference-in-Differences for Staged AI Feature Rollouts

Companion code for FCC Article 1: *"Why A/B Testing Breaks for Staged AI Feature Rollouts (and How to Use Difference-in-Differences in Python Instead)"*.

## What this teaches

Measuring the causal effect of an AI feature that ships workspace by workspace in waves, instead of through randomized user assignment. Covers:

- Why staged rollouts violate A/B-testing assumptions
- A simple 2x2 DiD on a treated-vs-control, pre-vs-post table
- Regression DiD with an interaction term and cluster-robust standard errors
- Parallel-trends check (visual plot + formal placebo test)

## Run

```bash
# From the repo root
python data/generate_data.py --seed 42 --n-users 10000 --out data/synthetic_llm_logs.csv
python 01_did_staged_rollouts/did_demo.py
```

## What you should see

- Simple 2x2 DiD estimate around +0.08
- Regression DiD with clustered SE around +0.066, p ≈ 0.017
- Parallel-trends plot saved as `parallel_trends.png`
- Pre-trend placebo test passes (slope near zero, p > 0.5)

The ground-truth effect baked into `data/generate_data.py` is +0.05 percentage points for post-treatment users. The regression estimator recovers it inside sampling noise. The simple 2x2 version runs higher because it collapses all within-cell variation into four means.

## Files

- `did_demo.py` — self-contained script that reproduces every code block from the article
- `parallel_trends.png` — generated on first run
- `README.md` — this file
