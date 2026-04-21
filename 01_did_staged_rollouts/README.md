# 01 — Difference-in-Differences for Staged AI Feature Rollouts

Companion notebook for the freeCodeCamp article *"Product Experimentation for AI Rollouts: Why A/B Testing Breaks and How Difference-in-Differences in Python Fixes It"*.

## What this covers

Measuring the causal effect of an AI feature that ships workspace by workspace in waves, without randomized assignment. Covers:

- Why staged rollouts violate A/B testing assumptions
- A simple 2x2 DiD estimate from four cell means
- Regression DiD with an interaction term and cluster-robust standard errors
- Parallel-trends check (visual plot + formal placebo test)

## Run

```bash
# From the repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
jupyter notebook 01_did_staged_rollouts/did_demo.ipynb
```

## What you should see

- Simple 2x2 DiD estimate: **+0.0527**
- Regression DiD with clustered SE: **+0.0541**, p < 0.001
- Pre-trend placebo test passes: slope = -0.00095, p = 0.44 (parallel trends hold)
- Figure 2 saved to `images/article-1/parallel_trends.png`

The ground-truth effect baked into `data/generate_data.py` is **+5 percentage points** for post-treatment users. Both estimators recover it inside sampling noise.

## Files

- `did_demo.ipynb` — executable notebook with pre-saved outputs; reproduces every code block from the article
- `README.md` — this file
