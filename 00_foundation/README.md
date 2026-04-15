# 00 — Foundation: From A/B Tests to Causal Inference

Companion code for freeCodeCamp Article 0: [From A/B Tests to Causal Inference: How to Measure What Your LLM Features Actually Do in Production](https://www.freecodecamp.org/news/author/rudrendupaul/).

This folder contains all four technique examples from Article 0 in one runnable script (`foundation_intro.py`). Each of the next ten article folders (`01_regression/`, `02_counterfactual/`, etc.) will go deep on one technique.

## Run

```bash
# From repo root
python data/generate_data.py --seed 42 --n-users 10000 --out data/synthetic_llm_logs.csv
python 00_foundation/foundation_intro.py
```

## What you should see

- **Regression (prompt variant)**: estimate near 3–4 pp, ground truth 4 pp
- **IPW (agent mode opt-in)**: naive difference ~21 pp (contaminated by selection) vs. IPW-adjusted ~7 pp, ground truth 8 pp
- **DiD (wave 1 staged rollout)**: estimate in the ballpark of 5–8 pp, ground truth 5 pp
- **RDD (premium model routing)**: estimate near 6–8 pp, ground truth 6 pp

Exact numbers depend on seed and sample size. Sanity check: the four methods should produce estimates within a few percentage points of the ground truths baked into `data/generate_data.py`.
