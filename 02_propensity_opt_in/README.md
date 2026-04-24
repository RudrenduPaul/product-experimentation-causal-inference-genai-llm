# 02 — Propensity Score Methods for AI Opt-In Features

Companion code for FCC Article 2: *"Product Experimentation with Propensity Scores: Causal Inference for LLM-Based Features in Python"*.

## What this teaches

Measuring the causal effect of an AI feature that ships behind a user-controlled opt-in toggle, where the naive "opted in vs. didn't" comparison is contaminated by selection bias. Covers:

- Why opt-in comparisons overstate feature impact
- Estimating the propensity score with logistic regression (+ AUC check)
- Inverse-probability weighting (IPW) for ATE and ATT
- 1-NN propensity score matching (with replacement)
- Covariate-balance diagnostics with standardized mean difference
- Bootstrap 95% confidence intervals for all three estimators

## Run

```bash
# From the repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 02_propensity_opt_in/psm_demo.py
```

Or open the pre-executed notebook on GitHub: [`psm_demo.ipynb`](psm_demo.ipynb).

## What you should see

| Quantity | Value |
|---|---|
| Naive opt-in effect | +0.2106 (heavily contaminated) |
| Ground-truth effect | +0.08 (+8 percentage points) |
| Propensity model AUC | 0.744 |
| IPW ATE | +0.0851, 95% CI [+0.0745, +0.0954] |
| IPW ATT | +0.0770, 95% CI [+0.0687, +0.0865] |
| 1-NN matching ATT | +0.0752, 95% CI [+0.0659, +0.0940] |
| Raw SMD on `engagement_tier_heavy` | +0.742 (large imbalance) |
| Weighted SMD on `engagement_tier_heavy` | +0.002 (balanced) |

The ground-truth causal effect baked into `data/generate_data.py` is +8 percentage points on task completion for users who opted in. All three estimators recover it with 95% CIs that cover the truth and exclude the naive +0.21 by a wide margin.

## Files

- `psm_demo.py` — self-contained script that reproduces every code block from the article plus bootstrap CIs
- `psm_demo.ipynb` — pre-executed Jupyter notebook with all cell outputs and both figures saved
- `generate_psm_charts.py` — regenerates the two article figures (conceptual overlap + data-driven density)
- `psm_overlap_conceptual.png` — Figure 1 (conceptual propensity distributions)
- `psm_overlap_density.png` — Figure 2 (data-driven overlap density)
- `README.md` — this file
