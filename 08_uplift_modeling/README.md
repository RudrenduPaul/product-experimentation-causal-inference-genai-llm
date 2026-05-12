# Article 8 — Uplift Modeling for Personalized AI Rollouts

Teaches T-learner and X-learner meta-learner approaches for estimating conditional average treatment effects (CATEs), Qini curve evaluation, and segmented rollout decision rules.

## What this teaches

- Why average treatment effects mislead for personalized AI features
- How T-learner and X-learner meta-learners estimate per-user CATEs
- The corrected Kunzel et al. (2019) X-learner propensity weighting formula
- Building a Qini curve to evaluate CATE model ranking
- Translating a CATE model into a segmented rollout rule
- Bootstrap 95% confidence intervals for CATE estimates

## Run

```bash
# From repo root — dataset must exist first
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv

# Run the full analysis
python 08_uplift_modeling/uplift_demo.py

# Generate article figures
python 08_uplift_modeling/generate_uplift_charts.py

# Rebuild and execute the notebook
python 08_uplift_modeling/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace \
    08_uplift_modeling/uplift_demo.ipynb \
    --ExecutePreprocessor.timeout=600
```

## What you should see

| Metric | Value |
|--------|-------|
| Total users | 50,000 |
| Treated (opt-in) | 13,451 |
| Control | 36,549 |
| Opt-in rate — heavy | 64.7% |
| Opt-in rate — medium | 35.3% |
| Opt-in rate — light | 12.0% |
| Naive ATE (confounded) | +0.2106 |
| T-learner mean CATE | +0.0847 |
| X-learner mean CATE | +0.0847 |
| CATE — light tier | +0.0954 |
| CATE — medium tier | +0.0744 |
| CATE — heavy tier | +0.0665 |
| Mean CATE 95% CI | [+0.0744, +0.0951] |
| Light tier 95% CI | [+0.0781, +0.1125] |
| Medium tier 95% CI | [+0.0596, +0.0892] |
| Heavy tier 95% CI | [+0.0483, +0.0842] |
| Qini top-10% uplift | +0.0895 |
| Qini top-20% uplift | +0.1018 |
| Rollout threshold | 0.085 |
| Users selected at 0.085 | 27,203 (54%, all light tier) |
| Users suppressed | 22,797 (46%) |

Bootstrap: 500 replicates, seed=7. All CIs from T-learner on 50k dataset, seed=42.

## Files

| File | Description |
|------|-------------|
| `uplift_demo.ipynb` | Executed companion notebook with all outputs |
| `uplift_demo.py` | Full analysis script — reproduces all article numbers |
| `generate_uplift_charts.py` | Generates Figure 1 (conceptual) and Figure 2 (data-driven) |
| `build_notebook.py` | Builds `uplift_demo.ipynb` from scratch using nbformat |
| `uplift_cate_conceptual.png` | Figure 1 — conceptual illustration of tier-level CATE heterogeneity |
| `uplift_cate_distribution.png` | Figure 2 — actual CATE distributions from 50k dataset |
| `qini_curve.png` | Qini curve generated during notebook execution |
| `README.md` | This file |
