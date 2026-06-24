# Article 9 — Regression Models for Causal Inference

Teaches OLS regression as a causal estimator for randomized A/B experiments: covariate balance checks, HC3 heteroskedasticity-robust standard errors, cluster-robust standard errors for workspace-correlated data, treatment-effect heterogeneity via interaction terms, and bootstrap confidence intervals.

## What this teaches

- Why OLS gives unbiased causal estimates under randomization (`E[ε|D] = 0`)
- How adding pre-treatment covariates shrinks the standard error without moving the point estimate
- HC3 vs HC0–HC2: why HC3 is preferred in finite samples
- Cluster-robust standard errors when users are correlated within workspaces
- Detecting treatment-effect heterogeneity across engagement tiers via interaction terms
- Bootstrap 95% CIs as a distributional-assumption-free cross-check

## Run

```bash
# From repo root — dataset must exist first
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv

# Run the full analysis
python 09_regression/regression_demo.py

# Generate article figures
python 09_regression/generate_regression_charts.py

# Rebuild and execute the notebook
python 09_regression/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace \
    09_regression/regression_demo.ipynb \
    --ExecutePreprocessor.timeout=600
```

## What you should see

| Metric | Value |
|--------|-------|
| Total users | 50,000 |
| Control arm | 25,000 |
| Treatment arm | 25,000 |
| Ground-truth causal effect (baked in) | +4 pp on `task_completed` |
| Control completion rate | 0.5727 |
| Treatment completion rate | 0.6192 |
| Naive mean difference | +0.0464 |
| t-statistic (naive) | 10.59 |
| OLS coeff — no covariates (HC3) | +0.0464 |
| HC3 SE — no covariates | 0.0044 |
| OLS coeff — with covariates (HC3) | +0.0453 |
| HC3 SE — with covariates | 0.0042 |
| R-squared (with covariates) | 0.1009 |
| Cluster-robust SE | 0.0042 |
| Number of workspaces | 50 |
| Users per workspace (avg) | 1,000 |
| Treatment effect — heavy tier | +0.0474 |
| Treatment effect — light tier | +0.0525 |
| Treatment effect — medium tier | +0.0311 |
| Joint F-test on interactions (p-value) | 0.1241 (not significant — uniform effect) |
| Bootstrap 95% CI | [+0.0377, +0.0545] |
| Bootstrap mean | +0.0453 |

Bootstrap: 500 replicates, seed=7. All estimates from 50k dataset, seed=42.

## Files

| File | Description |
|------|-------------|
| `regression_demo.ipynb` | Executed companion notebook with all outputs |
| `regression_demo.py` | Full analysis script — reproduces all article numbers |
| `generate_regression_charts.py` | Generates Figure 1 (conceptual) and Figure 2 (data-driven density plot) |
| `build_notebook.py` | Builds `regression_demo.ipynb` from scratch using nbformat |
| `regression_balance_conceptual.png` | Figure 1 — randomized vs biased covariate distributions (conceptual schematic) |
| `regression_balance_density.png` | Figure 2 — actual `query_confidence` density by arm on 50k dataset |
| `README.md` | This file |
