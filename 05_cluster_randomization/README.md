# 05 — Cluster Randomization and SUTVA for Collaborative AI Features

Companion code for FCC Article 5: *"Product Experimentation for Collaborative AI Features: Cluster Randomization, SUTVA, and Network Spillovers for LLM-Based Tools in Python"*.

## What this teaches

Measuring the causal effect of a collaborative AI feature (AI meeting summarizer, shared AI writing tool, AI code review suggestions) when users in the same workspace interfere with each other. User-level randomization violates the Stable Unit Treatment Value Assumption (SUTVA) because control teammates see AI-generated artifacts from treated teammates. Covers:

- Why user-level A/B randomization breaks under network interference
- SUTVA violations and the Hudgens & Halloran partial interference framework
- Workspace-level cluster assignment (25 treated, 25 control workspaces; K = 50)
- Naive user-level OLS (biased point estimate and a 19x-too-small standard error)
- Cluster-weighted least squares for an honest standard error based on K clusters
- Two-exposure decomposition that identifies direct and spillover effects separately
- Cluster-bootstrap 95% confidence intervals (resampling workspaces, not users)

## Run

```bash
# From the repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 05_cluster_randomization/cluster_randomization_demo.py
```

Or open the pre-executed notebook on GitHub: [`cluster_randomization_demo.ipynb`](cluster_randomization_demo.ipynb).

## What you should see

| Quantity | Value |
|---|---|
| Treated workspaces / control workspaces | 25 / 25 |
| Treated users | 24,937 |
| Pure-control users | 18,319 |
| Spillover-exposed control users | 6,744 |
| Workspace size (min / median / max) | 923 / 1002 / 1052 |
| Ground-truth direct effect | +0.80 min |
| Ground-truth spillover effect | +0.20 min |
| Naive OLS estimate | +0.6723 min (biased downward by 0.1277) |
| Naive OLS standard error (wrong) | 0.0034 |
| Cluster WLS estimate (K = 50) | +0.6723 min |
| Cluster WLS standard error (honest) | 0.0652 |
| Cluster WLS 95% CI (analytic) | [+0.5412, +0.8035] |
| Two-exposure direct estimate | +0.7284 min, cluster-robust SE 0.0647 |
| Two-exposure direct 95% CI (analytic) | [+0.6016, +0.8552] |
| Two-exposure spillover estimate | +0.2083 min, cluster-robust SE 0.0038 |
| Two-exposure spillover 95% CI (analytic) | [+0.2008, +0.2158] |
| Naive OLS bootstrap 95% CI | [+0.5386, +0.7966] (misses +0.80) |
| Cluster WLS bootstrap 95% CI | [+0.5386, +0.7966] (misses +0.80) |
| Direct effect bootstrap 95% CI | [+0.5931, +0.8519] (covers +0.80) |
| Spillover effect bootstrap 95% CI | [+0.2008, +0.2164] (covers +0.20) |

The naive user-level estimator is biased downward because spillover-exposed users in the control group inflate the control baseline, shrinking the observed difference. Fixing the standard error with cluster WLS does not fix the bias; the point estimate is identical. Only the two-exposure decomposition separates direct and spillover effects, and its 95% CI is the only one that covers the ground truth.

## Files

- `cluster_randomization_demo.py` — self-contained script that reproduces every code block from the article plus cluster bootstrap CIs
- `cluster_randomization_demo.ipynb` — pre-executed Jupyter notebook with all cell outputs and the inline diagnostic figure
- `generate_cluster_randomization_charts.py` — regenerates the two article figures (SUTVA schematic + three-group outcome distribution)
- `build_notebook.py` — template script that builds the notebook from an nbformat definition
- `cluster_randomization_conceptual.png` — Figure 1 (conceptual SUTVA violation schematic)
- `cluster_randomization_density.png` — Figure 2 (data-driven three-group outcome distribution)
- `README.md` — this file
