# 04 — Synthetic Control for Global LLM Rollouts

Companion code for FCC Article 4: *"Product Experimentation with Synthetic Control: Causal Inference for Global LLM Rollouts in Python"*.

## What this teaches

Measuring the causal effect of a global LLM model upgrade when every user receives the new model simultaneously and no randomized holdout exists. Covers:

- Why a global rollout breaks A/B inference and what to do instead
- Constructing a workspace-by-week panel from user-level logs
- SLSQP donor-weight optimization with convex-combination constraints
- Pre-period fit as the identification diagnostic (convex-hull condition)
- In-space placebo permutation test and pseudo p-value computation
- Leave-one-out donor sensitivity for checking whether a single donor drives the result
- User-level cluster bootstrap 95% confidence intervals as the Step 5 uncertainty quantification (classical bootstrap does not apply cleanly to single-treated-unit synthetic control)

## Run

```bash
# From the repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 04_synthetic_control/synthetic_control_demo.py
```

Or open the pre-executed notebook: [`synthetic_control_demo.ipynb`](synthetic_control_demo.ipynb).

To regenerate the article figures:

```bash
python 04_synthetic_control/generate_synthetic_control_charts.py
```

## What you should see

### Dataset setup

| Quantity | Value |
|---|---|
| Rows / columns | 50,000 / 16 |
| Wave 1 workspaces (treated, IDs 0-24) | 25 |
| Wave 2 workspaces (donor pool, IDs 25-49) | 25 |
| Users per workspace-week (weeks 0-29) | ~19.2 |
| Ground-truth effect (staged rollout post-period) | +0.0500 (+5 pp) |
| Wave 1 pre-period mean (weeks 0-19) | 0.5927 |
| Wave 1 post-period mean (weeks 20-29) | 0.6421 |
| Naive before/after gap for wave 1 | +0.0515 |

### Step 1 — SLSQP donor-weight optimization

| Quantity | Value |
|---|---|
| Non-zero donor weights (|w| > 0.001) | 12 |
| Pre-period MSE | 0.001400 |
| Pre-period RMSE | 0.0374 (3.74 pp) |
| Observed post-period gap | +0.0829 |
| Ground truth | +0.0500 |
| Top 5 donor weights | ws 35 (0.2016), ws 40 (0.1900), ws 25 (0.1638), ws 32 (0.0872), ws 36 (0.0784) |

### Step 2 — Weekly gap decomposition (post-period)

| Week | Treated | Synthetic | Gap |
|---|---|---|---|
| 20 | 0.6659 | 0.6262 | +0.0398 |
| 21 | 0.6769 | 0.5106 | +0.1663 |
| 22 | 0.6366 | 0.5347 | +0.1019 |
| 23 | 0.6359 | 0.4824 | +0.1535 |
| 24 | 0.6092 | 0.5022 | +0.1071 |
| 25 | 0.6761 | 0.5713 | +0.1047 |
| 26 | 0.5863 | 0.5439 | +0.0424 |
| 27 | 0.6453 | 0.6127 | +0.0326 |
| 28 | 0.6277 | 0.5949 | +0.0327 |
| 29 | 0.6610 | 0.6131 | +0.0479 |
| Mean | 0.6421 | 0.5592 | +0.0829 |

### Step 3 — In-space placebo permutation test

| Quantity | Value |
|---|---|
| Observed gap | +0.0829 |
| Placebo mean gap | -0.0008 |
| Placebo std gap | 0.0380 |
| Placebo gap range | [-0.0748, +0.0707] |
| Count &#124;placebo&#124; ≥ &#124;observed&#124; | 0 of 25 |
| Pseudo p-value | 0.0385 |

The observed gap is larger in absolute value than every placebo gap, yielding pseudo p = 1/(25+1) = 0.0385. The placebo distribution is centered near zero, which is the expected behavior under the null.

### Step 4 — Leave-one-out donor sensitivity (non-zero weights only)

| Dropped workspace | Dropped weight | New gap |
|---|---|---|
| 35 | 0.2016 | +0.0945 |
| 40 | 0.1900 | +0.0756 |
| 25 | 0.1638 | +0.0932 |
| 32 | 0.0872 | +0.0868 |
| 36 | 0.0784 | +0.0739 |
| 31 | 0.0718 | +0.0858 |
| 29 | 0.0648 | +0.0782 |
| 26 | 0.0439 | +0.0786 |
| 27 | 0.0364 | +0.0867 |
| 46 | 0.0350 | +0.0794 |
| 39 | 0.0192 | +0.0848 |
| 42 | 0.0078 | +0.0839 |

LOO gap range: [+0.0739, +0.0945]. The estimate is stable across donor subsets; no single donor drives the positive result.

### Step 5 — Cluster bootstrap 95% confidence interval

| Quantity | Value |
|---|---|
| Replicates | 500 |
| Seed | 7 |
| Post-period gap 95% CI | [+0.0511, +0.1215] |
| Zero inside CI | NO |
| Ground truth +0.05 inside CI | NO (lower bound 0.0511 is just above 0.05) |

Interpretation: the placebo pseudo p-value of 0.0385 rejects the null of no effect at the 5% level. The LOO sensitivity is tight. The bootstrap CI is positive and excludes zero. The point estimate overshoots the ground truth by about 3 pp because each donor workspace carries more week-to-week noise than the 25-workspace treated average; the weighted combination does not fully cancel post-period donor idiosyncrasies. The inferential conclusion (real positive effect) is correct; the point estimate has known upward finite-sample bias characteristic of synthetic control on small donor panels.

## Files

- `synthetic_control_demo.py` — self-contained script reproducing every code block from the article plus LOO sensitivity and cluster bootstrap CI
- `synthetic_control_demo.ipynb` — pre-executed Jupyter notebook with all cell outputs and the inline trajectory figure
- `build_notebook.py` — rebuilds `synthetic_control_demo.ipynb` from scratch using `nbformat`
- `generate_synthetic_control_charts.py` — regenerates the two article figures (conceptual + data-driven)
- `synthetic_control_conceptual.png` — Figure 1 (treated vs synthetic counterfactual schematic)
- `synthetic_control_density.png` — Figure 2 (data-driven trajectory + placebo distribution)
- `README.md` — this file
