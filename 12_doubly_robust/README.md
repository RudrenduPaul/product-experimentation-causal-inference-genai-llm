# 12 — Doubly Robust Estimation (AIPW) for Noisy LLM Product Experiments

Companion code for FCC Article 12: *"Product Experimentation with Doubly Robust Estimation: When Both Your Models Are Wrong in LLM Applications"*.

## What this teaches

Measuring the causal effect of an opt-in AI feature when neither your propensity model nor your outcome model is verifiably correct. Covers:

- Why both propensity-only (IPW) and regression-only methods fail when their models are misspecified
- The AIPW formula and the double-robust guarantee (consistent if either model is correct)
- Fitting a propensity model (logistic regression) and two outcome models (linear regression, treated/control separately)
- Combining both into the AIPW influence function
- Bootstrap 95% confidence intervals (500 replicates, seed=7)
- Empirical proof of double robustness via deliberate misspecification of one model at a time

## Run

```bash
# From the repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 12_doubly_robust/aipw_demo.py
```

Or open the pre-executed notebook on GitHub: [`aipw_demo.ipynb`](aipw_demo.ipynb).

## What you should see

| Quantity | Value |
|---|---|
| Dataset | 50,000 users |
| N treated (opted in) | 13,451 |
| N control (did not opt in) | 36,549 |
| Naive ATE (unadjusted) | +0.2106 (heavily inflated by selection bias) |
| Ground-truth effect | +0.0800 (+8 percentage points) |
| Propensity range | 0.114 to 0.675 |
| Mean propensity (treated) | 0.401 |
| Mean propensity (control) | 0.220 |
| Regression adjustment ATE | +0.0847 |
| AIPW ATE | +0.0847 |
| 95% Bootstrap CI | [+0.0744, +0.0952] |
| Bootstrap std dev | 0.0053 |
| Scenario 1: IPW with wrong propensity (e=0.3) | +0.2106 (breaks, as expected) |
| Scenario 1: AIPW with wrong propensity | +0.0847 (holds, outcome model carries it) |
| Scenario 2: Regression with wrong outcome (m=0.5) | +0.0000 (collapses, as expected) |
| Scenario 2: AIPW with wrong outcome models | +0.0849 (holds, propensity arm carries it) |

The ground-truth causal effect baked into `data/generate_data.py` is +8 percentage points on task completion for users who opted into agent mode. Both the regression adjustment and AIPW estimates recover it (+0.0847), and the 95% CI [+0.0744, +0.0952] contains the truth while excluding the naive +0.2106 by a wide margin.

## Files

- `aipw_demo.py` — self-contained script that reproduces every code block from the article plus bootstrap CIs and misspecification scenarios
- `aipw_demo.ipynb` — pre-executed Jupyter notebook with all cell outputs and the propensity overlap figure
- `build_notebook.py` — script that generates `aipw_demo.ipynb` using nbformat
- `generate_aipw_charts.py` — regenerates the two article figures (conceptual AIPW structure + data-driven propensity overlap)
- `aipw_structure_conceptual.png` — Figure 1 (conceptual: dual-model structure of AIPW)
- `aipw_propensity_overlap.png` — Figure 2 (data-driven: propensity score overlap on the 50,000-user dataset)
- `README.md` — this file
