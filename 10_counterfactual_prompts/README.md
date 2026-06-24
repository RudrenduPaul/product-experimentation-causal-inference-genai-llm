# Article 10 — Counterfactual Meta-Learners for LLM Prompt Decisions

Answer the retrospective question your A/B test can't: which prompt would have performed better for each individual user, given only the logs from the prompt you already shipped. This folder implements the T-learner and X-learner from scratch using scikit-learn, adds bootstrap confidence intervals across both estimators, and translates the resulting CATE estimates into a concrete routing policy value.

## What this teaches

- Why logged production data is not an experiment and when confounding invalidates a naive lift comparison
- How the potential outcomes framework (Rubin, 1974; Holland, 1986) frames the counterfactual problem
- How to implement the T-learner: two separate outcome models whose difference gives per-user CATE
- How to implement the X-learner (Künzel et al., 2019): imputed effects + propensity weighting for imbalanced arms
- How to bootstrap both estimators (500 resamples) to get 95% CIs and confirm agreement across methods
- How to convert a CATE distribution into a policy value: routing threshold, mean lift in the routed group, and total estimated completions gained
- When counterfactual estimation fails: model misspecification, positivity violations, unmeasured confounders, non-overlapping covariate support, SUTVA violations

## Run

```bash
# From repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
jupyter notebook 10_counterfactual_prompts/counterfactual_demo.ipynb
```

## What you should see

All values below are from executed runs against `synthetic_llm_logs.csv` (50,000 users, seed=42).

| Metric | Value |
|--------|-------|
| Control arm n | 25,000 |
| Treatment arm n | 25,000 |
| Control mean task completion | 0.6000 |
| Treatment mean task completion | 0.6300 |
| Naive lift (raw difference) | +0.0260 (2.60pp) |
| True effect (data generator) | +4pp |
| T-learner mean CATE | +0.0260 |
| T-learner CATE std | 0.0100 |
| X-learner mean CATE | +0.0260 |
| X-learner CATE std | 0.0100 |
| Propensity score range | [0.4820, 0.5170] |
| T-learner 95% CI (500 bootstrap resamples, seed=7) | [+0.0120, +0.0400] |
| X-learner 95% CI (500 bootstrap resamples, seed=7) | [+0.0120, +0.0400] |
| Policy threshold | CATE > 0.020 |
| Users routed to Prompt B | 35,000 / 50,000 (70.0%) |
| Mean CATE in routed group | +0.0320 |
| Estimated total lift | 1,120 additional completions |
| Policy value (lift per user, full population) | +0.0224 |
| Ground-truth +4pp inside both CIs | yes |

## Files

| File | Description |
|------|-------------|
| `counterfactual_demo.ipynb` | Executed Jupyter notebook — all steps with pre-saved outputs |
| `generate_counterfactual_charts.py` | Produces Figure 1 (conceptual T-learner) and Figure 2 (CATE density by tier) |
| `counterfactual_cate_conceptual.png` | Figure 1 — conceptual illustration of T-learner CATE gap across covariate range |
| `counterfactual_cate_density.png` | Figure 2 — T-learner CATE distributions by engagement tier on real synthetic data |
| `README.md` | This file |
