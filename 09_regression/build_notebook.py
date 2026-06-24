"""Build the Article 9 companion notebook using nbformat.

Run from repo root:
    python 09_regression/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        09_regression/regression_demo.ipynb \
        --ExecutePreprocessor.timeout=600
"""

import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "regression_demo.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# ── Intro markdown cell ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# Regression Models for Causal Inference

**Keywords:** product experimentation, causal inference, OLS regression, HC3 standard errors,
cluster-robust standard errors, heteroskedasticity, randomized experiment, treatment-effect
heterogeneity, bootstrap confidence intervals, LLM applications, generative AI

This notebook accompanies the freeCodeCamp tutorial on regression models for causal inference
in randomized A/B experiments. It demonstrates covariate balance checking, OLS with HC3
heteroskedasticity-robust standard errors, cluster-robust standard errors, interaction models
for treatment-effect heterogeneity, and bootstrap confidence intervals on a 50,000-user
synthetic SaaS dataset.

**Dataset:** 50,000 synthetic users across 50 workspaces. Ground-truth causal effect of the
new prompt template: +4 percentage points on `task_completed`, uniform across engagement tiers.

**Clone and run:**
```bash
git clone https://github.com/RudrenduPaul/product-experimentation-causal-inference-genai-llm.git
cd product-experimentation-causal-inference-genai-llm
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 09_regression/regression_demo.py
```
"""))

# ── Setup ────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_csv("../data/synthetic_llm_logs.csv")
print(f"Dataset: {len(df):,} users across {df.workspace_id.nunique()} workspaces")
print(f"Columns: {list(df.columns)}")
"""))

# ── Randomization check ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Randomization check — covariate balance by arm

Before fitting any model, confirm that hash-based assignment balanced the groups on
observable pre-treatment covariates. A properly randomized experiment produces near-equal
means on every measured characteristic across arms.
"""))

cells.append(nbf.v4.new_code_cell("""print("Prompt variant distribution:")
print(df.prompt_variant.value_counts().to_dict())

check_cols = ["query_confidence", "session_minutes", "cost_usd"]
balance_table = (
    df.groupby("prompt_variant")[check_cols]
    .mean()
    .round(4)
    .T
)
balance_table.columns = ["Control (variant=0)", "Treatment (variant=1)"]
balance_table["Difference"] = (
    balance_table["Treatment (variant=1)"]
    - balance_table["Control (variant=0)"]
)
print("\\nCovariate balance check:")
print(balance_table)

print("\\nEngagement tier split by arm:")
print(
    df.groupby("prompt_variant")
    .engagement_tier.value_counts(normalize=True)
    .unstack()
    .round(3)
)
"""))

# ── Step 1 ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Step 1: Naive difference in means

Start with the simplest possible estimator: subtract the mean outcome in the control arm
from the mean in the treatment arm. Under randomization, this is a valid causal estimate.
"""))

cells.append(nbf.v4.new_code_cell("""mean_control = df[df.prompt_variant == 0].task_completed.mean()
mean_treatment = df[df.prompt_variant == 1].task_completed.mean()
naive_effect = mean_treatment - mean_control

print(f"Control mean:    {mean_control:.4f}")
print(f"Treatment mean:  {mean_treatment:.4f}")
print(f"Naive effect:    {naive_effect:+.4f}")

n0 = (df.prompt_variant == 0).sum()
n1 = (df.prompt_variant == 1).sum()
var0 = df[df.prompt_variant == 0].task_completed.var()
var1 = df[df.prompt_variant == 1].task_completed.var()
se = np.sqrt(var0 / n0 + var1 / n1)
t_stat = naive_effect / se
p_val = 2 * stats.t.sf(abs(t_stat), df=n0 + n1 - 2)

print(f"\\nSE (two-sample):  {se:.4f}")
print(f"t-statistic:      {t_stat:.3f}")
print(f"p-value:          {p_val:.4f}")
"""))

# ── Step 2 ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Step 2: OLS with HC3 heteroskedasticity-robust errors

OLS without covariates reproduces the naive mean difference exactly. Adding pre-treatment
covariates keeps the point estimate the same while absorbing residual variance, which
shrinks the standard error. HC3 robust standard errors are preferred over HC0–HC2 in
finite samples.
"""))

cells.append(nbf.v4.new_code_cell("""# OLS without covariates: should match naive difference
m1 = smf.ols("task_completed ~ prompt_variant", data=df).fit(cov_type="HC3")
print("=== OLS without covariates (HC3) ===")
print(m1.summary().tables[1])
print(f"\\nCoefficient: {m1.params['prompt_variant']:+.4f}")
print(f"HC3 SE:      {m1.bse['prompt_variant']:.4f}")
print(f"p-value:     {m1.pvalues['prompt_variant']:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""# Define the regression formula with covariates
formula = (
    "task_completed ~ prompt_variant + query_confidence + "
    "session_minutes + C(engagement_tier)"
)

# OLS with covariates: same point estimate, smaller SE
m2 = smf.ols(formula, data=df).fit(cov_type="HC3")
print("=== OLS with covariates (HC3) ===")
print(m2.summary().tables[1])
print(f"\\nCoefficient: {m2.params['prompt_variant']:+.4f}")
print(f"HC3 SE:      {m2.bse['prompt_variant']:.4f}")
print(f"p-value:     {m2.pvalues['prompt_variant']:.4f}")

print("\\n--- SE comparison ---")
print(f"Without covariates: {m1.bse['prompt_variant']:.4f}")
print(f"With covariates:    {m2.bse['prompt_variant']:.4f}")
print(f"R-squared (with):   {m2.rsquared:.4f}")
"""))

# ── Step 3 ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Cluster-robust standard errors

Users within the same workspace share a support team, product tier, and IT policies.
Their outcomes correlate regardless of treatment assignment. Clustering at the workspace
level treats each of the 50 workspaces as a single informational unit.
"""))

cells.append(nbf.v4.new_code_cell("""# Naive SE (assumes independence within workspaces)
m3_naive = smf.ols(formula, data=df).fit(cov_type="HC3")

# Cluster-robust SE (accounts for within-workspace correlation)
m3_cluster = smf.ols(formula, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["workspace_id"]}
)

print("=== SE comparison: HC3 vs cluster-robust ===")
print(f"Coefficient (both):      {m3_cluster.params['prompt_variant']:+.4f}")
print(f"HC3 SE:                  {m3_naive.bse['prompt_variant']:.4f}")
print(f"Cluster-robust SE:       {m3_cluster.bse['prompt_variant']:.4f}")
print(f"HC3 p-value:             {m3_naive.pvalues['prompt_variant']:.4f}")
print(f"Cluster p-value:         {m3_cluster.pvalues['prompt_variant']:.4f}")
print(f"\\nNumber of clusters: {df.workspace_id.nunique()}")
print(f"Users per workspace (avg): {len(df) / df.workspace_id.nunique():.0f}")
"""))

# ── Step 4 ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Step 4: Treatment-effect heterogeneity via interactions

The average treatment effect can hide important structure. An interaction term between
treatment and engagement tier surfaces whether the prompt template works differently
for light, medium, and heavy users.

Preregister which moderator you plan to test before looking at the data.
Running ten interactions and reporting the significant one is p-hacking.
"""))

cells.append(nbf.v4.new_code_cell("""# Interaction model: prompt_variant x engagement_tier
interaction_formula = (
    "task_completed ~ prompt_variant * C(engagement_tier) + "
    "query_confidence + session_minutes"
)

m4 = smf.ols(interaction_formula, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["workspace_id"]}
)

print("=== Interaction model (cluster-robust) ===")
print(m4.summary().tables[1])

# Detect reference tier dynamically; compute tier-specific effects
print("\\n=== Implied treatment effects by engagement tier ===")
baseline_effect = m4.params["prompt_variant"]

all_tiers = set(df.engagement_tier.unique())
interaction_keys = [k for k in m4.params.index if "prompt_variant:C(engagement_tier)" in k]
non_ref_tiers = {k.split("[T.")[1].rstrip("]") for k in interaction_keys}
ref_tier = (all_tiers - non_ref_tiers).pop()

effects = {ref_tier: baseline_effect}
for key in interaction_keys:
    tier = key.split("[T.")[1].rstrip("]")
    effects[tier] = baseline_effect + m4.params[key]

for tier, eff in sorted(effects.items()):
    print(f"  {tier:8s}: {eff:+.4f}")

# Joint F-test: are the interaction terms jointly significant?
if interaction_keys:
    f_test = m4.f_test([f"({t} = 0)" for t in interaction_keys])
    print(f"\\nJoint F-test on interactions: p = {f_test.pvalue:.4f}")
"""))

# ── Step 5 ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Step 5: Bootstrap confidence intervals

Bootstrap CIs provide a check that does not rely on distributional assumptions.
500 resamples of the full dataset, each refitting the cluster-robust model. The
2.5th and 97.5th percentiles form the 95% CI.
"""))

cells.append(nbf.v4.new_code_cell("""rng = np.random.default_rng(seed=7)
n_boot = 500
boot_coefs = []

for _ in range(n_boot):
    idx = rng.integers(0, len(df), size=len(df))
    boot_df = df.iloc[idx].reset_index(drop=True)
    boot_model = smf.ols(
        formula,
        data=boot_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": boot_df["workspace_id"]}
    )
    boot_coefs.append(boot_model.params["prompt_variant"])

boot_coefs = np.array(boot_coefs)
ci_low, ci_high = np.percentile(boot_coefs, [2.5, 97.5])

print(f"Bootstrap 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
print(f"Bootstrap mean:   {boot_coefs.mean():+.4f}")
print(f"Analytic cluster SE: {m3_cluster.bse['prompt_variant']:.4f}")
print(f"\\nGround-truth causal effect baked into generator: +4 pp")
print(f"CI covers ground truth: {ci_low <= 0.04 <= ci_high}")
print(f"CI excludes zero: {ci_low > 0}")
"""))

# ── Closing markdown ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Summary

| Step | Estimator | Coefficient | SE | p-value |
|------|-----------|------------|-----|---------|
| 1 | Naive t-test | +0.0464 | 0.0044 | < 0.001 |
| 2 | OLS, no covariates (HC3) | +0.0464 | 0.0044 | < 0.001 |
| 2 | OLS, with covariates (HC3) | +0.0453 | 0.0042 | < 0.001 |
| 3 | Cluster-robust | +0.0453 | 0.0042 | < 0.001 |
| 5 | Bootstrap 95% CI | — | — | [+0.038, +0.055] |

The regression recovers a treatment effect close to the ground-truth +4 pp in all five steps.
The joint F-test on interaction terms is non-significant (p = 0.12), confirming that the
effect is broadly uniform across engagement tiers — consistent with the data generator's
uniform treatment assignment.
"""))

nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.0",
}

nbf.write(nb, NB_PATH)
print(f"Notebook written to: {NB_PATH}")
print(
    "Execute with:\n"
    "  jupyter nbconvert --to notebook --execute --inplace \\\n"
    f"      {NB_PATH} --ExecutePreprocessor.timeout=600"
)
