"""
Regression models for causal inference.

Companion code for FCC Article 9:
"Regression models for causal inference: estimating LLM feature impact
with Python and statsmodels"

Runs every code block from the article against the shared synthetic dataset.
Produces: covariate balance check, naive mean difference, OLS with HC3,
cluster-robust standard errors, treatment-effect heterogeneity via
interactions, and bootstrap 95% confidence intervals.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \\
        --out data/synthetic_llm_logs.csv
    python 09_regression/regression_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_data() -> pd.DataFrame:
    csv_path = REPO_ROOT / "data" / "synthetic_llm_logs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run: python data/generate_data.py --seed 42 --n-users 50000 "
            "--out data/synthetic_llm_logs.csv"
        )
    return pd.read_csv(csv_path)


def check_balance(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RANDOMIZATION CHECK — Covariate balance by arm")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print("\nPrompt variant distribution:")
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
    print("\nCovariate balance check:")
    print(balance_table)

    print("\nEngagement tier split by arm:")
    print(
        df.groupby("prompt_variant")
        .engagement_tier.value_counts(normalize=True)
        .unstack()
        .round(3)
    )


def step1_naive_diff(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STEP 1 — Naive difference in means")
    print("=" * 60)

    mean_control = df[df.prompt_variant == 0].task_completed.mean()
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

    print(f"\nSE (two-sample):  {se:.4f}")
    print(f"t-statistic:      {t_stat:.3f}")
    print(f"p-value:          {p_val:.4f}")


def step2_ols_hc3(df: pd.DataFrame) -> str:
    print("\n" + "=" * 60)
    print("STEP 2 — OLS with HC3 heteroskedasticity-robust errors")
    print("=" * 60)

    m1 = smf.ols("task_completed ~ prompt_variant", data=df).fit(cov_type="HC3")
    print("=== OLS without covariates (HC3) ===")
    print(m1.summary().tables[1])
    print(f"\nCoefficient: {m1.params['prompt_variant']:+.4f}")
    print(f"HC3 SE:      {m1.bse['prompt_variant']:.4f}")
    print(f"p-value:     {m1.pvalues['prompt_variant']:.4f}")

    formula = (
        "task_completed ~ prompt_variant + query_confidence + "
        "session_minutes + C(engagement_tier)"
    )
    m2 = smf.ols(formula, data=df).fit(cov_type="HC3")
    print("\n=== OLS with covariates (HC3) ===")
    print(m2.summary().tables[1])
    print(f"\nCoefficient: {m2.params['prompt_variant']:+.4f}")
    print(f"HC3 SE:      {m2.bse['prompt_variant']:.4f}")
    print(f"p-value:     {m2.pvalues['prompt_variant']:.4f}")

    print("\n--- SE comparison ---")
    print(f"Without covariates: {m1.bse['prompt_variant']:.4f}")
    print(f"With covariates:    {m2.bse['prompt_variant']:.4f}")
    print(f"R-squared (with):   {m2.rsquared:.4f}")

    return formula


def step3_cluster_robust(df: pd.DataFrame, formula: str) -> object:
    print("\n" + "=" * 60)
    print("STEP 3 — Cluster-robust standard errors")
    print("=" * 60)

    m3_naive = smf.ols(formula, data=df).fit(cov_type="HC3")
    m3_cluster = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["workspace_id"]},
    )

    print("=== SE comparison: HC3 vs cluster-robust ===")
    print(f"Coefficient (both):      {m3_cluster.params['prompt_variant']:+.4f}")
    print(f"HC3 SE:                  {m3_naive.bse['prompt_variant']:.4f}")
    print(f"Cluster-robust SE:       {m3_cluster.bse['prompt_variant']:.4f}")
    print(f"HC3 p-value:             {m3_naive.pvalues['prompt_variant']:.4f}")
    print(f"Cluster p-value:         {m3_cluster.pvalues['prompt_variant']:.4f}")
    print(f"\nNumber of clusters: {df.workspace_id.nunique()}")
    print(f"Users per workspace (avg): {len(df) / df.workspace_id.nunique():.0f}")

    return m3_cluster


def step4_interactions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STEP 4 — Treatment-effect heterogeneity via interactions")
    print("=" * 60)

    interaction_formula = (
        "task_completed ~ prompt_variant * C(engagement_tier) + "
        "query_confidence + session_minutes"
    )
    m4 = smf.ols(interaction_formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["workspace_id"]},
    )

    print("=== Interaction model (cluster-robust) ===")
    print(m4.summary().tables[1])

    print("\n=== Implied treatment effects by engagement tier ===")
    baseline_effect = m4.params["prompt_variant"]

    # Detect the reference tier dynamically (the one absent from interaction params)
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

    interaction_terms = [k for k in m4.params.index if "prompt_variant:C" in k]
    if interaction_terms:
        f_test = m4.f_test([f"({t} = 0)" for t in interaction_terms])
        print(f"\nJoint F-test on interactions: p = {f_test.pvalue:.4f}")


def step5_bootstrap(df: pd.DataFrame, formula: str, m3_cluster: object) -> None:
    print("\n" + "=" * 60)
    print("STEP 5 — Bootstrap confidence intervals (500 replicates)")
    print("=" * 60)

    rng = np.random.default_rng(seed=7)
    n_boot = 500
    boot_coefs = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        boot_df = df.iloc[idx].reset_index(drop=True)
        boot_model = smf.ols(formula, data=boot_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": boot_df["workspace_id"]},
        )
        boot_coefs.append(boot_model.params["prompt_variant"])

    boot_coefs = np.array(boot_coefs)
    ci_low, ci_high = np.percentile(boot_coefs, [2.5, 97.5])

    print(f"Bootstrap 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"Bootstrap mean:   {boot_coefs.mean():+.4f}")
    print(f"Analytic cluster SE: {m3_cluster.bse['prompt_variant']:.4f}")


def main() -> None:
    df = load_data()
    check_balance(df)
    step1_naive_diff(df)
    formula = step2_ols_hc3(df)
    m3_cluster = step3_cluster_robust(df, formula)
    step4_interactions(df)
    step5_bootstrap(df, formula, m3_cluster)
    print("\n" + "=" * 60)
    print("Done. Ground-truth causal effect baked into generator: +4 pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
