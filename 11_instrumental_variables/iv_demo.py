"""
Instrumental variable analysis for confounded LLM routing.

Companion code for FCC Article 11:
"Unconfounding the AI product experiment: instrumental variable analysis
for multi-model LLM gateways in Python"

Runs every code block from the article against the shared synthetic dataset.
Produces: naive OLS (biased baseline), manual 2SLS from scratch,
first-stage F-statistic, LATE/complier analysis, and bootstrap
95% confidence intervals (500 replicates, seed=7).

The IV demonstration adds three simulated variables on top of the shared
dataset's user covariates:
  - query_complexity: an unobserved confounder (not logged) that makes naive
    OLS biased downward — complex queries route premium AND complete less often
  - rate_limit_fallback: a random infrastructure-driven instrument (Bernoulli
    with p=0.15, independent of all query characteristics)
  - task_completed_iv: outcome regenerated from the IV causal graph so that
    the true effect of premium routing is exactly +6 pp

This mirrors the real-world structure: the confound (query complexity) lives
outside the feature space, and the instrument (rate-limit event) is genuinely
random with respect to what a user asked.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \\
        --out data/synthetic_llm_logs.csv
    python 11_instrumental_variables/iv_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "synthetic_llm_logs.csv"

TRUE_EFFECT = 0.06  # ground-truth premium routing effect baked into DGP


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(seed_iv: int = 99) -> pd.DataFrame:
    """
    Load shared dataset and attach IV demonstration variables.

    Three additions beyond the base dataset columns:
      rate_limit_fallback   — binary instrument; Bernoulli(0.15), independent
                              of all query characteristics
      routed_to_premium_iv  — endogenous routing; probability depends on
                              query_confidence (observable) AND query_complexity
                              (unobserved) creating OLS-biasing confounding
      task_completed_iv     — outcome re-simulated from the IV causal graph;
                              true premium effect = +0.06, unobserved complexity
                              creates a downward bias in naive OLS
    """
    df = pd.read_csv(DATA_PATH)
    rng = np.random.default_rng(seed_iv)
    n = len(df)

    # --- Unobserved confounder ---
    # query_complexity is drawn from N(0, 1). High values mean a difficult query:
    # it increases the probability of premium routing (complex queries get the
    # better model) and decreases task completion (hard queries fail more often).
    query_complexity = rng.normal(0, 1, n)

    # --- Endogenous routing ---
    # Low-confidence queries route to premium; unobserved complexity pushes
    # borderline queries toward premium too. This is the confounding structure
    # that makes naive OLS upward-biased: premium arm receives harder queries.
    log_odds_routing = (
        -2.0
        + 4.0 * (1.0 - df["query_confidence"].values)   # low conf → premium
        + 0.6 * query_complexity                          # complex → premium
    )
    premium_prob = 1.0 / (1.0 + np.exp(-log_odds_routing))
    df["routed_to_premium_iv"] = rng.binomial(1, premium_prob).astype(int)

    # --- Instrument ---
    # Rate-limit fallback: fires with probability 0.15, completely at random,
    # independent of query_confidence, query_complexity, or any other variable.
    # When it fires on a premium-routed query, the gateway reroutes to cheap.
    df["rate_limit_fallback"] = rng.binomial(1, 0.15, n)

    # Actual routing after fallback override.
    # Compliers: queries that would have gone premium but get rerouted to cheap.
    df["routed_to_premium_actual"] = (
        df["routed_to_premium_iv"] * (1 - df["rate_limit_fallback"])
    ).astype(int)

    # --- Outcome ---
    # True causal structure: premium routing adds +0.06; unobserved complexity
    # subtracts up to ~0.08 from completion probability (the hidden confounder).
    engagement_base = np.where(
        df["engagement_tier"] == "heavy", 0.70,
        np.where(df["engagement_tier"] == "medium", 0.55, 0.35)
    )
    completion_prob = np.clip(
        engagement_base
        + TRUE_EFFECT * df["routed_to_premium_actual"].values
        - 0.04 * query_complexity                         # unobserved confounder
        + rng.normal(0, 0.02, n),
        0.01, 0.99,
    )
    df["task_completed_iv"] = rng.binomial(1, completion_prob).astype(int)

    # One-hot encode engagement tier for regression
    df = pd.get_dummies(df, columns=["engagement_tier"], drop_first=True)
    return df


def get_tier_dummies(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("engagement_tier_")]


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_ci(df: pd.DataFrame, n_reps: int = 500, seed: int = 7) -> dict:
    """Bootstrap 95% CIs for OLS and 2SLS estimates (percentile method)."""
    rng = np.random.default_rng(seed)
    tier_dummies = get_tier_dummies(df)
    covariate_str = " + ".join(["query_confidence"] + tier_dummies)

    ols_boot, tsls_boot = [], []

    for _ in range(n_reps):
        samp = df.sample(len(df), replace=True,
                         random_state=int(rng.integers(1_000_000_000)))

        # OLS
        ols_f = (
            f"task_completed_iv ~ routed_to_premium_actual + {covariate_str}"
        )
        ols_b = smf.ols(ols_f, data=samp).fit()
        ols_boot.append(ols_b.params["routed_to_premium_actual"])

        # Manual 2SLS
        s1_f = (
            f"routed_to_premium_actual ~ rate_limit_fallback + {covariate_str}"
        )
        s1_b = smf.ols(s1_f, data=samp).fit()
        samp = samp.copy()
        samp["rtp_hat"] = s1_b.fittedvalues
        s2_f = f"task_completed_iv ~ rtp_hat + {covariate_str}"
        s2_b = smf.ols(s2_f, data=samp).fit()
        tsls_boot.append(s2_b.params["rtp_hat"])

    ols_arr  = np.array(ols_boot)
    tsls_arr = np.array(tsls_boot)

    return {
        "ols":  (float(np.percentile(ols_arr,  2.5)),
                 float(np.percentile(ols_arr,  97.5))),
        "tsls": (float(np.percentile(tsls_arr, 2.5)),
                 float(np.percentile(tsls_arr, 97.5))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = build_dataset()
    tier_dummies  = get_tier_dummies(df)
    covariate_str = " + ".join(["query_confidence"] + tier_dummies)
    n = len(df)

    print("=" * 60)
    print("Article 11 — Instrumental Variable Analysis")
    print(f"Dataset: {n:,} rows  |  True premium effect: +{TRUE_EFFECT:.2f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    print("\n=== Setup: instrument independence check ===")
    fallback_rate = df["rate_limit_fallback"].mean()
    premium_rate  = df["routed_to_premium_actual"].mean()
    conf_fb1 = df.loc[df["rate_limit_fallback"] == 1, "query_confidence"].mean()
    conf_fb0 = df.loc[df["rate_limit_fallback"] == 0, "query_confidence"].mean()
    print(f"Rate-limit fallback rate:      {fallback_rate:.3f}")
    print(f"Premium routing rate (actual): {premium_rate:.3f}")
    print(f"Mean confidence | fallback=1:  {conf_fb1:.3f}")
    print(f"Mean confidence | fallback=0:  {conf_fb0:.3f}")
    print("(Similar means confirm the instrument is independent of query confidence.)")

    # ------------------------------------------------------------------
    print("\n=== Step 1: Naive OLS (biased baseline) ===")
    ols_formula = (
        f"task_completed_iv ~ routed_to_premium_actual + {covariate_str}"
    )
    ols_model = smf.ols(ols_formula, data=df).fit(cov_type="HC3")
    ols_coef  = ols_model.params["routed_to_premium_actual"]
    ols_se    = ols_model.bse["routed_to_premium_actual"]
    ols_pval  = ols_model.pvalues["routed_to_premium_actual"]
    print(f"OLS estimate of premium routing effect: {ols_coef:+.4f}")
    print(f"HC3 standard error:                      {ols_se:.4f}")
    print(f"p-value:                                 {ols_pval:.4f}")

    # ------------------------------------------------------------------
    print("\n=== Step 2: Two-stage least squares (manual 2SLS) ===")

    # Stage 1: regress routing on instrument + observed covariates
    stage1_formula = (
        f"routed_to_premium_actual ~ rate_limit_fallback + {covariate_str}"
    )
    stage1   = smf.ols(stage1_formula, data=df).fit(cov_type="HC3")
    s1_coef  = stage1.params["rate_limit_fallback"]
    s1_pval  = stage1.pvalues["rate_limit_fallback"]
    df["rtp_hat"] = stage1.fittedvalues

    print(f"Stage 1 — instrument coefficient: {s1_coef:+.4f}  (p={s1_pval:.4f})")
    print("  (Negative: fallback lowers probability of premium routing)")

    # Stage 2: regress outcome on predicted routing
    stage2_formula = f"task_completed_iv ~ rtp_hat + {covariate_str}"
    stage2    = smf.ols(stage2_formula, data=df).fit(cov_type="HC3")
    tsls_coef = stage2.params["rtp_hat"]
    tsls_se   = stage2.bse["rtp_hat"]

    print(f"\nStage 2 — 2SLS estimate:            {tsls_coef:+.4f}")
    print(f"  Stage-2 SE (underestimate):        {tsls_se:.4f}")
    print(f"\n--- OLS vs 2SLS comparison ---")
    print(f"OLS estimate (biased):   {ols_coef:+.4f}")
    print(f"2SLS estimate (IV):      {tsls_coef:+.4f}")
    print(f"True premium effect:    +{TRUE_EFFECT:.4f}")
    print(f"Bias in OLS:             {ols_coef - TRUE_EFFECT:+.4f}")

    # ------------------------------------------------------------------
    print("\n=== Step 3: Weak-instrument diagnostics ===")
    restricted_formula = f"routed_to_premium_actual ~ {covariate_str}"
    stage1_restricted  = smf.ols(restricted_formula, data=df).fit()
    f_stat, f_pval, _  = stage1.compare_f_test(stage1_restricted)
    print(f"First-stage F-statistic (instrument): {f_stat:.2f}")
    print(f"p-value:                               {f_pval:.4f}")
    if f_stat > 10:
        print("Instrument is STRONG (F > 10). 2SLS estimates are reliable.")
    elif f_stat > 4:
        print("Instrument is BORDERLINE WEAK (4 < F < 10). Interpret with caution.")
    else:
        print("Instrument is WEAK (F < 4). 2SLS estimates are unreliable.")

    print("\nEndogeneity direction check (OLS minus 2SLS gap):")
    gap = ols_coef - tsls_coef
    print(f"  Gap: {gap:+.4f}")
    if abs(gap) > 0.005:
        print("  Gap suggests endogeneity bias is present in OLS.")
    else:
        print("  Small gap: OLS and 2SLS broadly agree on this dataset.")
    print("  For a formal Hausman endogeneity test, use linearmodels IV2SLS.")

    # ------------------------------------------------------------------
    print("\n=== Step 4: LATE / complier population ===")
    compliers_mask = df["rate_limit_fallback"] == 1
    complier_count = compliers_mask.sum()
    complier_pct   = complier_count / n * 100
    print(f"Approximate complier population: {complier_count:,} ({complier_pct:.1f}% of queries)")
    comp_conf = df.loc[compliers_mask, "query_confidence"].mean()
    comp_tc   = df.loc[compliers_mask, "task_completed_iv"].mean()
    non_conf  = df.loc[~compliers_mask, "query_confidence"].mean()
    non_tc    = df.loc[~compliers_mask, "task_completed_iv"].mean()
    print(f"Complier mean confidence:     {comp_conf:.3f}")
    print(f"Complier task completion:     {comp_tc:.3f}")
    print(f"Non-complier mean confidence: {non_conf:.3f}")
    print(f"Non-complier task completion: {non_tc:.3f}")
    print(f"\n2SLS LATE estimate: {tsls_coef:+.4f}")
    print("This is the causal effect of premium routing for queries rerouted")
    print("by rate-limit fallbacks, not all queries in the dataset.")

    # ------------------------------------------------------------------
    print("\n=== Step 5: Bootstrap 95% confidence intervals "
          "(500 replicates, seed=7) ===")
    cis = bootstrap_ci(df, n_reps=500, seed=7)
    print(f"OLS  95% CI: [{cis['ols'][0]:+.4f}, {cis['ols'][1]:+.4f}]")
    print(f"2SLS 95% CI: [{cis['tsls'][0]:+.4f}, {cis['tsls'][1]:+.4f}]")
    print(f"Ground truth: +{TRUE_EFFECT:.4f}")
    ols_covers  = cis["ols"][0]  <= TRUE_EFFECT <= cis["ols"][1]
    tsls_covers = cis["tsls"][0] <= TRUE_EFFECT <= cis["tsls"][1]
    print(f"OLS CI covers ground truth:  {ols_covers}")
    print(f"2SLS CI covers ground truth: {tsls_covers}")


if __name__ == "__main__":
    main()
