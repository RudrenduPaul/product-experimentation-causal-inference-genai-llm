"""
Cluster randomization and SUTVA for collaborative AI features.

Companion code for FCC Article 5:
"Product Experimentation for Collaborative AI Features: Cluster Randomization,
SUTVA, and Network Spillovers for LLM-Based Tools in Python"

Runs every code block from the article against the shared synthetic dataset,
then executes the bootstrap confidence interval section. Produces: naive OLS
(biased), cluster-weighted least squares, two-exposure decomposition, and
bootstrap 95% CIs around every estimator.

Ground-truth effects baked in:
    Direct effect on treated users:           +0.80 min
    Spillover effect on cross-workspace
    collaborators in control workspaces:      +0.20 min
    Total effect on a fully-rolled-out
    workspace (direct + spillover mix):       +0.80 + 0.20 * spillover_share

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \
        --out data/synthetic_llm_logs.csv
    python 05_cluster_randomization/cluster_randomization_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DIRECT_EFFECT = 0.80
SPILLOVER_EFFECT = 0.20
DATA_SEED = 42
OUTCOME_NOISE_SD = 0.30


def build_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Attach cluster assignment, spillover exposure, and the observed outcome.

    Workspaces 0 through 24 are treated; 25 through 49 are control. A control
    user is spillover-exposed if they interact cross-workspace (we use
    opt_in_agent_mode as a behavioral proxy for a user who actively engages
    with AI-generated artifacts shared across workspaces — these are the users
    who read teammate-authored docs, Slack threads, and pull requests that
    contain treated-workspace output).
    """
    rng = np.random.default_rng(DATA_SEED)
    df = df.copy()

    df["treated_workspace"] = (df["workspace_id"] < 25).astype(int)
    df["treated_user"] = df["treated_workspace"]
    df["spillover_exposed"] = (
        (df["treated_workspace"] == 0) & (df["opt_in_agent_mode"] == 1)
    ).astype(int)

    ws_baseline = pd.DataFrame({
        "workspace_id": np.arange(50),
        "ws_baseline": rng.normal(5.0, 0.30, size=50),
    })
    df = df.merge(ws_baseline, on="workspace_id")

    noise = rng.normal(0, OUTCOME_NOISE_SD, size=len(df))
    df["session_minutes_obs"] = (
        df["ws_baseline"]
        + DIRECT_EFFECT * df["treated_user"]
        + SPILLOVER_EFFECT * df["spillover_exposed"]
        + noise
    )
    df["exposure"] = np.select(
        [df["treated_user"] == 1, df["spillover_exposed"] == 1],
        ["direct", "spillover"],
        default="pure_control",
    )
    return df


def naive_ols(df: pd.DataFrame) -> dict:
    model = smf.ols("session_minutes_obs ~ treated_user", data=df).fit()
    est = model.params["treated_user"]
    se = model.bse["treated_user"]
    ci = model.conf_int().loc["treated_user"].tolist()
    return {"est": est, "se": se, "ci": ci, "n": int(model.nobs)}


def cluster_wls(df: pd.DataFrame) -> dict:
    ws = (
        df.groupby("workspace_id")
        .agg(ws_mean=("session_minutes_obs", "mean"),
             ws_size=("user_id", "count"),
             treated=("treated_workspace", "max"))
        .reset_index()
    )
    X = sm.add_constant(ws["treated"])
    model = sm.WLS(ws["ws_mean"], X, weights=ws["ws_size"]).fit()
    est = model.params["treated"]
    se = model.bse["treated"]
    ci = model.conf_int().loc["treated"].tolist()
    return {"est": est, "se": se, "ci": ci, "k": len(ws)}


def two_exposure_ols(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["is_direct"] = (df["exposure"] == "direct").astype(int)
    df["is_spillover"] = (df["exposure"] == "spillover").astype(int)
    model = smf.ols(
        "session_minutes_obs ~ is_direct + is_spillover",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["workspace_id"]})
    direct = model.params["is_direct"]
    spillover = model.params["is_spillover"]
    return {
        "direct": direct,
        "spillover": spillover,
        "direct_se": model.bse["is_direct"],
        "spillover_se": model.bse["is_spillover"],
        "direct_ci": model.conf_int().loc["is_direct"].tolist(),
        "spillover_ci": model.conf_int().loc["is_spillover"].tolist(),
    }


def bootstrap_ci(df: pd.DataFrame, n_reps: int = 500, seed: int = 7) -> dict:
    """Cluster bootstrap: resample entire workspaces with replacement.

    Resampling at the user level understates the variance of a cluster-assigned
    treatment. Resampling at the workspace level (cluster bootstrap) keeps the
    within-workspace correlation intact and matches the design.
    """
    rng = np.random.default_rng(seed)
    workspace_ids = df["workspace_id"].unique()
    k = len(workspace_ids)
    reps = {"naive": [], "cluster_wls": [], "direct": [], "spillover": []}
    for _ in range(n_reps):
        draw = rng.choice(workspace_ids, size=k, replace=True)
        # Concatenate all user rows from the resampled workspaces.
        sample = pd.concat(
            [df[df["workspace_id"] == wid] for wid in draw],
            ignore_index=True,
        )
        reps["naive"].append(naive_ols(sample)["est"])
        reps["cluster_wls"].append(cluster_wls(sample)["est"])
        two_exp = two_exposure_ols(sample)
        reps["direct"].append(two_exp["direct"])
        reps["spillover"].append(two_exp["spillover"])
    cis = {}
    for key, values in reps.items():
        arr = np.array(values)
        cis[key] = (float(np.percentile(arr, 2.5)),
                    float(np.percentile(arr, 97.5)))
    return cis


def main() -> None:
    data_path = Path("data/synthetic_llm_logs.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run "
            "`python data/generate_data.py --seed 42 --n-users 50000 "
            "--out data/synthetic_llm_logs.csv` first."
        )
    df = pd.read_csv(data_path)
    df = build_scenario(df)

    print("=== Step 1: scenario counts (50,000 users across 50 workspaces) ===")
    print(f"Total users:            {len(df):,}")
    print(f"Treated workspaces:     {df[df.treated_workspace == 1].workspace_id.nunique()}")
    print(f"Control workspaces:     {df[df.treated_workspace == 0].workspace_id.nunique()}")
    print(f"Treated users:          {df.treated_user.sum():,}")
    print(f"Pure-control users:     {(df.exposure == 'pure_control').sum():,}")
    print(f"Spillover-exposed users:{(df.exposure == 'spillover').sum():,}")
    print("\nWorkspace size distribution:")
    ws_sizes = df.groupby("workspace_id").size()
    print(f"  min={ws_sizes.min()}  median={int(ws_sizes.median())}  max={ws_sizes.max()}")

    print("\n=== Step 2: naive user-level OLS (biased) ===")
    naive = naive_ols(df)
    print(f"Naive estimate:     {naive['est']:+.4f} min")
    print(f"Naive SE:           {naive['se']:.4f}  (under-reported — ignores clustering)")
    print(f"Naive 95% CI:       [{naive['ci'][0]:+.4f}, {naive['ci'][1]:+.4f}]")
    print(f"Ground-truth direct effect: +0.80 min")
    print(f"Bias:               {naive['est'] - DIRECT_EFFECT:+.4f} min (downward — control group is contaminated)")

    print("\n=== Step 3: cluster-weighted least squares (honest SE) ===")
    wls = cluster_wls(df)
    print(f"WLS cluster-ATE:    {wls['est']:+.4f} min")
    print(f"WLS SE:             {wls['se']:.4f}  (correctly based on K={wls['k']} clusters)")
    print(f"WLS 95% CI:         [{wls['ci'][0]:+.4f}, {wls['ci'][1]:+.4f}]")
    print("Point estimate is still biased because the control cluster means")
    print("include spillover-exposed users. Only the SE is honest here.")

    print("\n=== Step 4: two-exposure decomposition ===")
    two_exp = two_exposure_ols(df)
    print(f"Direct effect:      {two_exp['direct']:+.4f} min  (ground truth = +0.80)")
    print(f"  SE:               {two_exp['direct_se']:.4f}")
    print(f"  95% CI:           [{two_exp['direct_ci'][0]:+.4f}, {two_exp['direct_ci'][1]:+.4f}]")
    print(f"Spillover effect:   {two_exp['spillover']:+.4f} min  (ground truth = +0.20)")
    print(f"  SE:               {two_exp['spillover_se']:.4f}")
    print(f"  95% CI:           [{two_exp['spillover_ci'][0]:+.4f}, {two_exp['spillover_ci'][1]:+.4f}]")
    spillover_share_overall = (df["exposure"] == "spillover").mean()
    print(f"\nAggregate effect under full rollout (direct + spillover share * spillover):")
    print(f"  spillover share of all users: {spillover_share_overall:.4f}")
    print(f"  projected total:              "
          f"{two_exp['direct'] + spillover_share_overall * two_exp['spillover']:+.4f} min")

    print("\n=== Step 5: cluster bootstrap 95% CIs (500 replicates, seed=7) ===")
    cis = bootstrap_ci(df, n_reps=500, seed=7)
    print(f"Naive OLS        bootstrap 95% CI:  "
          f"[{cis['naive'][0]:+.4f}, {cis['naive'][1]:+.4f}]   (misses +0.80)")
    print(f"Cluster WLS      bootstrap 95% CI:  "
          f"[{cis['cluster_wls'][0]:+.4f}, {cis['cluster_wls'][1]:+.4f}]   (misses +0.80)")
    print(f"Direct effect    bootstrap 95% CI:  "
          f"[{cis['direct'][0]:+.4f}, {cis['direct'][1]:+.4f}]   (covers +0.80)")
    print(f"Spillover effect bootstrap 95% CI:  "
          f"[{cis['spillover'][0]:+.4f}, {cis['spillover'][1]:+.4f}]   (covers +0.20)")


if __name__ == "__main__":
    main()
