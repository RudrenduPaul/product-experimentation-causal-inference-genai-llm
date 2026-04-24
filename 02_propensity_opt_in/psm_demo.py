"""
Propensity score methods for opt-in AI features.

Companion code for FCC Article 2:
"Product experimentation with propensity scores: causal inference for
LLM-based features in Python"

Runs every code block from the article against the shared synthetic dataset.
Produces: naive comparison, propensity score model, IPW ATE and ATT,
1-NN matching ATT, covariate-balance diagnostics, and bootstrap
95% confidence intervals.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \
        --out data/synthetic_llm_logs.csv
    python 02_propensity_opt_in/psm_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def smd(treated_vals, control_vals, treated_w=None, control_w=None):
    """Standardized mean difference, optionally with weights."""
    if treated_w is None:
        treated_w = np.ones(len(treated_vals))
    if control_w is None:
        control_w = np.ones(len(control_vals))
    t_mean = np.average(treated_vals, weights=treated_w)
    c_mean = np.average(control_vals, weights=control_w)
    pooled_std = np.sqrt((treated_vals.var() + control_vals.var()) / 2)
    return (t_mean - c_mean) / pooled_std


def fit_propensity(df: pd.DataFrame) -> pd.Series:
    X = pd.get_dummies(
        df[["engagement_tier", "query_confidence"]], drop_first=True
    ).astype(float)
    model = LogisticRegression(max_iter=1000).fit(X, df.opt_in_agent_mode)
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index)


def estimate_all(df: pd.DataFrame) -> dict:
    """Return a dict with ATE (IPW), ATT (IPW), and ATT (1-NN) on df."""
    df = df.copy()
    df["propensity"] = fit_propensity(df)
    df["ipw"] = np.where(
        df.opt_in_agent_mode == 1,
        1 / df.propensity,
        1 / (1 - df.propensity),
    )
    df["ipw_att"] = np.where(
        df.opt_in_agent_mode == 1,
        1,
        df.propensity / (1 - df.propensity),
    )
    t = df[df.opt_in_agent_mode == 1]
    c = df[df.opt_in_agent_mode == 0]

    ate_ipw = (
        (t.task_completed * t.ipw).sum() / t.ipw.sum()
        - (c.task_completed * c.ipw).sum() / c.ipw.sum()
    )
    att_ipw = t.task_completed.mean() - (
        (c.task_completed * c.ipw_att).sum() / c.ipw_att.sum()
    )

    nn = NearestNeighbors(n_neighbors=1).fit(c[["propensity"]].values)
    _, idx = nn.kneighbors(t[["propensity"]].values)
    matched_c = c.task_completed.values[idx.flatten()]
    att_match = (t.task_completed.values - matched_c).mean()

    return {"ate_ipw": ate_ipw, "att_ipw": att_ipw, "att_match": att_match}


def bootstrap_ci(df: pd.DataFrame, n_reps: int = 500,
                 seed: int = 7) -> dict:
    """Non-parametric bootstrap 95% CI for each estimator."""
    rng = np.random.default_rng(seed)
    n = len(df)
    reps = {"ate_ipw": [], "att_ipw": [], "att_match": []}
    for _ in range(n_reps):
        sample = df.iloc[rng.integers(0, n, size=n)]
        est = estimate_all(sample)
        for key, value in est.items():
            reps[key].append(value)
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

    print("=== Selection pattern: opt-in rate by engagement tier ===")
    print(df.groupby("engagement_tier").opt_in_agent_mode.mean().round(3))

    print("\n=== Naive comparison ===")
    naive = (
        df[df.opt_in_agent_mode == 1].task_completed.mean()
        - df[df.opt_in_agent_mode == 0].task_completed.mean()
    )
    print(f"Naive opt-in effect: {naive:+.4f}  (ground truth = 0.08)")

    print("\n=== Step 1: propensity score model ===")
    X = pd.get_dummies(
        df[["engagement_tier", "query_confidence"]], drop_first=True
    ).astype(float)
    ps_model = LogisticRegression(max_iter=1000).fit(X, df.opt_in_agent_mode)
    df["propensity"] = ps_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(df.opt_in_agent_mode, df.propensity)
    print("Mean propensity by engagement tier:")
    print(df.groupby("engagement_tier").propensity.mean().round(3))
    print(
        f"Propensity range (treated):  "
        f"{df[df.opt_in_agent_mode == 1].propensity.min():.3f} - "
        f"{df[df.opt_in_agent_mode == 1].propensity.max():.3f}"
    )
    print(
        f"Propensity range (control):  "
        f"{df[df.opt_in_agent_mode == 0].propensity.min():.3f} - "
        f"{df[df.opt_in_agent_mode == 0].propensity.max():.3f}"
    )
    print(f"Propensity model AUC: {auc:.3f}")

    print("\n=== Step 2: inverse-probability weighting ===")
    df["ipw"] = np.where(
        df.opt_in_agent_mode == 1,
        1 / df.propensity,
        1 / (1 - df.propensity),
    )
    t = df[df.opt_in_agent_mode == 1]
    c = df[df.opt_in_agent_mode == 0]
    ate_ipw = (
        (t.task_completed * t.ipw).sum() / t.ipw.sum()
        - (c.task_completed * c.ipw).sum() / c.ipw.sum()
    )
    print(f"IPW ATE: {ate_ipw:+.4f}")

    df["ipw_att"] = np.where(
        df.opt_in_agent_mode == 1,
        1,
        df.propensity / (1 - df.propensity),
    )
    t = df[df.opt_in_agent_mode == 1]
    c = df[df.opt_in_agent_mode == 0]
    treated_mean = t.task_completed.mean()
    control_w_mean = (
        (c.task_completed * c.ipw_att).sum() / c.ipw_att.sum()
    )
    att_ipw = treated_mean - control_w_mean
    print(f"IPW ATT: {att_ipw:+.4f}")

    print("\n=== Step 3: nearest-neighbor matching ===")
    treated_ps = df[df.opt_in_agent_mode == 1][["propensity"]].values
    control_ps = df[df.opt_in_agent_mode == 0][["propensity"]].values
    nn = NearestNeighbors(n_neighbors=1).fit(control_ps)
    _, idx = nn.kneighbors(treated_ps)
    treated_outcomes = df[df.opt_in_agent_mode == 1].task_completed.values
    matched_control_outcomes = (
        df[df.opt_in_agent_mode == 0].task_completed.values[idx.flatten()]
    )
    att_match = (treated_outcomes - matched_control_outcomes).mean()
    print(f"1-NN matching ATT: {att_match:+.4f}")

    print("\n=== Step 4: covariate balance (|SMD| < 0.1 is good) ===")
    engagement_heavy = (df.engagement_tier == "heavy").astype(float).values
    qc = df.query_confidence.values
    tr = (df.opt_in_agent_mode == 1).values
    covariates = {
        "engagement_tier_heavy": engagement_heavy,
        "query_confidence": qc,
    }
    print(f"{'Covariate':<30} {'Raw SMD':>10} {'Weighted SMD':>15}")
    for name, vals in covariates.items():
        smd_raw = smd(vals[tr], vals[~tr])
        smd_weighted = smd(
            vals[tr], vals[~tr],
            treated_w=df[tr].ipw.values,
            control_w=df[~tr].ipw.values,
        )
        print(f"{name:<30} {smd_raw:>+10.3f} {smd_weighted:>+15.3f}")

    print("\n=== Step 5: bootstrap 95% confidence intervals "
          "(500 replicates, ~2 min) ===")
    cis = bootstrap_ci(df, n_reps=500, seed=7)
    print(f"IPW ATE       95% CI: "
          f"[{cis['ate_ipw'][0]:+.4f}, {cis['ate_ipw'][1]:+.4f}]")
    print(f"IPW ATT       95% CI: "
          f"[{cis['att_ipw'][0]:+.4f}, {cis['att_ipw'][1]:+.4f}]")
    print(f"1-NN matching 95% CI: "
          f"[{cis['att_match'][0]:+.4f}, {cis['att_match'][1]:+.4f}]")


if __name__ == "__main__":
    main()
