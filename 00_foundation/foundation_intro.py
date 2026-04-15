"""
Article 0 companion script — all four techniques in one file.

Runs every code snippet from the article against the shared synthetic dataset
and prints estimates side-by-side. Use this as a sanity check before extending
into individual article notebooks.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 10000 --out data/synthetic_llm_logs.csv
    python 00_foundation/foundation_intro.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression


def main() -> None:
    data_path = Path("data/synthetic_llm_logs.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run "
            "`python data/generate_data.py --seed 42 --n-users 10000 "
            "--out data/synthetic_llm_logs.csv` first."
        )
    df = pd.read_csv(data_path)

    print("=== Technique 1: Regression for clean experiments ===")
    naive_prompt = (
        df[df.prompt_variant == 1].task_completed.mean()
        - df[df.prompt_variant == 0].task_completed.mean()
    )
    print(f"Naive prompt effect (truth = 0.04): {naive_prompt:.4f}")
    model = smf.ols(
        "task_completed ~ prompt_variant + C(engagement_tier) + query_confidence",
        data=df,
    ).fit(cov_type="HC3")
    print(model.summary().tables[1])

    print("\n=== Technique 2: Propensity scores for opt-in features ===")
    naive_opt_in = (
        df[df.opt_in_agent_mode == 1].task_completed.mean()
        - df[df.opt_in_agent_mode == 0].task_completed.mean()
    )
    print(f"Naive opt-in effect (contaminated): {naive_opt_in:.4f}")
    X = pd.get_dummies(
        df[["engagement_tier", "query_confidence"]], drop_first=True
    ).astype(float)
    ps_model = LogisticRegression(max_iter=1000).fit(X, df.opt_in_agent_mode)
    df["propensity"] = ps_model.predict_proba(X)[:, 1]
    df["ipw"] = np.where(
        df.opt_in_agent_mode == 1, 1 / df.propensity, 1 / (1 - df.propensity)
    )
    treated = df[df.opt_in_agent_mode == 1]
    control = df[df.opt_in_agent_mode == 0]
    ipw_effect = (
        (treated.task_completed * treated.ipw).sum() / treated.ipw.sum()
        - (control.task_completed * control.ipw).sum() / control.ipw.sum()
    )
    print(f"IPW-adjusted opt-in effect (truth = 0.08): {ipw_effect:.4f}")

    print("\n=== Technique 3: Difference-in-differences for staged rollouts ===")
    wave1_pre = df[(df.wave == 1) & (df.signup_week < 20)].task_completed.mean()
    wave1_post = df[
        (df.wave == 1) & (df.signup_week >= 20) & (df.signup_week < 30)
    ].task_completed.mean()
    wave2_pre = df[(df.wave == 2) & (df.signup_week < 20)].task_completed.mean()
    wave2_post = df[
        (df.wave == 2) & (df.signup_week >= 20) & (df.signup_week < 30)
    ].task_completed.mean()
    did_effect = (wave1_post - wave1_pre) - (wave2_post - wave2_pre)
    print(f"DiD wave-1 launch effect (truth = 0.05): {did_effect:.4f}")

    print("\n=== Technique 4: Regression discontinuity for threshold routing ===")
    bandwidth = 0.10
    near_cutoff = df[
        (df.query_confidence > 0.85 - bandwidth)
        & (df.query_confidence < 0.85 + bandwidth)
    ].copy()
    near_cutoff["below_cutoff"] = (near_cutoff.query_confidence < 0.85).astype(int)
    near_cutoff["running_centered"] = near_cutoff.query_confidence - 0.85
    rdd_model = smf.ols(
        "task_completed ~ below_cutoff + running_centered "
        "+ below_cutoff:running_centered",
        data=near_cutoff,
    ).fit(cov_type="HC3")
    print(
        f"RDD premium-routing effect (truth = 0.06): "
        f"{rdd_model.params['below_cutoff']:.4f}"
    )


if __name__ == "__main__":
    main()
