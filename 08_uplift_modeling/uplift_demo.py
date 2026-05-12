"""
Uplift modeling for personalized AI rollouts.

Companion code for FCC Article 8:
"Targeting your AI rollout: uplift modeling for product experiments that
find the users your LLM feature actually helps"

Runs every code block from the article against the shared synthetic dataset.
Produces: naive ATE, naive per-tier gaps, T-learner CATE, X-learner CATE
(Kunzel et al. 2019, corrected formula), Qini curve values, segmented rollout
rule, and bootstrap 95% confidence intervals for all CATE estimates.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \
        --out data/synthetic_llm_logs.csv
    python 08_uplift_modeling/uplift_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    X_full = pd.get_dummies(
        df[["query_confidence", "engagement_tier"]],
        drop_first=False,
    ).astype(float)
    return X_full.values, X_full.columns.tolist()


def fit_tlearner(df: pd.DataFrame, X_all: np.ndarray
                 ) -> tuple[np.ndarray, object, object]:
    treated_mask = df.opt_in_agent_mode == 1
    X1 = X_all[treated_mask]
    Y1 = df[treated_mask].task_completed.values
    X0 = X_all[~treated_mask]
    Y0 = df[~treated_mask].task_completed.values
    m1 = LinearRegression().fit(X1, Y1)
    m0 = LinearRegression().fit(X0, Y0)
    cate = m1.predict(X_all) - m0.predict(X_all)
    return cate, m1, m0


def fit_xlearner(df: pd.DataFrame, X_all: np.ndarray,
                 m1: object, m0: object) -> np.ndarray:
    treated_mask = df.opt_in_agent_mode == 1
    X1 = X_all[treated_mask]
    Y1 = df[treated_mask].task_completed.values
    X0 = X_all[~treated_mask]
    Y0 = df[~treated_mask].task_completed.values

    D1 = Y1 - m0.predict(X1)
    D0 = m1.predict(X0) - Y0

    tau1_model = LinearRegression().fit(X1, D1)
    tau0_model = LinearRegression().fit(X0, D0)

    ps_model = LogisticRegression(max_iter=1000).fit(
        X_all, df.opt_in_agent_mode.values
    )
    e_x = ps_model.predict_proba(X_all)[:, 1]

    tau1_all = tau1_model.predict(X_all)
    tau0_all = tau0_model.predict(X_all)
    # Kunzel et al. (2019): tau(x) = g(x)*tau_1(x) + (1 - g(x))*tau_0(x)
    cate = e_x * tau1_all + (1 - e_x) * tau0_all
    return cate


def qini_values(df: pd.DataFrame, cate_col: str) -> list[float]:
    df_sorted = df.sort_values(cate_col, ascending=False).copy()
    n = len(df_sorted)
    top_ks = np.arange(0.01, 1.01, 0.01)
    vals = []
    for k in top_ks:
        top_n = max(1, int(k * n))
        sub = df_sorted.iloc[:top_n]
        t_sub = sub[sub.opt_in_agent_mode == 1]
        c_sub = sub[sub.opt_in_agent_mode == 0]
        if len(t_sub) > 0 and len(c_sub) > 0:
            vals.append(t_sub.task_completed.mean() - c_sub.task_completed.mean())
        else:
            vals.append(np.nan)
    return vals


def bootstrap_cate_ci(df: pd.DataFrame, X_all: np.ndarray,
                      n_reps: int = 500, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df)
    tier_reps = {"light": [], "medium": [], "heavy": []}
    mean_reps = []
    for _ in range(n_reps):
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        X_b = X_all[idx]
        cate_b, m1_b, m0_b = fit_tlearner(df_b, X_b)
        df_b = df_b.copy()
        df_b["cate"] = cate_b
        for tier in tier_reps:
            tier_reps[tier].append(df_b[df_b.engagement_tier == tier].cate.mean())
        mean_reps.append(cate_b.mean())
    cis = {}
    for tier, vals in tier_reps.items():
        arr = np.array(vals)
        cis[tier] = (float(np.percentile(arr, 2.5)),
                     float(np.percentile(arr, 97.5)))
    arr = np.array(mean_reps)
    cis["mean"] = (float(np.percentile(arr, 2.5)),
                   float(np.percentile(arr, 97.5)))
    return cis


def main() -> None:
    data_path = REPO_ROOT / "data" / "synthetic_llm_logs.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run "
            "`python data/generate_data.py --seed 42 --n-users 50000 "
            "--out data/synthetic_llm_logs.csv` first."
        )
    df = pd.read_csv(data_path)

    print("=== Setup: dataset overview ===")
    print(f"Total users: {len(df):,}")
    print(f"Treated (opt-in): {(df.opt_in_agent_mode == 1).sum():,}")
    print(f"Control:          {(df.opt_in_agent_mode == 0).sum():,}")

    print("\n=== Opt-in rate by engagement tier ===")
    print(df.groupby("engagement_tier").opt_in_agent_mode.mean().round(3))

    naive_ate = (
        df[df.opt_in_agent_mode == 1].task_completed.mean()
        - df[df.opt_in_agent_mode == 0].task_completed.mean()
    )
    print(f"\nNaive ATE (treated - control): {naive_ate:+.4f}")

    print("\n=== Naive per-tier gap (confounded) ===")
    for tier in ["light", "medium", "heavy"]:
        sub = df[df.engagement_tier == tier]
        t = sub[sub.opt_in_agent_mode == 1].task_completed.mean()
        c = sub[sub.opt_in_agent_mode == 0].task_completed.mean()
        print(f"  {tier:8s}: treated={t:.3f}, control={c:.3f}, diff={t - c:+.3f}")

    X_all, feature_cols = build_features(df)
    print(f"\nFeature columns: {feature_cols}")

    print("\n=== Step 1: T-learner ===")
    cate_t, m1, m0 = fit_tlearner(df, X_all)
    df["cate_tlearner"] = cate_t
    print(f"Mean CATE (T-learner): {cate_t.mean():+.4f}")
    print("Mean predicted CATE by engagement tier:")
    print(df.groupby("engagement_tier").cate_tlearner.mean().round(4))

    print("\n=== Step 2: X-learner ===")
    cate_x = fit_xlearner(df, X_all, m1, m0)
    df["cate_xlearner"] = cate_x
    print(f"Mean CATE (X-learner): {cate_x.mean():+.4f}")
    print("Mean predicted CATE by engagement tier:")
    print(df.groupby("engagement_tier").cate_xlearner.mean().round(4))
    print("\nT-learner vs X-learner per tier:")
    comp = df.groupby("engagement_tier")[["cate_tlearner", "cate_xlearner"]].mean().round(4)
    print(comp)

    print("\n=== Step 3: Qini curve values ===")
    qini_vals = qini_values(df, "cate_tlearner")
    for target_k in [10, 20, 30, 50, 70, 100]:
        idx = target_k - 1
        print(f"  Top {target_k:3d}%: observed uplift = {qini_vals[idx]:.4f}")

    print("\n=== Step 4: CATE distribution and rollout rule ===")
    print("CATE distribution (T-learner):")
    print(pd.Series(df.cate_tlearner).describe().round(4))

    threshold = 0.07
    selected = df[df.cate_tlearner >= threshold]
    suppressed = df[df.cate_tlearner < threshold]
    print(f"\nRollout threshold: CATE >= {threshold}")
    print(f"Users selected: {len(selected):,} ({100*len(selected)/len(df):.0f}%)")
    print(f"Users suppressed: {len(suppressed):,} ({100*len(suppressed)/len(df):.0f}%)")
    print("\nTier composition of selected group:")
    print((selected.groupby("engagement_tier").size() / len(selected)).round(3))
    print(f"\nMean predicted CATE (selected):   {selected.cate_tlearner.mean():.4f}")
    print(f"Mean predicted CATE (suppressed): {suppressed.cate_tlearner.mean():.4f}")

    print("\n=== should_show_feature examples ===")
    def should_show_feature(query_confidence, engagement_tier, threshold=0.07):
        x = pd.get_dummies(
            pd.DataFrame([{"query_confidence": query_confidence,
                           "engagement_tier": engagement_tier}]),
            drop_first=False,
        ).reindex(columns=feature_cols, fill_value=0).astype(float).values
        cate = m1.predict(x)[0] - m0.predict(x)[0]
        return cate >= threshold, round(cate, 4)

    for qc, tier in [(0.72, "heavy"), (0.72, "light"), (0.45, "medium")]:
        show, cate = should_show_feature(qc, tier)
        print(f"  {tier} user, conf={qc}: show={show}, CATE={cate}")

    print("\n=== Step 5: Bootstrap 95% CI (500 replicates, seed=7) ===")
    print("  Running bootstrap... (this takes ~60 seconds)")
    cis = bootstrap_cate_ci(df, X_all, n_reps=500, seed=7)
    print(f"  Mean CATE  95% CI: [{cis['mean'][0]:+.4f}, {cis['mean'][1]:+.4f}]")
    print(f"  Light tier 95% CI: [{cis['light'][0]:+.4f}, {cis['light'][1]:+.4f}]")
    print(f"  Medium tier 95% CI: [{cis['medium'][0]:+.4f}, {cis['medium'][1]:+.4f}]")
    print(f"  Heavy tier 95% CI: [{cis['heavy'][0]:+.4f}, {cis['heavy'][1]:+.4f}]")


if __name__ == "__main__":
    main()
