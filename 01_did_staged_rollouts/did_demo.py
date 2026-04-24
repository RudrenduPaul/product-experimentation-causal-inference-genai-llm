"""
Difference-in-differences for staged AI feature rollouts.

Companion code for FCC Article 1:
"Why A/B Testing Breaks for Staged AI Feature Rollouts (and How to Use
Difference-in-Differences in Python Instead)"

Runs every code block from the article against the shared synthetic dataset.
Produces: printed DiD estimates, a parallel-trends plot, a pre-trend placebo test.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \
        --out data/synthetic_llm_logs.csv
    python 01_did_staged_rollouts/did_demo.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("data/synthetic_llm_logs.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run "
            "`python data/generate_data.py --seed 42 --n-users 50000 "
            "--out data/synthetic_llm_logs.csv` first."
        )
    df = pd.read_csv(data_path)

    print("=== Dataset shape and wave structure ===")
    print(f"Shape: {df.shape}")
    print(f"Wave sizes: {df.wave.value_counts().to_dict()}")
    print(
        "Treatment weeks per wave: "
        f"{df.groupby('wave').treatment_week.first().to_dict()}"
    )

    # Restrict to users who signed up before the wave 2 launch.
    analysis = df[df.signup_week < 30].copy()
    analysis["post"] = (analysis.signup_week >= 20).astype(int)
    analysis["treated"] = (analysis.wave == 1).astype(int)

    print("\n=== 2x2 cell means ===")
    print(
        analysis.groupby(["treated", "post"])
        .agg(n=("user_id", "count"),
             mean_completion=("task_completed", "mean"))
        .round(3)
    )

    print("\n=== Step 1: simple 2x2 DiD ===")
    cells = analysis.groupby(["treated", "post"]).task_completed.mean()
    w2_pre, w2_post = cells.loc[(0, 0)], cells.loc[(0, 1)]
    w1_pre, w1_post = cells.loc[(1, 0)], cells.loc[(1, 1)]
    did_simple = (w1_post - w1_pre) - (w2_post - w2_pre)
    print(f"Wave 1 change: {w1_post - w1_pre:+.4f}")
    print(f"Wave 2 change: {w2_post - w2_pre:+.4f}")
    print(f"DiD effect:    {did_simple:+.4f}  (ground truth = 0.05)")

    print("\n=== Step 2: regression DiD with clustered SE ===")
    did_model = smf.ols(
        "task_completed ~ treated * post + C(engagement_tier)",
        data=analysis,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis.workspace_id},
    )
    coef = did_model.params["treated:post"]
    pval = did_model.pvalues["treated:post"]
    print(f"treated:post coefficient: {coef:+.4f}")
    print(f"p-value (clustered):      {pval:.4f}")

    print("\n=== Step 3a: parallel-trends visual check ===")
    pre = df[df.signup_week < 30].copy()
    weekly = (
        pre.groupby(["signup_week", "wave"]).task_completed.mean()
        .reset_index()
        .pivot(index="signup_week", columns="wave", values="task_completed")
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(weekly.index, weekly[1], marker="o",
            label="Wave 1 (treated at week 20)")
    ax.plot(weekly.index, weekly[2], marker="s",
            label="Wave 2 (control during this window)")
    ax.axvline(20, color="gray", linestyle="--",
               label="Wave 1 treatment starts")
    ax.set_xlabel("Signup week")
    ax.set_ylabel("Mean task completion rate")
    ax.set_title("Parallel-trends visual check")
    ax.legend()
    plt.tight_layout()
    out_path = Path("01_did_staged_rollouts/parallel_trends.png")
    plt.savefig(out_path, dpi=140)
    print(f"Saved {out_path}")

    print("\n=== Step 3b: formal pre-trend placebo ===")
    pre_only = analysis[analysis.post == 0].copy()
    pre_only["weeks_since_start"] = pre_only.signup_week - 10
    placebo = smf.ols(
        "task_completed ~ treated * weeks_since_start + C(engagement_tier)",
        data=pre_only,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pre_only.workspace_id},
    )
    slope = placebo.params["treated:weeks_since_start"]
    p = placebo.pvalues["treated:weeks_since_start"]
    print(f"Pre-trend slope difference: {slope:+.5f}")
    print(f"p-value:                    {p:.4f}")
    verdict = "PASSES" if p > 0.1 else "FAILS"
    print(f"Parallel-trends placebo test {verdict}.")


if __name__ == "__main__":
    main()
