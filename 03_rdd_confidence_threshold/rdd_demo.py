"""
Regression discontinuity design (RDD) for LLM confidence-threshold routing.

Companion code for FCC Article 3:
"Product experimentation with regression discontinuity: how an LLM
confidence threshold creates a natural product experiment in Python"

Runs every code block from the article against the shared synthetic
50,000-user dataset. Produces: naive comparison, sharp local-linear RDD
at bw=0.10, bandwidth sensitivity sweep, density check around the
threshold, quadratic robustness specification, and bootstrap 95%
confidence intervals for every point estimate.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \
        --out data/synthetic_llm_logs.csv
    python 03_rdd_confidence_threshold/rdd_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


CUTOFF = 0.85


def fit_sharp_rdd(df: pd.DataFrame, cutoff: float, bw: float,
                  quadratic: bool = False):
    """Fit a sharp RDD at a given bandwidth. Returns (result, near_df)."""
    near = df[(df.query_confidence > cutoff - bw) &
              (df.query_confidence < cutoff + bw)].copy()
    near["below_cutoff"] = (near.query_confidence < cutoff).astype(int)
    near["rc"] = near.query_confidence - cutoff
    if quadratic:
        near["rc2"] = near.rc ** 2
        formula = ("task_completed ~ below_cutoff + rc + below_cutoff:rc"
                   " + rc2 + below_cutoff:rc2")
    else:
        formula = "task_completed ~ below_cutoff + rc + below_cutoff:rc"
    return smf.ols(formula, data=near).fit(cov_type="HC3"), near


def bootstrap_ci(df: pd.DataFrame, cutoff: float, bw: float,
                 quadratic: bool = False, n_reps: int = 500,
                 seed: int = 7) -> tuple[float, float]:
    """Non-parametric bootstrap 95% CI for the sharp-RDD jump at a cutoff.

    Resamples the bandwidth-restricted slice with replacement, refits the
    RDD on each replicate, collects the `below_cutoff` coefficient, and
    returns the (2.5th, 97.5th) percentile interval.
    """
    rng = np.random.default_rng(seed)
    near = df[(df.query_confidence > cutoff - bw) &
              (df.query_confidence < cutoff + bw)].copy()
    near["below_cutoff"] = (near.query_confidence < cutoff).astype(int)
    near["rc"] = near.query_confidence - cutoff
    if quadratic:
        near["rc2"] = near.rc ** 2
        formula = ("task_completed ~ below_cutoff + rc + below_cutoff:rc"
                   " + rc2 + below_cutoff:rc2")
    else:
        formula = "task_completed ~ below_cutoff + rc + below_cutoff:rc"

    n = len(near)
    estimates = np.empty(n_reps)
    for i in range(n_reps):
        sample = near.iloc[rng.integers(0, n, size=n)]
        m = smf.ols(formula, data=sample).fit()
        estimates[i] = m.params["below_cutoff"]
    return (float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5)))


def main() -> None:
    data_path = Path("data/synthetic_llm_logs.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run "
            "`python data/generate_data.py --seed 42 --n-users 50000 "
            "--out data/synthetic_llm_logs.csv` first."
        )
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    print("\n=== Routing breakdown ===")
    counts = df.routed_to_premium.value_counts().to_dict()
    print(f"Premium-routed (confidence < 0.85):  {counts.get(1, 0):,}")
    print(f"Cheap-routed   (confidence >= 0.85): {counts.get(0, 0):,}")
    print("\nQuery confidence distribution:")
    print(df.query_confidence.describe().round(3))

    print("\n=== Naive comparison (no controls) ===")
    naive = (
        df[df.routed_to_premium == 1].task_completed.mean()
        - df[df.routed_to_premium == 0].task_completed.mean()
    )
    print(f"Naive premium-vs-cheap effect: {naive:+.4f}  (ground truth = +0.06)")
    print("Premium-routed users have systematically lower-confidence queries,")
    print("so this comparison confounds the routing decision with query difficulty.")

    print("\n=== Step 1: Sharp RDD with local linear regression (bw=0.10) ===")
    rdd_model, near = fit_sharp_rdd(df, CUTOFF, bw=0.10)
    effect = float(rdd_model.params["below_cutoff"])
    pval = float(rdd_model.pvalues["below_cutoff"])
    se = float(rdd_model.bse["below_cutoff"])
    print(f"RDD effect at cutoff (LATE): {effect:+.4f}")
    print(f"Std error (HC3):             {se:.4f}")
    print(f"p-value:                     {pval:.4f}")
    print(f"N users in [0.75, 0.95):     {len(near):,}")

    print("\n=== Step 2: Bandwidth sensitivity ===")
    rows = []
    for bw in [0.05, 0.10, 0.15, 0.20]:
        m, sub = fit_sharp_rdd(df, CUTOFF, bw=bw)
        rows.append({
            "bandwidth": bw,
            "n": len(sub),
            "effect": float(m.params["below_cutoff"]),
            "se": float(m.bse["below_cutoff"]),
            "p": float(m.pvalues["below_cutoff"]),
        })
    bw_table = pd.DataFrame(rows)
    print(bw_table.round(4).to_string(index=False))

    print("\n=== Step 3: Density check near the cutoff (manipulation diagnostic) ===")
    print("User counts in 2-percentage-point bins around 0.85:")
    bins = [(0.80, 0.82), (0.82, 0.84), (0.84, 0.86), (0.86, 0.88), (0.88, 0.90)]
    bin_counts = []
    for lo, hi in bins:
        cnt = int(((df.query_confidence >= lo)
                   & (df.query_confidence < hi)).sum())
        bin_counts.append(cnt)
        print(f"  [{lo:.2f}, {hi:.2f}):  n = {cnt:,}")
    spread = max(bin_counts) - min(bin_counts)
    print(f"Spread across 5 bins: {spread:,} users "
          f"(no sharp spike at 0.85 = no manipulation evidence)")

    print("\n=== Step 4: Quadratic robustness check (bw=0.10) ===")
    quad_model, _ = fit_sharp_rdd(df, CUTOFF, bw=0.10, quadratic=True)
    quad_effect = float(quad_model.params["below_cutoff"])
    quad_p = float(quad_model.pvalues["below_cutoff"])
    quad_se = float(quad_model.bse["below_cutoff"])
    print(f"Linear RDD    effect: {effect:+.4f}  p = {pval:.4f}")
    print(f"Quadratic RDD effect: {quad_effect:+.4f}  p = {quad_p:.4f}  "
          f"(SE = {quad_se:.4f})")
    diff = effect - quad_effect
    print(f"Linear-vs-quadratic gap: {diff:+.4f}")

    print("\n=== Step 5: Bootstrap 95% confidence intervals "
          "(500 replicates, seed=7) ===")
    print("Recomputes the RDD on each bootstrap sample at every bandwidth.")
    print("Each interval should cover the +0.06 ground truth and exclude the")
    print(f"naive estimate of {naive:+.4f}.\n")

    print("Linear RDD (Step 1, bw=0.10):")
    lin_lo, lin_hi = bootstrap_ci(df, CUTOFF, bw=0.10)
    print(f"  effect = {effect:+.4f}   95% CI: [{lin_lo:+.4f}, {lin_hi:+.4f}]")

    print("\nBandwidth sensitivity (Step 2):")
    for row in rows:
        bw = row["bandwidth"]
        lo, hi = bootstrap_ci(df, CUTOFF, bw=bw)
        print(f"  bw = {bw:.2f}, n = {row['n']:>6,}   "
              f"effect = {row['effect']:+.4f}   "
              f"95% CI: [{lo:+.4f}, {hi:+.4f}]")

    print("\nQuadratic RDD (Step 4, bw=0.10):")
    quad_lo, quad_hi = bootstrap_ci(df, CUTOFF, bw=0.10, quadratic=True)
    print(f"  effect = {quad_effect:+.4f}   "
          f"95% CI: [{quad_lo:+.4f}, {quad_hi:+.4f}]")


if __name__ == "__main__":
    main()
