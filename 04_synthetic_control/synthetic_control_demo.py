"""
Synthetic control for global LLM rollouts.

Companion code for FCC Article 4:
"Product experimentation for global LLM rollouts: synthetic control
methods for causal inference when every user gets the upgrade"

Runs every code block from the article against the shared synthetic
50,000-user dataset. Produces: workspace-week panel construction,
SLSQP donor-weight optimization, pre-period fit, post-period gap,
in-space placebo permutation test, leave-one-out (LOO) donor
sensitivity, and user-level cluster bootstrap 95% confidence intervals
for the post-treatment gap.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \
        --out data/synthetic_llm_logs.csv
    python 04_synthetic_control/synthetic_control_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PRE = 20        # weeks 0-19 are pre-treatment
TREATMENT_WEEK = 20
WINDOW = 30     # analysis window weeks 0-29


def build_panel(df: pd.DataFrame):
    """Return (treated_series, donor_matrix, wave1_ws, wave2_ws)."""
    df_window = df[df.signup_week < WINDOW].copy()

    panel = (
        df_window
        .groupby(["workspace_id", "signup_week"])["task_completed"]
        .mean()
        .reset_index()
    )
    panel.columns = ["workspace_id", "week", "task_completed"]

    pivot = panel.pivot(
        index="week", columns="workspace_id", values="task_completed"
    )
    pivot = pivot.interpolate(method="linear", axis=0).ffill().bfill()

    ws_wave = df.groupby("workspace_id").wave.first()
    wave1_ws = sorted(ws_wave[ws_wave == 1].index.tolist())
    wave2_ws = sorted(ws_wave[ws_wave == 2].index.tolist())

    treated_series = pivot[wave1_ws].mean(axis=1).values
    donor_matrix = pivot[wave2_ws].values
    return treated_series, donor_matrix, wave1_ws, wave2_ws


def fit_sc(treated_series: np.ndarray, donor_matrix: np.ndarray,
           pre: int = PRE) -> np.ndarray:
    """SLSQP optimization: convex combination of donors that tracks
    treated series in the pre-period."""
    n_donors = donor_matrix.shape[1]
    Y_pre = treated_series[:pre]
    D_pre = donor_matrix[:pre, :]

    def objective(w):
        return np.mean((Y_pre - D_pre @ w) ** 2)

    w0 = np.ones(n_donors) / n_donors
    bounds = [(0, 1)] * n_donors
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(
        objective, w0, method="SLSQP", bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    return res.x


def post_gap(treated_series: np.ndarray, donor_matrix: np.ndarray,
             w: np.ndarray, pre: int = PRE) -> float:
    """Mean post-period gap between treated and synthetic control."""
    synth = donor_matrix @ w
    return float((treated_series[pre:] - synth[pre:]).mean())


def placebo_permutation(treated_series: np.ndarray,
                        donor_matrix: np.ndarray, pre: int = PRE):
    """In-space placebo test: each donor takes its turn as the
    placebo treated unit, re-fit, record post-period gap."""
    n_donors = donor_matrix.shape[1]
    gaps = np.empty(n_donors)
    for j in range(n_donors):
        placebo_treated = donor_matrix[:, j]
        placebo_pool = np.delete(donor_matrix, j, axis=1)
        w_p = fit_sc(placebo_treated, placebo_pool, pre=pre)
        gaps[j] = post_gap(placebo_treated, placebo_pool, w_p, pre=pre)
    return gaps


def loo_sensitivity(treated_series: np.ndarray,
                    donor_matrix: np.ndarray, w: np.ndarray,
                    wave2_ws: list, pre: int = PRE) -> pd.DataFrame:
    """Drop each non-zero-weight donor in turn, refit, record gap."""
    rows = []
    nz = np.where(w > 0.001)[0]
    for j in nz:
        kept = np.delete(donor_matrix, j, axis=1)
        w_new = fit_sc(treated_series, kept, pre=pre)
        gap_new = post_gap(treated_series, kept, w_new, pre=pre)
        rows.append({
            "dropped_workspace": int(wave2_ws[j]),
            "dropped_weight": float(w[j]),
            "new_gap": float(gap_new),
        })
    return pd.DataFrame(rows).sort_values("dropped_weight", ascending=False)


def cluster_bootstrap_ci(df: pd.DataFrame, n_reps: int = 500,
                         seed: int = 7) -> tuple[float, float]:
    """User-level cluster bootstrap 95% CI for the post-period gap.

    Resamples users with replacement, rebuilds the workspace-week
    panel, re-fits donor weights on the pre-period, and records the
    post-period gap. Returns the 2.5th/97.5th percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    gaps = np.empty(n_reps)
    for i in range(n_reps):
        sample = df.iloc[rng.integers(0, n, size=n)]
        t, d, _, _ = build_panel(sample)
        w = fit_sc(t, d)
        gaps[i] = post_gap(t, d, w)
    return (float(np.percentile(gaps, 2.5)),
            float(np.percentile(gaps, 97.5)))


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

    print("\n=== Wave breakdown ===")
    print(df.wave.value_counts().to_dict())
    print(f"Workspace count: {df.workspace_id.nunique()}")

    print("\n=== Naive before/after for wave 1 ===")
    w1 = df[df.wave == 1]
    pre_mean = w1[w1.signup_week < TREATMENT_WEEK].task_completed.mean()
    post_mean = w1[(w1.signup_week >= TREATMENT_WEEK)
                   & (w1.signup_week < WINDOW)].task_completed.mean()
    naive_gap = post_mean - pre_mean
    print(f"Wave 1 pre-period mean  (weeks 0-19):  {pre_mean:.4f}")
    print(f"Wave 1 post-period mean (weeks 20-29): {post_mean:.4f}")
    print(f"Naive before/after gap: {naive_gap:+.4f}  "
          f"(ground truth = +0.05)")
    print("Confounded by any other week-to-week variation.")

    print("\n=== Building workspace-week panel ===")
    treated, donors, wave1_ws, wave2_ws = build_panel(df)
    print(f"Treated series shape: {treated.shape}")
    print(f"Donor matrix shape:   {donors.shape}")
    print(f"Wave 1 workspaces: {len(wave1_ws)} IDs in {wave1_ws[:3]}...")
    print(f"Wave 2 workspaces: {len(wave2_ws)} IDs in {wave2_ws[:3]}...")
    users_per_ws_week = len(df[df.signup_week < WINDOW]) / (50 * WINDOW)
    print(f"Users per workspace-week: ~{users_per_ws_week:.1f}")
    print(f"Pre-period treated mean  (weeks 0-19):  {treated[:PRE].mean():.4f}")
    print(f"Post-period treated mean (weeks 20-29): {treated[PRE:].mean():.4f}")

    print("\n=== Step 1: SLSQP donor-weight optimization ===")
    w_opt = fit_sc(treated, donors)
    pre_mse = float(np.mean((treated[:PRE] - donors[:PRE] @ w_opt) ** 2))
    pre_rmse = float(np.sqrt(pre_mse))
    nz = int((w_opt > 0.001).sum())
    print(f"Non-zero donor weights (|w| > 0.001): {nz}")
    print(f"Pre-period MSE:  {pre_mse:.6f}")
    print(f"Pre-period RMSE: {pre_rmse:.4f}  "
          f"({pre_rmse * 100:.2f} percentage points)")
    print("\nTop 5 donor weights:")
    nz_pairs = sorted(
        [(ws, w_opt[i]) for i, ws in enumerate(wave2_ws) if w_opt[i] > 0.001],
        key=lambda x: -x[1]
    )
    for ws_id, weight in nz_pairs[:5]:
        print(f"  workspace {ws_id}: w = {weight:.4f}")

    gap = post_gap(treated, donors, w_opt)
    print(f"\nObserved post-period gap: {gap:+.4f}  (ground truth = +0.05)")

    print("\n=== Step 2: Weekly gap decomposition ===")
    synth = donors @ w_opt
    for wk in range(PRE, WINDOW):
        print(f"  week {wk}: treated={treated[wk]:.4f}  "
              f"synth={synth[wk]:.4f}  gap={treated[wk] - synth[wk]:+.4f}")

    print("\n=== Step 3: Placebo permutation test ===")
    placebo_gaps = placebo_permutation(treated, donors)
    observed = gap
    print(f"Observed gap:         {observed:+.4f}")
    print(f"Placebo mean gap:     {placebo_gaps.mean():+.4f}")
    print(f"Placebo std gap:      {placebo_gaps.std():.4f}")
    print(f"Placebo gap range:    [{placebo_gaps.min():+.4f}, "
          f"{placebo_gaps.max():+.4f}]")
    rank = int((np.abs(placebo_gaps) >= abs(observed)).sum())
    pseudo_p = (rank + 1) / (len(placebo_gaps) + 1)
    print(f"|placebo gap| >= |observed|: {rank} of {len(placebo_gaps)}")
    print(f"Pseudo p-value: {pseudo_p:.4f}")

    print("\n=== Step 4: Leave-one-out (LOO) donor sensitivity ===")
    loo_tbl = loo_sensitivity(treated, donors, w_opt, wave2_ws)
    print(loo_tbl.round(4).to_string(index=False))
    print(f"\nLOO gap range:  [{loo_tbl.new_gap.min():+.4f}, "
          f"{loo_tbl.new_gap.max():+.4f}]")
    print(f"Original gap:   {gap:+.4f}")

    print("\n=== Step 5: Cluster bootstrap 95% CI (500 reps, seed=7) ===")
    print("Resamples users with replacement, rebuilds the panel,")
    print("re-fits donor weights, records the post-period gap.")
    lo, hi = cluster_bootstrap_ci(df, n_reps=500, seed=7)
    print(f"Post-period gap 95% CI: [{lo:+.4f}, {hi:+.4f}]")
    print(f"Ground truth +0.05 inside CI: "
          f"{'YES' if lo <= 0.05 <= hi else 'NO'}")
    print(f"Naive gap {naive_gap:+.4f} inside CI: "
          f"{'YES' if lo <= naive_gap <= hi else 'NO'}")


if __name__ == "__main__":
    main()
