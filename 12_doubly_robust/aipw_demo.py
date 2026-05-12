"""
Doubly robust estimation (AIPW) for noisy LLM product experiments.

Companion code for FCC Article 12:
"Product Experimentation with Doubly Robust Estimation: When Both Your
Models Are Wrong in LLM Applications"

Runs every code block from the article against the shared synthetic dataset.
Produces: naive ATE, regression adjustment ATE, AIPW ATE, bootstrap 95% CI,
and deliberate misspecification scenarios proving the double-robust property.

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \
        --out data/synthetic_llm_logs.csv
    python 12_doubly_robust/aipw_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "synthetic_llm_logs.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.get_dummies(df[["engagement_tier", "query_confidence"]], drop_first=True)
        .astype(float)
        .values
    )


# ---------------------------------------------------------------------------
# Step 1: Propensity model
# ---------------------------------------------------------------------------

def fit_propensity(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, T)
    e_hat = model.predict_proba(X)[:, 1]
    return np.clip(e_hat, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Step 2: Outcome models
# ---------------------------------------------------------------------------

def fit_outcome_models(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    m1_model = LinearRegression().fit(X[T == 1], Y[T == 1])
    m0_model = LinearRegression().fit(X[T == 0], Y[T == 0])
    m1_hat = m1_model.predict(X)
    m0_hat = m0_model.predict(X)
    return m1_hat, m0_hat


# ---------------------------------------------------------------------------
# Step 3: AIPW estimator
# ---------------------------------------------------------------------------

def aipw_ate(
    Y: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
    m1_hat: np.ndarray,
    m0_hat: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Augmented Inverse-Probability Weighting (AIPW) estimator.

    Returns (ate_estimate, per_observation_influence_values).
    """
    ipw_treated = T * (Y - m1_hat) / e_hat
    ipw_control = (1 - T) * (Y - m0_hat) / (1 - e_hat)
    phi = m1_hat - m0_hat + ipw_treated - ipw_control
    return phi.mean(), phi


# ---------------------------------------------------------------------------
# Step 4: Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def aipw_bootstrap(
    df: pd.DataFrame,
    x_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    n_bootstrap: int = 500,
    seed: int = 7,
) -> tuple[np.ndarray, float, float]:
    """Bootstrap AIPW ATE with 95% percentile confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(df)
    X_all = (
        pd.get_dummies(df[x_cols], drop_first=True).astype(float).values
    )
    T_all = df[treatment_col].values
    Y_all = df[outcome_col].values

    boot_estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b, T_b, Y_b = X_all[idx], T_all[idx], Y_all[idx]

        ps = LogisticRegression(max_iter=1000, C=1.0)
        ps.fit(X_b, T_b)
        e_b = np.clip(ps.predict_proba(X_b)[:, 1], 0.01, 0.99)

        m1 = LinearRegression().fit(X_b[T_b == 1], Y_b[T_b == 1])
        m0 = LinearRegression().fit(X_b[T_b == 0], Y_b[T_b == 0])
        m1_b = m1.predict(X_b)
        m0_b = m0.predict(X_b)

        ate_b, _ = aipw_ate(Y_b, T_b, e_b, m1_b, m0_b)
        boot_estimates.append(ate_b)

    boot_arr = np.array(boot_estimates)
    ci_low = np.percentile(boot_arr, 2.5)
    ci_high = np.percentile(boot_arr, 97.5)
    return boot_arr, ci_low, ci_high


# ---------------------------------------------------------------------------
# Step 5: Double-robust property — deliberate misspecification
# ---------------------------------------------------------------------------

def misspecification_scenario_1(
    Y: np.ndarray,
    T: np.ndarray,
    m1_hat: np.ndarray,
    m0_hat: np.ndarray,
) -> tuple[float, float]:
    """Wrong propensity (constant e=0.3), correct outcome models."""
    e_wrong = np.full(len(Y), 0.3)
    t_mask = T == 1
    c_mask = T == 0

    ate_ipw_wrong = (
        (Y[t_mask] / e_wrong[t_mask]).sum() / (1 / e_wrong[t_mask]).sum()
        - (Y[c_mask] / (1 - e_wrong[c_mask])).sum()
        / (1 / (1 - e_wrong[c_mask])).sum()
    )
    ate_aipw_wrong_ps, _ = aipw_ate(Y, T, e_wrong, m1_hat, m0_hat)
    return ate_ipw_wrong, ate_aipw_wrong_ps


def misspecification_scenario_2(
    Y: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
) -> tuple[float, float]:
    """Correct propensity, wrong outcome models (constant 0.5)."""
    m1_wrong = np.full(len(Y), 0.5)
    m0_wrong = np.full(len(Y), 0.5)
    t_mask = T == 1
    c_mask = T == 0

    ate_regression_wrong = (m1_wrong - m0_wrong).mean()
    ate_ipw_correct = (
        (Y[t_mask] / e_hat[t_mask]).sum() / (1 / e_hat[t_mask]).sum()
        - (Y[c_mask] / (1 - e_hat[c_mask])).sum()
        / (1 / (1 - e_hat[c_mask])).sum()
    )
    ate_aipw_wrong_out, _ = aipw_ate(Y, T, e_hat, m1_wrong, m0_wrong)
    return ate_regression_wrong, ate_ipw_correct, ate_aipw_wrong_out


# ---------------------------------------------------------------------------
# Main — prints every number the article quotes
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_data()
    T = df["opt_in_agent_mode"].values
    Y = df["task_completed"].values
    X = build_feature_matrix(df)

    print("=" * 60)
    print("Article 12: Doubly Robust Estimation (AIPW)")
    print(f"Dataset: {len(df):,} users")
    print("=" * 60)

    # --- Naive ATE ---
    naive_ate = Y[T == 1].mean() - Y[T == 0].mean()
    print(f"\n[Setup] Naive ATE (unadjusted): {naive_ate:+.4f}")
    print(f"[Setup] N treated: {T.sum()}, N control: {(1-T).sum()}")

    # --- Step 1: Propensity model ---
    e_hat = fit_propensity(X, T)
    print(f"\n[Step 1] Propensity range: {e_hat.min():.3f} to {e_hat.max():.3f}")
    print(f"[Step 1] Mean propensity (treated): {e_hat[T == 1].mean():.3f}")
    print(f"[Step 1] Mean propensity (control): {e_hat[T == 0].mean():.3f}")

    # --- Step 2: Outcome models ---
    m1_hat, m0_hat = fit_outcome_models(X, T, Y)
    ate_regression = (m1_hat - m0_hat).mean()
    print(f"\n[Step 2] Regression adjustment ATE: {ate_regression:+.4f}")

    # --- Step 3: AIPW ---
    ate_aipw, phi_obs = aipw_ate(Y, T, e_hat, m1_hat, m0_hat)
    print(f"\n[Step 3] AIPW ATE:            {ate_aipw:+.4f}")
    print(f"[Step 3] Naive ATE:           {naive_ate:+.4f}")
    print(f"[Step 3] Regression-only ATE: {ate_regression:+.4f}")
    print(f"[Step 3] Ground truth:        +0.0800")

    # --- Step 4: Bootstrap CI ---
    print("\n[Step 4] Running bootstrap (n=500, seed=7) ...")
    boot_dist, ci_lo, ci_hi = aipw_bootstrap(
        df,
        x_cols=["engagement_tier", "query_confidence"],
        treatment_col="opt_in_agent_mode",
        outcome_col="task_completed",
        n_bootstrap=500,
        seed=7,
    )
    print(f"[Step 4] AIPW ATE:          {ate_aipw:+.4f}")
    print(f"[Step 4] 95% Bootstrap CI:  [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"[Step 4] Bootstrap std dev: {boot_dist.std():.4f}")

    # --- Step 5: Misspecification scenarios ---
    ate_ipw_wrong, ate_aipw_wrong_ps = misspecification_scenario_1(
        Y, T, m1_hat, m0_hat
    )
    print("\n[Step 5 — Scenario 1: constant propensity e=0.3]")
    print(f"  IPW with wrong propensity:         {ate_ipw_wrong:+.4f}  (should be wrong)")
    print(f"  Regression adjustment (unchanged): {ate_regression:+.4f}  (should be ~{ate_regression:.3f})")
    print(f"  AIPW with wrong propensity:        {ate_aipw_wrong_ps:+.4f}  (should stay ~{ate_regression:.3f})")
    print(f"  Ground truth:                      +0.0800")

    ate_reg_wrong, ate_ipw_correct, ate_aipw_wrong_out = misspecification_scenario_2(
        Y, T, e_hat
    )
    print("\n[Step 5 — Scenario 2: constant outcome models m1=m0=0.5]")
    print(f"  Regression with wrong outcome models: {ate_reg_wrong:+.4f}  (should be 0.0)")
    print(f"  IPW with correct propensity:          {ate_ipw_correct:+.4f}  (should be ~{ate_aipw:.3f})")
    print(f"  AIPW with wrong outcome models:       {ate_aipw_wrong_out:+.4f}  (should stay ~{ate_aipw:.3f})")
    print(f"  Ground truth:                         +0.0800")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"  Naive ATE:                {naive_ate:+.4f}")
    print(f"  Regression adjustment:    {ate_regression:+.4f}")
    print(f"  AIPW ATE:                 {ate_aipw:+.4f}")
    print(f"  95% Bootstrap CI:         [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Bootstrap std dev:        {boot_dist.std():.4f}")
    print(f"  Ground truth:             +0.0800")


if __name__ == "__main__":
    main()
