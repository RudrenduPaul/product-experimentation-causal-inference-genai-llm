"""
Switchback experiment analysis for AI routing features in multi-user LLM products.

Scenario: A SaaS AI product tests an intelligent query-routing feature that decides
whether to send each user's queries to a standard model or a premium model. Because
all users share the same premium-model capacity pool, you cannot randomize at the
user level -- when half your users get AI routing, they claim the best capacity
windows first, which degrades response quality for everyone else. Instead, you
randomize time slots: the full platform runs with AI routing enabled for one
30-minute slot, then disabled for the next.

This script implements the full switchback pipeline:
  Step 1: Build the switchback time series from session logs
  Step 2: Naive estimate (ignoring carryover)
  Step 3: Carryover-adjusted OLS regression
  Step 4: HAC standard errors for time-series data
  Step 5: Bootstrap 95% confidence intervals

Run from repo root:
    python 06_switchback/switchback_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.stats.stattools import durbin_watson

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "synthetic_llm_logs.csv"

TRUE_EFFECT = 0.060   # AI routing raises task completion by 6 percentage points
CARRYOVER   = 0.030   # Residual routing effect persists into the following slot
N_SLOTS     = 48      # 48 synthetic 30-minute slots (8 full 3-on / 3-off cycles)


# ---------------------------------------------------------------------------
# Step 1: Build the switchback time series
# ---------------------------------------------------------------------------

def build_slots() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(df[["user_id", "task_completed", "cost_usd", "session_minutes"]].head(3).round(3))

    # Shuffle to eliminate row-ordering bias before slot assignment
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assign hour slots: 48 slots, each containing ~1,042 sessions
    df["hour_slot"] = df.index % N_SLOTS

    # Treatment schedule: 3-slot blocks (on, on, on, off, off, off, ...)
    # Using 3-slot blocks gives the platform time to settle into each state
    # and breaks the perfect collinearity between ai_on and its one-period lag.
    ai_on_schedule = np.tile([1, 1, 1, 0, 0, 0], N_SLOTS // 6)
    df["ai_on"] = ai_on_schedule[df["hour_slot"]]

    # Aggregate to slot level: mean outcome, mean cost, treatment indicator, session count
    slots = df.groupby("hour_slot").agg(
        mean_task_completed=("task_completed", "mean"),
        mean_cost=("cost_usd", "mean"),
        ai_on=("ai_on", "first"),
        n_obs=("user_id", "count"),
    ).reset_index()

    print(f"\nSlot-level data: {len(slots)} slots")
    print(slots[["hour_slot", "ai_on", "mean_task_completed", "mean_cost", "n_obs"]].head(8).round(4))
    print(f"\nAI-on slots: {slots['ai_on'].sum()},  AI-off slots: {(1 - slots['ai_on']).sum()}")

    return slots, df["task_completed"].mean()


def inject_ground_truth(slots: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    slots = slots.copy()

    # Replace slot means with synthetic balanced base rates.
    # The slot noise std matches the central-limit-theorem variance of aggregating
    # ~1,042 Bernoulli(base_rate) sessions, simulating realistic slot-to-slot
    # demand variation without systematic treatment-group imbalance.
    slot_noise_std = np.sqrt(base_rate * (1 - base_rate) / slots["n_obs"].iloc[0])
    rng = np.random.default_rng(42)
    slots["mean_task_completed"] = base_rate + rng.normal(0, slot_noise_std, size=N_SLOTS)

    # Lag the treatment indicator: did the previous slot have AI routing on?
    slots["ai_on_lag1"] = slots["ai_on"].shift(1).fillna(0).astype(int)

    # Observed outcome = base outcome + treatment effect + carryover from prior slot
    slots["mean_task_completed"] = (
        slots["mean_task_completed"]
        + TRUE_EFFECT * slots["ai_on"]
        + CARRYOVER   * slots["ai_on_lag1"]
    )

    print("Post-injection slot data:")
    print(slots[["hour_slot", "ai_on", "ai_on_lag1", "mean_task_completed"]].head(8).round(4))
    return slots


# ---------------------------------------------------------------------------
# Step 2: Naive estimate (ignoring time structure)
# ---------------------------------------------------------------------------

def naive_estimate(slots: pd.DataFrame) -> tuple:
    X_naive = sm.add_constant(slots["ai_on"])
    naive_model = sm.OLS(slots["mean_task_completed"], X_naive).fit()

    naive_ate = naive_model.params["ai_on"]
    naive_se  = naive_model.bse["ai_on"]

    print("=== Naive estimate (no carryover control) ===")
    print(f"  ATE estimate : {naive_ate:.4f}")
    print(f"  Std error    : {naive_se:.4f}")
    print(f"  95% CI       : [{naive_ate - 1.96*naive_se:.4f},  {naive_ate + 1.96*naive_se:.4f}]")
    print(f"\n  True effect  : {TRUE_EFFECT}")
    print(f"  Bias         : {naive_ate - TRUE_EFFECT:+.4f}")

    return naive_ate, naive_se, naive_model


# ---------------------------------------------------------------------------
# Step 3: Carryover-adjusted OLS regression
# ---------------------------------------------------------------------------

def carryover_adjusted_estimate(slots: pd.DataFrame, naive_ate: float) -> tuple:
    X_adj = sm.add_constant(slots[["ai_on", "ai_on_lag1"]])
    adj_model = sm.OLS(slots["mean_task_completed"], X_adj).fit()

    adj_ate       = adj_model.params["ai_on"]
    adj_carryover = adj_model.params["ai_on_lag1"]
    adj_se        = adj_model.bse["ai_on"]

    print("=== Carryover-adjusted estimate ===")
    print(adj_model.summary().tables[1])

    print(f"\n  Direct ATE estimate  : {adj_ate:.4f}  (true: {TRUE_EFFECT})")
    print(f"  Carryover estimate   : {adj_carryover:.4f}  (true: {CARRYOVER})")
    print(f"  Residual bias        : {adj_ate - TRUE_EFFECT:+.4f}")
    print(f"\n  Bias removed vs naive: {naive_ate - adj_ate:+.4f}")

    return adj_ate, adj_carryover, adj_se, adj_model


# ---------------------------------------------------------------------------
# Step 4: HAC standard errors for time-series data
# ---------------------------------------------------------------------------

def hac_standard_errors(adj_model, adj_ate: float) -> tuple:
    dw_stat = durbin_watson(adj_model.resid)
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    print("  DW near 2.0 = little autocorrelation in residuals.")
    print("  DW < 1.5 = positive serial correlation.")
    print("  DW > 2.5 = negative serial correlation.")
    print("  Apply HAC standard errors regardless -- DW only tests AR(1) structure.")

    hac_cov = cov_hac(adj_model, nlags=3)
    hac_se  = np.sqrt(np.diag(hac_cov))

    print("\n=== Standard error comparison ===")
    print(f"  OLS SE on ai_on  : {adj_model.bse['ai_on']:.4f}")
    print(f"  HAC SE on ai_on  : {hac_se[1]:.4f}")
    print(f"  OLS t-stat       : {adj_model.tvalues['ai_on']:.2f}")
    print(f"  HAC t-stat       : {adj_ate / hac_se[1]:.2f}")

    hac_ci_lower = adj_ate - 1.96 * hac_se[1]
    hac_ci_upper = adj_ate + 1.96 * hac_se[1]
    print(f"\n  HAC 95% CI: [{hac_ci_lower:.4f},  {hac_ci_upper:.4f}]")
    print(f"  True effect {TRUE_EFFECT} inside CI: {hac_ci_lower < TRUE_EFFECT < hac_ci_upper}")

    return hac_se, hac_ci_lower, hac_ci_upper, dw_stat


# ---------------------------------------------------------------------------
# Step 5: Bootstrap 95% confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(slots: pd.DataFrame, B: int = 500, seed: int = 7) -> dict:
    """Bootstrap CIs treating each slot as an independent observation.

    Each slot's ai_on_lag1 value is fixed from the original schedule
    (it is a known property of the slot, not recomputed from the resampled series).
    This is the correct approach for switchback data: the lag covariate reflects
    the original treatment schedule, not the resampled ordering.
    """
    rng = np.random.default_rng(seed)
    n = len(slots)

    naive_ates    = []
    adj_ates      = []
    carryover_ests = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        s   = slots.iloc[idx]  # ai_on_lag1 stays as the original slot's value

        X_n = sm.add_constant(s["ai_on"])
        naive_ates.append(sm.OLS(s["mean_task_completed"], X_n).fit().params["ai_on"])

        X_a = sm.add_constant(s[["ai_on", "ai_on_lag1"]])
        m   = sm.OLS(s["mean_task_completed"], X_a).fit()
        adj_ates.append(m.params["ai_on"])
        carryover_ests.append(m.params["ai_on_lag1"])

    naive_ci     = np.percentile(naive_ates,     [2.5, 97.5])
    adj_ci       = np.percentile(adj_ates,       [2.5, 97.5])
    carryover_ci = np.percentile(carryover_ests, [2.5, 97.5])

    print(f"\n=== Bootstrap 95% confidence intervals (B={B}, seed={seed}) ===")
    print(f"  Naive ATE        : [{naive_ci[0]:.4f},  {naive_ci[1]:.4f}]  "
          f"(covers {TRUE_EFFECT}: {naive_ci[0] < TRUE_EFFECT < naive_ci[1]})")
    print(f"  Adjusted ATE     : [{adj_ci[0]:.4f},  {adj_ci[1]:.4f}]  "
          f"(covers {TRUE_EFFECT}: {adj_ci[0] < TRUE_EFFECT < adj_ci[1]})")
    print(f"  Carryover effect : [{carryover_ci[0]:.4f},  {carryover_ci[1]:.4f}]  "
          f"(covers {CARRYOVER}: {carryover_ci[0] < CARRYOVER < carryover_ci[1]})")

    return {"naive_ci": naive_ci, "adj_ci": adj_ci, "carryover_ci": carryover_ci}


# ---------------------------------------------------------------------------
# Validation table
# ---------------------------------------------------------------------------

def validation_table(naive_ate: float, adj_ate: float, adj_carryover: float) -> None:
    print("=" * 52)
    print(f"{'Estimator':<30} {'Estimate':>8}  {'True':>6}  {'Bias':>7}")
    print("-" * 52)
    print(f"{'Naive OLS (no lag)':<30} {naive_ate:>8.4f}  {TRUE_EFFECT:>6.4f}  {naive_ate - TRUE_EFFECT:>+7.4f}")
    print(f"{'Carryover-adjusted OLS':<30} {adj_ate:>8.4f}  {TRUE_EFFECT:>6.4f}  {adj_ate - TRUE_EFFECT:>+7.4f}")
    print(f"{'Carryover coefficient':<30} {adj_carryover:>8.4f}  {CARRYOVER:>6.4f}  {adj_carryover - CARRYOVER:>+7.4f}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    slots, base_rate = build_slots()
    slots            = inject_ground_truth(slots, base_rate)
    naive_ate, naive_se, naive_model = naive_estimate(slots)
    adj_ate, adj_carryover, adj_se, adj_model = carryover_adjusted_estimate(slots, naive_ate)
    hac_se, hac_ci_lower, hac_ci_upper, dw_stat = hac_standard_errors(adj_model, adj_ate)
    bootstrap_results = bootstrap_ci(slots)
    validation_table(naive_ate, adj_ate, adj_carryover)
