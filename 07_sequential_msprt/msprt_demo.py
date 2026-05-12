"""
mSPRT and sequential testing for AI product experiments.

Companion code for FCC Article 7:
"Product Experimentation: stop early without p-hacking using mSPRT
and sequential testing for AI product experiments in Python"

Implements the mixture Sequential Probability Ratio Test (mSPRT) using
always-valid e-values for Bernoulli outcomes. Runs against the shared
synthetic LLM product dataset.

Steps:
  1  Simulate the peeking problem — show inflated false positive rate
  2  Implement the mSPRT e-value via log beta function
  3  Apply mSPRT to the real dataset, find earliest valid stopping day
  4  Compare power against a fixed-sample t-test
     Validate against ground truth
  5  Bootstrap 95% confidence intervals (500 replicates, seed=7)

Usage (from repo root):
    python data/generate_data.py --seed 42 --n-users 50000 \\
        --out data/synthetic_llm_logs.csv
    python 07_sequential_msprt/msprt_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln
from scipy.stats import ttest_ind

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "synthetic_llm_logs.csv"

ALPHA = 0.05
THRESHOLD = 1 / ALPHA          # = 20
ALPHA_PRIOR = 1.0
BETA_PRIOR = 1.0
USERS_PER_ARM_PER_DAY = 60
N_DAYS_RUN = 30
N_PER_ARM = USERS_PER_ARM_PER_DAY * N_DAYS_RUN   # 1 800


# ---------------------------------------------------------------------------
# Core e-value computation
# ---------------------------------------------------------------------------

def compute_evalue_running(outcomes_treated, outcomes_control,
                           alpha_prior=1.0, beta_prior=1.0):
    """Return running mSPRT e-value array for two Bernoulli arms.

    At each step t, computes the Bayes factor comparing:
      H1: each arm has its own Beta(alpha_prior, beta_prior) completion rate
      H0: both arms share a single Beta(alpha_prior, beta_prior) rate

    The resulting process is a valid e-value (E[e_t] <= 1 under H0 for each t
    and the process satisfies the nonneg supermartingale property under H0,
    enabling the Ville time-uniform bound).

    Parameters
    ----------
    outcomes_treated, outcomes_control : array-like of 0/1
    alpha_prior, beta_prior : Beta prior hyperparameters (default: uniform)

    Returns
    -------
    e_values : np.ndarray, shape (n,)
    """
    ot = np.asarray(outcomes_treated, dtype=float)
    oc = np.asarray(outcomes_control, dtype=float)
    n = min(len(ot), len(oc))

    cum_t = np.cumsum(ot[:n])
    cum_c = np.cumsum(oc[:n])
    t_arr = np.arange(1, n + 1, dtype=float)

    log_ml_t = (betaln(alpha_prior + cum_t,
                       beta_prior + t_arr - cum_t)
                - betaln(alpha_prior, beta_prior))

    log_ml_c = (betaln(alpha_prior + cum_c,
                       beta_prior + t_arr - cum_c)
                - betaln(alpha_prior, beta_prior))

    pooled_successes = cum_t + cum_c
    pooled_n = 2 * t_arr
    log_ml_h0 = (betaln(alpha_prior + pooled_successes,
                        beta_prior + pooled_n - pooled_successes)
                 - betaln(alpha_prior, beta_prior))

    return np.exp(log_ml_t + log_ml_c - log_ml_h0)


def first_crossing(e_values, threshold):
    """Return index of first e_value >= threshold, or None."""
    idx = np.where(e_values >= threshold)[0]
    return int(idx[0]) if len(idx) > 0 else None


# ---------------------------------------------------------------------------
# Step 1: Simulate the peeking problem
# ---------------------------------------------------------------------------

def step1_peeking_simulation():
    print("\n=== Step 1: Peeking simulation ===")
    np.random.seed(42)
    N_SIMS = 1_000
    NULL_RATE = 0.60

    false_pos_peeking = 0
    false_pos_single = 0

    for _ in range(N_SIMS):
        control_obs, treated_obs = [], []
        stopped = False
        for _ in range(N_DAYS_RUN):
            control_obs.extend(np.random.binomial(
                1, NULL_RATE, USERS_PER_ARM_PER_DAY))
            treated_obs.extend(np.random.binomial(
                1, NULL_RATE, USERS_PER_ARM_PER_DAY))
            if len(control_obs) >= 10:
                _, p = stats.ttest_ind(treated_obs, control_obs)
                if p < ALPHA and not stopped:
                    false_pos_peeking += 1
                    stopped = True
        _, p_final = stats.ttest_ind(treated_obs, control_obs)
        if p_final < ALPHA:
            false_pos_single += 1

    fpr_peeking = false_pos_peeking / N_SIMS
    fpr_single = false_pos_single / N_SIMS
    print(f"False positive rate (peeking daily):  {fpr_peeking:.1%}")
    print(f"False positive rate (single look):    {fpr_single:.1%}")
    return fpr_peeking, fpr_single


# ---------------------------------------------------------------------------
# Step 2: Sanity check on null data
# ---------------------------------------------------------------------------

def step2_null_sanity_check():
    print("\n=== Step 2: Null sanity check ===")
    np.random.seed(0)
    null_t = np.random.binomial(1, 0.60, 500)
    null_c = np.random.binomial(1, 0.60, 500)
    ev_null = compute_evalue_running(null_t, null_c)
    print(f"E-value at end under null (should be near 1): {ev_null[-1]:.3f}")
    print(f"Max e-value under null: {ev_null.max():.3f}")
    return ev_null


# ---------------------------------------------------------------------------
# Step 3: Apply mSPRT to real dataset
# ---------------------------------------------------------------------------

def step3_apply_to_dataset(treated, control):
    print("\n=== Step 3: Apply mSPRT to real dataset ===")
    np.random.seed(42)
    treated_sh = treated.copy()
    control_sh = control.copy()
    np.random.shuffle(treated_sh)
    np.random.shuffle(control_sh)

    treated_seq = treated_sh[:N_PER_ARM]
    control_seq = control_sh[:N_PER_ARM]

    e_values = compute_evalue_running(treated_seq, control_seq)
    days = np.arange(1, len(e_values) + 1) / USERS_PER_ARM_PER_DAY

    cross_idx = first_crossing(e_values, THRESHOLD)
    if cross_idx is not None:
        stopping_day = days[cross_idx]
        print(f"mSPRT stopping day: {stopping_day:.1f}")
        print(f"E-value at stopping: {e_values[cross_idx]:.1f}")
    else:
        stopping_day = None
        print("mSPRT did not cross threshold in 30-day window")

    print(f"Final e-value on day {N_DAYS_RUN}: {e_values[-1]:.2f}")
    return e_values, days, stopping_day


# ---------------------------------------------------------------------------
# Step 4: Power comparison
# ---------------------------------------------------------------------------

def step4_power_comparison():
    print("\n=== Step 4: Power comparison ===")
    np.random.seed(42)
    N_SIMS = 1_000
    TRUE_EFFECT = 0.05
    BASE_RATE = 0.60

    msprt_stopping_days = []
    msprt_detected = 0
    ttest_detected = 0

    for _ in range(N_SIMS):
        t_obs = np.random.binomial(1, BASE_RATE + TRUE_EFFECT, N_PER_ARM)
        c_obs = np.random.binomial(1, BASE_RATE, N_PER_ARM)

        e_vals = compute_evalue_running(t_obs, c_obs)
        days = np.arange(1, N_PER_ARM + 1) / USERS_PER_ARM_PER_DAY
        cross = first_crossing(e_vals, THRESHOLD)
        if cross is not None:
            msprt_detected += 1
            msprt_stopping_days.append(days[cross])
        else:
            msprt_stopping_days.append(float(N_DAYS_RUN))

        _, p = ttest_ind(t_obs, c_obs)
        if p < ALPHA:
            ttest_detected += 1

    msprt_power = msprt_detected / N_SIMS
    ttest_power = ttest_detected / N_SIMS
    msprt_stop_arr = np.array(msprt_stopping_days)
    median_stop = float(np.median(msprt_stop_arr))
    pct_early = float(np.mean(msprt_stop_arr < N_DAYS_RUN))

    print(f"mSPRT power:               {msprt_power:.1%}")
    print(f"Fixed-sample t-test power: {ttest_power:.1%}")
    print(f"Median mSPRT stop day:     {median_stop:.1f} / {N_DAYS_RUN}")
    print(f"Fraction stopping early:   {pct_early:.1%}")
    return msprt_power, ttest_power, median_stop, pct_early, msprt_stop_arr


# ---------------------------------------------------------------------------
# Validate against ground truth
# ---------------------------------------------------------------------------

def validate_ground_truth(treated, control):
    print("\n=== Validate against ground truth ===")
    np.random.seed(0)
    treated_sh = treated.copy()
    control_sh = control.copy()
    np.random.shuffle(treated_sh)
    np.random.shuffle(control_sh)
    n = min(len(treated_sh), len(control_sh))

    e_full = compute_evalue_running(treated_sh[:n], control_sh[:n])
    days_full = np.arange(1, n + 1) / USERS_PER_ARM_PER_DAY

    cross_idx = first_crossing(e_full, THRESHOLD)
    true_effect = treated.mean() - control.mean()
    print(f"True effect in data: {true_effect:.4f} ({true_effect*100:.2f}pp)")
    if cross_idx is not None:
        print(f"mSPRT correctly detected the effect.")
        print(f"Could have stopped on day {days_full[cross_idx]:.1f}")
        print(f"E-value at stopping point: {e_full[cross_idx]:.1f}")
    else:
        print("mSPRT did not cross threshold on full data slice.")
    return e_full, days_full, cross_idx, true_effect


# ---------------------------------------------------------------------------
# Step 5: Bootstrap 95% confidence intervals
# ---------------------------------------------------------------------------

def step5_bootstrap_ci(treated, control, n_boot=500, seed=7):
    """Bootstrap 95% CI for the treatment effect on task completion.

    Resamples treated and control arrays independently with replacement.
    Returns (point_estimate, lower_ci, upper_ci).
    """
    print("\n=== Step 5: Bootstrap 95% CI ===")
    rng = np.random.default_rng(seed)
    point_est = treated.mean() - control.mean()
    print(f"Point estimate (treated - control): {point_est:.4f} "
          f"({point_est*100:.2f}pp)")

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        t_boot = rng.choice(treated, size=len(treated), replace=True)
        c_boot = rng.choice(control, size=len(control), replace=True)
        boot_diffs[i] = t_boot.mean() - c_boot.mean()

    lower = float(np.percentile(boot_diffs, 2.5))
    upper = float(np.percentile(boot_diffs, 97.5))
    print(f"95% bootstrap CI: [{lower:.4f}, {upper:.4f}]  "
          f"([{lower*100:.2f}pp, {upper*100:.2f}pp])")
    print(f"Ground-truth effect (0.05 / 5pp) is "
          f"{'inside' if lower <= 0.05 <= upper else 'OUTSIDE'} the CI.")
    print(f"Naive (unweighted) effect is "
          f"{'outside' if not (lower <= point_est <= upper) else 'inside'} "
          f"range context: point est = {point_est*100:.2f}pp")
    return point_est, lower, upper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    treated = df[df["wave"] == 1]["task_completed"].values
    control = df[df["wave"] == 2]["task_completed"].values

    print(f"Treated: n={len(treated):,}, mean={treated.mean():.4f}")
    print(f"Control: n={len(control):,}, mean={control.mean():.4f}")
    print(f"Observed lift: {treated.mean() - control.mean():.4f}")

    step1_peeking_simulation()
    step2_null_sanity_check()
    e_values, days, stopping_day = step3_apply_to_dataset(treated, control)
    step4_power_comparison()
    validate_ground_truth(treated, control)
    step5_bootstrap_ci(treated, control, n_boot=500, seed=7)

    print("\n=== All steps complete ===")


if __name__ == "__main__":
    main()
