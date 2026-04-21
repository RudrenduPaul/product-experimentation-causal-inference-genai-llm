"""
Synthetic LLM product dataset generator.

Simulates a realistic SaaS product that ships an AI assistant feature. Users have
heterogeneous engagement levels, the company rolls out a new prompt template
through a staged launch, and users can opt in to an "agent mode" that uses a
more expensive model. Outcomes include task completion, thumbs-up feedback,
session length, and 7-day retention.

The dataset is designed to demonstrate every causal inference technique in the
series:
  - Regression (clean confounding)
  - Counterfactual estimation (logged-only decisions)
  - Propensity score methods (opt-in selection bias)
  - Uplift modeling (heterogeneous treatment effects)
  - Synthetic control (full-population rollouts)
  - Instrumental variables (confounded routing)
  - Regression discontinuity (confidence-threshold routing)
  - Difference-in-differences (staged rollouts)
  - Doubly robust estimation (noisy experiments)

Usage:
    python generate_data.py --seed 42 --n-users 10000 --out synthetic_llm_logs.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(seed: int = 42, n_users: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Baseline user traits
    user_id = np.arange(n_users)
    engagement_tier = rng.choice(["light", "medium", "heavy"], size=n_users,
                                 p=[0.55, 0.30, 0.15])
    workspace_id = rng.integers(0, 50, size=n_users)
    signup_week = rng.integers(0, 52, size=n_users)

    # Confidence score for each user's typical query (used by threshold routing)
    query_confidence = np.clip(rng.beta(5, 2, size=n_users), 0, 1)

    # Staged rollout: workspace-level treatment, wave 1 at week 20, wave 2 at week 30
    wave = np.where(workspace_id < 25, 1, 2)
    treatment_week = np.where(wave == 1, 20, 30)
    treated_post = (signup_week >= treatment_week).astype(int)

    # Opt-in to agent mode (endogenous: heavy users more likely to opt in)
    opt_in_prob = np.where(engagement_tier == "heavy", 0.65,
                  np.where(engagement_tier == "medium", 0.35, 0.12))
    opt_in_agent_mode = rng.binomial(1, opt_in_prob)

    # Model routing: queries above 0.85 confidence go to the cheap model
    routed_to_premium = (query_confidence < 0.85).astype(int)

    # Prompt variant A/B (randomized on user_id hash)
    prompt_variant = (user_id % 2).astype(int)  # 0 = control, 1 = new prompt

    # Outcome model (with known ground-truth effects for teaching)
    base_completion = {
        "light": 0.35, "medium": 0.55, "heavy": 0.70
    }
    base = np.array([base_completion[t] for t in engagement_tier])

    # True causal effects baked in
    effect_new_prompt = 0.04         # +4pp from new prompt
    effect_agent_mode = 0.08         # +8pp from opting into agent mode
    effect_premium_model = 0.06      # +6pp from premium routing
    effect_staged_rollout = 0.05     # +5pp from the staged feature launch

    # Smooth AR(1) shared week signal: both waves experience the same
    # correlated market/seasonal fluctuations. AR(1) with rho=0.75 creates
    # smooth week-to-week movement (no abrupt spikes) so the weekly aggregates
    # look like gently rolling parallel lines rather than jagged noise.
    n_weeks_total = 52
    ar_signal = np.zeros(n_weeks_total)
    for t in range(1, n_weeks_total):
        ar_signal[t] = 0.75 * ar_signal[t - 1] + rng.normal(0, 0.010)
    ar_signal -= ar_signal.mean()   # center so it doesn't shift overall level
    week_signal_per_user = ar_signal[np.clip(signup_week, 0, n_weeks_total - 1)]

    # Small wave-1 level offset so wave 1 sits consistently above wave 2
    # in the pre-period (more realistic and easier to read on the chart).
    # This is absorbed by the `treated` dummy in regression — no bias.
    wave_offset = np.where(wave == 1, 0.03, 0.0)

    task_completion_prob = np.clip(
        base
        + wave_offset                    # wave-1 level difference (time-invariant)
        + effect_new_prompt * prompt_variant
        + effect_agent_mode * opt_in_agent_mode
        + effect_premium_model * routed_to_premium
        + effect_staged_rollout * treated_post
        + week_signal_per_user           # smooth shared AR(1) signal
        + rng.normal(0, 0.015, size=n_users),  # small individual noise
        0.01, 0.99
    )
    task_completed = rng.binomial(1, task_completion_prob)

    # Thumbs-up correlates with completion but not perfectly
    thumbs_up_prob = np.clip(0.2 + 0.6 * task_completion_prob +
                             rng.normal(0, 0.05, size=n_users), 0.01, 0.99)
    thumbs_up = rng.binomial(1, thumbs_up_prob)

    # Session length (minutes) and cost (USD)
    session_minutes = np.clip(
        5 + 3 * (engagement_tier == "heavy").astype(int)
        + 2 * opt_in_agent_mode
        + rng.normal(0, 1.5, size=n_users), 0.5, None
    )
    cost_usd = np.round(
        0.002 * session_minutes
        + 0.015 * opt_in_agent_mode
        + 0.008 * routed_to_premium, 4
    )

    # 7-day retention (lagged outcome)
    retention_prob = np.clip(
        0.4 + 0.35 * task_completion_prob + 0.1 * thumbs_up
        - 0.05 * (engagement_tier == "light").astype(int)
        + rng.normal(0, 0.05, size=n_users), 0.01, 0.99
    )
    retained_7d = rng.binomial(1, retention_prob)

    df = pd.DataFrame({
        "user_id": user_id,
        "workspace_id": workspace_id,
        "engagement_tier": engagement_tier,
        "signup_week": signup_week,
        "wave": wave,
        "treatment_week": treatment_week,
        "treated_post": treated_post,
        "prompt_variant": prompt_variant,
        "opt_in_agent_mode": opt_in_agent_mode,
        "query_confidence": np.round(query_confidence, 4),
        "routed_to_premium": routed_to_premium,
        "task_completed": task_completed,
        "thumbs_up": thumbs_up,
        "session_minutes": np.round(session_minutes, 2),
        "cost_usd": cost_usd,
        "retained_7d": retained_7d,
    })
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-users", type=int, default=10000)
    parser.add_argument("--out", type=str, default="synthetic_llm_logs.csv")
    args = parser.parse_args()

    df = generate(seed=args.seed, n_users=args.n_users)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
