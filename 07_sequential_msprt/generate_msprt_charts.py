"""Generate the two article figures for the mSPRT sequential testing tutorial.

Figure 1 (conceptual): schematic e-value trajectories under H1 (real effect,
rising to threshold) vs H0 (null, bouncing near 1) with the stopping boundary.

Figure 2 (data-driven): the actual mSPRT e-value trajectory on the real
synthetic dataset, showing the stopping day on the log-scaled e-value axis.

Run from repo root:
    python 07_sequential_msprt/generate_msprt_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betaln

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "07_sequential_msprt"
IMAGES_DIR = REPO_ROOT / "images" / "article-7"

THRESHOLD = 20.0
USERS_PER_ARM_PER_DAY = 60
N_DAYS_RUN = 30
N_PER_ARM = USERS_PER_ARM_PER_DAY * N_DAYS_RUN


def save_figure(fig, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    for dest in (IMAGES_DIR / name, METHOD_DIR / name):
        fig.savefig(dest, dpi=150, bbox_inches="tight")
    print(f"  Saved: {name}")


def compute_evalue_running(outcomes_treated, outcomes_control,
                           alpha_prior=1.0, beta_prior=1.0):
    ot = np.asarray(outcomes_treated, dtype=float)
    oc = np.asarray(outcomes_control, dtype=float)
    n = min(len(ot), len(oc))
    cum_t = np.cumsum(ot[:n])
    cum_c = np.cumsum(oc[:n])
    t_arr = np.arange(1, n + 1, dtype=float)
    log_ml_t = (betaln(alpha_prior + cum_t, beta_prior + t_arr - cum_t)
                - betaln(alpha_prior, beta_prior))
    log_ml_c = (betaln(alpha_prior + cum_c, beta_prior + t_arr - cum_c)
                - betaln(alpha_prior, beta_prior))
    pooled_s = cum_t + cum_c
    log_ml_h0 = (betaln(alpha_prior + pooled_s,
                        beta_prior + 2 * t_arr - pooled_s)
                 - betaln(alpha_prior, beta_prior))
    return np.exp(log_ml_t + log_ml_c - log_ml_h0)


def make_figure_1_conceptual() -> None:
    """Schematic: three example e-value paths — H1 crossing threshold,
    H1 near-miss, and H0 meandering near 1.

    Top panel: log-scale e-value trajectories.
    Bottom panel: experiment day strip with 'stop zone' marker.
    """
    rng = np.random.default_rng(7)
    days = np.arange(1, N_DAYS_RUN + 1)

    # Simulate conceptual paths with fixed seeds for reproducibility
    # H1 path that crosses — strong effect
    np.random.seed(7)
    t1 = np.random.binomial(1, 0.65, N_PER_ARM)
    c1 = np.random.binomial(1, 0.60, N_PER_ARM)
    ev_h1_cross = compute_evalue_running(t1, c1)
    ev_h1_cross_daily = ev_h1_cross[USERS_PER_ARM_PER_DAY - 1::
                                    USERS_PER_ARM_PER_DAY]

    # H1 path that grows but doesn't quite cross in 30 days — moderate effect
    np.random.seed(13)
    t2 = np.random.binomial(1, 0.64, N_PER_ARM)
    c2 = np.random.binomial(1, 0.60, N_PER_ARM)
    ev_h1_slow = compute_evalue_running(t2, c2)
    ev_h1_slow_daily = ev_h1_slow[USERS_PER_ARM_PER_DAY - 1::
                                  USERS_PER_ARM_PER_DAY]

    # H0 path — null, bouncing near 1
    np.random.seed(99)
    t0 = np.random.binomial(1, 0.60, N_PER_ARM)
    c0 = np.random.binomial(1, 0.60, N_PER_ARM)
    ev_h0 = compute_evalue_running(t0, c0)
    ev_h0_daily = ev_h0[USERS_PER_ARM_PER_DAY - 1::USERS_PER_ARM_PER_DAY]

    color_h1_cross = "#2563eb"    # blue — crosses
    color_h1_slow = "#7c3aed"     # purple — growing
    color_h0 = "#6b7280"          # grey — null
    color_threshold = "#dc2626"   # red dashed

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 5.6),
        gridspec_kw={"height_ratios": [5.0, 1.0], "hspace": 0.0},
        sharex=True,
    )

    # Top — log-scale trajectories
    ax_top.semilogy(days, ev_h1_cross_daily, color=color_h1_cross,
                    linewidth=2.2, label="H₁: real effect (crosses)")
    ax_top.semilogy(days, ev_h1_slow_daily, color=color_h1_slow,
                    linewidth=1.8, linestyle="--",
                    label="H₁: weaker effect (growing)")
    ax_top.semilogy(days, ev_h0_daily, color=color_h0,
                    linewidth=1.5, linestyle=":", label="H₀: no effect")
    ax_top.axhline(THRESHOLD, color=color_threshold, linestyle="--",
                   linewidth=1.8, label=f"Stopping boundary (1/α = {THRESHOLD:.0f})")

    # Find crossing day for the blue path
    cross_idx = np.where(ev_h1_cross_daily >= THRESHOLD)[0]
    if len(cross_idx) > 0:
        cd = days[cross_idx[0]]
        ax_top.axvline(cd, color=color_h1_cross, linestyle=":",
                       linewidth=1.2, alpha=0.7)

    ax_top.set_ylabel("E-value (log scale)", fontsize=11)
    ax_top.set_title("mSPRT e-value trajectories: when to stop and when not to",
                     fontsize=12, pad=10)
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.set_ylim(bottom=0.05)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines["bottom"].set_visible(False)

    # Bottom — day strip with region labels
    ax_bot.set_xlim(days[0], days[-1])
    ax_bot.set_ylim(0, 1)
    ax_bot.set_yticks([])
    ax_bot.spines[["left", "right", "top"]].set_visible(False)

    # Shaded zone where the blue path stays above threshold
    if len(cross_idx) > 0:
        ax_bot.axvspan(days[cross_idx[0]], days[-1],
                       alpha=0.15, color=color_threshold,
                       label="Stop zone (e > 20)")
        ax_bot.text(days[cross_idx[0]] + 0.3, 0.5, "stop zone",
                    va="center", ha="left", fontsize=8,
                    color=color_threshold)

    ax_bot.text(days[0] + 0.3, 0.5, "accumulating evidence",
                va="center", ha="left", fontsize=8, color=color_h0)

    ax_bot.set_xlabel("Experiment day", fontsize=11)

    plt.tight_layout()
    save_figure(fig, "msprt_evalue_schematic.png")
    plt.close(fig)


def make_figure_2_real_data() -> None:
    """Data-driven: actual e-value trajectory on the real synthetic dataset.

    Top panel: e-value on log scale with stopping threshold and stopping day.
    Bottom panel: cumulative completion rates per arm.
    """
    data_path = REPO_ROOT / "data" / "synthetic_llm_logs.csv"
    df = pd.read_csv(data_path)
    treated = df[df["wave"] == 1]["task_completed"].values
    control = df[df["wave"] == 2]["task_completed"].values

    np.random.seed(42)
    treated_sh = treated.copy()
    control_sh = control.copy()
    np.random.shuffle(treated_sh)
    np.random.shuffle(control_sh)

    treated_seq = treated_sh[:N_PER_ARM]
    control_seq = control_sh[:N_PER_ARM]

    e_values = compute_evalue_running(treated_seq, control_seq)

    # Downsample to daily resolution
    ev_daily = e_values[USERS_PER_ARM_PER_DAY - 1::USERS_PER_ARM_PER_DAY]
    days = np.arange(1, len(ev_daily) + 1)

    # Cumulative completion rates
    cum_t = np.cumsum(treated_seq) / np.arange(1, len(treated_seq) + 1)
    cum_c = np.cumsum(control_seq) / np.arange(1, len(control_seq) + 1)
    cum_t_daily = cum_t[USERS_PER_ARM_PER_DAY - 1::USERS_PER_ARM_PER_DAY]
    cum_c_daily = cum_c[USERS_PER_ARM_PER_DAY - 1::USERS_PER_ARM_PER_DAY]

    # Find stopping day
    cross_indices = np.where(ev_daily >= THRESHOLD)[0]
    stopping_day = int(days[cross_indices[0]]) if len(cross_indices) > 0 else None

    color_ev = "#2563eb"
    color_t = "#16a34a"
    color_c = "#6b7280"
    color_threshold = "#dc2626"

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 6.4),
        gridspec_kw={"height_ratios": [4.0, 2.0], "hspace": 0.0},
        sharex=True,
    )

    # Top: e-value trajectory
    ax_top.semilogy(days, ev_daily, color=color_ev, linewidth=2.2,
                    label="mSPRT e-value")
    ax_top.axhline(THRESHOLD, color=color_threshold, linestyle="--",
                   linewidth=1.8,
                   label=f"Stopping threshold (1/α = {THRESHOLD:.0f})")
    if stopping_day is not None:
        ax_top.axvline(stopping_day, color=color_ev, linestyle=":",
                       linewidth=1.5, alpha=0.8,
                       label=f"Could stop: day {stopping_day}")
        ax_top.annotate(
            f"Stop day {stopping_day}",
            xy=(stopping_day, ev_daily[stopping_day - 1]),
            xytext=(stopping_day + 1.5, ev_daily[stopping_day - 1] * 2.0),
            fontsize=9, color=color_ev,
            arrowprops=dict(arrowstyle="->", color=color_ev, lw=1.2),
        )

    ax_top.set_ylabel("E-value (log scale)", fontsize=11)
    ax_top.set_title(
        "mSPRT on the real synthetic dataset: LLM task completion experiment\n"
        f"Treated: {len(treated):,} users (wave 1)  |  "
        f"Control: {len(control):,} users (wave 2)",
        fontsize=11, pad=8,
    )
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.set_ylim(bottom=0.05)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines["bottom"].set_visible(False)

    # Bottom: cumulative completion rates
    ax_bot.plot(days, cum_t_daily * 100, color=color_t, linewidth=2.0,
                label=f"Treatment (wave 1, final {cum_t_daily[-1]*100:.1f}%)")
    ax_bot.plot(days, cum_c_daily * 100, color=color_c, linewidth=1.8,
                linestyle="--",
                label=f"Control (wave 2, final {cum_c_daily[-1]*100:.1f}%)")
    ax_bot.set_ylabel("Completion rate (%)", fontsize=10)
    ax_bot.set_xlabel("Experiment day", fontsize=11)
    ax_bot.legend(fontsize=9, loc="lower right")
    ax_bot.spines["top"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "msprt_evalue_real_data.png")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Figure 1 (conceptual)...")
    make_figure_1_conceptual()
    print("Generating Figure 2 (data-driven)...")
    make_figure_2_real_data()
    print("Done.")
