"""Generate the two article figures for the switchback experiments tutorial.

Figure 1 (conceptual): schematic 3-slot switchback design showing treatment blocks,
carryover contamination, and why the naive comparison overestimates.

Figure 2 (data-driven): actual slot-level task-completion time series from the
synthetic dataset, annotated with treatment blocks, carryover slots, and the
naive vs. adjusted ATE estimates.

Run from repo root:
    python 06_switchback/generate_switchback_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "06_switchback"
IMAGES_DIR = REPO_ROOT / "images" / "article-6"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette (matches series style)
C_ON       = "#4C78A8"   # muted blue for AI-on
C_OFF      = "#E8EDF2"   # light grey for AI-off fill
C_CARRY    = "#F58518"   # orange for carryover slots
C_TRUE     = "#54A24B"   # green for true effect line
C_NAIVE    = "#E45756"   # red for naive estimate
C_ADJ      = "#4C78A8"   # blue for adjusted estimate
C_GRID     = "#E0E0E0"


def save_figure(fig: plt.Figure, name: str) -> None:
    for dest in [METHOD_DIR, IMAGES_DIR]:
        path = dest / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1 — Conceptual: 3-slot switchback with carryover annotation
# ---------------------------------------------------------------------------

def make_figure_1_conceptual() -> None:
    rng = np.random.default_rng(7)
    n_cycles = 4
    n_slots  = n_cycles * 6      # 24 slots for illustration

    # Base outcome (smooth background)
    base = 0.60 + 0.005 * rng.standard_normal(n_slots)
    true_effect = 0.060
    carryover   = 0.030

    ai_on = np.tile([1, 1, 1, 0, 0, 0], n_cycles)
    lag1  = np.roll(ai_on, 1)
    lag1[0] = 0

    outcome = base + true_effect * ai_on + carryover * lag1

    fig, (ax_main, ax_label) = plt.subplots(
        2, 1,
        figsize=(11, 5),
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.0},
    )

    x = np.arange(n_slots)

    # Shade on/off blocks
    for i in range(n_slots):
        if ai_on[i]:
            ax_main.axvspan(i - 0.5, i + 0.5, color=C_ON, alpha=0.18, linewidth=0)
        elif lag1[i]:  # carryover slot
            ax_main.axvspan(i - 0.5, i + 0.5, color=C_CARRY, alpha=0.22, linewidth=0)

    # Outcome line
    ax_main.plot(x, outcome, color="#333333", linewidth=1.6, zorder=3)

    # True-effect reference band
    true_band_top = base + true_effect
    ax_main.fill_between(x, base, true_band_top,
                         where=(ai_on == 1),
                         alpha=0.30, color=C_TRUE, label="True direct effect (6 pp)")

    # Carryover annotation arrows (first off-slot of each cycle)
    carry_slots = [i for i in range(n_slots) if lag1[i] and not ai_on[i]]
    for s in carry_slots:
        ax_main.annotate(
            "", xy=(s, outcome[s]), xytext=(s, base[s] + true_effect),
            arrowprops=dict(arrowstyle="->", color=C_CARRY, lw=1.4),
        )

    ax_main.set_xlim(-0.7, n_slots - 0.3)
    y_lo, y_hi = outcome.min() - 0.012, outcome.max() + 0.025
    ax_main.set_ylim(y_lo, y_hi)
    ax_main.set_ylabel("Task completion rate", fontsize=11)
    ax_main.set_xticks([])
    ax_main.yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax_main.set_axisbelow(True)
    ax_main.spines[["top", "right", "bottom"]].set_visible(False)

    # Legend
    patch_on     = mpatches.Patch(color=C_ON, alpha=0.5, label="AI routing ON (3-slot block)")
    patch_carry  = mpatches.Patch(color=C_CARRY, alpha=0.5, label="Carryover slot (first AI-off slot)")
    patch_effect = mpatches.Patch(color=C_TRUE, alpha=0.5, label="True direct effect (6 pp)")
    ax_main.legend(
        handles=[patch_on, patch_carry, patch_effect],
        loc="upper right", fontsize=9, framealpha=0.85,
    )
    ax_main.set_title(
        "3-slot switchback: ON / OFF blocks with carryover into first OFF slot",
        fontsize=12, pad=8,
    )

    # Bottom label strip
    ax_label.set_xlim(-0.7, n_slots - 0.3)
    ax_label.set_ylim(0, 1)
    ax_label.axis("off")
    for c in range(n_cycles):
        start = c * 6
        # "ON" label
        ax_label.text(start + 1, 0.5, "AI ON", ha="center", va="center",
                      fontsize=8.5, color=C_ON, fontweight="bold")
        # "OFF" label
        ax_label.text(start + 4, 0.5, "AI OFF", ha="center", va="center",
                      fontsize=8.5, color="#888888")
        # Vertical dividers
        for v in [start - 0.5, start + 2.5, start + 5.5]:
            ax_label.axvline(v, color="#CCCCCC", linewidth=0.8)

    plt.tight_layout()
    save_figure(fig, "switchback_design_conceptual.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Data-driven: real slot time series + ATE comparison
# ---------------------------------------------------------------------------

def make_figure_2_data_driven() -> None:
    df = pd.read_csv(REPO_ROOT / "data" / "synthetic_llm_logs.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    N_SLOTS = 48
    TRUE_EFFECT = 0.060
    CARRYOVER   = 0.030
    base_rate   = df["task_completed"].mean()

    df["hour_slot"] = df.index % N_SLOTS
    ai_on_schedule = np.tile([1, 1, 1, 0, 0, 0], N_SLOTS // 6)
    df["ai_on"] = ai_on_schedule[df["hour_slot"]]

    slots = df.groupby("hour_slot").agg(
        mean_task_completed=("task_completed", "mean"),
        ai_on=("ai_on", "first"),
        n_obs=("user_id", "count"),
    ).reset_index()

    slot_noise_std = np.sqrt(base_rate * (1 - base_rate) / slots["n_obs"].iloc[0])
    rng = np.random.default_rng(42)
    slots["mean_task_completed"] = base_rate + rng.normal(0, slot_noise_std, size=N_SLOTS)
    slots["ai_on_lag1"] = slots["ai_on"].shift(1).fillna(0).astype(int)
    slots["mean_task_completed"] = (
        slots["mean_task_completed"]
        + TRUE_EFFECT * slots["ai_on"]
        + CARRYOVER   * slots["ai_on_lag1"]
    )

    # Point estimates from the probe run
    naive_ate = 0.0688
    adj_ate   = 0.0607
    naive_ci  = (0.0596, 0.0783)
    adj_ci    = (0.0541, 0.0683)

    fig, (ax_ts, ax_bar) = plt.subplots(
        1, 2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [3, 1.2]},
    )

    # ---- Left panel: slot time series ----
    x = slots["hour_slot"].values
    y = slots["mean_task_completed"].values
    ai_on_arr  = slots["ai_on"].values
    lag1_arr   = slots["ai_on_lag1"].values

    for i in x:
        if ai_on_arr[i]:
            ax_ts.axvspan(i - 0.5, i + 0.5, color=C_ON, alpha=0.15, linewidth=0)
        elif lag1_arr[i]:
            ax_ts.axvspan(i - 0.5, i + 0.5, color=C_CARRY, alpha=0.20, linewidth=0)

    ax_ts.plot(x, y, color="#333333", linewidth=1.5, zorder=3, label="Slot outcome")

    # Mark carryover slots
    carry_x = x[(ai_on_arr == 0) & (lag1_arr == 1)]
    carry_y = y[(ai_on_arr == 0) & (lag1_arr == 1)]
    ax_ts.scatter(carry_x, carry_y, color=C_CARRY, s=45, zorder=5,
                  label="Carryover slot (ai_on=0, lag=1)")

    ax_ts.set_xlabel("Hour slot", fontsize=11)
    ax_ts.set_ylabel("Mean task completion rate", fontsize=11)
    ax_ts.set_title("Slot-level outcomes: 48 switchback slots\n(orange = first AI-off slot after an AI-on block)",
                    fontsize=11, pad=6)
    ax_ts.yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax_ts.set_axisbelow(True)
    ax_ts.spines[["top", "right"]].set_visible(False)
    ax_ts.legend(fontsize=9, loc="lower left", framealpha=0.85)

    # ---- Right panel: ATE comparison ----
    labels   = ["Naive OLS\n(no lag)", "Carryover-\nadjusted OLS"]
    ates     = [naive_ate, adj_ate]
    lo_errs  = [naive_ate - naive_ci[0], adj_ate - adj_ci[0]]
    hi_errs  = [naive_ci[1] - naive_ate, adj_ci[1] - adj_ate]
    colors   = [C_NAIVE, C_ADJ]

    bars = ax_bar.bar(labels, ates, color=colors, width=0.45,
                      yerr=[lo_errs, hi_errs],
                      error_kw={"elinewidth": 1.8, "capsize": 6, "capthick": 1.8, "ecolor": "#555555"},
                      zorder=3)
    ax_bar.axhline(TRUE_EFFECT, color=C_TRUE, linewidth=2.0, linestyle="--",
                   label=f"True effect ({TRUE_EFFECT})", zorder=4)
    ax_bar.set_ylim(0.03, 0.095)
    ax_bar.set_ylabel("Average treatment effect (pp)", fontsize=11)
    ax_bar.set_title("ATE estimates\nwith 95% bootstrap CI", fontsize=11, pad=6)
    ax_bar.yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.legend(fontsize=9, framealpha=0.85)

    # Bias annotation on naive bar
    ax_bar.annotate(
        f"+{(naive_ate - TRUE_EFFECT)*100:.1f} pp bias",
        xy=(0, naive_ate + hi_errs[0] + 0.001),
        ha="center", va="bottom", fontsize=8.5, color=C_NAIVE,
    )

    plt.tight_layout(pad=1.5)
    save_figure(fig, "switchback_estimates_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Figure 1 (conceptual)...")
    make_figure_1_conceptual()
    print("\nGenerating Figure 2 (data-driven)...")
    make_figure_2_data_driven()
    print("\nDone.")
