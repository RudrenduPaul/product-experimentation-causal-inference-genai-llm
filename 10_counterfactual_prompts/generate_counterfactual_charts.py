"""Generate the two article figures for the counterfactual estimation tutorial.

Figure 1 (conceptual): T-learner mechanics — two separate outcome models
  (m0 for Prompt A, m1 for Prompt B) as smooth functions of query_confidence,
  with the CATE shown as the gap between them.

Figure 2 (data-driven): CATE distribution by engagement tier on the
  50,000-user synthetic dataset, showing heterogeneous treatment effects.

Run from repo root:
    python 10_counterfactual_prompts/generate_counterfactual_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "10_counterfactual_prompts"
IMAGES_DIR = REPO_ROOT / "images" / "article-10"


# ── shared palette ────────────────────────────────────────────────────────────
PROMPT_A_COLOR = "#4C72B0"   # blue  – Prompt A / control
PROMPT_B_COLOR = "#C44E52"   # red   – Prompt B / treatment
CATE_COLOR     = "#2E8B57"   # green – CATE gap
TIER_COLORS = {
    "light":  "#8FA9C7",
    "medium": "#B8A886",
    "heavy":  "#C88B8B",
}


# ── Figure 1: conceptual T-learner ───────────────────────────────────────────
def make_figure_1_conceptual() -> None:
    """Conceptual illustration of the T-learner.

    Top panel   : smooth sigmoid curves for m0 (Prompt A) and m1 (Prompt B)
                  as a function of query_confidence.
    Bottom panel: CATE = m1 − m0 across the covariate range, with a shaded
                  region marking where the effect is largest.
    """
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 400)

    # Smooth sigmoid outcome models (schematic, not real data)
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    m0 = sigmoid(2.5 * x - 0.8)          # Prompt A: lower completion rate
    m1 = sigmoid(2.5 * x - 0.8 + 0.35)   # Prompt B: lifted by ~+0.04 overall
    cate = m1 - m0

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 6.4),
        gridspec_kw={"height_ratios": [3.0, 1.8], "hspace": 0.0},
        sharex=True,
    )

    # ── top panel: outcome curves ─────────────────────────────────────────
    ax_top.fill_between(x, m0, alpha=0.15, color=PROMPT_A_COLOR)
    ax_top.fill_between(x, m1, alpha=0.15, color=PROMPT_B_COLOR)
    ax_top.plot(x, m0, color=PROMPT_A_COLOR, lw=2.0,
                label="m0: predicted P(complete | Prompt A, X)")
    ax_top.plot(x, m1, color=PROMPT_B_COLOR, lw=2.0,
                label="m1: predicted P(complete | Prompt B, X)")

    # Shade the CATE gap between the two curves
    ax_top.fill_between(x, m0, m1, alpha=0.25, color=CATE_COLOR, zorder=1)

    # Annotate the gap — label anchored in the upper-left white margin
    mid = 0.50
    mid_idx = np.argmin(np.abs(x - mid))
    ax_top.annotate(
        "CATE = m1(x) − m0(x)",
        xy=(mid, (m0[mid_idx] + m1[mid_idx]) / 2),
        xytext=(0.08, 0.90),
        fontsize=10, color=CATE_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=CATE_COLOR, lw=1.2,
                        connectionstyle="arc3,rad=-0.25"),
    )

    ax_top.set_ylim(0.0, 1.05)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylabel("P(task completed)", fontsize=10)
    ax_top.set_title(
        "T-learner: two outcome models produce a counterfactual gap at every covariate value",
        fontsize=12.0, loc="left", pad=8,
    )
    ax_top.legend(
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=10,
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    # ── bottom panel: CATE curve ──────────────────────────────────────────
    ax_bot.fill_between(x, 0, cate, alpha=0.35, color=CATE_COLOR)
    ax_bot.plot(x, cate, color=CATE_COLOR, lw=2.0)
    ax_bot.axhline(0, color="#999999", lw=0.8, ls="--")

    # Mark the ground-truth +4 pp band
    ax_bot.axhline(0.04, color="#888888", lw=1.0, ls=":", alpha=0.7)
    ax_bot.text(
        0.02, 0.041, "+4 pp ground truth",
        fontsize=9, color="#555555", va="bottom",
    )

    ax_bot.set_ylim(-0.01, cate.max() * 1.35)
    ax_bot.set_ylabel("CATE", fontsize=10)
    ax_bot.set_xlabel(
        "query_confidence  (user covariate: 0 = low complexity, 1 = high complexity)",
        fontsize=10,
    )
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, "counterfactual_cate_conceptual.png")


# ── Figure 2: data-driven CATE distribution ──────────────────────────────────
def make_figure_2_data_driven() -> None:
    """CATE distribution by engagement tier on the 50,000-user synthetic dataset.

    Top panel   : KDE of T-learner CATE estimates for each engagement tier,
                  showing heterogeneous treatment effects.
    Bottom panel: mean CATE per tier as a horizontal bar chart with the
                  overall mean CATE reference line.
    """
    from scipy.stats import gaussian_kde

    df = pd.read_csv(REPO_ROOT / "data" / "synthetic_llm_logs.csv")

    # Fit the T-learner (same as in the article)
    X_cols = ["engagement_tier", "query_confidence"]
    X = pd.get_dummies(df[X_cols], drop_first=True).astype(float)
    X_arr = X.values
    treatment = df["prompt_variant"].values
    outcome   = df["task_completed"].values

    m0 = LogisticRegression(max_iter=1000)
    m1 = LogisticRegression(max_iter=1000)
    m0.fit(X_arr[treatment == 0], outcome[treatment == 0])
    m1.fit(X_arr[treatment == 1], outcome[treatment == 1])

    mu0 = m0.predict_proba(X_arr)[:, 1]
    mu1 = m1.predict_proba(X_arr)[:, 1]
    df["cate"] = mu1 - mu0

    tier_order = ["light", "medium", "heavy"]
    overall_mean = df["cate"].mean()

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 7.0),
        gridspec_kw={"height_ratios": [2.8, 1.0], "hspace": 0.65},
    )

    # ── top panel: KDE per tier ───────────────────────────────────────────
    grid = np.linspace(df["cate"].min() - 0.005, df["cate"].max() + 0.005, 400)
    peak_max = 0.0
    for tier in tier_order:
        vals = df[df["engagement_tier"] == tier]["cate"].values
        kde = gaussian_kde(vals, bw_method=0.25)
        dens = kde(grid)
        peak_max = max(peak_max, dens.max())
        ax_top.fill_between(
            grid, dens, alpha=0.30, color=TIER_COLORS[tier],
        )
        ax_top.plot(
            grid, dens, color=TIER_COLORS[tier], lw=1.8,
            label=f"{tier.capitalize()} (mean CATE = {vals.mean():+.3f})",
        )

    y_top = peak_max * 1.40
    ax_top.set_ylim(0, y_top)

    # Overall mean CATE reference line — label anchored to axes fraction (top-right)
    ax_top.axvline(overall_mean, color="#555555", lw=1.2, ls="--", alpha=0.8)
    ax_top.text(
        0.99, 0.97,
        f"Overall mean CATE = {overall_mean:+.3f}",
        fontsize=9.5, color="#333333", va="top", ha="right",
        transform=ax_top.transAxes,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#BBBBBB", lw=0.7),
    )

    ax_top.legend(
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=10,
    )
    ax_top.set_xlabel("Estimated individual treatment effect (CATE)", fontsize=10)
    ax_top.set_ylabel("Density", fontsize=10)
    ax_top.set_title(
        "T-learner CATE distributions on the 50,000-user synthetic dataset: "
        "heavy users benefit most",
        fontsize=12.0, loc="left", pad=8,
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # ── bottom panel: mean CATE bar chart ─────────────────────────────────
    tier_means = [
        df[df["engagement_tier"] == t]["cate"].mean() for t in tier_order
    ]
    y_pos = np.arange(len(tier_order))[::-1]

    for y, tier, mean_val in zip(y_pos, tier_order, tier_means):
        ax_bot.barh(
            y, mean_val, color=TIER_COLORS[tier],
            alpha=0.80, edgecolor="white", height=0.55,
        )
        ax_bot.text(
            mean_val + 0.0015, y,
            f"{mean_val:+.3f}",
            ha="left", va="center", fontsize=10, color="#333333", fontweight="bold",
        )

    ax_bot.axvline(overall_mean, color="#555555", lw=1.2, ls="--", alpha=0.8)
    ax_bot.axvline(0, color="#999999", lw=0.8)

    ax_bot.set_yticks(y_pos)
    ax_bot.set_yticklabels([t.capitalize() for t in tier_order], fontsize=10)
    ax_bot.set_xlabel("Mean CATE", fontsize=10)
    ax_bot.set_title(
        "Mean CATE per engagement tier: selective routing targets high-CATE users",
        fontsize=10.5, loc="left", pad=6,
    )
    ax_bot.set_xlim(0, max(tier_means) * 1.50)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.tick_params(axis="y", length=0)

    fig.tight_layout()
    save_figure(fig, "counterfactual_cate_density.png")


# ── shared save helper ────────────────────────────────────────────────────────
def save_figure(fig, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for target in (IMAGES_DIR / name, METHOD_DIR / name):
        fig.savefig(target, dpi=150, bbox_inches="tight")
        print(f"  wrote {target.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    print("Generating Figure 1 (conceptual)...")
    make_figure_1_conceptual()
    print("Generating Figure 2 (data-driven)...")
    make_figure_2_data_driven()
    print("Done.")


if __name__ == "__main__":
    main()
