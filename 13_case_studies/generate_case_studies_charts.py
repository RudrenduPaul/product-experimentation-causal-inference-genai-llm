"""Generate the two article figures for the production case studies article.

Figure 1 (conceptual): method-selection map — four deployment scenarios, each
linked to its identifying assumption and causal method.

Figure 2 (data-driven): IPW weight distribution diagnostic from the Lyft case
study, showing the 99th-percentile trim threshold and ATE comparison.

Run from repo root:
    python 13_case_studies/generate_case_studies_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "13_case_studies"
IMAGES_DIR = REPO_ROOT / "images" / "article-13"


def save_figure(fig, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for target in (IMAGES_DIR / name, METHOD_DIR / name):
        fig.savefig(target, dpi=150, bbox_inches="tight")
        print(f"  wrote {target.relative_to(REPO_ROOT)}")
    plt.close(fig)


def make_figure_1_conceptual() -> None:
    """Method-selection map: deployment scenario → assumption → method → company."""

    rows = [
        {
            "scenario": "Staged rollout\n(waves, cohorts, regions)",
            "assumption": "Parallel pre-treatment\ntrends",
            "method": "Difference-in-\nDifferences (DiD)",
            "company": "Airbnb",
            "color": "#4C72B0",
        },
        {
            "scenario": "Threshold-gated routing\n(score cutoff assigns treatment)",
            "assumption": "No manipulation\nof the running variable",
            "method": "Regression\nDiscontinuity (RDD)",
            "company": "Netflix · Uber",
            "color": "#C44E52",
        },
        {
            "scenario": "Full-population upgrade\n(no holdout group available)",
            "assumption": "Good pre-period fit\nbetween actual and synthetic",
            "method": "Synthetic\nControl",
            "company": "Netflix",
            "color": "#55A868",
        },
        {
            "scenario": "Opt-in / observational\n(users self-select into feature)",
            "assumption": "Unconfoundedness\n(all confounders observed)",
            "method": "Propensity Score\nMethods (IPW / AIPW)",
            "company": "Lyft",
            "color": "#DD8452",
        },
    ]

    fig_h = 1.0 + len(rows) * 1.3
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, len(rows) * 1.3 + 0.2)
    ax.axis("off")

    col_x = [0.15, 3.85, 7.55, 10.7]
    col_labels = ["Deployment scenario", "Identifying assumption", "Causal method", "Team"]
    col_widths = [3.5, 3.5, 3.0, 1.2]

    # Header row
    for cx, label, cw in zip(col_x, col_labels, col_widths):
        ax.text(
            cx + cw / 2, len(rows) * 1.3 - 0.05,
            label,
            ha="center", va="bottom", fontsize=10.5,
            fontweight="bold", color="#222222",
        )

    ax.axhline(len(rows) * 1.3 - 0.15, color="#CCCCCC", lw=1.0)

    for i, row in enumerate(rows):
        y = (len(rows) - 1 - i) * 1.3 + 0.15
        color = row["color"]

        # Scenario box (left)
        rect_s = mpatches.FancyBboxPatch(
            (col_x[0], y), col_widths[0], 1.05,
            boxstyle="round,pad=0.07",
            linewidth=1.2, edgecolor=color, facecolor="#F8F8F8",
        )
        ax.add_patch(rect_s)
        ax.text(
            col_x[0] + col_widths[0] / 2, y + 0.525,
            row["scenario"],
            ha="center", va="center", fontsize=9.5, color="#111111",
            multialignment="center",
        )

        # Arrow → assumption
        ax.annotate(
            "", xy=(col_x[1], y + 0.525), xytext=(col_x[0] + col_widths[0] + 0.04, y + 0.525),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
        )

        # Assumption box (middle-left)
        rect_a = mpatches.FancyBboxPatch(
            (col_x[1], y), col_widths[1], 1.05,
            boxstyle="round,pad=0.07",
            linewidth=1.2, edgecolor="#AAAAAA", facecolor="white",
        )
        ax.add_patch(rect_a)
        ax.text(
            col_x[1] + col_widths[1] / 2, y + 0.525,
            row["assumption"],
            ha="center", va="center", fontsize=9.0, color="#444444",
            fontstyle="italic", multialignment="center",
        )

        # Arrow → method
        ax.annotate(
            "", xy=(col_x[2], y + 0.525), xytext=(col_x[1] + col_widths[1] + 0.04, y + 0.525),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
        )

        # Method box (middle-right)
        rect_m = mpatches.FancyBboxPatch(
            (col_x[2], y), col_widths[2], 1.05,
            boxstyle="round,pad=0.07",
            linewidth=1.8, edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(rect_m)
        ax.text(
            col_x[2] + col_widths[2] / 2, y + 0.525,
            row["method"],
            ha="center", va="center", fontsize=9.5,
            color=color, fontweight="bold", multialignment="center",
        )

        # Company label (right, no box)
        ax.text(
            col_x[3] + col_widths[3] / 2, y + 0.525,
            row["company"],
            ha="center", va="center", fontsize=9.0, color="#555555",
            multialignment="center",
        )

        # Thin separator line between rows
        if i < len(rows) - 1:
            ax.axhline(y - 0.1, color="#EEEEEE", lw=0.8)

    ax.set_title(
        "Deployment structure determines identification strategy",
        fontsize=12.5, loc="left", pad=10, fontweight="normal", color="#111111",
    )

    fig.tight_layout(pad=0.8)
    save_figure(fig, "case_studies_method_map.png")


def make_figure_2_data_driven() -> None:
    """IPW weight distribution diagnostic with ATE comparison panel."""

    df = pd.read_csv(REPO_ROOT / "data" / "synthetic_llm_logs.csv")

    X = pd.get_dummies(
        df[["engagement_tier", "query_confidence"]], drop_first=True
    ).astype(float)
    ps_model = LogisticRegression(max_iter=1000).fit(X, df["opt_in_agent_mode"])
    df["propensity"] = ps_model.predict_proba(X)[:, 1]

    df["ipw"] = np.where(
        df.opt_in_agent_mode == 1,
        1 / df.propensity,
        1 / (1 - df.propensity),
    )
    trim_threshold = np.percentile(df.ipw, 99)
    df["ipw_trimmed"] = df.ipw.clip(upper=trim_threshold)

    def weighted_ate(data):
        t = data[data.opt_in_agent_mode == 1]
        c = data[data.opt_in_agent_mode == 0]
        return (
            (t.task_completed * t.ipw_trimmed).sum() / t.ipw_trimmed.sum()
            - (c.task_completed * c.ipw_trimmed).sum() / c.ipw_trimmed.sum()
        )

    ate_untrimmed = (
        (df[df.opt_in_agent_mode == 1].task_completed * df[df.opt_in_agent_mode == 1].ipw).sum()
        / df[df.opt_in_agent_mode == 1].ipw.sum()
        - (df[df.opt_in_agent_mode == 0].task_completed * df[df.opt_in_agent_mode == 0].ipw).sum()
        / df[df.opt_in_agent_mode == 0].ipw.sum()
    )
    ate_trimmed = weighted_ate(df)

    main_color = "#4C72B0"
    trim_color = "#C44E52"

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 6.5),
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.55},
    )

    # --- TOP: weight distribution histogram ---
    weights_clipped = df.ipw[df.ipw <= trim_threshold * 1.05]
    ax_top.hist(
        df.ipw.clip(upper=trim_threshold * 1.05),
        bins=70, color=main_color, alpha=0.65, edgecolor="none",
    )
    ax_top.axvline(
        trim_threshold, color=trim_color, lw=2.0, ls="--",
        label=f"99th pct trim threshold  ({trim_threshold:.2f})",
        zorder=3,
    )

    y_max = ax_top.get_ylim()[1]

    # Shade extreme weights region
    ax_top.axvspan(trim_threshold, df.ipw.max(), color=trim_color, alpha=0.08, zorder=0)
    ax_top.text(
        trim_threshold + 0.08, y_max * 0.75,
        f"Extreme weights\ntrimmed here\n({(df.ipw > trim_threshold).sum():,} obs.)",
        ha="left", va="top", fontsize=9.0, color=trim_color,
    )

    # Stats annotation in the near-zero region (bulk of the distribution)
    ax_top.text(
        trim_threshold * 0.38, y_max * 0.82,
        f"Median: {np.median(df.ipw):.2f}\n90th pct: {np.percentile(df.ipw, 90):.2f}\n99th pct: {trim_threshold:.2f}",
        ha="center", va="top", fontsize=9.0, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.7),
    )

    ax_top.set_xlabel("IPW weight")
    ax_top.set_ylabel("Count")
    ax_top.set_title(
        "IPW weight distribution: checking for extreme observations\n"
        "(Lyft diagnostic — 50,000-user synthetic dataset)",
        fontsize=11.5, loc="left",
    )
    ax_top.legend(frameon=False, fontsize=10, loc="upper right")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # --- BOTTOM: ATE comparison bars ---
    labels = ["ATE (untrimmed)", "ATE (trimmed 99th pct)"]
    values = [ate_untrimmed, ate_trimmed]
    bar_colors = [main_color, "#55A868"]

    bars = ax_bot.barh(
        labels, values, color=bar_colors, alpha=0.80, height=0.45, edgecolor="none",
    )
    for bar, val in zip(bars, values):
        ax_bot.text(
            val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center", ha="left", fontsize=10, color="#222222",
        )

    ax_bot.axvline(0, color="#AAAAAA", lw=0.8)
    ax_bot.set_xlim(0, max(values) * 1.25)
    ax_bot.set_xlabel("ATE estimate (task completion rate)")
    ax_bot.set_title(
        "Trimming has minimal effect: extreme weights don't dominate this estimate",
        fontsize=10.0, loc="left",
    )
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.tick_params(axis="y", length=0)

    fig.tight_layout()
    save_figure(fig, "case_studies_weight_distribution.png")


def main() -> None:
    print("Generating Figure 1 (conceptual — method-selection map)...")
    make_figure_1_conceptual()
    print("Generating Figure 2 (data-driven — IPW weight distribution)...")
    make_figure_2_data_driven()
    print("Done.")


if __name__ == "__main__":
    main()
