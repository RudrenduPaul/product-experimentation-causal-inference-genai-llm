"""Generate the two article figures for the RDD confidence-threshold tutorial.

Figure 1 (conceptual): idealized jump at the routing threshold, showing how
RDD reads the discontinuity in task completion at confidence = 0.85.

Figure 2 (data-driven): KDE of query_confidence on the 50,000-user dataset
with routing groups annotated, plus a horizontal bar chart of counts in
narrow bins around the threshold (manipulation diagnostic).

Run from repo root:
    python 03_rdd_confidence_threshold/generate_rdd_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "03_rdd_confidence_threshold"
IMAGES_DIR = REPO_ROOT / "images" / "article-3"

CUTOFF = 0.85
RED = "#C44E52"
BLUE = "#4C72B0"
GRAY = "#888888"


def save_figure(fig, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for target in (IMAGES_DIR / name, METHOD_DIR / name):
        fig.savefig(target, dpi=150, bbox_inches="tight")
        print(f"  wrote {target.relative_to(REPO_ROOT)}")
    plt.close(fig)


def make_figure_1_conceptual() -> None:
    """Conceptual schematic: idealized RDD jump at confidence = 0.85.

    Two stacked panels with zero overlap between text and plot data:
      (top)    smooth idealized outcome curves on each side of the threshold
      (bottom) a narrow label strip with region brackets below the x-axis
    """
    rng = np.random.default_rng(7)

    # Build idealized smooth outcome curves on each side of the cutoff
    # Premium side (left): confidence in [0.50, 0.85), outcome rises from ~0.55 to ~0.71
    x_prem = np.linspace(0.50, 0.85 - 1e-4, 300)
    y_prem = 0.55 + (0.71 - 0.55) * (x_prem - 0.50) / (0.85 - 0.50)

    # Cheap side (right): confidence in (0.85, 1.00], outcome rises from ~0.65 to ~0.78
    x_cheap = np.linspace(0.85 + 1e-4, 1.00, 300)
    y_cheap = 0.65 + (0.78 - 0.65) * (x_cheap - 0.85) / (1.00 - 0.85)

    y_top_val = max(y_prem.max(), y_cheap.max()) * 1.40

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 5.6),
        gridspec_kw={"height_ratios": [5.0, 1.0], "hspace": 0.0},
        sharex=True,
        layout="constrained",
    )

    # --- TOP: smooth outcome curves ---
    ax_top.fill_between(x_prem, y_prem, alpha=0.15, color=RED)
    ax_top.plot(x_prem, y_prem, color=RED, lw=2,
                label="Premium routing (confidence < 0.85)")

    ax_top.fill_between(x_cheap, y_cheap, alpha=0.15, color=BLUE)
    ax_top.plot(x_cheap, y_cheap, color=BLUE, lw=2,
                label="Cheap routing (confidence >= 0.85)")

    # Vertical cutoff line
    ax_top.axvline(CUTOFF, color="#555555", ls="--", lw=1.5, zorder=3)

    # +6pp causal-effect bracket at x = CUTOFF + small offset.
    # Endpoints: premium curve ends at ~0.71, cheap curve starts at ~0.65.
    # Bottom arrow inset above the blue cheap line so its head doesn't
    # touch the data line.
    y_prem_end = y_prem[-1]
    y_cheap_start = y_cheap[0] + 0.015
    y_mid = (y_prem_end + y_cheap_start) / 2

    bx = CUTOFF + 0.012          # bracket sits just right of the cutoff
    ax_top.annotate(
        "", xy=(bx, y_cheap_start), xytext=(bx, y_prem_end),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4),
    )
    # Label moved into the top-right empty headroom (well clear of both
    # curves) with a thin leader line back to the bracket midpoint.
    ax_top.annotate(
        "+6pp causal effect",
        xy=(bx, y_mid),
        xytext=(0.97, y_top_val * 0.92),
        ha="right", va="bottom",
        fontsize=10, color="#333333", fontweight="bold",
        arrowprops=dict(
            arrowstyle="-",
            color="#888888", lw=0.7,
            shrinkA=0, shrinkB=2,
            connectionstyle="arc3,rad=-0.2",
        ),
    )

    ax_top.set_xlim(0.50, 1.00)
    ax_top.set_ylim(0.45, y_top_val)
    ax_top.set_ylabel("Task completion rate")
    ax_top.set_title(
        "Regression discontinuity reads the jump at the routing threshold",
        fontsize=12.5, loc="left",
    )
    ax_top.legend(
        frameon=False, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.0), fontsize=10,
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    # --- BOTTOM: region label strip ---
    ax_bot.set_xlim(0.50, 1.00)
    ax_bot.set_ylim(0, 1)
    ax_bot.set_yticks([])
    for side in ("top", "left", "right"):
        ax_bot.spines[side].set_visible(False)
    ax_bot.spines["bottom"].set_visible(True)

    # Premium-routed region bracket
    for x0, x1, label, color in [
        (0.50, CUTOFF, "Premium-routed region", RED),
        (CUTOFF, 1.00, "Cheap-routed region", BLUE),
    ]:
        mid = (x0 + x1) / 2
        ax_bot.annotate(
            "", xy=(x0 + 0.004, 0.78), xytext=(x1 - 0.004, 0.78),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.2, alpha=0.7),
        )
        ax_bot.plot([x0 + 0.004, x0 + 0.004], [0.68, 0.88],
                    color=color, lw=1.2, alpha=0.7)
        ax_bot.plot([x1 - 0.004, x1 - 0.004], [0.68, 0.88],
                    color=color, lw=1.2, alpha=0.7)
        ax_bot.text(mid, 0.32, label, ha="center", va="center",
                    fontsize=9.5, color="#333333")

    # Routing cutoff marker with plain-English rule just below
    ax_bot.axvline(CUTOFF, color="#555555", ls="--", lw=1.2, ymin=0.55,
                   ymax=0.95)
    ax_bot.text(
        CUTOFF, 0.05,
        "Routing cutoff (0.85)\nqueries below 0.85 → premium model",
        ha="center", va="bottom", fontsize=8.5, color="#555555",
    )

    ax_bot.set_xlabel(
        "Query confidence score (running variable)", fontsize=10,
    )

    save_figure(fig, "rdd_threshold_conceptual.png")


def make_figure_2_density() -> None:
    """Data-driven: KDE of query_confidence + bin-count bar chart.

    Two-panel layout:
      (top)    smooth KDE curves for all / premium / cheap routing groups
      (bottom) horizontal count bars for 5 narrow bins near the threshold
    """
    df = pd.read_csv(REPO_ROOT / "data" / "synthetic_llm_logs.csv")

    prem_conf = df[df.routed_to_premium == 1].query_confidence.values
    cheap_conf = df[df.routed_to_premium == 0].query_confidence.values
    all_conf = df.query_confidence.values

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 7.0),
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.70},
    )

    # --- TOP: KDE curves ---
    grid = np.linspace(0, 1, 500)

    kde_all = gaussian_kde(all_conf, bw_method=0.15)
    kde_prem = gaussian_kde(prem_conf, bw_method=0.25)
    kde_cheap = gaussian_kde(cheap_conf, bw_method=0.25)

    dens_all = kde_all(grid)
    dens_prem = kde_prem(grid)
    dens_cheap = kde_cheap(grid)

    # Full population (gray background)
    ax_top.fill_between(grid, dens_all, alpha=0.4, color="#CCCCCC",
                        label="All queries (n=50,000)")
    ax_top.plot(grid, dens_all, color="#AAAAAA", lw=1.0)

    # Premium-routed (red)
    ax_top.fill_between(grid, dens_prem, alpha=0.55, color=RED,
                        label=f"Premium-routed (n={len(prem_conf):,})")
    ax_top.plot(grid, dens_prem, color=RED, lw=1.4)

    # Cheap-routed (blue)
    ax_top.fill_between(grid, dens_cheap, alpha=0.55, color=BLUE,
                        label=f"Cheap-routed (n={len(cheap_conf):,})")
    ax_top.plot(grid, dens_cheap, color=BLUE, lw=1.4)

    # Routing cutoff
    y_top = max(dens_all.max(), dens_prem.max(), dens_cheap.max()) * 1.40
    ax_top.axvline(CUTOFF, color="#555555", ls="--", lw=1.5, zorder=3)
    ax_top.text(
        CUTOFF + 0.012, y_top * 0.93,
        "Routing cutoff (0.85)",
        ha="left", va="top", fontsize=9.5, color="#555555",
    )

    ax_top.set_ylim(0, y_top)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylabel("Density")
    ax_top.set_title(
        "Query confidence distribution on the 50,000-user dataset",
        fontsize=12.5, loc="left", pad=10,
    )
    ax_top.legend(
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=10,
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # --- BOTTOM: horizontal count bars for 5 narrow bins ---
    bins = [(0.80, 0.82), (0.82, 0.84), (0.84, 0.86), (0.86, 0.88), (0.88, 0.90)]
    bin_labels = ["[0.80,0.82)", "[0.82,0.84)", "[0.84,0.86)",
                  "[0.86,0.88)", "[0.88,0.90)"]
    bin_colors = [RED, RED, GRAY, BLUE, BLUE]   # straddles cutoff = gray

    bin_counts = []
    for lo, hi in bins:
        cnt = int(((df.query_confidence >= lo) & (df.query_confidence < hi)).sum())
        bin_counts.append(cnt)

    y_pos = np.arange(len(bins))
    bars = ax_bot.barh(y_pos, bin_counts, color=bin_colors, height=0.55,
                       edgecolor="white")

    # Annotate each bar with count (white bold inside)
    for bar, cnt in zip(bars, bin_counts):
        w = bar.get_width()
        ax_bot.text(
            w * 0.5, bar.get_y() + bar.get_height() / 2,
            f"n={cnt:,}",
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold",
        )

    ax_bot.set_yticks(y_pos)
    ax_bot.set_yticklabels(bin_labels, fontsize=9)
    ax_bot.set_xlabel("User count")
    ax_bot.set_title(
        "Counts roughly uniform across bins near 0.85 — no manipulation spike",
        fontsize=10.5, loc="left", pad=6,
    )
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.tick_params(axis="y", length=0)

    fig.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.97)
    save_figure(fig, "rdd_threshold_density.png")


def main() -> None:
    print("Generating Figure 1 (conceptual)...")
    make_figure_1_conceptual()
    print("Generating Figure 2 (data-driven density)...")
    make_figure_2_density()
    print("Done.")


if __name__ == "__main__":
    main()
