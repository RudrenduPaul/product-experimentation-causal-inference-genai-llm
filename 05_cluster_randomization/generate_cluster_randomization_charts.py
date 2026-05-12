"""Generate the two article figures for the cluster randomization tutorial.

Figure 1 (conceptual): a SUTVA violation schematic. Two rows of circles
represent workspaces. Users in a treated workspace (top row) get the AI
feature. Users in a control workspace (bottom row) who collaborate
cross-workspace leak treated-workspace output back into the control group,
contaminating the counterfactual.

Figure 2 (data-driven): the actual three-group outcome distribution on the
50,000-user dataset, showing pure-control (lowest), spillover-exposed
(middle, shifted by +0.20 min), and treated (highest, shifted by +0.80 min).

Run from repo root:
    python 05_cluster_randomization/generate_cluster_randomization_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Make the demo-module importable when running from the repo root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_randomization_demo import build_scenario

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "05_cluster_randomization"
IMAGES_DIR = REPO_ROOT / "images" / "article-5"

DIRECT_COLOR = "#C44E52"        # muted red — treated workspaces
SPILLOVER_COLOR = "#DD8452"     # orange — contaminated control users
CONTROL_COLOR = "#4C72B0"       # blue — pure control
GREY = "#888888"


def save_figure(fig, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    for path in (IMAGES_DIR / name, METHOD_DIR / name):
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")


def make_figure_1_conceptual() -> None:
    """Conceptual: two rows of workspace 'cells' with cross-row arrows
    illustrating cross-workspace spillover that breaks SUTVA."""
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    # Treated row (top) and control row (bottom). 6 workspaces per row.
    n_per_row = 6
    x_positions = np.linspace(0.8, 9.2, n_per_row)
    y_treated = 3.3
    y_control = 1.3

    # Draw each workspace as a rounded rectangle with 8 user dots inside.
    for x, is_treated in [(x, True) for x in x_positions] + [(x, False) for x in x_positions]:
        y_center = y_treated if is_treated else y_control
        color = DIRECT_COLOR if is_treated else CONTROL_COLOR
        face = "#FBE5E4" if is_treated else "#E2ECF6"
        rect = patches.FancyBboxPatch(
            (x - 0.55, y_center - 0.45), 1.10, 0.90,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=color, facecolor=face, zorder=1,
        )
        ax.add_patch(rect)
        # 8 user dots in a 2x4 grid inside each workspace
        for i, (dx, dy) in enumerate([
            (-0.35, 0.15), (-0.15, 0.15), (0.05, 0.15), (0.25, 0.15),
            (-0.35, -0.15), (-0.15, -0.15), (0.05, -0.15), (0.25, -0.15),
        ]):
            # In control workspaces, dots 0-1 are "spillover-exposed" (orange),
            # the rest are pure control (blue).
            if is_treated:
                dot_color = DIRECT_COLOR
            else:
                dot_color = SPILLOVER_COLOR if i < 2 else CONTROL_COLOR
            ax.scatter(x + dx, y_center + dy, s=26, color=dot_color, zorder=2)

    # Row labels on the left
    ax.text(-0.1, y_treated, "Treated\nworkspaces\n(feature ON)",
            ha="right", va="center", fontsize=10.5,
            color=DIRECT_COLOR, fontweight="bold")
    ax.text(-0.1, y_control, "Control\nworkspaces\n(feature OFF)",
            ha="right", va="center", fontsize=10.5,
            color=CONTROL_COLOR, fontweight="bold")

    # Spillover arrows from 3 treated workspaces down to the orange dots
    # in control workspaces on the same x-column.
    arrow_pairs = [(0, 0), (2, 2), (4, 4)]
    for i_from, i_to in arrow_pairs:
        ax.annotate(
            "",
            xy=(x_positions[i_to] - 0.32, y_control + 0.18),
            xytext=(x_positions[i_from] - 0.32, y_treated - 0.18),
            arrowprops=dict(
                arrowstyle="->", color=SPILLOVER_COLOR,
                lw=1.6, alpha=0.85,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=3,
        )

    # Caption label on middle arrow
    ax.text(5.0, 2.3,
            "shared Slack / Docs / PRs\ncarry AI artifacts\nacross workspaces",
            ha="center", va="center", fontsize=9.0,
            color=SPILLOVER_COLOR, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=SPILLOVER_COLOR, linewidth=0.8))

    # Title and subtitle as suptitle + subtitle pattern
    ax.set_title(
        "SUTVA breaks when control users collaborate with treated users",
        fontsize=13.0, loc="left", pad=12, fontweight="bold",
    )

    # Legend strip at bottom
    legend_y = 0.15
    for x_label, color, label in [
        (1.0, DIRECT_COLOR, "treated user"),
        (4.0, SPILLOVER_COLOR, "control user exposed to spillover"),
        (7.6, CONTROL_COLOR, "pure-control user"),
    ]:
        ax.scatter(x_label, legend_y, s=70, color=color, zorder=2)
        ax.text(x_label + 0.3, legend_y, label,
                ha="left", va="center", fontsize=10.0)

    ax.set_xlim(-1.5, 10.3)
    ax.set_ylim(-0.4, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    save_figure(fig, "cluster_randomization_conceptual.png")
    plt.close(fig)


def make_figure_2_data_driven() -> None:
    """Data-driven: outcome distribution by exposure group on the 50k dataset.
    Top panel: KDE curves for pure_control, spillover, direct.
    Bottom panel: horizontal bar of counts per exposure group.
    """
    from scipy.stats import gaussian_kde

    data_path = REPO_ROOT / "data" / "synthetic_llm_logs.csv"
    df = pd.read_csv(data_path)
    df = build_scenario(df)

    groups = {
        "pure_control": ("Pure control (no exposure)", CONTROL_COLOR),
        "spillover":    ("Control, exposed to spillover", SPILLOVER_COLOR),
        "direct":       ("Treated (direct effect)",      DIRECT_COLOR),
    }

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 6.4),
        gridspec_kw={"height_ratios": [5.0, 1.4], "hspace": 0.55},
    )

    # TOP: KDE curves for session_minutes_obs per group
    lo, hi = 3.5, 7.0
    grid = np.linspace(lo, hi, 400)
    peak = 0.0
    for key, (label, color) in groups.items():
        values = df.loc[df["exposure"] == key, "session_minutes_obs"].values
        kde = gaussian_kde(values, bw_method=0.25)
        dens = kde(grid)
        peak = max(peak, dens.max())
        ax_top.fill_between(grid, dens, alpha=0.38, color=color, label=label)
        ax_top.plot(grid, dens, color=color, lw=1.6)

    # Ground-truth means as vertical reference lines
    for key, (_, color) in groups.items():
        mean_val = df.loc[df["exposure"] == key, "session_minutes_obs"].mean()
        ax_top.axvline(mean_val, color=color, lw=1.2, ls="--", alpha=0.75)
        ax_top.text(mean_val, peak * 1.26,
                    f"mean\n{mean_val:.2f}",
                    ha="center", va="bottom", fontsize=9.0, color=color)

    ax_top.set_ylim(0, peak * 1.55)
    ax_top.set_xlim(lo, hi)
    ax_top.set_xlabel("Observed session minutes (per user)", fontsize=10.5)
    ax_top.set_ylabel("Density", fontsize=10.5)
    ax_top.set_title(
        "Three outcome distributions reveal the spillover contamination",
        fontsize=12.5, loc="left", fontweight="bold",
    )
    ax_top.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # BOTTOM: count bars per exposure group
    counts = [
        (df["exposure"] == "pure_control").sum(),
        (df["exposure"] == "spillover").sum(),
        (df["exposure"] == "direct").sum(),
    ]
    labels = ["pure_control", "spillover", "direct"]
    colors = [CONTROL_COLOR, SPILLOVER_COLOR, DIRECT_COLOR]

    bars = ax_bot.barh(labels[::-1], counts[::-1], color=colors[::-1], alpha=0.85)
    for bar, n in zip(bars, counts[::-1]):
        ax_bot.text(bar.get_width() + 400, bar.get_y() + bar.get_height() / 2,
                    f"n = {n:,}", va="center", ha="left", fontsize=10.0)

    ax_bot.set_xlim(0, max(counts) * 1.25)
    ax_bot.set_xlabel("Users per exposure group", fontsize=10.0)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "cluster_randomization_density.png")
    plt.close(fig)


def main() -> None:
    make_figure_1_conceptual()
    make_figure_2_data_driven()


if __name__ == "__main__":
    main()
