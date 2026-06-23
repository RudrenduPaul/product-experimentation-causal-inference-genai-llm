"""Generate the two article figures for the regression causal inference tutorial.

Figure 1 (conceptual): Under randomization, covariate distributions are identical
across treatment and control arms. Under selection bias, they diverge. Two-panel
comparison showing why randomization makes OLS causal.

Figure 2 (data-driven): Actual query_confidence density by treatment arm on the
50,000-user synthetic dataset, confirming that covariate balance holds.

Run from repo root:
    python 09_regression/generate_regression_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "09_regression"
IMAGES_DIR = REPO_ROOT / "images" / "article-9"

CONTROL_COLOR = "#4C72B0"
TREATMENT_COLOR = "#C44E52"
GRID_COLOR = "#e8e8e8"

RNG = np.random.default_rng(42)


def save_figure(fig: plt.Figure, name: str) -> None:
    for dest in [IMAGES_DIR, METHOD_DIR]:
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {name}")


def make_figure_1_conceptual() -> None:
    """Conceptual two-panel: randomized (overlapping) vs biased (separated) distributions."""

    x = np.linspace(0, 1, 500)

    # Randomized: both arms drawn from same distribution
    rand_control = np.clip(RNG.beta(4, 4, 6000), 0.01, 0.99)
    rand_treat   = np.clip(RNG.beta(4, 4, 6000), 0.01, 0.99)

    # Biased: treated users have higher covariate values (selection on engagement)
    bias_control = np.clip(RNG.beta(3, 6, 6000), 0.01, 0.99)
    bias_treat   = np.clip(RNG.beta(6, 3, 6000), 0.01, 0.99)

    bw = 0.10

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    fig.patch.set_facecolor("#fafafa")

    titles = ["Randomized experiment\n(OLS is causal)", "Observational data\n(OLS is biased)"]
    datasets = [(rand_control, rand_treat), (bias_control, bias_treat)]
    subtexts = [
        "Covariate distributions overlap — treatment\nassignment is independent of user characteristics.",
        "Treated users have systematically higher\nconfidence scores — confounding is present.",
    ]

    for ax, title, (ctrl, trt), subtext in zip(axes, titles, datasets, subtexts):
        ax.set_facecolor("#fafafa")
        ax.set_xlim(0, 1)

        kde_ctrl = gaussian_kde(ctrl, bw_method=bw)
        kde_trt  = gaussian_kde(trt,  bw_method=bw)

        y_ctrl = kde_ctrl(x)
        y_trt  = kde_trt(x)
        y_top  = max(y_ctrl.max(), y_trt.max()) * 1.35

        ax.fill_between(x, y_ctrl, alpha=0.22, color=CONTROL_COLOR)
        ax.fill_between(x, y_trt,  alpha=0.22, color=TREATMENT_COLOR)
        ax.plot(x, y_ctrl, color=CONTROL_COLOR,   lw=2.0, label="Control")
        ax.plot(x, y_trt,  color=TREATMENT_COLOR, lw=2.0, label="Treatment")

        ax.set_ylim(0, y_top)
        ax.set_xlabel("Covariate value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper center", fontsize=10, framealpha=0.8)
        ax.text(
            0.5, -0.22, subtext,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9.5, color="#444444",
            style="italic",
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Why OLS gives causal estimates under randomization",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, "regression_balance_conceptual.png")
    plt.close(fig)


def make_figure_2_data_driven() -> None:
    """Data-driven: query_confidence density by arm on the 50k synthetic dataset."""

    csv_path = REPO_ROOT / "data" / "synthetic_llm_logs.csv"
    df = pd.read_csv(csv_path)

    ctrl = df.loc[df.prompt_variant == 0, "query_confidence"].values
    trt  = df.loc[df.prompt_variant == 1, "query_confidence"].values

    x = np.linspace(0, 1, 500)
    bw = 0.11

    kde_ctrl = gaussian_kde(ctrl, bw_method=bw)
    kde_trt  = gaussian_kde(trt,  bw_method=bw)
    y_ctrl   = kde_ctrl(x)
    y_trt    = kde_trt(x)
    y_top    = max(y_ctrl.max(), y_trt.max()) * 1.40

    n_ctrl = (df.prompt_variant == 0).sum()
    n_trt  = (df.prompt_variant == 1).sum()

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 5.6),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.0},
        sharex=True,
    )
    fig.patch.set_facecolor("#fafafa")

    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#fafafa")

    # Top panel — density curves
    ax_top.fill_between(x, y_ctrl, alpha=0.22, color=CONTROL_COLOR)
    ax_top.fill_between(x, y_trt,  alpha=0.22, color=TREATMENT_COLOR)
    ax_top.plot(x, y_ctrl, color=CONTROL_COLOR,   lw=2.0, label=f"Control  (n={n_ctrl:,})")
    ax_top.plot(x, y_trt,  color=TREATMENT_COLOR, lw=2.0, label=f"Treatment (n={n_trt:,})")

    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, y_top)
    ax_top.set_ylabel("Density", fontsize=11)
    ax_top.set_title(
        "query_confidence by treatment arm — 50,000-user synthetic dataset",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_top.legend(loc="upper left", fontsize=10, framealpha=0.85)
    ax_top.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax_top.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_top.spines[spine].set_visible(False)
    ax_top.tick_params(bottom=False)

    # Mean difference annotation in a quiet zone
    mean_diff = trt.mean() - ctrl.mean()
    ax_top.text(
        0.98, 0.94,
        f"Mean diff: {mean_diff:+.4f}",
        transform=ax_top.transAxes,
        ha="right", va="top",
        fontsize=10, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )

    # Bottom panel — bracket annotation strip
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis("off")

    bracket_y = 0.55
    label_y   = 0.05
    BRACKET_COLOR = "#666666"
    for x0, x1, label, color in [
        (0.0,  0.33, "Low confidence", BRACKET_COLOR),
        (0.33, 0.67, "Mid confidence", BRACKET_COLOR),
        (0.67, 1.0,  "High confidence", BRACKET_COLOR),
    ]:
        xm = (x0 + x1) / 2
        ax_bot.plot([x0 + 0.01, x1 - 0.01], [bracket_y, bracket_y], color=color, lw=1.8)
        ax_bot.plot([x0 + 0.01, x0 + 0.01], [bracket_y - 0.12, bracket_y], color=color, lw=1.8)
        ax_bot.plot([x1 - 0.01, x1 - 0.01], [bracket_y - 0.12, bracket_y], color=color, lw=1.8)
        ax_bot.text(xm, label_y, label, ha="center", va="bottom",
                    fontsize=9.5, color=color, fontweight="bold")

    ax_bot.set_xlabel("query_confidence", fontsize=11)

    caption = (
        "Figure 2: query_confidence density by treatment arm. The two curves overlap almost exactly,\n"
        "confirming that hash-based random assignment produced balance on this covariate."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=9.5, color="#555555", style="italic")

    fig.tight_layout()
    save_figure(fig, "regression_balance_density.png")
    plt.close(fig)


if __name__ == "__main__":
    make_figure_1_conceptual()
    make_figure_2_data_driven()
    print("Done.")
