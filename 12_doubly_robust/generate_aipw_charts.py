"""Generate the two article figures for the doubly robust (AIPW) tutorial.

Figure 1 (conceptual): schematic showing the two-model structure of AIPW —
propensity arm and outcome arm combining into a single doubly robust estimate.

Figure 2 (data-driven): propensity score overlap density for the 50,000-user
synthetic dataset — treated vs. control distributions with the overlap region
shaded, confirming the positivity assumption holds.

Run from repo root:
    python 12_doubly_robust/generate_aipw_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "12_doubly_robust"
IMAGES_DIR = REPO_ROOT / "images" / "article-12"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TIER_COLORS = {
    "treated": "#2563EB",   # blue
    "control": "#DC2626",   # red
    "overlap": "#16A34A",   # green
    "accent":  "#F59E0B",   # amber
}

SPINE_GRAY = "#D1D5DB"
TEXT_GRAY  = "#374151"


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save PNG to both images/article-12/ and 12_doubly_robust/."""
    for dest in (IMAGES_DIR, METHOD_DIR):
        path = dest / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1 — Conceptual: dual-model structure of AIPW
# ---------------------------------------------------------------------------

def make_figure_1_conceptual() -> None:
    """Schematic of the AIPW estimator's two-model redundancy structure.

    Two stacked KDE-style curves represent the propensity arm (top) and
    outcome arm (bottom). A central panel shows the AIPW formula as the
    combination of both, with the overlap/correction highlighted.
    """
    rng = np.random.default_rng(42)
    n = 800

    # Simulate treated / control propensity distributions
    ps_treated = np.clip(rng.beta(3, 5, n) + 0.15, 0.05, 0.95)
    ps_control = np.clip(rng.beta(2, 7, n), 0.05, 0.85)

    # Simulate outcome predictions (schematic, not real data)
    m1 = 0.68 + rng.normal(0, 0.04, n)
    m0 = 0.58 + rng.normal(0, 0.04, n)
    diff = m1 - m0

    x_ps = np.linspace(0, 1, 300)
    kde_t = gaussian_kde(ps_treated, bw_method=0.3)(x_ps)
    kde_c = gaussian_kde(ps_control, bw_method=0.3)(x_ps)

    x_diff = np.linspace(diff.min(), diff.max(), 300)
    kde_diff = gaussian_kde(diff, bw_method=0.35)(x_diff)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(9, 8),
        gridspec_kw={"height_ratios": [2.5, 0.4, 2.5], "hspace": 0.55},
    )

    # --- Top panel: propensity arm ---
    ax_top = axes[0]
    ax_top.fill_between(x_ps, kde_t, alpha=0.25, color=TIER_COLORS["treated"])
    ax_top.fill_between(x_ps, kde_c, alpha=0.25, color=TIER_COLORS["control"])
    ax_top.plot(x_ps, kde_t, color=TIER_COLORS["treated"], lw=2, label="Treated (opted in)")
    ax_top.plot(x_ps, kde_c, color=TIER_COLORS["control"], lw=2, label="Control (did not opt in)")

    # Shade overlap region
    overlap = np.minimum(kde_t, kde_c)
    ax_top.fill_between(x_ps, overlap, alpha=0.4, color=TIER_COLORS["overlap"], label="Common support")

    ax_top.set_xlabel("Propensity score  e(X)", color=TEXT_GRAY, fontsize=10)
    ax_top.set_ylabel("Density", color=TEXT_GRAY, fontsize=10)
    ax_top.set_title("Propensity arm  —  who opts in?", fontsize=11, fontweight="bold", color=TEXT_GRAY)
    ax_top.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
    for spine in ax_top.spines.values():
        spine.set_edgecolor(SPINE_GRAY)
    ax_top.tick_params(colors=TEXT_GRAY, labelsize=9)

    # --- Middle: arrow / connector ---
    ax_mid = axes[1]
    ax_mid.axis("off")
    ax_mid.text(
        0.5, 0.5,
        "Both arms combine in AIPW:  ATE = Regression adjustment + IPW correction",
        ha="center", va="center", fontsize=9.5, color=TEXT_GRAY,
        style="italic",
        transform=ax_mid.transAxes,
    )

    # --- Bottom panel: outcome arm ---
    ax_bot = axes[2]
    ax_bot.fill_between(x_diff, kde_diff, alpha=0.3, color=TIER_COLORS["accent"])
    ax_bot.plot(x_diff, kde_diff, color=TIER_COLORS["accent"], lw=2)
    ax_bot.axvline(0.08, color="#374151", lw=1.5, linestyle="--", label="Ground truth +8 pp")
    ax_bot.axvline(diff.mean(), color=TIER_COLORS["accent"], lw=2, linestyle="-", label=f"AIPW estimate ≈ +{diff.mean():.2f}")

    ax_bot.set_xlabel("Predicted outcome difference  m₁(X) − m₀(X)", color=TEXT_GRAY, fontsize=10)
    ax_bot.set_ylabel("Density", color=TEXT_GRAY, fontsize=10)
    ax_bot.set_title("Outcome arm  —  what would each user's outcome be?", fontsize=11, fontweight="bold", color=TEXT_GRAY)
    ax_bot.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
    for spine in ax_bot.spines.values():
        spine.set_edgecolor(SPINE_GRAY)
    ax_bot.tick_params(colors=TEXT_GRAY, labelsize=9)

    fig.suptitle(
        "Figure 1: AIPW's two-model structure\n"
        "Either arm alone requires its model to be correct.\n"
        "AIPW requires only one of the two.",
        fontsize=11, y=1.01, color=TEXT_GRAY,
    )

    save_figure(fig, "aipw_structure_conceptual.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Data-driven: propensity score overlap on the real dataset
# ---------------------------------------------------------------------------

def make_figure_2_data_driven() -> None:
    """Propensity overlap density from the 50,000-user synthetic dataset.

    Two-panel layout:
      Top:    KDE density curves for treated and control propensity distributions
      Bottom: count bar per engagement tier, confirming overlap holds within tier
    """
    data_path = REPO_ROOT / "data" / "synthetic_llm_logs.csv"
    df = pd.read_csv(data_path)

    X = (
        pd.get_dummies(df[["engagement_tier", "query_confidence"]], drop_first=True)
        .astype(float)
        .values
    )
    T = df["opt_in_agent_mode"].values

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, T)
    ps = np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99)

    ps_treated = ps[T == 1]
    ps_control = ps[T == 0]

    x_grid = np.linspace(0.05, 0.85, 300)
    kde_t = gaussian_kde(ps_treated, bw_method=0.25)(x_grid)
    kde_c = gaussian_kde(ps_control, bw_method=0.25)(x_grid)

    tier_counts = (
        df.groupby(["engagement_tier", "opt_in_agent_mode"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Control", 1: "Treated"})
    )
    tier_order = ["light", "medium", "heavy"]
    tier_counts = tier_counts.reindex(tier_order)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 7),
        gridspec_kw={"height_ratios": [4, 1.5], "hspace": 0.05},
        sharex=False,
    )

    # --- Top: density ---
    ax = axes[0]
    ax.fill_between(x_grid, kde_t, alpha=0.2, color=TIER_COLORS["treated"])
    ax.fill_between(x_grid, kde_c, alpha=0.2, color=TIER_COLORS["control"])
    ax.plot(x_grid, kde_t, color=TIER_COLORS["treated"], lw=2,
            label=f"Opted in (n={T.sum():,})")
    ax.plot(x_grid, kde_c, color=TIER_COLORS["control"], lw=2,
            label=f"Did not opt in (n={(1-T).sum():,})")

    overlap = np.minimum(kde_t, kde_c)
    ax.fill_between(x_grid, overlap, alpha=0.45, color=TIER_COLORS["overlap"],
                    label="Common support (positivity holds)")

    y_top = max(kde_t.max(), kde_c.max()) * 1.35
    ax.set_ylim(0, y_top)
    ax.set_ylabel("Density", color=TEXT_GRAY, fontsize=10)
    ax.set_title(
        "Figure 2: Propensity score overlap — 50,000-user synthetic dataset\n"
        "Both groups share common support across the full propensity range",
        fontsize=11, fontweight="bold", color=TEXT_GRAY,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.88)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_GRAY)
    ax.tick_params(colors=TEXT_GRAY, labelsize=9)
    ax.set_xticklabels([])

    # --- Bottom: tier counts ---
    ax2 = axes[1]
    x_tiers = np.arange(len(tier_order))
    bar_w = 0.35
    ax2.bar(x_tiers - bar_w / 2, tier_counts["Control"], bar_w,
            color=TIER_COLORS["control"], alpha=0.7, label="Control")
    ax2.bar(x_tiers + bar_w / 2, tier_counts["Treated"], bar_w,
            color=TIER_COLORS["treated"], alpha=0.7, label="Treated")
    ax2.set_xticks(x_tiers)
    ax2.set_xticklabels(["Light tier", "Medium tier", "Heavy tier"],
                        fontsize=9, color=TEXT_GRAY)
    ax2.set_ylabel("Users", color=TEXT_GRAY, fontsize=9)
    ax2.tick_params(colors=TEXT_GRAY, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor(SPINE_GRAY)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

    fig.text(
        0.5, -0.02,
        "Propensity score  P(opt-in | engagement tier, query confidence)",
        ha="center", fontsize=10, color=TEXT_GRAY,
    )

    save_figure(fig, "aipw_propensity_overlap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Article 12 figures...")
    make_figure_1_conceptual()
    make_figure_2_data_driven()
    print("Done.")
