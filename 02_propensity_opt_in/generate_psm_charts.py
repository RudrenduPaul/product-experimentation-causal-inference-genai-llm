"""Generate the two article figures for the propensity score tutorial.

Figure 1 (conceptual): two overlapping propensity score distributions showing
selection bias and the region of common support.

Figure 2 (data-driven): the actual propensity score overlap on the synthetic
dataset, with shaded common-support region.

Run from repo root:
    python 02_propensity_opt_in/generate_psm_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "02_propensity_opt_in"
IMAGES_DIR = REPO_ROOT / "images" / "article-2"


def make_figure_1_conceptual() -> None:
    """Conceptual: treated skews right, control skews left; overlap shaded.

    Two stacked panels with zero overlap between text and plot data:
      (top)    smooth KDE curves with only a legend in the corner
      (bottom) a narrow label strip under the x-axis with three region
               brackets (control-heavy, common support, treatment-heavy)
    """
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(7)
    treated = np.clip(rng.beta(6, 3, size=5000), 0.02, 0.98)
    control = np.clip(rng.beta(3, 6, size=5000), 0.02, 0.98)

    lo, hi = 0.15, 0.85  # common support region
    control_color = "#4C72B0"
    treated_color = "#C44E52"

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 5.6),
        gridspec_kw={"height_ratios": [5.0, 1.0], "hspace": 0.0},
        sharex=True,
    )

    # --- TOP: smooth KDE curves ---
    grid = np.linspace(0, 1, 400)
    dens_c = gaussian_kde(control, bw_method=0.35)(grid)
    dens_t = gaussian_kde(treated, bw_method=0.35)(grid)

    ax_top.fill_between(grid, dens_c, alpha=0.45, color=control_color,
                        label="Did not opt in")
    ax_top.fill_between(grid, dens_t, alpha=0.45, color=treated_color,
                        label="Opted in")
    ax_top.plot(grid, dens_c, color=control_color, lw=1.5)
    ax_top.plot(grid, dens_t, color=treated_color, lw=1.5)
    ax_top.axvspan(lo, hi, color="#EFEFEF", alpha=0.5, zorder=0)

    y_top = max(dens_c.max(), dens_t.max()) * 1.20
    ax_top.set_ylim(0, y_top)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylabel("Density")
    ax_top.set_title(
        "Propensity score distributions separate treated and control groups",
        fontsize=12.5, loc="left",
    )
    ax_top.legend(frameon=False, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, 1.0), fontsize=10)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    # --- BOTTOM: region label strip ---
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.set_yticks([])
    for side in ("top", "left", "right"):
        ax_bot.spines[side].set_visible(False)
    ax_bot.spines["bottom"].set_visible(True)

    # Three region brackets with labels beneath
    regions = [
        (0.00, lo, "Control-heavy region\n(few treated users)",
         control_color, 0.5),
        (lo, hi, "Region of common support\n(both groups present)",
         "#555555", 0.7),
        (hi, 1.00, "Treatment-heavy region\n(few controls)",
         treated_color, 0.5),
    ]
    for x0, x1, label, color, alpha in regions:
        mid = (x0 + x1) / 2
        # Horizontal bracket line
        ax_bot.annotate(
            "", xy=(x0 + 0.005, 0.78), xytext=(x1 - 0.005, 0.78),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.2,
                            alpha=alpha),
        )
        ax_bot.plot([x0 + 0.005, x0 + 0.005], [0.70, 0.86],
                    color=color, lw=1.2, alpha=alpha)
        ax_bot.plot([x1 - 0.005, x1 - 0.005], [0.70, 0.86],
                    color=color, lw=1.2, alpha=alpha)
        ax_bot.text(mid, 0.35, label, ha="center", va="center",
                    fontsize=9.5, color="#333333")

    ax_bot.set_xlabel(
        "Propensity score  (predicted probability of opting in)",
        fontsize=10,
    )

    fig.tight_layout()
    save_figure(fig, "psm_overlap_conceptual.png")


def make_figure_2_data_driven() -> None:
    """Actual propensity overlap on the synthetic dataset.

    Two-panel layout because the positivity story has two parts:
      (top)    smooth KDE curves of the propensity distributions
      (bottom) stacked horizontal count bars per engagement tier showing
               that BOTH groups exist at every tier (the visual positivity
               check that matters for downstream weighting).
    """
    from scipy.stats import gaussian_kde

    df = pd.read_csv(REPO_ROOT / "data" / "synthetic_llm_logs.csv")

    X = pd.get_dummies(
        df[["engagement_tier", "query_confidence"]], drop_first=True
    ).astype(float)
    model = LogisticRegression(max_iter=1000).fit(X, df.opt_in_agent_mode)
    df["propensity"] = model.predict_proba(X)[:, 1]

    treated_ps = df[df.opt_in_agent_mode == 1].propensity.values
    control_ps = df[df.opt_in_agent_mode == 0].propensity.values

    common_lo = max(treated_ps.min(), control_ps.min())
    common_hi = min(treated_ps.max(), control_ps.max())

    tier_colors = {
        "light": "#8FA9C7",
        "medium": "#B8A886",
        "heavy": "#C88B8B",
    }
    tier_order = ["light", "medium", "heavy"]

    control_color = "#4C72B0"
    treated_color = "#C44E52"

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 6.6),
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.35},
    )

    # --- TOP: smooth KDE curves ---
    grid = np.linspace(0.05, 0.75, 400)
    kde_c = gaussian_kde(control_ps, bw_method=0.25)
    kde_t = gaussian_kde(treated_ps, bw_method=0.25)
    dens_c = kde_c(grid)
    dens_t = kde_t(grid)

    ax_top.fill_between(
        grid, dens_c, alpha=0.45, color=control_color,
        label=f"Did not opt in  (n = {len(control_ps):,})",
    )
    ax_top.fill_between(
        grid, dens_t, alpha=0.45, color=treated_color,
        label=f"Opted in  (n = {len(treated_ps):,})",
    )
    ax_top.plot(grid, dens_c, color=control_color, lw=1.4)
    ax_top.plot(grid, dens_t, color=treated_color, lw=1.4)

    # Mark the common support band
    ax_top.axvspan(
        common_lo, common_hi, color="#EFEFEF", alpha=0.5, zorder=0,
    )
    y_top = max(dens_c.max(), dens_t.max()) * 1.25
    ax_top.set_ylim(0, y_top)

    # Mark the three engagement-tier cluster centers
    for tier in tier_order:
        tier_df = df[df.engagement_tier == tier]
        center = tier_df.propensity.mean()
        ax_top.axvline(
            center, color=tier_colors[tier], lw=1.2, ls="--", alpha=0.9,
            zorder=1,
        )
        ax_top.text(
            center, y_top * 0.96,
            f"{tier}\n(p ≈ {center:.2f})",
            ha="center", va="top", fontsize=9, color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=tier_colors[tier], lw=0.8),
        )

    # Common support label placed in the low-density valley near p = 0.5
    ax_top.text(
        0.50, y_top * 0.55,
        f"Common support: [{common_lo:.2f}, {common_hi:.2f}]\n"
        "both groups present across this range",
        ha="center", va="center", fontsize=9.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB", lw=0.6),
    )

    ax_top.set_xlabel("Propensity score  (predicted probability of opting in)")
    ax_top.set_ylabel("Density")
    ax_top.set_title(
        "Propensity score overlap on the 50,000-user synthetic dataset",
        fontsize=12.5, loc="left",
    )
    ax_top.set_xlim(0.05, 0.75)
    ax_top.legend(frameon=False, loc="upper left", fontsize=9.5,
                  bbox_to_anchor=(0.01, 0.75))
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # --- BOTTOM: stacked count bars per tier ---
    tier_totals = []
    for tier in tier_order:
        tier_df = df[df.engagement_tier == tier]
        tier_totals.append({
            "tier": tier,
            "treated": int((tier_df.opt_in_agent_mode == 1).sum()),
            "control": int((tier_df.opt_in_agent_mode == 0).sum()),
        })

    y_pos = np.arange(len(tier_order))[::-1]
    max_total = max(t["treated"] + t["control"] for t in tier_totals)
    for y, row in zip(y_pos, tier_totals):
        ax_bot.barh(
            y, row["control"], color=control_color, alpha=0.75,
            edgecolor="white", height=0.6,
        )
        ax_bot.barh(
            y, row["treated"], left=row["control"], color=treated_color,
            alpha=0.75, edgecolor="white", height=0.6,
        )
        # In-bar count labels
        ax_bot.text(
            row["control"] / 2, y,
            f"{row['control']:,}",
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold",
        )
        ax_bot.text(
            row["control"] + row["treated"] / 2, y,
            f"{row['treated']:,}",
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold",
        )
        # Opt-in rate annotation on the right
        total = row["control"] + row["treated"]
        rate = row["treated"] / total
        ax_bot.text(
            total + max_total * 0.015, y,
            f"{rate:.0%} opt-in",
            ha="left", va="center", fontsize=9, color="#333333",
        )

    ax_bot.set_yticks(y_pos)
    ax_bot.set_yticklabels([r["tier"].capitalize() for r in tier_totals])
    ax_bot.set_xlim(0, max_total * 1.18)
    ax_bot.set_xlabel("User count")
    ax_bot.set_title(
        "Both groups exist at every engagement tier → positivity holds",
        fontsize=10.5, loc="left", pad=6,
    )
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.tick_params(axis="y", length=0)

    fig.tight_layout()
    save_figure(fig, "psm_overlap_density.png")


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
