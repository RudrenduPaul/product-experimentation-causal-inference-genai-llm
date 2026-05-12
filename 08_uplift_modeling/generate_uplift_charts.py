"""Generate the two article figures for the uplift modeling tutorial.

Figure 1 (conceptual): three engagement-tier groups each showing a different
treatment effect size, demonstrating that an ATE hides within-group
heterogeneity.

Figure 2 (data-driven): actual predicted CATE distributions per engagement
tier from the 50,000-user synthetic dataset, confirming that light users
have the highest treatment effect and heavy users the lowest.

Run from repo root:
    python 08_uplift_modeling/generate_uplift_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "08_uplift_modeling"
IMAGES_DIR = REPO_ROOT / "images" / "article-8"

TIER_COLORS = {
    "light": "#4C72B0",
    "medium": "#DD8452",
    "heavy": "#55A868",
}
TIER_ORDER = ["light", "medium", "heavy"]


def save_figure(fig, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for target in (IMAGES_DIR / name, METHOD_DIR / name):
        fig.savefig(target, dpi=150, bbox_inches="tight")
        print(f"  wrote {target.relative_to(REPO_ROOT)}")
    plt.close(fig)


def make_figure_1_conceptual() -> None:
    """Conceptual figure: three tiers with different treatment effect sizes.

    Shows control and treated outcome distributions for light, medium, and
    heavy users. The gap between treated and control shrinks from light to
    heavy, illustrating that the ATE (a single number) masks the spread.
    Two-panel layout: top shows KDE curves; bottom shows ATE vs. per-tier
    CATE bars as a reference.
    """
    rng = np.random.default_rng(42)

    # Schematic outcome distributions per tier (not from real data)
    # Control: lower completion rates; Treated: higher, but varying gap
    tier_params = {
        "light":  dict(mu_c=0.46, sigma_c=0.12, mu_t=0.56, sigma_t=0.12),
        "medium": dict(mu_c=0.67, sigma_c=0.10, mu_t=0.74, sigma_t=0.10),
        "heavy":  dict(mu_c=0.82, sigma_c=0.08, mu_t=0.89, sigma_t=0.08),
    }
    n_schematic = 3000

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(11.0, 6.8),
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.55},
    )

    grid = np.linspace(0.0, 1.2, 500)
    max_y = 0.0
    for tier in TIER_ORDER:
        p = tier_params[tier]
        c_vals = np.clip(rng.normal(p["mu_c"], p["sigma_c"], n_schematic), 0, 1)
        t_vals = np.clip(rng.normal(p["mu_t"], p["sigma_t"], n_schematic), 0, 1)
        kde_c = gaussian_kde(c_vals, bw_method=0.25)(grid)
        kde_t = gaussian_kde(t_vals, bw_method=0.25)(grid)
        max_y = max(max_y, kde_c.max(), kde_t.max())

    for i, tier in enumerate(TIER_ORDER):
        p = tier_params[tier]
        c_vals = np.clip(rng.normal(p["mu_c"], p["sigma_c"], n_schematic), 0, 1)
        t_vals = np.clip(rng.normal(p["mu_t"], p["sigma_t"], n_schematic), 0, 1)
        kde_c = gaussian_kde(c_vals, bw_method=0.25)(grid)
        kde_t = gaussian_kde(t_vals, bw_method=0.25)(grid)

        color = TIER_COLORS[tier]
        label_c = f"{tier.capitalize()} – no feature" if i == 0 else None
        label_t = f"{tier.capitalize()} – feature on" if i == 0 else None

        ax_top.fill_between(grid, kde_c, alpha=0.18, color=color)
        ax_top.fill_between(grid, kde_t, alpha=0.35, color=color)
        ax_top.plot(grid, kde_c, color=color, lw=1.3, ls="--", alpha=0.8)
        ax_top.plot(grid, kde_t, color=color, lw=1.8, alpha=1.0)

        # Label the gap with a bracket between the two peak means
        ax_top.annotate(
            "",
            xy=(p["mu_t"], max_y * (0.82 - 0.22 * i)),
            xytext=(p["mu_c"], max_y * (0.82 - 0.22 * i)),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.4),
        )
        cate_label = p["mu_t"] - p["mu_c"]
        ax_top.text(
            (p["mu_c"] + p["mu_t"]) / 2,
            max_y * (0.84 - 0.22 * i),
            f"{tier.capitalize()}  CATE ≈ {cate_label:+.2f}",
            ha="center", va="bottom", fontsize=9.5, color=color,
            fontweight="bold",
        )

    # Legend via proxy artists
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="gray", lw=1.3, ls="--", label="No feature (control)"),
        Line2D([0], [0], color="gray", lw=1.8, label="Feature on (treated)"),
    ]
    ax_top.legend(handles=legend_els, frameon=False, loc="upper right", fontsize=9.5)
    ax_top.set_xlim(0.15, 1.05)
    ax_top.set_ylim(0, max_y * 1.05)
    ax_top.set_xlabel("Task completion rate")
    ax_top.set_ylabel("Density")
    ax_top.set_title(
        "Heterogeneous treatment effects: each tier responds differently to the AI feature",
        fontsize=12.5, loc="left",
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Bottom: ATE vs. per-tier CATE bars
    tier_cates = [p["mu_t"] - p["mu_c"] for p in tier_params.values()]
    ate_approx = np.mean(tier_cates)
    x_pos = np.arange(len(TIER_ORDER))
    bars = ax_bot.bar(
        x_pos, tier_cates,
        color=[TIER_COLORS[t] for t in TIER_ORDER],
        width=0.5, alpha=0.85,
    )
    ax_bot.axhline(ate_approx, color="#333333", lw=1.8, ls="--",
                   label=f"ATE ≈ {ate_approx:+.2f}")
    for bar, val in zip(bars, tier_cates):
        ax_bot.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.003, f"{val:+.2f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_bot.legend(frameon=False, fontsize=9.5)
    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels([t.capitalize() for t in TIER_ORDER])
    ax_bot.set_ylabel("CATE")
    ax_bot.set_title(
        "ATE collapses the spread — the average hides that light users benefit most",
        fontsize=10.5, loc="left",
    )
    ax_bot.set_ylim(0, max(tier_cates) * 1.35)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, "uplift_cate_conceptual.png")


def make_figure_2_data_driven() -> None:
    """Actual predicted CATE distributions from the 50k synthetic dataset.

    Two-panel layout:
      (top)    Smooth KDE of predicted CATE per engagement tier — confirms
               light > medium > heavy ordering.
      (bottom) Mean CATE per tier with 95% bootstrap CI bars, plus the
               naive ATE as a reference line.
    """
    df = pd.read_csv(REPO_ROOT / "data" / "synthetic_llm_logs.csv")

    X_full = pd.get_dummies(
        df[["query_confidence", "engagement_tier"]], drop_first=False
    ).astype(float)
    X_all = X_full.values
    treated_mask = df.opt_in_agent_mode == 1
    m1 = LinearRegression().fit(X_all[treated_mask], df[treated_mask].task_completed.values)
    m0 = LinearRegression().fit(X_all[~treated_mask], df[~treated_mask].task_completed.values)
    df["cate"] = m1.predict(X_all) - m0.predict(X_all)

    naive_ate = (
        df[df.opt_in_agent_mode == 1].task_completed.mean()
        - df[df.opt_in_agent_mode == 0].task_completed.mean()
    )

    # Bootstrap CIs for mean CATE per tier (200 reps, seed=7 for speed in chart generation)
    rng = np.random.default_rng(7)
    n = len(df)
    tier_boot = {t: [] for t in TIER_ORDER}
    for _ in range(200):
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        X_b = X_all[idx]
        m1_b = LinearRegression().fit(X_b[treated_mask[idx].values], df_b[treated_mask[idx].values].task_completed.values)
        m0_b = LinearRegression().fit(X_b[~treated_mask[idx].values], df_b[~treated_mask[idx].values].task_completed.values)
        cate_b = m1_b.predict(X_b) - m0_b.predict(X_b)
        df_b["cate"] = cate_b
        for tier in TIER_ORDER:
            tier_boot[tier].append(df_b[df_b.engagement_tier == tier].cate.mean())

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 7.0),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.60},
    )

    # --- TOP: smooth KDE of CATE per tier ---
    grid = np.linspace(0.03, 0.12, 400)
    max_dens = 0.0
    for tier in TIER_ORDER:
        vals = df[df.engagement_tier == tier].cate.values
        dens = gaussian_kde(vals, bw_method=0.25)(grid)
        max_dens = max(max_dens, dens.max())

    for tier in TIER_ORDER:
        vals = df[df.engagement_tier == tier].cate.values
        dens = gaussian_kde(vals, bw_method=0.25)(grid)
        color = TIER_COLORS[tier]
        mean_cate = vals.mean()
        ax_top.fill_between(grid, dens, alpha=0.35, color=color)
        ax_top.plot(grid, dens, color=color, lw=1.8,
                    label=f"{tier.capitalize()}  (mean CATE = {mean_cate:+.4f})")
        ax_top.axvline(mean_cate, color=color, lw=1.2, ls="--", alpha=0.8)

    ax_top.axvline(0.085, color="#888888", lw=1.5, ls=":",
                   label="Rollout threshold = 0.085")
    ax_top.set_xlim(0.03, 0.12)
    ax_top.set_ylim(0, max_dens * 1.30)
    ax_top.set_xlabel("Predicted CATE (T-learner)")
    ax_top.set_ylabel("Density")
    ax_top.set_title(
        "Predicted CATE distributions by engagement tier — 50,000-user synthetic dataset",
        fontsize=12.5, loc="left",
    )
    ax_top.legend(frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9.5)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # --- BOTTOM: mean CATE per tier with 95% CI ---
    tier_means = [df[df.engagement_tier == t].cate.mean() for t in TIER_ORDER]
    tier_ci_lo = [float(np.percentile(tier_boot[t], 2.5)) for t in TIER_ORDER]
    tier_ci_hi = [float(np.percentile(tier_boot[t], 97.5)) for t in TIER_ORDER]
    x_pos = np.arange(len(TIER_ORDER))
    yerr_lo = [m - lo for m, lo in zip(tier_means, tier_ci_lo)]
    yerr_hi = [hi - m for m, hi in zip(tier_means, tier_ci_hi)]

    ax_bot.bar(x_pos, tier_means,
               color=[TIER_COLORS[t] for t in TIER_ORDER],
               width=0.5, alpha=0.85)
    ax_bot.errorbar(x_pos, tier_means,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="#333333", lw=1.8, capsize=5)
    ax_bot.axhline(naive_ate, color="#888888", lw=1.5, ls="--",
                   label=f"Naive ATE = {naive_ate:+.4f} (confounded)")
    ax_bot.axhline(0.08, color="#CC5500", lw=1.2, ls=":",
                   label="Ground truth ≈ +0.08")
    for x, m, lo, hi in zip(x_pos, tier_means, tier_ci_lo, tier_ci_hi):
        ax_bot.text(x, hi + 0.003, f"{m:+.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax_bot.legend(frameon=False, fontsize=9, loc="upper right")
    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels([t.capitalize() for t in TIER_ORDER])
    ax_bot.set_ylabel("Mean predicted CATE")
    ax_bot.set_title(
        "Mean CATE with 95% CI: light users respond most, heavy users least",
        fontsize=10.5, loc="left",
    )
    ax_bot.set_ylim(0, naive_ate * 1.15)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, "uplift_cate_distribution.png")


def main() -> None:
    print("Generating Figure 1 (conceptual)...")
    make_figure_1_conceptual()
    print("Generating Figure 2 (data-driven)...")
    make_figure_2_data_driven()
    print("Done.")


if __name__ == "__main__":
    main()
