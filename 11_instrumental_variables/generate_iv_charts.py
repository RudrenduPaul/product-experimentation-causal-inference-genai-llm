"""
Generate the two article figures for the instrumental variable tutorial.

Figure 1 (conceptual): causal DAG showing the IV structure — instrument,
endogenous treatment, unobserved confounder, and outcome.

Figure 2 (data-driven): first-stage relationship on the synthetic dataset —
routing rates by fallback group and the OLS vs 2SLS estimate comparison.

Run from repo root:
    python 11_instrumental_variables/generate_iv_charts.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Resolve paths relative to this file (works from any working directory)
REPO_ROOT   = Path(__file__).resolve().parent.parent
METHOD_DIR  = REPO_ROOT / "11_instrumental_variables"
IMAGES_DIR  = REPO_ROOT / "images" / "article-11"
DATA_PATH   = REPO_ROOT / "data" / "synthetic_llm_logs.csv"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# --- Palette (matches series style) ---
BLUE   = "#4C72B0"
RED    = "#C44E52"
GREEN  = "#55A868"
GRAY   = "#8c8c8c"
ORANGE = "#DD8452"
BG     = "#FAFAFA"


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save PNG to both images/article-11/ and 11_instrumental_variables/."""
    for dest in [IMAGES_DIR / name, METHOD_DIR / name]:
        fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {name}")


# ---------------------------------------------------------------------------
# Figure 1 — Conceptual IV DAG
# ---------------------------------------------------------------------------

def make_figure_1_conceptual() -> None:
    """
    Draw the IV causal graph.

    Nodes: Z (instrument), D (treatment), Y (outcome), U (unobserved confounder)
    Arrows:
      Z → D           (relevance — solid blue)
      D → Y           (causal effect of interest — solid blue)
      U → D, U → Y    (confounder paths — dashed red)
      X → D           (observed covariate — solid gray, thin)
    The exclusion restriction is shown by the ABSENCE of a Z → Y arrow.
    """
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Node positions: Z, D, Y in a horizontal line; U above; X below D
    nodes = {
        "Z": (1.5, 3.0),
        "D": (5.0, 3.0),
        "Y": (8.5, 3.0),
        "U": (5.0, 5.2),
        "X": (3.0, 1.0),
    }

    def draw_node(ax, label, pos, color, desc, desc2=""):
        cx, cy = pos
        circle = plt.Circle((cx, cy), 0.55, color=color, zorder=3,
                             linewidth=1.5, fill=True, alpha=0.92)
        ax.add_patch(circle)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white", zorder=4)
        # Description below or above the node
        offset_y = -0.85 if label not in ("U",) else 0.85
        ax.text(cx, cy + offset_y, desc, ha="center", va="center",
                fontsize=8.5, color="#333333", zorder=4)
        if desc2:
            ax.text(cx, cy + offset_y - 0.38, desc2, ha="center", va="center",
                    fontsize=7.5, color="#555555", style="italic", zorder=4)

    draw_node(ax, "Z", nodes["Z"], BLUE,   "Rate-limit fallback", "(instrument)")
    draw_node(ax, "D", nodes["D"], BLUE,   "Premium routing",     "(endogenous)")
    draw_node(ax, "Y", nodes["Y"], GREEN,  "Task completion",     "(outcome)")
    draw_node(ax, "U", nodes["U"], RED,    "Query complexity",    "(unobserved)")
    draw_node(ax, "X", nodes["X"], GRAY,   "Query confidence",    "(observed)")

    def arrow(ax, src, dst, color, style="solid", label="", lw=2.0, rad=0.0):
        sx, sy = nodes[src]
        dx, dy = nodes[dst]
        # Adjust endpoints to node radius
        r = 0.55
        vx, vy = dx - sx, dy - sy
        norm = (vx**2 + vy**2) ** 0.5
        ux, uy = vx / norm, vy / norm
        sx2 = sx + r * ux
        sy2 = sy + r * uy
        dx2 = dx - r * ux
        dy2 = dy - r * uy
        arrowstyle = dict(
            arrowstyle="-|>", color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        )
        if style == "dashed":
            arrowstyle["linestyle"] = "dashed"
        patch = mpatches.FancyArrowPatch(
            (sx2, sy2), (dx2, dy2),
            mutation_scale=16,
            **arrowstyle,
            zorder=2,
        )
        ax.add_patch(patch)
        if label:
            mx = (sx2 + dx2) / 2
            my = (sy2 + dy2) / 2 + 0.28
            ax.text(mx, my, label, fontsize=8, color=color,
                    ha="center", va="bottom", zorder=5)

    # Z → D  (relevance)
    arrow(ax, "Z", "D", BLUE,  label="+Relevance")
    # D → Y  (causal effect)
    arrow(ax, "D", "Y", BLUE,  label="+0.06 pp  ")
    # U → D  (confounder → treatment, dashed red)
    arrow(ax, "U", "D", RED,   style="dashed", rad=-0.15, label="confounder")
    # U → Y  (confounder → outcome, dashed red)
    arrow(ax, "U", "Y", RED,   style="dashed", rad=0.15)
    # X → D  (observed covariate, gray, thin)
    arrow(ax, "X", "D", GRAY,  lw=1.2, label="routing rule")

    # Exclusion-restriction annotation: cross-out line (no Z→Y arrow)
    # Draw a curved dotted line with an X marker at the midpoint
    zx, zy = nodes["Z"]
    yx, yy = nodes["Y"]
    mid_x, mid_y = (zx + yx) / 2, (zy + yy) / 2 + 1.6
    ax.annotate(
        "",
        xy=(yx - 0.6, yy + 0.3), xytext=(zx + 0.6, zy + 0.3),
        arrowprops=dict(
            arrowstyle="-", color=RED, lw=1.5,
            connectionstyle="arc3,rad=-0.5",
            linestyle=(0, (4, 4)),
        ),
        zorder=1,
    )
    ax.text(5.0, 4.35, "✗", fontsize=18, color=RED, ha="center", va="center", zorder=5)
    ax.text(5.0, 1.55, "Exclusion restriction: no direct path Z → Y",
            ha="center", va="center", fontsize=8.5,
            color=RED, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, alpha=0.8))

    ax.set_title(
        "Instrumental variable causal structure\n"
        "Z satisfies relevance, exclusion, and independence",
        fontsize=11, color="#222222", pad=8,
    )

    fig.tight_layout()
    save_figure(fig, "iv_causal_dag.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Data-driven: first-stage + OLS vs 2SLS
# ---------------------------------------------------------------------------

def make_figure_2_data_driven() -> None:
    """
    Two-panel data-driven figure.

    Left panel:  routing rate by fallback group (bar chart showing the first
                 stage — fallback=1 group has near-zero premium routing).
    Right panel: point estimates + 95% CI for OLS and 2SLS, with the true
                 effect marked, showing OLS misses and 2SLS recovers it.
    """
    # --- Build dataset ---
    df = pd.read_csv(DATA_PATH)
    rng = np.random.default_rng(99)
    n = len(df)
    query_complexity   = rng.normal(0, 1, n)
    log_odds = -2.0 + 4.0 * (1.0 - df["query_confidence"]) + 0.6 * query_complexity
    premium_prob       = 1.0 / (1.0 + np.exp(-log_odds))
    df["routed_to_premium_iv"]     = rng.binomial(1, premium_prob).astype(int)
    df["rate_limit_fallback"]      = rng.binomial(1, 0.15, n)
    df["routed_to_premium_actual"] = (
        df["routed_to_premium_iv"] * (1 - df["rate_limit_fallback"])
    ).astype(int)
    engagement_base = np.where(df.engagement_tier == "heavy", 0.70,
                      np.where(df.engagement_tier == "medium", 0.55, 0.35))
    completion_prob = np.clip(
        engagement_base + 0.06 * df["routed_to_premium_actual"] - 0.04 * query_complexity
        + rng.normal(0, 0.02, n), 0.01, 0.99)
    df["task_completed_iv"] = rng.binomial(1, completion_prob).astype(int)
    df = pd.get_dummies(df, columns=["engagement_tier"], drop_first=True)
    tier_dummies  = [c for c in df.columns if c.startswith("engagement_tier_")]
    covariate_str = " + ".join(["query_confidence"] + tier_dummies)

    # --- First stage values ---
    rate_fb0 = df.loc[df.rate_limit_fallback == 0, "routed_to_premium_actual"].mean()
    rate_fb1 = df.loc[df.rate_limit_fallback == 1, "routed_to_premium_actual"].mean()

    # --- OLS and 2SLS estimates ---
    ols_m = smf.ols(
        f"task_completed_iv ~ routed_to_premium_actual + {covariate_str}", data=df
    ).fit(cov_type="HC3")
    ols_est = ols_m.params["routed_to_premium_actual"]
    ols_ci  = ols_m.conf_int().loc["routed_to_premium_actual"]

    s1 = smf.ols(
        f"routed_to_premium_actual ~ rate_limit_fallback + {covariate_str}", data=df
    ).fit()
    df["rtp_hat"] = s1.fittedvalues
    s2 = smf.ols(
        f"task_completed_iv ~ rtp_hat + {covariate_str}", data=df
    ).fit(cov_type="HC3")
    tsls_est = s2.params["rtp_hat"]

    # Bootstrap 95% CI for 2SLS
    rng2 = np.random.default_rng(7)
    tsls_boot = []
    for _ in range(500):
        samp = df.sample(n, replace=True, random_state=int(rng2.integers(1_000_000_000)))
        s1b = smf.ols(
            f"routed_to_premium_actual ~ rate_limit_fallback + {covariate_str}", data=samp
        ).fit()
        samp = samp.copy()
        samp["rtp_hat"] = s1b.fittedvalues
        s2b = smf.ols(f"task_completed_iv ~ rtp_hat + {covariate_str}", data=samp).fit()
        tsls_boot.append(s2b.params["rtp_hat"])
    tsls_arr = np.array(tsls_boot)
    tsls_ci  = (np.percentile(tsls_arr, 2.5), np.percentile(tsls_arr, 97.5))

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5.0), facecolor=BG,
        gridspec_kw={"width_ratios": [1, 1.3]},
    )
    fig.patch.set_facecolor(BG)

    # Left: first-stage bar chart
    ax1.set_facecolor(BG)
    bars = ax1.bar(
        ["No fallback\n(Z = 0)", "Rate-limit fallback\n(Z = 1)"],
        [rate_fb0, rate_fb1],
        color=[BLUE, ORANGE], width=0.5, zorder=2,
    )
    ax1.set_ylim(0, max(rate_fb0, rate_fb1) * 1.35)
    ax1.set_ylabel("Fraction routed to premium model", fontsize=9.5)
    ax1.set_title("First stage: instrument predicts routing\n"
                  "(relevance assumption check)", fontsize=9.5, pad=6)
    ax1.grid(axis="y", color="#e0e0e0", zorder=1)
    ax1.spines[["top", "right"]].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.annotate(
        f"First-stage gap: {rate_fb0 - rate_fb1:.3f} pp",
        xy=(0.5, max(rate_fb0, rate_fb1) * 1.22),
        xycoords="data", ha="center", fontsize=8.5,
        color=GRAY, style="italic",
    )

    # Right: OLS vs 2SLS comparison
    ax2.set_facecolor(BG)
    true_eff = 0.06
    methods  = ["Naive OLS\n(biased)", "2SLS\n(IV estimate)", "True effect\n(ground truth)"]
    estimates = [ols_est,  tsls_est, true_eff]
    cis_lo = [ols_ci[0], tsls_ci[0], true_eff]
    cis_hi = [ols_ci[1], tsls_ci[1], true_eff]
    colors = [RED, GREEN, BLUE]
    y_pos  = [2, 1, 0]

    for y, est, lo, hi, col, lab in zip(
        y_pos, estimates, cis_lo, cis_hi, colors, methods
    ):
        ax2.plot([lo, hi], [y, y], color=col, lw=3.5, solid_capstyle="round", zorder=2)
        ax2.scatter([est], [y], color=col, s=90, zorder=3)
        ax2.text(hi + 0.001, y, f"{est:+.3f}", va="center",
                 fontsize=9, color=col, fontweight="bold")

    ax2.axvline(true_eff, color=BLUE, lw=1.4, ls="--", alpha=0.7, zorder=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=9.5)
    ax2.set_xlabel("Estimated effect of premium routing on task completion", fontsize=9)
    ax2.set_title("OLS vs 2SLS: recovering the causal effect\n"
                  "(OLS CI misses true effect; 2SLS CI covers it)", fontsize=9.5, pad=6)
    ax2.grid(axis="x", color="#e0e0e0", zorder=1)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_xlim(
        min(ols_ci[0], tsls_ci[0]) - 0.01,
        max(ols_ci[1], tsls_ci[1]) + 0.025,
    )

    fig.suptitle(
        "Synthetic dataset: 50,000 users | True premium routing effect = +0.06 pp",
        fontsize=9, color=GRAY, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "iv_first_stage_estimates.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Figure 1 — conceptual IV DAG...")
    make_figure_1_conceptual()
    print("Generating Figure 2 — first stage + OLS vs 2SLS...")
    make_figure_2_data_driven()
    print("Done.")
