"""Generate the two article figures for the synthetic control tutorial.

Figure 1 (conceptual): idealized treated unit vs synthetic control with
pre/post regions and the counterfactual gap annotation.

Figure 2 (data-driven): two-panel layout from the 50,000-user dataset.
Top panel: treated unit vs fitted synthetic control over weeks 0-29.
Bottom panel: placebo-gap distribution with observed gap marked.

Run from repo root:
    python 04_synthetic_control/generate_synthetic_control_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_DIR = REPO_ROOT / "04_synthetic_control"
IMAGES_DIR = REPO_ROOT / "images" / "article-4"

PRE = 20
WINDOW = 30
RED = "#C44E52"
BLUE = "#4C72B0"
GRAY = "#888888"


def save_figure(fig, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for target in (IMAGES_DIR / name, METHOD_DIR / name):
        fig.savefig(target, dpi=150, bbox_inches="tight")
        print(f"  wrote {target.relative_to(REPO_ROOT)}")
    plt.close(fig)


def build_panel(df: pd.DataFrame):
    df_window = df[df.signup_week < WINDOW].copy()
    panel = (
        df_window.groupby(["workspace_id", "signup_week"])["task_completed"]
        .mean().reset_index()
    )
    panel.columns = ["workspace_id", "week", "task_completed"]
    pivot = panel.pivot(
        index="week", columns="workspace_id", values="task_completed"
    )
    pivot = pivot.interpolate(method="linear", axis=0).ffill().bfill()
    ws_wave = df.groupby("workspace_id").wave.first()
    wave1_ws = sorted(ws_wave[ws_wave == 1].index.tolist())
    wave2_ws = sorted(ws_wave[ws_wave == 2].index.tolist())
    treated = pivot[wave1_ws].mean(axis=1).values
    donors = pivot[wave2_ws].values
    return treated, donors, wave2_ws


def fit_sc(treated, donors, pre=PRE):
    n = donors.shape[1]
    def objective(w):
        return np.mean((treated[:pre] - donors[:pre] @ w) ** 2)
    w0 = np.ones(n) / n
    res = minimize(
        objective, w0, method="SLSQP", bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    return res.x


def placebo_permutation(treated, donors, pre=PRE):
    n = donors.shape[1]
    gaps = np.empty(n)
    for j in range(n):
        pt = donors[:, j]
        pp = np.delete(donors, j, axis=1)
        w = fit_sc(pt, pp, pre=pre)
        synth = pp @ w
        gaps[j] = (pt[pre:] - synth[pre:]).mean()
    return gaps


def make_figure_1_conceptual() -> None:
    """Conceptual schematic: smooth treated vs synthetic vs donor fan.

    Two stacked panels:
      (top)    idealized treated (solid), synthetic (dashed), donor fan
      (bottom) region label strip with pre/post brackets
    """
    rng = np.random.default_rng(7)
    weeks = np.arange(WINDOW)

    # Idealized synthetic control: smooth AR(1)-looking pre-period trend,
    # continues the same trend into the post-period (counterfactual)
    ar = np.zeros(WINDOW)
    for t in range(1, WINDOW):
        ar[t] = 0.78 * ar[t - 1] + rng.normal(0, 0.010)
    synth = 0.55 + ar

    # Treated matches synth in pre-period, diverges upward post-treatment
    treated = synth.copy()
    # Apply a +5pp treatment effect at week 20 with a small onset + noise
    post = np.arange(WINDOW) >= PRE
    ramp = np.clip((np.arange(WINDOW) - PRE) / 2, 0, 1) * post
    treated = treated + 0.05 * ramp + rng.normal(0, 0.006, WINDOW) * post

    # Donor fan: three representative donor trajectories around synth
    donors_show = np.array([
        synth - 0.035 + rng.normal(0, 0.012, WINDOW),
        synth + 0.040 + rng.normal(0, 0.012, WINDOW),
        synth - 0.015 + rng.normal(0, 0.012, WINDOW),
    ])

    y_min = min(treated.min(), synth.min(), donors_show.min()) - 0.02
    y_max = max(treated.max(), synth.max(), donors_show.max()) * 1.20
    y_top = y_max

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 5.8),
        gridspec_kw={"height_ratios": [5.0, 1.0], "hspace": 0.0},
        sharex=True,
        layout="constrained",
    )

    # --- TOP: outcome trajectories ---
    # Donors (faint gray fan)
    for i, d in enumerate(donors_show):
        ax_top.plot(weeks, d, color=GRAY, lw=1.0, alpha=0.45,
                    label="Donor workspaces (wave 2)" if i == 0 else None)

    # Synthetic control (dashed blue)
    ax_top.plot(weeks, synth, color=BLUE, lw=2.4, ls="--",
                label="Synthetic control (weighted donors)")

    # Treated unit (solid red)
    ax_top.plot(weeks, treated, color=RED, lw=2.4,
                label="Treated unit (wave 1)")

    # Vertical treatment line
    ax_top.axvline(PRE, color="#555555", ls=":", lw=1.5, zorder=3)

    # Gap annotation at last week
    wk_last = WINDOW - 1
    ax_top.annotate(
        "", xy=(wk_last - 0.3, synth[wk_last]),
        xytext=(wk_last - 0.3, treated[wk_last]),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4),
    )
    ax_top.text(
        wk_last - 1.2, (synth[wk_last] + treated[wk_last]) / 2,
        "causal effect\n(gap)",
        ha="right", va="center", fontsize=10, color="#333333",
        fontweight="bold",
    )

    ax_top.set_xlim(0, WINDOW - 1)
    ax_top.set_ylim(y_min, y_top)
    ax_top.set_ylabel("Task completion rate")
    ax_top.set_title(
        "Synthetic control rebuilds the missing counterfactual",
        fontsize=12.5, loc="left",
    )
    ax_top.legend(
        frameon=False, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.0), fontsize=9.5,
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    # --- BOTTOM: region label strip ---
    ax_bot.set_xlim(0, WINDOW - 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.set_yticks([])
    for side in ("top", "left", "right"):
        ax_bot.spines[side].set_visible(False)
    ax_bot.spines["bottom"].set_visible(True)

    for x0, x1, label, color in [
        (0, PRE, "Pre-treatment (weight-fitting window)", BLUE),
        (PRE, WINDOW - 1, "Post-treatment (gap emerges)", RED),
    ]:
        mid = (x0 + x1) / 2
        ax_bot.annotate(
            "", xy=(x0 + 0.15, 0.78), xytext=(x1 - 0.15, 0.78),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.2, alpha=0.75),
        )
        ax_bot.plot([x0 + 0.15, x0 + 0.15], [0.68, 0.88],
                    color=color, lw=1.2, alpha=0.75)
        ax_bot.plot([x1 - 0.15, x1 - 0.15], [0.68, 0.88],
                    color=color, lw=1.2, alpha=0.75)
        ax_bot.text(mid, 0.32, label, ha="center", va="center",
                    fontsize=9.5, color="#333333")

    ax_bot.axvline(PRE, color="#555555", ls=":", lw=1.2,
                   ymin=0.55, ymax=0.95)
    ax_bot.text(
        PRE, 0.05, "Treatment date (week 20)",
        ha="center", va="bottom", fontsize=8.5, color="#555555",
    )

    ax_bot.set_xlabel("Signup week", fontsize=10)

    save_figure(fig, "synthetic_control_conceptual.png")


def make_figure_2_density() -> None:
    """Data-driven: two-panel figure from the 50k dataset.

    Top:    treated vs synthetic control trajectories (weeks 0-29)
    Bottom: placebo gap distribution with observed gap marked (red line)
    """
    df = pd.read_csv(REPO_ROOT / "data" / "synthetic_llm_logs.csv")
    treated, donors, wave2_ws = build_panel(df)
    w_opt = fit_sc(treated, donors)
    synth = donors @ w_opt
    pre_rmse = float(np.sqrt(np.mean((treated[:PRE] - synth[:PRE]) ** 2)))
    observed_gap = float((treated[PRE:] - synth[PRE:]).mean())
    placebo_gaps = placebo_permutation(treated, donors)

    weeks = np.arange(WINDOW)

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(10.0, 7.0),
        gridspec_kw={"height_ratios": [2.2, 1.3], "hspace": 0.55},
    )

    # --- TOP: trajectories ---
    ax_top.plot(weeks, treated, color=RED, lw=2.2, marker="o", ms=4.5,
                label="Treated unit (wave 1 mean, 25 workspaces)")
    ax_top.plot(weeks, synth, color=BLUE, lw=2.2, ls="--", marker="s", ms=4.0,
                label="Synthetic control (weighted wave 2 donors)")
    ax_top.axvline(PRE, color="#555555", ls=":", lw=1.4, zorder=3)

    y_top_val = max(treated.max(), synth.max()) * 1.12
    y_bot_val = min(treated.min(), synth.min()) * 0.92

    # Annotation dead zone (upper-left, empty area)
    ax_top.text(
        1.0, y_top_val * 0.97,
        f"Pre-period RMSE: {pre_rmse * 100:.2f} pp",
        ha="left", va="top", fontsize=10, color="#333333",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC"),
    )
    ax_top.text(
        PRE + 0.3, y_top_val * 0.97,
        f"Post-period gap: {observed_gap:+.4f}",
        ha="left", va="top", fontsize=10, color="#333333",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC"),
    )
    ax_top.text(
        PRE, y_bot_val * 1.01, "Treatment\n(week 20)",
        ha="center", va="bottom", fontsize=8.5, color="#555555",
    )

    ax_top.set_xlim(0, WINDOW - 1)
    ax_top.set_ylim(y_bot_val, y_top_val)
    ax_top.set_ylabel("Mean task completion rate")
    ax_top.set_xlabel("Signup week")
    ax_top.set_title(
        "Treated unit vs. fitted synthetic control (50,000-user dataset)",
        fontsize=12.5, loc="left", pad=8,
    )
    ax_top.legend(
        frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=10,
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # --- BOTTOM: placebo distribution ---
    from scipy.stats import gaussian_kde
    grid = np.linspace(placebo_gaps.min() - 0.03,
                       max(placebo_gaps.max(), observed_gap) + 0.03, 400)
    kde_pl = gaussian_kde(placebo_gaps, bw_method=0.35)
    dens_pl = kde_pl(grid)

    ax_bot.fill_between(grid, dens_pl, alpha=0.55, color=GRAY,
                        label=f"Placebo gaps (n={len(placebo_gaps)} donors)")
    ax_bot.plot(grid, dens_pl, color="#555555", lw=1.2)

    # Observed gap as a tall thin marker
    ax_bot.axvline(observed_gap, color=RED, lw=2.4,
                   label=f"Observed gap ({observed_gap:+.4f})")
    ax_bot.axvline(0, color="#555555", ls=":", lw=1.0)

    y_bot_top = dens_pl.max() * 1.35
    ax_bot.set_ylim(0, y_bot_top)
    ax_bot.set_xlabel("Mean post-period gap")
    ax_bot.set_ylabel("Density")
    ax_bot.set_title(
        "In-space placebo test: observed gap sits outside the "
        "placebo distribution",
        fontsize=11.5, loc="left", pad=6,
    )
    ax_bot.legend(
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=10,
    )
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig.subplots_adjust(top=0.94, bottom=0.14, left=0.08, right=0.97)
    save_figure(fig, "synthetic_control_density.png")


def main() -> None:
    print("Generating Figure 1 (conceptual)...")
    make_figure_1_conceptual()
    print("Generating Figure 2 (data-driven density)...")
    make_figure_2_density()
    print("Done.")


if __name__ == "__main__":
    main()
