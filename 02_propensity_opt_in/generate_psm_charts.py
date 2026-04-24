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
    """Conceptual: treated skews right, control skews left; overlap shaded."""
    rng = np.random.default_rng(7)
    treated = np.clip(rng.beta(6, 3, size=5000), 0.02, 0.98)
    control = np.clip(rng.beta(3, 6, size=5000), 0.02, 0.98)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bins = np.linspace(0, 1, 40)

    ax.hist(
        control, bins=bins, density=True, alpha=0.55,
        color="#4C72B0", label="Did not opt in",
    )
    ax.hist(
        treated, bins=bins, density=True, alpha=0.55,
        color="#C44E52", label="Opted in",
    )

    ax.axvspan(0.15, 0.85, color="#DDDDDD", alpha=0.35, zorder=0)
    ax.text(
        0.50, 2.15, "Region of common support",
        ha="center", va="center", fontsize=11, color="#333333",
    )
    ax.annotate(
        "Control-heavy region\n(few treated users here)",
        xy=(0.10, 1.7), xytext=(0.02, 2.5),
        fontsize=9, color="#333333",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
    )
    ax.annotate(
        "Treatment-heavy region\n(few controls here)",
        xy=(0.90, 1.7), xytext=(0.70, 2.5),
        fontsize=9, color="#333333",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
    )

    ax.set_xlabel("Propensity score (predicted probability of opting in)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Propensity score distributions separate treated and control groups",
        fontsize=12, loc="left",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.0)
    ax.legend(frameon=False, loc="upper center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, "psm_overlap_conceptual.png")


def make_figure_2_data_driven() -> None:
    """Actual propensity overlap on the synthetic dataset."""
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

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bins = np.linspace(0.05, 0.75, 30)

    ax.hist(
        control_ps, bins=bins, density=True, alpha=0.55,
        color="#4C72B0", label=f"Did not opt in (n={len(control_ps):,})",
    )
    ax.hist(
        treated_ps, bins=bins, density=True, alpha=0.55,
        color="#C44E52", label=f"Opted in (n={len(treated_ps):,})",
    )

    ax.axvspan(common_lo, common_hi, color="#DDDDDD", alpha=0.35, zorder=0)
    ax.text(
        (common_lo + common_hi) / 2, 3.9,
        f"Common support: [{common_lo:.2f}, {common_hi:.2f}]",
        ha="center", va="center", fontsize=10, color="#333333",
    )

    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.set_title(
        "Propensity score overlap on the 50,000-user synthetic dataset",
        fontsize=12, loc="left",
    )
    ax.set_xlim(0.05, 0.75)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
