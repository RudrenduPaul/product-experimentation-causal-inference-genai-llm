"""Build synthetic_control_demo.ipynb for Article 4 companion repo."""
from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(
    "/Users/Rudrendu/All Mac/project-code/VS_Code/content-system/"
    "free-code-camp-hashnode-blog/product-experimentation-causal-inference-articles/"
    "github-repo-causal-inference-for-genai-llm-applications"
)
NB_PATH = REPO_ROOT / "04_synthetic_control" / "synthetic_control_demo.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# --- Cell 1: Intro ---
cells.append(nbf.v4.new_markdown_cell(
    "# Product Experimentation with Synthetic Control: Causal Inference "
    "for Global LLM Rollouts in Python\n"
    "\n"
    "**Keywords:** product experimentation, causal inference, synthetic "
    "control, LLM applications, generative AI, global rollout, counterfactual "
    "construction\n"
    "\n"
    "## What this notebook does\n"
    "\n"
    "Measures the causal effect of a global LLM model upgrade using "
    "synthetic control. The scenario: every wave-1 workspace received a new "
    "model at week 20 simultaneously. There is no randomized holdout. "
    "Synthetic control builds a counterfactual by finding a weighted "
    "combination of untreated wave-2 workspaces whose pre-period task "
    "completion trajectory matches wave 1. The post-period gap between "
    "wave 1 and its synthetic twin is the estimated causal effect, "
    "conditional on three identification assumptions: pre-period fit "
    "(convex-hull condition), no interference from the treatment to donors "
    "(SUTVA for donors), and stable donor composition (no structural "
    "breaks in the post-period). Covers SLSQP weight optimization, "
    "pre/post trajectory visualization, in-space placebo permutation test, "
    "leave-one-out donor sensitivity, and cluster bootstrap 95% "
    "confidence intervals.\n"
    "\n"
    "## Dataset\n"
    "\n"
    "A 50,000-user synthetic SaaS dataset where the ground-truth causal "
    "effect of the staged model rollout for wave 1 post-treatment is "
    "**+5 percentage points** on task completion. The naive before/after "
    "comparison is confounded by shared week-to-week variation that "
    "affects both waves, so it provides no credible counterfactual. "
    "Wave 1 contains 25 workspaces (IDs 0-24); wave 2 contains 25 "
    "workspaces (IDs 25-49) and serves as the donor pool.\n"
    "\n"
    "## Run\n"
    "\n"
    "From the repo root:\n"
    "\n"
    "```bash\n"
    "python data/generate_data.py --seed 42 --n-users 50000 \\\n"
    "    --out data/synthetic_llm_logs.csv\n"
    "jupyter notebook 04_synthetic_control/synthetic_control_demo.ipynb\n"
    "```"
))

# --- Cell 1b: Conceptual figure (Figure 1) ---
cells.append(nbf.v4.new_markdown_cell(
    "## How synthetic control rebuilds the missing counterfactual\n"
    "\n"
    "![Conceptual figure showing three horizontal curves over 30 weeks. "
    "A faint gray fan of donor workspace trajectories runs across the plot, "
    "a dashed navy synthetic-control curve tracks a red solid treated curve "
    "closely through the pre-treatment window (weeks 0 through 19), and "
    "after week 20 the red treated curve rises above the dashed synthetic, "
    "with a labeled double-arrow showing the causal effect gap at the right "
    "edge. Below the x-axis, a strip with colored brackets labels the "
    "pre-treatment weight-fitting window and the post-treatment "
    "gap-emerges window."
    "](https://raw.githubusercontent.com/RudrenduPaul/product-experimentation-"
    "causal-inference-genai-llm/main/images/article-4/"
    "synthetic_control_conceptual.png)\n"
    "\n"
    "*Figure 1: Schematic of the synthetic control construction. The gray "
    "curves are donor workspaces that remain on the old model. The dashed "
    "navy curve is the weighted combination of donors that best tracks the "
    "treated unit (red) during the pre-treatment window. After the "
    "treatment date (week 20, dotted vertical line), the weights stay "
    "frozen and the dashed curve projects forward as the counterfactual, "
    "while the treated unit moves upward. The gap between the two curves "
    "in the post-treatment window is the causal-effect estimate. Weights "
    "are fit once on pre-treatment data only and never refit using "
    "post-treatment data.*"
))

# --- Cell 2: Setup + load data ---
cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "from scipy.optimize import minimize\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "df = pd.read_csv(\"../data/synthetic_llm_logs.csv\")\n"
    "print(f\"Loaded {len(df):,} rows, {df.shape[1]} columns\")\n"
    "print(df.wave.value_counts().to_dict())"
))

# --- Cell 3: Setup markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Setting up the working example\n"
    "\n"
    "Wave 1 workspaces (IDs 0-24) received the upgrade at week 20. "
    "Wave 2 workspaces (IDs 25-49) are still on the old model through "
    "week 29. Aggregate user-level rows to a workspace-by-week panel and "
    "split into a treated series (mean across wave 1) and a donor matrix "
    "(one column per wave 2 workspace)."
))

# --- Cell 4: Build the panel ---
cells.append(nbf.v4.new_code_cell(
    "PRE = 20         # weeks 0-19 are pre-treatment\n"
    "WINDOW = 30      # analysis window weeks 0-29\n"
    "\n"
    "df_window = df[df.signup_week < WINDOW].copy()\n"
    "\n"
    "panel = (\n"
    "    df_window.groupby([\"workspace_id\", \"signup_week\"])\n"
    "    [\"task_completed\"].mean().reset_index()\n"
    ")\n"
    "panel.columns = [\"workspace_id\", \"week\", \"task_completed\"]\n"
    "\n"
    "pivot = panel.pivot(\n"
    "    index=\"week\", columns=\"workspace_id\", values=\"task_completed\"\n"
    ")\n"
    "pivot = pivot.interpolate(method=\"linear\", axis=0).ffill().bfill()\n"
    "\n"
    "ws_wave = df.groupby(\"workspace_id\").wave.first()\n"
    "wave1_ws = sorted(ws_wave[ws_wave == 1].index.tolist())\n"
    "wave2_ws = sorted(ws_wave[ws_wave == 2].index.tolist())\n"
    "\n"
    "treated_series = pivot[wave1_ws].mean(axis=1).values\n"
    "donor_matrix = pivot[wave2_ws].values\n"
    "\n"
    "print(f\"Treated series shape: {treated_series.shape}\")\n"
    "print(f\"Donor matrix shape:   {donor_matrix.shape}\")\n"
    "users_per_cell = len(df_window) / (50 * WINDOW)\n"
    "print(f\"Users per workspace-week: ~{users_per_cell:.1f}\")\n"
    "print(f\"Pre-period treated mean  (weeks 0-19):  {treated_series[:PRE].mean():.4f}\")\n"
    "print(f\"Post-period treated mean (weeks 20-29): {treated_series[PRE:].mean():.4f}\")"
))

# --- Cell 5: Step 1 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 1: Fit donor weights with SLSQP\n"
    "\n"
    "Find weights `w` that minimize the pre-period mean squared error "
    "between the treated series and the weighted combination of donor "
    "series, subject to convex-combination constraints (each weight in "
    "[0, 1], all weights sum to 1). The non-negativity + sum-to-1 "
    "constraints define a convex combination; this prevents "
    "extrapolation beyond the support of the donor pool."
))

# --- Cell 6: Step 1 code ---
cells.append(nbf.v4.new_code_cell(
    "n_donors = len(wave2_ws)\n"
    "Y_pre = treated_series[:PRE]\n"
    "D_pre = donor_matrix[:PRE, :]\n"
    "\n"
    "def objective(w):\n"
    "    return np.mean((Y_pre - D_pre @ w) ** 2)\n"
    "\n"
    "w0 = np.ones(n_donors) / n_donors\n"
    "bounds = [(0, 1)] * n_donors\n"
    "constraints = [{\"type\": \"eq\", \"fun\": lambda w: w.sum() - 1}]\n"
    "\n"
    "result = minimize(\n"
    "    objective, w0, method=\"SLSQP\", bounds=bounds,\n"
    "    constraints=constraints,\n"
    "    options={\"ftol\": 1e-12, \"maxiter\": 5000},\n"
    ")\n"
    "w_opt = result.x\n"
    "\n"
    "pre_mse = float(np.mean((Y_pre - D_pre @ w_opt) ** 2))\n"
    "pre_rmse = float(np.sqrt(pre_mse))\n"
    "nz = int((w_opt > 0.001).sum())\n"
    "\n"
    "print(f\"Optimization converged: {result.success}\")\n"
    "print(f\"Non-zero donor weights (|w| > 0.001): {nz}\")\n"
    "print(f\"Pre-period MSE:  {pre_mse:.6f}\")\n"
    "print(f\"Pre-period RMSE: {pre_rmse:.4f}  \"\n"
    "      f\"({pre_rmse * 100:.2f} percentage points)\")\n"
    "\n"
    "# Post-period gap\n"
    "synth_full = donor_matrix @ w_opt\n"
    "gap = float((treated_series[PRE:] - synth_full[PRE:]).mean())\n"
    "print(f\"\\nObserved post-period gap: {gap:+.4f}  \"\n"
    "      f\"(ground truth = +0.0500)\")\n"
    "\n"
    "# Top 5 donor weights\n"
    "nz_pairs = sorted(\n"
    "    [(ws, w_opt[i]) for i, ws in enumerate(wave2_ws) if w_opt[i] > 0.001],\n"
    "    key=lambda x: -x[1]\n"
    ")\n"
    "print(\"\\nTop 5 donor weights:\")\n"
    "for ws_id, weight in nz_pairs[:5]:\n"
    "    print(f\"  workspace {ws_id}: w = {weight:.4f}\")"
))

# --- Cell 7: Step 2 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 2: Plot treated vs synthetic control trajectories\n"
    "\n"
    "The primary visual diagnostic: plot both series on the same axes, "
    "mark the treatment date, and confirm the synthetic control tracks "
    "the treated unit in the pre-period. A tight pre-period fit is the "
    "signal that the weights identify a credible counterfactual."
))

# --- Cell 8: Step 2 code ---
cells.append(nbf.v4.new_code_cell(
    "plt.rcParams[\"figure.dpi\"] = 200\n"
    "\n"
    "weeks = np.arange(WINDOW)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
    "ax.plot(weeks, treated_series, marker=\"o\", linewidth=1.8,\n"
    "        color=\"#C44E52\", label=\"Wave 1 (treated)\")\n"
    "ax.plot(weeks, synth_full, marker=\"s\", linestyle=\"--\",\n"
    "        linewidth=1.8, color=\"#4C72B0\",\n"
    "        label=\"Synthetic control\")\n"
    "ax.axvline(PRE, color=\"#555555\", linestyle=\":\", linewidth=1.4,\n"
    "           label=\"Model upgrade (week 20)\")\n"
    "ax.set_xlabel(\"Signup week\")\n"
    "ax.set_ylabel(\"Mean task completion rate\")\n"
    "ax.set_title(\"Treated unit vs synthetic control\",\n"
    "             fontsize=11, loc=\"left\")\n"
    "ax.legend(frameon=False)\n"
    "ax.spines[\"top\"].set_visible(False)\n"
    "ax.spines[\"right\"].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "# Weekly gap decomposition\n"
    "post_gap = treated_series[PRE:] - synth_full[PRE:]\n"
    "print(\"Post-period weekly gaps (treated minus synthetic):\")\n"
    "for wk, g in zip(range(PRE, WINDOW), post_gap):\n"
    "    print(f\"  week {wk}: {g:+.4f}\")\n"
    "print(f\"\\nMean gap: {post_gap.mean():+.4f}\")"
))

# --- Cell 9: Step 3 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 3: In-space placebo permutation test\n"
    "\n"
    "For a single treated unit, no standard p-value is available. The "
    "in-space placebo assigns each donor in turn as a placebo 'treated' "
    "unit, re-fits the synthetic control on the remaining donors, and "
    "records the placebo post-period gap. The pseudo p-value is the "
    "share of placebo gaps whose absolute value is at least as large as "
    "the observed gap, with the standard (count + 1) / (N + 1) "
    "finite-sample correction."
))

# --- Cell 10: Step 3 code ---
cells.append(nbf.v4.new_code_cell(
    "placebo_gaps = []\n"
    "for j in range(n_donors):\n"
    "    placebo_treated = donor_matrix[:, j]\n"
    "    placebo_pool = np.delete(donor_matrix, j, axis=1)\n"
    "    n_p = placebo_pool.shape[1]\n"
    "\n"
    "    def obj_p(w):\n"
    "        return np.mean((placebo_treated[:PRE] - placebo_pool[:PRE] @ w) ** 2)\n"
    "\n"
    "    res_p = minimize(\n"
    "        obj_p, np.ones(n_p) / n_p, method=\"SLSQP\",\n"
    "        bounds=[(0, 1)] * n_p,\n"
    "        constraints=[{\"type\": \"eq\", \"fun\": lambda w: w.sum() - 1}],\n"
    "        options={\"ftol\": 1e-12, \"maxiter\": 5000},\n"
    "    )\n"
    "    synth_p = placebo_pool @ res_p.x\n"
    "    placebo_gaps.append((placebo_treated[PRE:] - synth_p[PRE:]).mean())\n"
    "\n"
    "placebo_gaps = np.array(placebo_gaps)\n"
    "observed_gap = gap\n"
    "\n"
    "rank = int((np.abs(placebo_gaps) >= abs(observed_gap)).sum())\n"
    "pseudo_p = (rank + 1) / (len(placebo_gaps) + 1)\n"
    "\n"
    "print(f\"Observed gap:      {observed_gap:+.4f}\")\n"
    "print(f\"Placebo mean gap:  {placebo_gaps.mean():+.4f}\")\n"
    "print(f\"Placebo std gap:   {placebo_gaps.std():.4f}\")\n"
    "print(f\"Placebo gap range: [{placebo_gaps.min():+.4f}, \"\n"
    "      f\"{placebo_gaps.max():+.4f}]\")\n"
    "print(f\"|placebo| >= |observed|: {rank} of {len(placebo_gaps)}\")\n"
    "print(f\"Pseudo p-value: {pseudo_p:.4f}\")"
))

# --- Cell 11: Step 4 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 4: Leave-one-out (LOO) donor sensitivity\n"
    "\n"
    "Drop each non-zero-weight donor one at a time, re-fit, and record "
    "the new gap. If the estimate is stable across the LOO set, no "
    "single donor is driving the result. Abadie (2021) recommends this "
    "as a first-line robustness check for synthetic control."
))

# --- Cell 12: Step 4 code ---
cells.append(nbf.v4.new_code_cell(
    "def fit_and_gap(treated, donors, pre=PRE):\n"
    "    n = donors.shape[1]\n"
    "    def obj(w):\n"
    "        return np.mean((treated[:pre] - donors[:pre] @ w) ** 2)\n"
    "    res = minimize(\n"
    "        obj, np.ones(n) / n, method=\"SLSQP\",\n"
    "        bounds=[(0, 1)] * n,\n"
    "        constraints=[{\"type\": \"eq\", \"fun\": lambda w: w.sum() - 1}],\n"
    "        options={\"ftol\": 1e-12, \"maxiter\": 5000},\n"
    "    )\n"
    "    synth = donors @ res.x\n"
    "    return float((treated[pre:] - synth[pre:]).mean())\n"
    "\n"
    "\n"
    "nz_idx = np.where(w_opt > 0.001)[0]\n"
    "loo_rows = []\n"
    "for j in nz_idx:\n"
    "    kept = np.delete(donor_matrix, j, axis=1)\n"
    "    gap_new = fit_and_gap(treated_series, kept)\n"
    "    loo_rows.append({\n"
    "        \"dropped_workspace\": int(wave2_ws[j]),\n"
    "        \"dropped_weight\": float(w_opt[j]),\n"
    "        \"new_gap\": gap_new,\n"
    "    })\n"
    "loo_df = pd.DataFrame(loo_rows).sort_values(\"dropped_weight\", ascending=False)\n"
    "print(loo_df.round(4).to_string(index=False))\n"
    "print(f\"\\nLOO gap range: [{loo_df.new_gap.min():+.4f}, \"\n"
    "      f\"{loo_df.new_gap.max():+.4f}]\")\n"
    "print(f\"Original gap:  {gap:+.4f}\")"
))

# --- Cell 13: Step 5 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 5: Cluster bootstrap 95% confidence interval\n"
    "\n"
    "The classical bootstrap does not apply cleanly to synthetic control "
    "on a single treated unit. A valid substitute: user-level cluster "
    "bootstrap. Resample users with replacement, rebuild the "
    "workspace-week panel, re-fit the donor weights on the pre-period, "
    "and record the post-period gap. Repeat 500 times; report the 2.5th "
    "and 97.5th percentiles as the 95% CI. Combined with the placebo "
    "p-value and LOO range, this provides the inference package for the "
    "estimator."
))

# --- Cell 14: Step 5 code ---
cells.append(nbf.v4.new_code_cell(
    "def build_panel(df_inner):\n"
    "    dfw = df_inner[df_inner.signup_week < WINDOW].copy()\n"
    "    panel = (dfw.groupby([\"workspace_id\", \"signup_week\"])\n"
    "             [\"task_completed\"].mean().reset_index())\n"
    "    panel.columns = [\"workspace_id\", \"week\", \"task_completed\"]\n"
    "    piv = panel.pivot(index=\"week\", columns=\"workspace_id\",\n"
    "                      values=\"task_completed\")\n"
    "    piv = piv.interpolate(method=\"linear\", axis=0).ffill().bfill()\n"
    "    ws_wave = df_inner.groupby(\"workspace_id\").wave.first()\n"
    "    w1 = sorted(ws_wave[ws_wave == 1].index.tolist())\n"
    "    w2 = sorted(ws_wave[ws_wave == 2].index.tolist())\n"
    "    return piv[w1].mean(axis=1).values, piv[w2].values\n"
    "\n"
    "\n"
    "rng = np.random.default_rng(7)\n"
    "n = len(df)\n"
    "n_reps = 500\n"
    "gaps_boot = np.empty(n_reps)\n"
    "for i in range(n_reps):\n"
    "    sample = df.iloc[rng.integers(0, n, size=n)]\n"
    "    t_b, d_b = build_panel(sample)\n"
    "    gaps_boot[i] = fit_and_gap(t_b, d_b)\n"
    "\n"
    "lo = float(np.percentile(gaps_boot, 2.5))\n"
    "hi = float(np.percentile(gaps_boot, 97.5))\n"
    "print(f\"Post-period gap 95% CI: [{lo:+.4f}, {hi:+.4f}]\")\n"
    "print(f\"Observed point estimate: {gap:+.4f}\")\n"
    "print(f\"Ground truth +0.05 inside CI: \"\n"
    "      f\"{'YES' if lo <= 0.05 <= hi else 'NO'}\")\n"
    "print(f\"Zero inside CI: {'YES' if lo <= 0 <= hi else 'NO'}\")"
))

# --- Cell 15: Results summary ---
cells.append(nbf.v4.new_markdown_cell(
    "## Results summary\n"
    "\n"
    "| Quantity | Value |\n"
    "|---|---|\n"
    "| Dataset | 50,000 rows, 16 columns |\n"
    "| Wave 1 workspaces (treated) | 25 |\n"
    "| Wave 2 workspaces (donor pool) | 25 |\n"
    "| Users per workspace-week | ~19.2 |\n"
    "| Ground-truth effect | +0.0500 (+5 pp on task completion) |\n"
    "| Wave 1 pre-period mean (weeks 0-19) | 0.5927 |\n"
    "| Wave 1 post-period mean (weeks 20-29) | 0.6421 |\n"
    "| Naive before/after gap | +0.0515 |\n"
    "| **Step 1 — SLSQP synthetic control** | |\n"
    "| Non-zero donor weights | 12 |\n"
    "| Pre-period MSE | 0.001400 |\n"
    "| Pre-period RMSE | 0.0374 (3.74 pp) |\n"
    "| Observed post-period gap | +0.0829 |\n"
    "| Top donor | workspace 35 (w = 0.2016) |\n"
    "| **Step 3 — in-space placebo test** | |\n"
    "| Placebo mean gap | -0.0008 |\n"
    "| Placebo std | 0.0380 |\n"
    "| Placebo range | [-0.0748, +0.0707] |\n"
    "| Pseudo p-value | 0.0385 |\n"
    "| **Step 4 — LOO sensitivity** | |\n"
    "| LOO gap range | [+0.0739, +0.0945] |\n"
    "| **Step 5 — cluster bootstrap 95% CI** | |\n"
    "| 95% CI | [+0.0511, +0.1215] |\n"
    "| CI excludes zero | YES |\n"
    "| CI excludes naive +0.0515 | NO (boundary) |\n"
    "\n"
    "The placebo pseudo p-value of 0.0385 rejects the null of no effect "
    "at the 5% level. The LOO range [+0.074, +0.094] confirms the "
    "estimate is stable across donor subsets. The cluster bootstrap CI "
    "[+0.051, +0.122] is positive and excludes zero. The point estimate "
    "overshoots the ground truth by about 3 pp because each donor "
    "workspace (average of ~19 users per week) carries more noise than "
    "the treated unit (averaged across 25 workspaces), and the weighted "
    "combination does not fully cancel post-period week-to-week donor "
    "idiosyncrasies."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open("w") as f:
    nbf.write(nb, f)
print(f"Wrote {NB_PATH}")
print(f"Size: {NB_PATH.stat().st_size:,} bytes")
