"""Build psm_demo.ipynb for Article 2 companion repo."""
from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(
    "/Users/Rudrendu/All Mac/project-code/VS_Code/content-system/"
    "free-code-camp-hashnode-blog/product-experimentation-causal-inference-articles/"
    "github-repo-causal-inference-for-genai-llm-applications"
)
NB_PATH = REPO_ROOT / "02_propensity_opt_in" / "psm_demo.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# Intro cell — keywords, title, dataset, clone/run. NO "Companion notebook"
# link back to the article. NO "GitHub repo:" line (reader is in the repo).
cells.append(nbf.v4.new_markdown_cell(
    "# Product Experimentation with Propensity Scores: Causal Inference "
    "for LLM-Based Features in Python\n"
    "\n"
    "**Keywords:** product experimentation, causal inference, propensity "
    "score matching, inverse-probability weighting, LLM applications, "
    "generative AI\n"
    "\n"
    "## What this notebook does\n"
    "\n"
    "Measures the causal effect of an AI feature (\"agent mode\") that "
    "ships behind a user-controlled opt-in toggle, where the naive "
    "opted-in vs. non-opted-in comparison is contaminated by selection "
    "bias. Covers propensity score estimation, inverse-probability "
    "weighting (ATE and ATT), 1-nearest-neighbor matching, covariate "
    "balance via standardized mean differences, and bootstrap 95% "
    "confidence intervals.\n"
    "\n"
    "## Dataset\n"
    "\n"
    "A 50,000-user synthetic SaaS dataset where the ground-truth causal "
    "effect of opting into agent mode is **+8 percentage points** on task "
    "completion. The naive comparison inflates this to around +21 "
    "percentage points because heavy-engagement users opt in at 65% "
    "while light users opt in at 12%.\n"
    "\n"
    "## Run\n"
    "\n"
    "From the repo root:\n"
    "\n"
    "```bash\n"
    "python data/generate_data.py --seed 42 --n-users 50000 \\\n"
    "    --out data/synthetic_llm_logs.csv\n"
    "jupyter notebook 02_propensity_opt_in/psm_demo.ipynb\n"
    "```"
))

# Setup cell
cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.metrics import roc_auc_score\n"
    "from sklearn.neighbors import NearestNeighbors\n"
    "\n"
    "df = pd.read_csv(\"../data/synthetic_llm_logs.csv\")\n"
    "print(f\"Loaded {len(df):,} rows, {df.shape[1]} columns\")"
))

# Section 0 — selection pattern + naive
cells.append(nbf.v4.new_markdown_cell(
    "## Setting up the working example\n"
    "\n"
    "Opt-in rates differ sharply by engagement tier. The naive comparison "
    "folds selection bias into the effect estimate."
))
cells.append(nbf.v4.new_code_cell(
    "print(df.groupby(\"engagement_tier\").opt_in_agent_mode.mean().round(3))\n"
    "\n"
    "naive_effect = (\n"
    "    df[df.opt_in_agent_mode == 1].task_completed.mean()\n"
    "    - df[df.opt_in_agent_mode == 0].task_completed.mean()\n"
    ")\n"
    "print(f\"\\nNaive opt-in effect: {naive_effect:+.4f}\")"
))

# Step 1
cells.append(nbf.v4.new_markdown_cell(
    "## Step 1: Estimate the propensity score\n"
    "\n"
    "Logistic regression predicting opt-in from engagement tier and query "
    "confidence. AUC confirms the model discriminates above chance."
))
cells.append(nbf.v4.new_code_cell(
    "X = pd.get_dummies(\n"
    "    df[[\"engagement_tier\", \"query_confidence\"]],\n"
    "    drop_first=True\n"
    ").astype(float)\n"
    "y_treat = df.opt_in_agent_mode\n"
    "\n"
    "ps_model = LogisticRegression(max_iter=1000).fit(X, y_treat)\n"
    "df[\"propensity\"] = ps_model.predict_proba(X)[:, 1]\n"
    "\n"
    "print(df.groupby(\"engagement_tier\").propensity.mean().round(3))\n"
    "print(\n"
    "    f\"\\nPropensity range (treated):  \"\n"
    "    f\"{df[df.opt_in_agent_mode == 1].propensity.min():.3f} - \"\n"
    "    f\"{df[df.opt_in_agent_mode == 1].propensity.max():.3f}\"\n"
    ")\n"
    "print(\n"
    "    f\"Propensity range (control):  \"\n"
    "    f\"{df[df.opt_in_agent_mode == 0].propensity.min():.3f} - \"\n"
    "    f\"{df[df.opt_in_agent_mode == 0].propensity.max():.3f}\"\n"
    ")\n"
    "print(f\"Propensity model AUC: {roc_auc_score(y_treat, df.propensity):.3f}\")"
))

# Overlap plot cell
cells.append(nbf.v4.new_markdown_cell(
    "### Propensity score overlap (positivity check)\n"
    "\n"
    "Both groups should have overlapping propensity distributions. If "
    "treated users cluster near 1 and controls near 0, positivity fails "
    "and no amount of weighting can recover a causal effect."
))
cells.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "from scipy.stats import gaussian_kde\n"
    "\n"
    "treated_ps = df[df.opt_in_agent_mode == 1].propensity.values\n"
    "control_ps = df[df.opt_in_agent_mode == 0].propensity.values\n"
    "common_lo = max(treated_ps.min(), control_ps.min())\n"
    "common_hi = min(treated_ps.max(), control_ps.max())\n"
    "\n"
    "tier_colors = {\"light\": \"#8FA9C7\", \"medium\": \"#B8A886\",\n"
    "               \"heavy\": \"#C88B8B\"}\n"
    "tier_order = [\"light\", \"medium\", \"heavy\"]\n"
    "\n"
    "fig, (ax_top, ax_bot) = plt.subplots(\n"
    "    nrows=2, figsize=(10.0, 7.0),\n"
    "    gridspec_kw={\"height_ratios\": [2.4, 1.0], \"hspace\": 0.70}\n"
    ")\n"
    "\n"
    "# Top: smooth KDE curves\n"
    "grid = np.linspace(0.05, 0.75, 400)\n"
    "dens_c = gaussian_kde(control_ps, bw_method=0.25)(grid)\n"
    "dens_t = gaussian_kde(treated_ps, bw_method=0.25)(grid)\n"
    "ax_top.fill_between(grid, dens_c, alpha=0.45, color=\"#4C72B0\",\n"
    "                    label=f\"Did not opt in  (n = {len(control_ps):,})\")\n"
    "ax_top.fill_between(grid, dens_t, alpha=0.45, color=\"#C44E52\",\n"
    "                    label=f\"Opted in  (n = {len(treated_ps):,})\")\n"
    "ax_top.plot(grid, dens_c, color=\"#4C72B0\", lw=1.4)\n"
    "ax_top.plot(grid, dens_t, color=\"#C44E52\", lw=1.4)\n"
    "ax_top.axvspan(common_lo, common_hi, color=\"#EFEFEF\", alpha=0.5, zorder=0)\n"
    "\n"
    "y_top = max(dens_c.max(), dens_t.max()) * 1.40\n"
    "ax_top.set_ylim(0, y_top)\n"
    "for tier in tier_order:\n"
    "    tier_df = df[df.engagement_tier == tier]\n"
    "    center = tier_df.propensity.mean()\n"
    "    ax_top.axvline(center, color=tier_colors[tier], lw=1.2, ls=\"--\",\n"
    "                   alpha=0.9)\n"
    "    ax_top.text(center, y_top * 0.97,\n"
    "                f\"{tier}  (p \\u2248 {center:.2f})\",\n"
    "                ha=\"center\", va=\"top\", fontsize=9.5,\n"
    "                bbox=dict(boxstyle=\"round,pad=0.28\", fc=\"white\",\n"
    "                          ec=tier_colors[tier], lw=0.9))\n"
    "\n"
    "ax_top.text(0.50, y_top * 0.55,\n"
    "            f\"Common support band: [{common_lo:.2f}, {common_hi:.2f}]\\n\"\n"
    "            \"both groups present across this range\",\n"
    "            ha=\"center\", va=\"center\", fontsize=9.5,\n"
    "            bbox=dict(boxstyle=\"round,pad=0.35\", fc=\"white\",\n"
    "                      ec=\"#BBBBBB\", lw=0.6))\n"
    "ax_top.set_xlabel(\"Propensity score  (predicted probability of opting in)\")\n"
    "ax_top.set_ylabel(\"Density\")\n"
    "ax_top.set_title(\"Propensity score overlap on the 50,000-user synthetic dataset\",\n"
    "                 fontsize=12.5, loc=\"left\", pad=10)\n"
    "ax_top.set_xlim(0.05, 0.75)\n"
    "ax_top.legend(frameon=False, loc=\"upper center\",\n"
    "              bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=10)\n"
    "ax_top.spines[\"top\"].set_visible(False)\n"
    "ax_top.spines[\"right\"].set_visible(False)\n"
    "\n"
    "# Bottom: stacked count bars per tier\n"
    "rows = [{\"tier\": t,\n"
    "         \"control\": int(((df.engagement_tier == t) & (df.opt_in_agent_mode == 0)).sum()),\n"
    "         \"treated\": int(((df.engagement_tier == t) & (df.opt_in_agent_mode == 1)).sum())}\n"
    "        for t in tier_order]\n"
    "y_pos = np.arange(len(tier_order))[::-1]\n"
    "max_total = max(r[\"control\"] + r[\"treated\"] for r in rows)\n"
    "for y, r in zip(y_pos, rows):\n"
    "    ax_bot.barh(y, r[\"control\"], color=\"#4C72B0\", alpha=0.75,\n"
    "                edgecolor=\"white\", height=0.6)\n"
    "    ax_bot.barh(y, r[\"treated\"], left=r[\"control\"], color=\"#C44E52\",\n"
    "                alpha=0.75, edgecolor=\"white\", height=0.6)\n"
    "    ax_bot.text(r[\"control\"] / 2, y, f\"{r['control']:,}\",\n"
    "                ha=\"center\", va=\"center\", fontsize=9,\n"
    "                color=\"white\", fontweight=\"bold\")\n"
    "    ax_bot.text(r[\"control\"] + r[\"treated\"] / 2, y, f\"{r['treated']:,}\",\n"
    "                ha=\"center\", va=\"center\", fontsize=9,\n"
    "                color=\"white\", fontweight=\"bold\")\n"
    "    total = r[\"control\"] + r[\"treated\"]\n"
    "    ax_bot.text(total + max_total * 0.015, y,\n"
    "                f\"{r['treated'] / total:.0%} opt-in\",\n"
    "                ha=\"left\", va=\"center\", fontsize=9)\n"
    "ax_bot.set_yticks(y_pos)\n"
    "ax_bot.set_yticklabels([r[\"tier\"].capitalize() for r in rows])\n"
    "ax_bot.set_xlim(0, max_total * 1.18)\n"
    "ax_bot.set_xlabel(\"User count\")\n"
    "ax_bot.set_title(\"Both groups exist at every engagement tier → positivity holds\",\n"
    "                 fontsize=10.5, loc=\"left\", pad=6)\n"
    "ax_bot.spines[\"top\"].set_visible(False)\n"
    "ax_bot.spines[\"right\"].set_visible(False)\n"
    "ax_bot.tick_params(axis=\"y\", length=0)\n"
    "plt.show()"
))

# Step 2
cells.append(nbf.v4.new_markdown_cell(
    "## Step 2: Inverse-probability weighting\n"
    "\n"
    "IPW weights rebalance the sample so opted-in and non-opted-in groups "
    "look similar on observables. ATE is the effect on a random user; "
    "ATT is the effect on users who actually opted in."
))
cells.append(nbf.v4.new_code_cell(
    "df[\"ipw\"] = np.where(\n"
    "    df.opt_in_agent_mode == 1,\n"
    "    1 / df.propensity,\n"
    "    1 / (1 - df.propensity)\n"
    ")\n"
    "\n"
    "t = df[df.opt_in_agent_mode == 1]\n"
    "c = df[df.opt_in_agent_mode == 0]\n"
    "ate_ipw = (\n"
    "    (t.task_completed * t.ipw).sum() / t.ipw.sum()\n"
    "    - (c.task_completed * c.ipw).sum() / c.ipw.sum()\n"
    ")\n"
    "print(f\"IPW average treatment effect (ATE): {ate_ipw:+.4f}\")\n"
    "\n"
    "df[\"ipw_att\"] = np.where(\n"
    "    df.opt_in_agent_mode == 1,\n"
    "    1,\n"
    "    df.propensity / (1 - df.propensity)\n"
    ")\n"
    "t = df[df.opt_in_agent_mode == 1]\n"
    "c = df[df.opt_in_agent_mode == 0]\n"
    "treated_mean = t.task_completed.mean()\n"
    "control_w_mean = (c.task_completed * c.ipw_att).sum() / c.ipw_att.sum()\n"
    "att_ipw = treated_mean - control_w_mean\n"
    "print(f\"IPW average treatment effect on treated (ATT): {att_ipw:+.4f}\")"
))

# Step 3
cells.append(nbf.v4.new_markdown_cell(
    "## Step 3: Nearest-neighbor matching\n"
    "\n"
    "Pair each opted-in user with the closest non-opted-in user by "
    "propensity score. `NearestNeighbors` allows the same control user "
    "to match multiple treated users (matching with replacement)."
))
cells.append(nbf.v4.new_code_cell(
    "treated_ps = df[df.opt_in_agent_mode == 1][[\"propensity\"]].values\n"
    "control_ps = df[df.opt_in_agent_mode == 0][[\"propensity\"]].values\n"
    "\n"
    "nn = NearestNeighbors(n_neighbors=1).fit(control_ps)\n"
    "_, idx = nn.kneighbors(treated_ps)\n"
    "\n"
    "treated_outcomes = df[df.opt_in_agent_mode == 1].task_completed.values\n"
    "matched_control_outcomes = (\n"
    "    df[df.opt_in_agent_mode == 0].task_completed.values[idx.flatten()]\n"
    ")\n"
    "\n"
    "att_match = (treated_outcomes - matched_control_outcomes).mean()\n"
    "print(f\"1-NN matching ATT: {att_match:+.4f}\")"
))

# Step 4
cells.append(nbf.v4.new_markdown_cell(
    "## Step 4: Check covariate balance\n"
    "\n"
    "Standardized mean differences (SMDs) measure how imbalanced each "
    "covariate is between groups. |SMD| < 0.1 after weighting is the "
    "conventional bar for \"balanced enough\"."
))
cells.append(nbf.v4.new_code_cell(
    "def smd(treated_vals, control_vals, treated_w=None, control_w=None):\n"
    "    \"\"\"Standardized mean difference, optionally with weights.\"\"\"\n"
    "    if treated_w is None:\n"
    "        treated_w = np.ones(len(treated_vals))\n"
    "    if control_w is None:\n"
    "        control_w = np.ones(len(control_vals))\n"
    "    t_mean = np.average(treated_vals, weights=treated_w)\n"
    "    c_mean = np.average(control_vals, weights=control_w)\n"
    "    pooled_std = np.sqrt((treated_vals.var() + control_vals.var()) / 2)\n"
    "    return (t_mean - c_mean) / pooled_std\n"
    "\n"
    "engagement_heavy = (df.engagement_tier == \"heavy\").astype(float).values\n"
    "qc = df.query_confidence.values\n"
    "tr = (df.opt_in_agent_mode == 1).values\n"
    "\n"
    "covariates = {\n"
    "    \"engagement_tier_heavy\": engagement_heavy,\n"
    "    \"query_confidence\": qc,\n"
    "}\n"
    "\n"
    "print(f\"{'Covariate':<30} {'Raw SMD':>10} {'Weighted SMD':>15}\")\n"
    "for name, vals in covariates.items():\n"
    "    smd_raw = smd(vals[tr], vals[~tr])\n"
    "    smd_weighted = smd(\n"
    "        vals[tr], vals[~tr],\n"
    "        treated_w=df[tr].ipw.values,\n"
    "        control_w=df[~tr].ipw.values,\n"
    "    )\n"
    "    print(f\"{name:<30} {smd_raw:>+10.3f} {smd_weighted:>+15.3f}\")"
))

# Step 5: Bootstrap
cells.append(nbf.v4.new_markdown_cell(
    "## Step 5: Bootstrap 95% confidence intervals\n"
    "\n"
    "Non-parametric bootstrap with 500 replicates. For each replicate, "
    "resample with replacement, refit the propensity model, and "
    "recompute all three estimators. The 2.5th and 97.5th percentiles "
    "of the bootstrap distribution give the 95% CI. Takes ~1–2 minutes."
))
cells.append(nbf.v4.new_code_cell(
    "def estimate_all(sample):\n"
    "    \"\"\"Return (ATE_IPW, ATT_IPW, ATT_match) on a bootstrap sample.\"\"\"\n"
    "    s = sample.copy()\n"
    "    X_s = pd.get_dummies(\n"
    "        s[[\"engagement_tier\", \"query_confidence\"]], drop_first=True\n"
    "    ).astype(float)\n"
    "    ps = LogisticRegression(max_iter=1000).fit(X_s, s.opt_in_agent_mode)\n"
    "    s[\"p\"] = ps.predict_proba(X_s)[:, 1]\n"
    "\n"
    "    s[\"w_ate\"] = np.where(\n"
    "        s.opt_in_agent_mode == 1, 1 / s.p, 1 / (1 - s.p)\n"
    "    )\n"
    "    s[\"w_att\"] = np.where(\n"
    "        s.opt_in_agent_mode == 1, 1, s.p / (1 - s.p)\n"
    "    )\n"
    "    t, c = s[s.opt_in_agent_mode == 1], s[s.opt_in_agent_mode == 0]\n"
    "\n"
    "    ate = (\n"
    "        (t.task_completed * t.w_ate).sum() / t.w_ate.sum()\n"
    "        - (c.task_completed * c.w_ate).sum() / c.w_ate.sum()\n"
    "    )\n"
    "    att = t.task_completed.mean() - (\n"
    "        (c.task_completed * c.w_att).sum() / c.w_att.sum()\n"
    "    )\n"
    "    nn_b = NearestNeighbors(n_neighbors=1).fit(c[[\"p\"]].values)\n"
    "    _, idx_b = nn_b.kneighbors(t[[\"p\"]].values)\n"
    "    match = (\n"
    "        t.task_completed.values\n"
    "        - c.task_completed.values[idx_b.flatten()]\n"
    "    ).mean()\n"
    "    return ate, att, match\n"
    "\n"
    "rng = np.random.default_rng(7)\n"
    "n_reps = 500\n"
    "results = np.zeros((n_reps, 3))\n"
    "for i in range(n_reps):\n"
    "    boot = df.iloc[rng.integers(0, len(df), size=len(df))]\n"
    "    results[i] = estimate_all(boot)\n"
    "\n"
    "for name, col in zip([\"IPW ATE\", \"IPW ATT\", \"1-NN ATT\"], range(3)):\n"
    "    lo, hi = np.percentile(results[:, col], [2.5, 97.5])\n"
    "    print(f\"{name:<10} 95% CI: [{lo:+.4f}, {hi:+.4f}]\")"
))

# Results summary
cells.append(nbf.v4.new_markdown_cell(
    "## Results summary\n"
    "\n"
    "| Quantity | Value |\n"
    "|---|---|\n"
    "| Naive opt-in effect | +0.2106 (heavily contaminated) |\n"
    "| Ground-truth effect | +0.08 (+8 percentage points) |\n"
    "| Propensity model AUC | 0.744 |\n"
    "| IPW ATE | +0.0851, 95% CI [+0.0745, +0.0954] |\n"
    "| IPW ATT | +0.0770, 95% CI [+0.0687, +0.0865] |\n"
    "| 1-NN matching ATT | +0.0752, 95% CI [+0.0659, +0.0940] |\n"
    "| Raw SMD on engagement_tier_heavy | +0.742 (large imbalance) |\n"
    "| Weighted SMD on engagement_tier_heavy | +0.002 (balanced) |\n"
    "\n"
    "All three estimators recover the true +0.08 effect. The naive "
    "estimate of +0.2106 is 2.6x the true effect and is excluded from "
    "every 95% confidence interval. This is the textbook pattern: "
    "selection bias dominates when you ignore it, propensity score "
    "methods correct it when their assumptions hold."
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
