"""Build cluster_randomization_demo.ipynb for Article 5 companion repo."""
from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(
    "/Users/Rudrendu/All Mac/project-code/VS_Code/content-system/"
    "free-code-camp-hashnode-blog/product-experimentation-causal-inference-articles/"
    "github-repo-causal-inference-for-genai-llm-applications"
)
NB_PATH = REPO_ROOT / "05_cluster_randomization" / "cluster_randomization_demo.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# Intro
cells.append(nbf.v4.new_markdown_cell(
    "# Product Experimentation for Collaborative AI Features: Cluster "
    "Randomization, SUTVA, and Network Spillovers for LLM-Based Tools in Python\n"
    "\n"
    "**Keywords:** product experimentation, causal inference, cluster "
    "randomization, SUTVA, network interference, partial interference, "
    "spillover effects, LLM applications, generative AI\n"
    "\n"
    "## What this notebook does\n"
    "\n"
    "Measures the causal effect of a collaborative AI feature (AI meeting "
    "summarizer, shared AI writing tool, AI code review) when users in the "
    "same team workspace interfere with each other. User-level "
    "randomization breaks the Stable Unit Treatment Value Assumption (SUTVA) "
    "because control teammates see AI-generated artifacts from treated "
    "teammates. Covers cluster assignment at the workspace level, the naive "
    "(biased) user-level OLS, cluster-weighted least squares for honest "
    "standard errors, a two-exposure decomposition that separates the "
    "direct effect from the spillover effect, and cluster-bootstrap 95% "
    "confidence intervals.\n"
    "\n"
    "## Dataset\n"
    "\n"
    "A 50,000-user synthetic SaaS dataset with 50 workspaces. The "
    "collaborative feature ships to 25 randomly assigned treated workspaces "
    "at full coverage. Ground-truth effects baked in: **+0.80 min direct "
    "effect** on treated users, **+0.20 min spillover effect** on control "
    "users who collaborate cross-workspace. The naive user-level estimator "
    "is biased downward because spillover-exposed control users contaminate "
    "the control baseline.\n"
    "\n"
    "## Run\n"
    "\n"
    "From the repo root:\n"
    "\n"
    "```bash\n"
    "python data/generate_data.py --seed 42 --n-users 50000 \\\n"
    "    --out data/synthetic_llm_logs.csv\n"
    "jupyter notebook 05_cluster_randomization/cluster_randomization_demo.ipynb\n"
    "```"
))

# Setup cell
cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import statsmodels.api as sm\n"
    "import statsmodels.formula.api as smf\n"
    "\n"
    "DIRECT_EFFECT = 0.80\n"
    "SPILLOVER_EFFECT = 0.20\n"
    "DATA_SEED = 42\n"
    "OUTCOME_NOISE_SD = 0.30\n"
    "\n"
    "df = pd.read_csv(\"../data/synthetic_llm_logs.csv\")\n"
    "print(f\"Loaded {len(df):,} rows, {df.shape[1]} columns\")"
))

# Step 1 — build scenario
cells.append(nbf.v4.new_markdown_cell(
    "## Step 1: Build the cluster assignment and spillover exposure\n"
    "\n"
    "Workspaces 0 through 24 are treated at full coverage; 25 through 49 "
    "are control. A control user is spillover-exposed when they "
    "collaborate cross-workspace. `opt_in_agent_mode` is a defensible "
    "behavioral proxy: users who actively opt into AI tooling are the ones "
    "who read teammate-authored docs, Slack threads, and pull requests "
    "where treated-workspace AI output surfaces. In a real deployment, "
    "this proxy would be replaced with an observed collaboration graph.\n"
    "\n"
    "The observed outcome `session_minutes_obs` is constructed so the "
    "ground truth is known: each workspace has a baseline session time, "
    "treated users get +0.80 min, spillover-exposed users get +0.20 min."
))
cells.append(nbf.v4.new_code_cell(
    "rng = np.random.default_rng(DATA_SEED)\n"
    "\n"
    "df[\"treated_workspace\"] = (df[\"workspace_id\"] < 25).astype(int)\n"
    "df[\"treated_user\"] = df[\"treated_workspace\"]\n"
    "df[\"spillover_exposed\"] = (\n"
    "    (df[\"treated_workspace\"] == 0) & (df[\"opt_in_agent_mode\"] == 1)\n"
    ").astype(int)\n"
    "\n"
    "ws_baseline = pd.DataFrame({\n"
    "    \"workspace_id\": np.arange(50),\n"
    "    \"ws_baseline\": rng.normal(5.0, 0.30, size=50),\n"
    "})\n"
    "df = df.merge(ws_baseline, on=\"workspace_id\")\n"
    "noise = rng.normal(0, OUTCOME_NOISE_SD, size=len(df))\n"
    "df[\"session_minutes_obs\"] = (\n"
    "    df[\"ws_baseline\"]\n"
    "    + DIRECT_EFFECT * df[\"treated_user\"]\n"
    "    + SPILLOVER_EFFECT * df[\"spillover_exposed\"]\n"
    "    + noise\n"
    ")\n"
    "df[\"exposure\"] = np.select(\n"
    "    [df[\"treated_user\"] == 1, df[\"spillover_exposed\"] == 1],\n"
    "    [\"direct\", \"spillover\"],\n"
    "    default=\"pure_control\",\n"
    ")\n"
    "\n"
    "print(f\"Total users:             {len(df):,}\")\n"
    "print(f\"Treated workspaces:      {df[df.treated_workspace == 1].workspace_id.nunique()}\")\n"
    "print(f\"Control workspaces:      {df[df.treated_workspace == 0].workspace_id.nunique()}\")\n"
    "print(f\"Treated users:           {df.treated_user.sum():,}\")\n"
    "print(f\"Pure-control users:      {(df.exposure == 'pure_control').sum():,}\")\n"
    "print(f\"Spillover-exposed users: {(df.exposure == 'spillover').sum():,}\")\n"
    "ws_sizes = df.groupby(\"workspace_id\").size()\n"
    "print(f\"\\nWorkspace size: min={ws_sizes.min()}  median={int(ws_sizes.median())}  max={ws_sizes.max()}\")"
))

# Diagnostic chart
cells.append(nbf.v4.new_markdown_cell(
    "### Outcome distribution by exposure group\n"
    "\n"
    "Three distinct distributions: pure-control (lowest mean), "
    "spillover-exposed (shifted by about +0.20 min), and treated (shifted "
    "by about +0.80 min). The spillover distribution sits between the two. "
    "That middle bump is the contamination the naive user-level estimator "
    "would fold into the control baseline."
))
cells.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "from scipy.stats import gaussian_kde\n"
    "\n"
    "DIRECT_COLOR = \"#C44E52\"\n"
    "SPILLOVER_COLOR = \"#DD8452\"\n"
    "CONTROL_COLOR = \"#4C72B0\"\n"
    "\n"
    "groups = {\n"
    "    \"pure_control\": (\"Pure control (no exposure)\", CONTROL_COLOR),\n"
    "    \"spillover\":    (\"Control, exposed to spillover\", SPILLOVER_COLOR),\n"
    "    \"direct\":       (\"Treated (direct effect)\", DIRECT_COLOR),\n"
    "}\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, 4.8))\n"
    "grid = np.linspace(3.5, 7.0, 400)\n"
    "peak = 0.0\n"
    "for key, (label, color) in groups.items():\n"
    "    values = df.loc[df[\"exposure\"] == key, \"session_minutes_obs\"].values\n"
    "    dens = gaussian_kde(values, bw_method=0.25)(grid)\n"
    "    peak = max(peak, dens.max())\n"
    "    ax.fill_between(grid, dens, alpha=0.38, color=color, label=label)\n"
    "    ax.plot(grid, dens, color=color, lw=1.6)\n"
    "for key, (_, color) in groups.items():\n"
    "    mean_val = df.loc[df[\"exposure\"] == key, \"session_minutes_obs\"].mean()\n"
    "    ax.axvline(mean_val, color=color, lw=1.2, ls=\"--\", alpha=0.75)\n"
    "    ax.text(mean_val, peak * 1.22, f\"mean\\n{mean_val:.2f}\",\n"
    "            ha=\"center\", va=\"bottom\", fontsize=9.5, color=color)\n"
    "ax.set_ylim(0, peak * 1.55)\n"
    "ax.set_xlim(3.5, 7.0)\n"
    "ax.set_xlabel(\"Observed session minutes (per user)\")\n"
    "ax.set_ylabel(\"Density\")\n"
    "ax.set_title(\"Three outcome distributions reveal the spillover contamination\",\n"
    "             fontsize=12.5, loc=\"left\", fontweight=\"bold\")\n"
    "ax.legend(frameon=False, loc=\"upper left\", fontsize=9.5)\n"
    "ax.spines[\"top\"].set_visible(False)\n"
    "ax.spines[\"right\"].set_visible(False)\n"
    "plt.show()"
))

# Step 2 — naive OLS
cells.append(nbf.v4.new_markdown_cell(
    "## Step 2: Naive user-level OLS (biased and overconfident)\n"
    "\n"
    "Fit an OLS regression of the observed outcome on the user's own "
    "treatment assignment, ignoring the cluster structure. Two things go "
    "wrong at once: the point estimate is biased downward because "
    "spillover contaminates the control group, and the standard error is "
    "far too small because it treats 50,000 users as independent when the "
    "treatment was only randomized across 50 clusters."
))
cells.append(nbf.v4.new_code_cell(
    "naive = smf.ols(\"session_minutes_obs ~ treated_user\", data=df).fit()\n"
    "print(f\"Naive estimate:  {naive.params['treated_user']:+.4f} min\")\n"
    "print(f\"Naive SE:        {naive.bse['treated_user']:.4f}  (under-reported)\")\n"
    "ci = naive.conf_int().loc[\"treated_user\"].tolist()\n"
    "print(f\"Naive 95% CI:    [{ci[0]:+.4f}, {ci[1]:+.4f}]\")\n"
    "print(f\"Ground truth:    +0.80\")\n"
    "print(f\"Bias:            {naive.params['treated_user'] - DIRECT_EFFECT:+.4f} min\")"
))

# Step 3 — cluster WLS
cells.append(nbf.v4.new_markdown_cell(
    "## Step 3: Cluster-weighted least squares (honest standard error)\n"
    "\n"
    "Aggregate the user-level data to 50 workspace means, then regress the "
    "workspace mean on the workspace-level treatment indicator, weighted "
    "by workspace size. The standard error now reflects a sample of 50 "
    "clusters. The point estimate is still biased (control cluster means "
    "include spillover-exposed users), but at least the inference is "
    "honest about precision."
))
cells.append(nbf.v4.new_code_cell(
    "ws = (\n"
    "    df.groupby(\"workspace_id\")\n"
    "    .agg(ws_mean=(\"session_minutes_obs\", \"mean\"),\n"
    "         ws_size=(\"user_id\", \"count\"),\n"
    "         treated=(\"treated_workspace\", \"max\"))\n"
    "    .reset_index()\n"
    ")\n"
    "X_ws = sm.add_constant(ws[\"treated\"])\n"
    "wls = sm.WLS(ws[\"ws_mean\"], X_ws, weights=ws[\"ws_size\"]).fit()\n"
    "wls_ci = wls.conf_int().loc[\"treated\"].tolist()\n"
    "print(f\"WLS cluster-ATE: {wls.params['treated']:+.4f} min\")\n"
    "print(f\"WLS SE:          {wls.bse['treated']:.4f}  (based on K=50 clusters)\")\n"
    "print(f\"WLS 95% CI:      [{wls_ci[0]:+.4f}, {wls_ci[1]:+.4f}]\")"
))

# Step 4 — two-exposure OLS
cells.append(nbf.v4.new_markdown_cell(
    "## Step 4: Two-exposure decomposition (unbiased direct and spillover)\n"
    "\n"
    "Categorize every user as one of three exposures: `direct` (treated "
    "workspace), `spillover` (control workspace but cross-workspace "
    "collaborator), or `pure_control` (control workspace, no "
    "cross-workspace exposure). Regress the outcome on `is_direct` and "
    "`is_spillover` with cluster-robust standard errors keyed to "
    "`workspace_id`. The pure-control group is now the omitted baseline, "
    "and both effects are identified separately."
))
cells.append(nbf.v4.new_code_cell(
    "df[\"is_direct\"] = (df[\"exposure\"] == \"direct\").astype(int)\n"
    "df[\"is_spillover\"] = (df[\"exposure\"] == \"spillover\").astype(int)\n"
    "two_exp = smf.ols(\n"
    "    \"session_minutes_obs ~ is_direct + is_spillover\",\n"
    "    data=df,\n"
    ").fit(cov_type=\"cluster\", cov_kwds={\"groups\": df[\"workspace_id\"]})\n"
    "direct = two_exp.params[\"is_direct\"]\n"
    "spillover = two_exp.params[\"is_spillover\"]\n"
    "direct_ci = two_exp.conf_int().loc[\"is_direct\"].tolist()\n"
    "spillover_ci = two_exp.conf_int().loc[\"is_spillover\"].tolist()\n"
    "print(f\"Direct effect:     {direct:+.4f} min  (ground truth = +0.80)\")\n"
    "print(f\"  SE:              {two_exp.bse['is_direct']:.4f}\")\n"
    "print(f\"  95% CI:          [{direct_ci[0]:+.4f}, {direct_ci[1]:+.4f}]\")\n"
    "print(f\"Spillover effect:  {spillover:+.4f} min  (ground truth = +0.20)\")\n"
    "print(f\"  SE:              {two_exp.bse['is_spillover']:.4f}\")\n"
    "print(f\"  95% CI:          [{spillover_ci[0]:+.4f}, {spillover_ci[1]:+.4f}]\")\n"
    "spillover_share = (df[\"exposure\"] == \"spillover\").mean()\n"
    "print(f\"\\nSpillover share of all users: {spillover_share:.4f}\")\n"
    "print(f\"Projected total under full rollout: {direct + spillover_share * spillover:+.4f} min\")"
))

# Step 5 — cluster bootstrap
cells.append(nbf.v4.new_markdown_cell(
    "## Step 5: Cluster-bootstrap 95% confidence intervals\n"
    "\n"
    "Resample entire workspaces (not users) with replacement. Resampling "
    "users would understate variance because users in the same workspace "
    "share the same cluster assignment and workspace-level baseline. The "
    "cluster bootstrap preserves the design and matches what a pre-"
    "registered analysis plan would specify. Takes about 1 minute at 500 "
    "replicates."
))
cells.append(nbf.v4.new_code_cell(
    "def naive_point(d):\n"
    "    return smf.ols(\"session_minutes_obs ~ treated_user\", data=d).fit().params[\"treated_user\"]\n"
    "\n"
    "def wls_point(d):\n"
    "    w = (d.groupby(\"workspace_id\").agg(\n"
    "            ws_mean=(\"session_minutes_obs\", \"mean\"),\n"
    "            ws_size=(\"user_id\", \"count\"),\n"
    "            treated=(\"treated_workspace\", \"max\")).reset_index())\n"
    "    X = sm.add_constant(w[\"treated\"])\n"
    "    return sm.WLS(w[\"ws_mean\"], X, weights=w[\"ws_size\"]).fit().params[\"treated\"]\n"
    "\n"
    "def two_exp_point(d):\n"
    "    fit = smf.ols(\"session_minutes_obs ~ is_direct + is_spillover\",\n"
    "                  data=d).fit(cov_type=\"cluster\",\n"
    "                              cov_kwds={\"groups\": d[\"workspace_id\"]})\n"
    "    return fit.params[\"is_direct\"], fit.params[\"is_spillover\"]\n"
    "\n"
    "rng_boot = np.random.default_rng(7)\n"
    "ws_ids = df[\"workspace_id\"].unique()\n"
    "k = len(ws_ids)\n"
    "reps = {\"naive\": [], \"cluster_wls\": [], \"direct\": [], \"spillover\": []}\n"
    "for _ in range(500):\n"
    "    draw = rng_boot.choice(ws_ids, size=k, replace=True)\n"
    "    sample = pd.concat([df[df[\"workspace_id\"] == wid] for wid in draw],\n"
    "                       ignore_index=True)\n"
    "    reps[\"naive\"].append(naive_point(sample))\n"
    "    reps[\"cluster_wls\"].append(wls_point(sample))\n"
    "    d_b, s_b = two_exp_point(sample)\n"
    "    reps[\"direct\"].append(d_b)\n"
    "    reps[\"spillover\"].append(s_b)\n"
    "\n"
    "for key, truth in [(\"naive\", 0.80), (\"cluster_wls\", 0.80),\n"
    "                   (\"direct\", 0.80), (\"spillover\", 0.20)]:\n"
    "    arr = np.array(reps[key])\n"
    "    lo, hi = np.percentile(arr, [2.5, 97.5])\n"
    "    covers = \"covers\" if lo <= truth <= hi else \"misses\"\n"
    "    print(f\"{key:<13} 95% CI: [{lo:+.4f}, {hi:+.4f}]   ({covers} {truth:+.2f})\")"
))

# Summary
cells.append(nbf.v4.new_markdown_cell(
    "## Results summary\n"
    "\n"
    "| Estimator | Point estimate | 95% CI (cluster bootstrap) | Covers truth? |\n"
    "|---|---|---|---|\n"
    "| Naive user-level OLS | +0.6723 | [+0.5386, +0.7966] | no (biased + CI misses +0.80) |\n"
    "| Cluster WLS | +0.6723 | [+0.5386, +0.7966] | no (same bias; honest SE) |\n"
    "| Two-exposure direct | +0.7284 | [+0.5931, +0.8519] | yes, covers +0.80 |\n"
    "| Two-exposure spillover | +0.2083 | [+0.2008, +0.2164] | yes, covers +0.20 |\n"
    "\n"
    "The naive estimator and the cluster WLS estimator share the same "
    "biased point estimate. Cluster WLS fixes the standard error but not "
    "the bias. Only the two-exposure decomposition separates the two "
    "effects cleanly and recovers the ground truth inside the 95% "
    "confidence interval. The lesson: when SUTVA is violated, the "
    "estimator needs to model the spillover, not just the clustering."
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
