"""Build rdd_demo.ipynb for Article 3 companion repo."""
from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(
    "/Users/Rudrendu/All Mac/project-code/VS_Code/content-system/"
    "free-code-camp-hashnode-blog/product-experimentation-causal-inference-articles/"
    "github-repo-causal-inference-for-genai-llm-applications"
)
NB_PATH = REPO_ROOT / "03_rdd_confidence_threshold" / "rdd_demo.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# --- Cell 1: Intro ---
cells.append(nbf.v4.new_markdown_cell(
    "# Product Experimentation with Regression Discontinuity: Causal Inference "
    "for LLM Confidence-Threshold Routing in Python\n"
    "\n"
    "**Keywords:** product experimentation, causal inference, regression "
    "discontinuity, LLM applications, generative AI, confidence-threshold routing\n"
    "\n"
    "## What this notebook does\n"
    "\n"
    "Measures the causal effect of routing low-confidence LLM queries to a "
    "premium model using a sharp regression discontinuity design (RDD). "
    "The routing rule — queries with confidence score below 0.85 go to the "
    "premium model — creates a natural experiment at the threshold. Users "
    "just below the cutoff are similar to users just above it, so the jump "
    "in task completion at 0.85 estimates the Local Average Treatment Effect "
    "(LATE) of premium routing. Covers local linear regression with HC3 "
    "standard errors, bandwidth sensitivity analysis, McCrary density check "
    "(manipulation diagnostic), quadratic robustness specification, and "
    "bootstrap 95% confidence intervals.\n"
    "\n"
    "## Dataset\n"
    "\n"
    "A 50,000-user synthetic SaaS dataset where the ground-truth causal "
    "effect of routing to the premium model is **+6 percentage points** on "
    "task completion. The naive comparison confounds query difficulty with "
    "the routing decision because harder queries (lower confidence) go to "
    "premium but are also harder to complete.\n"
    "\n"
    "## Run\n"
    "\n"
    "From the repo root:\n"
    "\n"
    "```bash\n"
    "python data/generate_data.py --seed 42 --n-users 50000 \\\n"
    "    --out data/synthetic_llm_logs.csv\n"
    "jupyter notebook 03_rdd_confidence_threshold/rdd_demo.ipynb\n"
    "```"
))

# --- Cell 2: Setup + load data ---
cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import statsmodels.formula.api as smf\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "df = pd.read_csv(\"../data/synthetic_llm_logs.csv\")\n"
    "print(f\"Loaded {len(df):,} rows, {df.shape[1]} columns\")"
))

# --- Cell 3: Setup markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Setting up the working example\n"
    "\n"
    "The routing rule is deterministic: any query with `query_confidence < 0.85` "
    "goes to the premium model (`routed_to_premium = 1`). "
    "The key columns are `query_confidence` (the running variable), "
    "`routed_to_premium` (the treatment), and `task_completed` (the outcome). "
    "The threshold at 0.85 is the discontinuity we exploit."
))

# --- Cell 4: Routing breakdown + naive comparison ---
cells.append(nbf.v4.new_code_cell(
    "CUTOFF = 0.85\n"
    "\n"
    "# Routing breakdown\n"
    "counts = df.routed_to_premium.value_counts().to_dict()\n"
    "print(f\"Premium-routed (confidence < 0.85):  {counts.get(1, 0):,}\")\n"
    "print(f\"Cheap-routed   (confidence >= 0.85): {counts.get(0, 0):,}\")\n"
    "\n"
    "# Confidence distribution\n"
    "print(\"\\nQuery confidence distribution:\")\n"
    "print(df.query_confidence.describe().round(3))\n"
    "\n"
    "# Naive comparison\n"
    "naive = (\n"
    "    df[df.routed_to_premium == 1].task_completed.mean()\n"
    "    - df[df.routed_to_premium == 0].task_completed.mean()\n"
    ")\n"
    "print(f\"\\nNaive premium-vs-cheap effect: {naive:+.4f}  (ground truth = +0.06)\")\n"
    "print(\"Premium-routed users have systematically lower-confidence queries,\")\n"
    "print(\"so this comparison confounds the routing decision with query difficulty.\")"
))

# --- Cell 5: Step 1 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 1: Sharp RDD with local linear regression\n"
    "\n"
    "Restrict analysis to users within bandwidth 0.10 of the cutoff "
    "(confidence in [0.75, 0.95)). Within this narrow window, assignment "
    "to premium routing is as-good-as-random. Fit a local linear regression "
    "with a dummy for being below the cutoff (`below_cutoff`) and allow the "
    "slope to differ on each side (`below_cutoff:rc`). The `below_cutoff` "
    "coefficient is the LATE. HC3 heteroskedasticity-robust standard errors "
    "guard against heteroskedasticity in the binary outcome."
))

# --- Cell 6: Step 1 code ---
cells.append(nbf.v4.new_code_cell(
    "bw = 0.10\n"
    "near = df[(df.query_confidence > CUTOFF - bw) &\n"
    "          (df.query_confidence < CUTOFF + bw)].copy()\n"
    "near[\"below_cutoff\"] = (near.query_confidence < CUTOFF).astype(int)\n"
    "near[\"rc\"] = near.query_confidence - CUTOFF\n"
    "\n"
    "formula = \"task_completed ~ below_cutoff + rc + below_cutoff:rc\"\n"
    "rdd_model = smf.ols(formula, data=near).fit(cov_type=\"HC3\")\n"
    "\n"
    "effect = float(rdd_model.params[\"below_cutoff\"])\n"
    "se = float(rdd_model.bse[\"below_cutoff\"])\n"
    "pval = float(rdd_model.pvalues[\"below_cutoff\"])\n"
    "\n"
    "print(f\"RDD effect at cutoff (LATE): {effect:+.4f}\")\n"
    "print(f\"Std error (HC3):             {se:.4f}\")\n"
    "print(f\"p-value:                     {pval:.4f}\")\n"
    "print(f\"N users in bandwidth:        {len(near):,}\")"
))

# --- Cell 7: Scatter plot markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "### RDD scatter plot (visual check)\n"
    "\n"
    "Bin query confidence into 50 equal-width bins, compute mean task "
    "completion per bin, and plot the result. The vertical dashed line at "
    "0.85 should show a visible jump between the two sides — the visual "
    "fingerprint of the discontinuity the local linear regression is estimating."
))

# --- Cell 8: Scatter plot code ---
cells.append(nbf.v4.new_code_cell(
    "plt.rcParams['figure.dpi'] = 200\n"
    "\n"
    "n_bins = 50\n"
    "df[\"conf_bin\"] = pd.cut(df.query_confidence, bins=n_bins)\n"
    "bin_means = df.groupby(\"conf_bin\", observed=True).agg(\n"
    "    completion_mean=(\"task_completed\", \"mean\"),\n"
    "    bin_mid=(\"query_confidence\", \"mean\"),\n"
    "    below=(\"routed_to_premium\", \"max\"),\n"
    ").reset_index()\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
    "prem_bins = bin_means[bin_means.below == 1]\n"
    "cheap_bins = bin_means[bin_means.below == 0]\n"
    "ax.scatter(prem_bins.bin_mid, prem_bins.completion_mean,\n"
    "           color=\"#C44E52\", alpha=0.8, s=28,\n"
    "           label=\"Premium-routed bins\")\n"
    "ax.scatter(cheap_bins.bin_mid, cheap_bins.completion_mean,\n"
    "           color=\"#4C72B0\", alpha=0.8, s=28,\n"
    "           label=\"Cheap-routed bins\")\n"
    "ax.axvline(CUTOFF, color=\"#555555\", ls=\"--\", lw=1.5)\n"
    "ax.text(CUTOFF + 0.01, 0.97, \"cutoff (0.85)\",\n"
    "        transform=ax.get_xaxis_transform(), fontsize=9, color=\"#555555\")\n"
    "ax.set_xlabel(\"Query confidence score\")\n"
    "ax.set_ylabel(\"Mean task completion rate\")\n"
    "ax.set_title(\"RDD scatter: mean completion by confidence bin\",\n"
    "             fontsize=11, loc=\"left\")\n"
    "ax.legend(frameon=False)\n"
    "ax.spines[\"top\"].set_visible(False)\n"
    "ax.spines[\"right\"].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# --- Cell 9: Step 2 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 2: Bandwidth sensitivity\n"
    "\n"
    "A wider bandwidth includes more observations but risks including users "
    "whose assignment is less comparable. A narrower bandwidth is cleaner "
    "but noisier. Stable estimates across bandwidths build confidence that "
    "the jump is real, not an artifact of the bandwidth choice."
))

# --- Cell 10: Step 2 code ---
cells.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for bandwidth in [0.05, 0.10, 0.15, 0.20]:\n"
    "    sub = df[(df.query_confidence > CUTOFF - bandwidth) &\n"
    "             (df.query_confidence < CUTOFF + bandwidth)].copy()\n"
    "    sub[\"below_cutoff\"] = (sub.query_confidence < CUTOFF).astype(int)\n"
    "    sub[\"rc\"] = sub.query_confidence - CUTOFF\n"
    "    m = smf.ols(\n"
    "        \"task_completed ~ below_cutoff + rc + below_cutoff:rc\",\n"
    "        data=sub\n"
    "    ).fit(cov_type=\"HC3\")\n"
    "    rows.append({\n"
    "        \"bandwidth\": bandwidth,\n"
    "        \"n\": len(sub),\n"
    "        \"effect\": float(m.params[\"below_cutoff\"]),\n"
    "        \"se\": float(m.bse[\"below_cutoff\"]),\n"
    "        \"p\": float(m.pvalues[\"below_cutoff\"]),\n"
    "    })\n"
    "\n"
    "bw_table = pd.DataFrame(rows)\n"
    "print(bw_table.round(4).to_string(index=False))"
))

# --- Cell 11: Step 3 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 3: Density check near the cutoff (manipulation diagnostic)\n"
    "\n"
    "If users could manipulate their confidence score to land just below 0.85 "
    "and get premium routing, you would see a spike in observations just below "
    "the threshold. Count users in 2-percentage-point bins around 0.85. "
    "Roughly uniform counts indicate no manipulation."
))

# --- Cell 12: Step 3 code ---
cells.append(nbf.v4.new_code_cell(
    "print(\"User counts in 2-percentage-point bins around 0.85:\")\n"
    "bins = [(0.80, 0.82), (0.82, 0.84), (0.84, 0.86), (0.86, 0.88), (0.88, 0.90)]\n"
    "bin_counts = []\n"
    "for lo, hi in bins:\n"
    "    cnt = int(((df.query_confidence >= lo) &\n"
    "               (df.query_confidence < hi)).sum())\n"
    "    bin_counts.append(cnt)\n"
    "    print(f\"  [{lo:.2f}, {hi:.2f}):  n = {cnt:,}\")\n"
    "\n"
    "spread = max(bin_counts) - min(bin_counts)\n"
    "print(f\"\\nSpread across 5 bins: {spread:,} users\")\n"
    "print(\"(No sharp spike at 0.85 = no manipulation evidence)\")"
))

# --- Cell 13: Step 4 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 4: Quadratic robustness check (bw=0.10)\n"
    "\n"
    "If the relationship between confidence and completion is curved rather "
    "than linear, a misspecified linear fit can create a spurious jump. "
    "Adding quadratic terms (`rc^2`) tests whether the linear specification "
    "was sufficient. Agreement between the two estimates builds confidence "
    "in the linear result."
))

# --- Cell 14: Step 4 code ---
cells.append(nbf.v4.new_code_cell(
    "bw = 0.10\n"
    "near = df[(df.query_confidence > CUTOFF - bw) &\n"
    "          (df.query_confidence < CUTOFF + bw)].copy()\n"
    "near[\"below_cutoff\"] = (near.query_confidence < CUTOFF).astype(int)\n"
    "near[\"rc\"] = near.query_confidence - CUTOFF\n"
    "near[\"rc2\"] = near.rc ** 2\n"
    "\n"
    "lin_model = smf.ols(\n"
    "    \"task_completed ~ below_cutoff + rc + below_cutoff:rc\",\n"
    "    data=near\n"
    ").fit(cov_type=\"HC3\")\n"
    "quad_model = smf.ols(\n"
    "    \"task_completed ~ below_cutoff + rc + below_cutoff:rc + rc2 + below_cutoff:rc2\",\n"
    "    data=near\n"
    ").fit(cov_type=\"HC3\")\n"
    "\n"
    "lin_effect = float(lin_model.params[\"below_cutoff\"])\n"
    "quad_effect = float(quad_model.params[\"below_cutoff\"])\n"
    "quad_se = float(quad_model.bse[\"below_cutoff\"])\n"
    "quad_p = float(quad_model.pvalues[\"below_cutoff\"])\n"
    "\n"
    "print(f\"Linear RDD    effect: {lin_effect:+.4f}  \"\n"
    "      f\"p = {float(lin_model.pvalues['below_cutoff']):.4f}\")\n"
    "print(f\"Quadratic RDD effect: {quad_effect:+.4f}  \"\n"
    "      f\"p = {quad_p:.4f}  SE = {quad_se:.4f}\")\n"
    "print(f\"Linear-vs-quadratic gap: {lin_effect - quad_effect:+.4f}\")"
))

# --- Cell 15: Step 5 markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## Step 5: Bootstrap 95% confidence intervals\n"
    "\n"
    "Non-parametric bootstrap with 500 replicates, seed=7. For each "
    "replicate, resample the bandwidth-restricted slice with replacement "
    "and refit the linear RDD. The 2.5th and 97.5th percentiles of the "
    "bootstrap distribution give the 95% confidence interval."
))

# --- Cell 16: Step 5 code ---
cells.append(nbf.v4.new_code_cell(
    "def bootstrap_rdd(df, cutoff, bw, quadratic=False, n_reps=500, seed=7):\n"
    "    \"\"\"Bootstrap 95% CI for the RDD jump estimate.\"\"\"\n"
    "    rng = np.random.default_rng(seed)\n"
    "    near = df[(df.query_confidence > cutoff - bw) &\n"
    "              (df.query_confidence < cutoff + bw)].copy()\n"
    "    near[\"below_cutoff\"] = (near.query_confidence < cutoff).astype(int)\n"
    "    near[\"rc\"] = near.query_confidence - cutoff\n"
    "    if quadratic:\n"
    "        near[\"rc2\"] = near.rc ** 2\n"
    "        formula = (\"task_completed ~ below_cutoff + rc + below_cutoff:rc\"\n"
    "                   \" + rc2 + below_cutoff:rc2\")\n"
    "    else:\n"
    "        formula = \"task_completed ~ below_cutoff + rc + below_cutoff:rc\"\n"
    "\n"
    "    n = len(near)\n"
    "    estimates = np.empty(n_reps)\n"
    "    for i in range(n_reps):\n"
    "        sample = near.iloc[rng.integers(0, n, size=n)]\n"
    "        m = smf.ols(formula, data=sample).fit()\n"
    "        estimates[i] = m.params[\"below_cutoff\"]\n"
    "    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))\n"
    "\n"
    "\n"
    "# Linear RDD bootstrap (bw=0.10)\n"
    "lin_lo, lin_hi = bootstrap_rdd(df, CUTOFF, bw=0.10)\n"
    "print(f\"Linear RDD (bw=0.10)  95% CI: [{lin_lo:+.4f}, {lin_hi:+.4f}]\")"
))

# --- Cell 17: Results summary ---
cells.append(nbf.v4.new_markdown_cell(
    "## Results summary\n"
    "\n"
    "| Quantity | Value |\n"
    "|---|---|\n"
    "| Dataset | 50,000 rows, 16 columns |\n"
    "| Premium-routed (confidence < 0.85) | 38,874 |\n"
    "| Cheap-routed (confidence >= 0.85) | 11,126 |\n"
    "| Confidence: mean / std | 0.715 / 0.159 |\n"
    "| Confidence: min / 25% / 50% / 75% / max | 0.078 / 0.611 / 0.736 / 0.838 / 0.998 |\n"
    "| Naive premium-vs-cheap effect | +0.0632 (ground truth +0.06) |\n"
    "| **Step 1 — linear RDD (bw=0.10)** | |\n"
    "| LATE | +0.0548 |\n"
    "| SE (HC3) | 0.0131 |\n"
    "| p-value | < 0.0001 |\n"
    "| N in bandwidth | 21,689 |\n"
    "| 95% bootstrap CI | [+0.0278, +0.0817] |\n"
    "| **Step 2 — bandwidth sensitivity** | |\n"
    "| bw=0.05, n=11,554 | effect +0.0635, SE 0.0183, p=0.0005, CI [+0.0244, +0.0986] |\n"
    "| bw=0.10, n=21,689 | effect +0.0548, SE 0.0131, p<0.0001, CI [+0.0278, +0.0817] |\n"
    "| bw=0.15, n=29,137 | effect +0.0618, SE 0.0112, p<0.0001, CI [+0.0381, +0.0823] |\n"
    "| bw=0.20, n=34,074 | effect +0.0614, SE 0.0107, p<0.0001, CI [+0.0420, +0.0808] |\n"
    "| **Step 3 — density check (2pp bins)** | |\n"
    "| [0.80, 0.82) | 2,461 |\n"
    "| [0.82, 0.84) | 2,481 |\n"
    "| [0.84, 0.86) | 2,335 |\n"
    "| [0.86, 0.88) | 2,229 |\n"
    "| [0.88, 0.90) | 2,048 |\n"
    "| Spread across 5 bins | 433 users |\n"
    "| **Step 4 — quadratic robustness (bw=0.10)** | |\n"
    "| Linear RDD effect | +0.0548 |\n"
    "| Quadratic RDD effect | +0.0569 (SE 0.0196, p=0.0036) |\n"
    "| Linear-vs-quadratic gap | -0.0022 |\n"
    "| Quadratic 95% CI | [+0.0205, +0.0959] |\n"
    "\n"
    "All bandwidth estimates cluster around the ground-truth +0.06 effect "
    "with CIs that cover the truth. The density check shows no manipulation "
    "spike at 0.85. Linear and quadratic specifications agree within 0.002 "
    "percentage points, confirming the linear fit is adequate."
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
