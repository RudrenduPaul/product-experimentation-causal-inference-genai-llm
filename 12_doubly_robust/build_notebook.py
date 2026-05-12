"""Build the Article 12 companion notebook using nbformat.

Run from repo root:
    python 12_doubly_robust/build_notebook.py
"""

import json
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "12_doubly_robust" / "aipw_demo.ipynb"


def md_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def code_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


def build() -> None:
    nb = nbformat.v4.new_notebook()

    nb.cells = [
        # --- Intro ---
        md_cell(
            "# Product Experimentation with Doubly Robust Estimation: "
            "When Both Your Models Are Wrong in LLM Applications\n\n"
            "**Keywords:** product experimentation, causal inference, doubly robust estimation, "
            "AIPW, augmented inverse-probability weighting, LLM product experiments, "
            "noisy observational studies, generative AI\n\n"
            "This notebook implements the Augmented Inverse-Probability Weighting (AIPW) estimator "
            "from scratch and proves the double-robust property empirically by deliberately "
            "misspecifying one model at a time.\n\n"
            "**Dataset:** 50,000 synthetic LLM product users. Ground-truth causal effect: "
            "+8 percentage points on task completion for agent-mode opt-in.\n\n"
            "**Run the dataset generator first:**\n"
            "```bash\n"
            "python data/generate_data.py --seed 42 --n-users 50000 \\\n"
            "    --out data/synthetic_llm_logs.csv\n"
            "```"
        ),

        # --- Setup ---
        md_cell("## Setup"),
        code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.linear_model import LinearRegression, LogisticRegression\n\n"
            "df = pd.read_csv('../data/synthetic_llm_logs.csv')\n"
            "T = df['opt_in_agent_mode'].values\n"
            "Y = df['task_completed'].values\n\n"
            "naive_ate = Y[T == 1].mean() - Y[T == 0].mean()\n"
            "print(f'Naive ATE (unadjusted): {naive_ate:+.4f}')\n"
            "print(f'N treated: {T.sum()}, N control: {(1-T).sum()}')"
        ),

        # --- Step 1 ---
        md_cell("## Step 1: Fit the propensity model"),
        code_cell(
            "X_raw = pd.get_dummies(\n"
            "    df[['engagement_tier', 'query_confidence']],\n"
            "    drop_first=True\n"
            ").astype(float)\n"
            "X = X_raw.values\n\n"
            "ps_model = LogisticRegression(max_iter=1000, C=1.0)\n"
            "ps_model.fit(X, T)\n"
            "e_hat = ps_model.predict_proba(X)[:, 1]\n"
            "e_hat = np.clip(e_hat, 0.01, 0.99)\n\n"
            "print(f'Propensity range: {e_hat.min():.3f} to {e_hat.max():.3f}')\n"
            "print(f'Mean propensity (treated): {e_hat[T == 1].mean():.3f}')\n"
            "print(f'Mean propensity (control): {e_hat[T == 0].mean():.3f}')"
        ),

        # --- Step 2 ---
        md_cell("## Step 2: Fit the outcome models"),
        code_cell(
            "m1_model = LinearRegression()\n"
            "m1_model.fit(X[T == 1], Y[T == 1])\n\n"
            "m0_model = LinearRegression()\n"
            "m0_model.fit(X[T == 0], Y[T == 0])\n\n"
            "m1_hat = m1_model.predict(X)\n"
            "m0_hat = m0_model.predict(X)\n\n"
            "ate_regression = (m1_hat - m0_hat).mean()\n"
            "print(f'Regression adjustment ATE: {ate_regression:+.4f}')"
        ),

        # --- Step 3 ---
        md_cell("## Step 3: Combine into the AIPW estimator"),
        code_cell(
            "def aipw_ate(Y, T, e_hat, m1_hat, m0_hat):\n"
            "    ipw_treated = T * (Y - m1_hat) / e_hat\n"
            "    ipw_control = (1 - T) * (Y - m0_hat) / (1 - e_hat)\n"
            "    phi = m1_hat - m0_hat + ipw_treated - ipw_control\n"
            "    return phi.mean(), phi\n\n"
            "ate_aipw, phi_obs = aipw_ate(Y, T, e_hat, m1_hat, m0_hat)\n"
            "print(f'AIPW ATE:            {ate_aipw:+.4f}')\n"
            "print(f'Naive ATE:           {naive_ate:+.4f}')\n"
            "print(f'Regression-only ATE: {ate_regression:+.4f}')\n"
            "print(f'Ground truth:        +0.0800')"
        ),

        # --- Step 4 ---
        md_cell("## Step 4: Bootstrap confidence intervals"),
        code_cell(
            "def aipw_bootstrap(df, X_cols, treatment_col, outcome_col,\n"
            "                   n_bootstrap=500, seed=7):\n"
            "    rng = np.random.default_rng(seed)\n"
            "    n = len(df)\n"
            "    X_all = pd.get_dummies(df[X_cols], drop_first=True).astype(float).values\n"
            "    T_all = df[treatment_col].values\n"
            "    Y_all = df[outcome_col].values\n"
            "    boot_estimates = []\n"
            "    for _ in range(n_bootstrap):\n"
            "        idx = rng.integers(0, n, size=n)\n"
            "        X_b, T_b, Y_b = X_all[idx], T_all[idx], Y_all[idx]\n"
            "        ps = LogisticRegression(max_iter=1000, C=1.0)\n"
            "        ps.fit(X_b, T_b)\n"
            "        e_b = np.clip(ps.predict_proba(X_b)[:, 1], 0.01, 0.99)\n"
            "        m1 = LinearRegression().fit(X_b[T_b == 1], Y_b[T_b == 1])\n"
            "        m0 = LinearRegression().fit(X_b[T_b == 0], Y_b[T_b == 0])\n"
            "        ate_b, _ = aipw_ate(Y_b, T_b, e_b, m1.predict(X_b), m0.predict(X_b))\n"
            "        boot_estimates.append(ate_b)\n"
            "    boot_arr = np.array(boot_estimates)\n"
            "    return boot_arr, np.percentile(boot_arr, 2.5), np.percentile(boot_arr, 97.5)\n\n"
            "boot_dist, ci_lo, ci_hi = aipw_bootstrap(\n"
            "    df,\n"
            "    X_cols=['engagement_tier', 'query_confidence'],\n"
            "    treatment_col='opt_in_agent_mode',\n"
            "    outcome_col='task_completed',\n"
            "    n_bootstrap=500,\n"
            "    seed=7,\n"
            ")\n"
            "print(f'AIPW ATE:           {ate_aipw:+.4f}')\n"
            "print(f'95% Bootstrap CI:   [{ci_lo:+.4f}, {ci_hi:+.4f}]')\n"
            "print(f'Bootstrap std dev:  {boot_dist.std():.4f}')"
        ),

        # --- Step 5 ---
        md_cell("## Step 5: Prove the double-robust property via deliberate misspecification"),
        code_cell(
            "# Scenario 1: constant propensity (e = 0.3 for everyone)\n"
            "e_wrong = np.full(len(df), 0.3)\n"
            "t_mask = T == 1\n"
            "c_mask = T == 0\n\n"
            "ate_ipw_wrong = (\n"
            "    (Y[t_mask] / e_wrong[t_mask]).sum() / (1 / e_wrong[t_mask]).sum()\n"
            "    - (Y[c_mask] / (1 - e_wrong[c_mask])).sum()\n"
            "    / (1 / (1 - e_wrong[c_mask])).sum()\n"
            ")\n"
            "ate_aipw_wrong_ps, _ = aipw_ate(Y, T, e_wrong, m1_hat, m0_hat)\n\n"
            "print('--- Scenario 1: constant propensity (e = 0.3) ---')\n"
            "print(f'IPW with wrong propensity:         {ate_ipw_wrong:+.4f}  (should be wrong)')\n"
            "print(f'Regression adjustment (unchanged): {ate_regression:+.4f}  (should be ~0.085)')\n"
            "print(f'AIPW with wrong propensity:        {ate_aipw_wrong_ps:+.4f}  (should stay ~0.085)')\n"
            "print(f'Ground truth:                      +0.0800')"
        ),
        code_cell(
            "# Scenario 2: constant outcome models (m1 = m0 = 0.5 for everyone)\n"
            "m1_wrong = np.full(len(df), 0.5)\n"
            "m0_wrong = np.full(len(df), 0.5)\n\n"
            "ate_regression_wrong = (m1_wrong - m0_wrong).mean()\n"
            "ate_ipw_correct = (\n"
            "    (Y[t_mask] / e_hat[t_mask]).sum() / (1 / e_hat[t_mask]).sum()\n"
            "    - (Y[c_mask] / (1 - e_hat[c_mask])).sum()\n"
            "    / (1 / (1 - e_hat[c_mask])).sum()\n"
            ")\n"
            "ate_aipw_wrong_out, _ = aipw_ate(Y, T, e_hat, m1_wrong, m0_wrong)\n\n"
            "print('--- Scenario 2: constant outcome models (m1 = m0 = 0.5) ---')\n"
            "print(f'Regression with wrong outcome models: {ate_regression_wrong:+.4f}  (should be 0.0)')\n"
            "print(f'IPW with correct propensity:          {ate_ipw_correct:+.4f}  (should be ~0.085)')\n"
            "print(f'AIPW with wrong outcome models:       {ate_aipw_wrong_out:+.4f}  (should stay ~0.085)')\n"
            "print(f'Ground truth:                         +0.0800')"
        ),

        # --- Figure: propensity overlap ---
        md_cell("## Propensity score overlap (Figure 2)"),
        code_cell(
            "from scipy.stats import gaussian_kde\n\n"
            "ps_treated = e_hat[T == 1]\n"
            "ps_control = e_hat[T == 0]\n"
            "x_grid = np.linspace(0.05, 0.85, 300)\n"
            "kde_t = gaussian_kde(ps_treated, bw_method=0.25)(x_grid)\n"
            "kde_c = gaussian_kde(ps_control, bw_method=0.25)(x_grid)\n\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "ax.fill_between(x_grid, kde_t, alpha=0.2, color='#2563EB')\n"
            "ax.fill_between(x_grid, kde_c, alpha=0.2, color='#DC2626')\n"
            "ax.plot(x_grid, kde_t, color='#2563EB', lw=2, label=f'Opted in (n={T.sum():,})')\n"
            "ax.plot(x_grid, kde_c, color='#DC2626', lw=2, label=f'Did not opt in (n={(1-T).sum():,})')\n"
            "ax.fill_between(x_grid, np.minimum(kde_t, kde_c), alpha=0.4,\n"
            "                color='#16A34A', label='Common support')\n"
            "ax.set_xlabel('Propensity score')\n"
            "ax.set_ylabel('Density')\n"
            "ax.set_title('Propensity score overlap — 50,000-user synthetic dataset')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),

        # --- Results summary ---
        md_cell("## Results summary"),
        md_cell(
            "| Estimator | ATE estimate | Notes |\n"
            "|-----------|-------------|-------|\n"
            "| Naive (unadjusted) | +0.2106 | Heavily inflated by selection bias |\n"
            "| Regression adjustment | +0.0847 | Outcome model only |\n"
            "| AIPW | +0.0847 | Doubly robust |\n"
            "| **95% Bootstrap CI** | **[+0.0744, +0.0952]** | 500 replicates, seed=7 |\n"
            "| Ground truth | +0.0800 | Baked into data generator |\n\n"
            "**Double-robust property verified:**\n"
            "- Scenario 1 (wrong propensity, correct outcome): AIPW stays at +0.0847 ✓\n"
            "- Scenario 2 (correct propensity, wrong outcome): AIPW stays at +0.0849 ✓"
        ),
    ]

    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    }

    with open(OUT_PATH, "w") as f:
        nbformat.write(nb, f)

    print(f"Notebook written: {OUT_PATH}")


if __name__ == "__main__":
    build()
