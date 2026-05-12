"""Build the Article 8 companion notebook using nbformat.

Run from repo root:
    python 08_uplift_modeling/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        08_uplift_modeling/uplift_demo.ipynb \
        --ExecutePreprocessor.timeout=600
"""

import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "uplift_demo.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# ── Intro markdown cell ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# Uplift Modeling for Personalized AI Rollouts

**Keywords:** product experimentation, causal inference, LLM applications, generative AI,
uplift modeling, heterogeneous treatment effects, CATE, T-learner, X-learner, Qini curve,
segmented rollout

This notebook accompanies the freeCodeCamp tutorial on uplift modeling for LLM-based product
experiments. It demonstrates T-learner and X-learner meta-learners, Qini curve evaluation,
and a segmented rollout decision rule on a 50,000-user synthetic SaaS dataset.

**Dataset:** 50,000 synthetic users. Ground-truth causal effect of the AI summary feature:
approximately +8 percentage points on `task_completed`, with per-tier variation across
engagement segments (light, medium, heavy). True CATE ordering: light > medium > heavy.

**Clone and run:**
```bash
git clone https://github.com/RudrenduPaul/product-experimentation-causal-inference-genai-llm.git
cd product-experimentation-causal-inference-genai-llm
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 08_uplift_modeling/uplift_demo.py
```
"""))

# ── Setup ────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv("../data/synthetic_llm_logs.csv")
print(f"Dataset: {len(df):,} users")
print(df[["engagement_tier", "opt_in_agent_mode", "task_completed"]].head(5))

print("\\nOpt-in rate by engagement tier:")
print(df.groupby("engagement_tier").opt_in_agent_mode.mean().round(3))

naive_ate = (
    df[df.opt_in_agent_mode == 1].task_completed.mean()
    - df[df.opt_in_agent_mode == 0].task_completed.mean()
)
print(f"\\nNaive ATE (treated - control): {naive_ate:+.4f}")
print(f"Treated users: {(df.opt_in_agent_mode == 1).sum():,}")
print(f"Control users: {(df.opt_in_agent_mode == 0).sum():,}")
"""))

# ── Naive per-tier gap ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""print("Naive per-tier treated vs. control completion rate:")
for tier in ["light", "medium", "heavy"]:
    sub = df[df.engagement_tier == tier]
    t = sub[sub.opt_in_agent_mode == 1].task_completed.mean()
    c = sub[sub.opt_in_agent_mode == 0].task_completed.mean()
    print(f"  {tier:8s}: treated={t:.3f}, control={c:.3f}, diff={t - c:+.3f}")
"""))

# ── T-learner ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# Build feature matrix
X_full = pd.get_dummies(
    df[["query_confidence", "engagement_tier"]], drop_first=False
).astype(float)
feature_cols = X_full.columns.tolist()
X_all = X_full.values
treated_mask = df.opt_in_agent_mode == 1

X1 = X_all[treated_mask]; Y1 = df[treated_mask].task_completed.values
X0 = X_all[~treated_mask]; Y0 = df[~treated_mask].task_completed.values

m1 = LinearRegression().fit(X1, Y1)
m0 = LinearRegression().fit(X0, Y0)
cate_t = m1.predict(X_all) - m0.predict(X_all)
df["cate_tlearner"] = cate_t

print(f"Mean CATE (T-learner): {cate_t.mean():+.4f}")
print("\\nMean predicted CATE by engagement tier:")
print(df.groupby("engagement_tier").cate_tlearner.mean().round(4))
"""))

# ── X-learner ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# X-learner (Kunzel et al. 2019)
D1 = Y1 - m0.predict(X1)
D0 = m1.predict(X0) - Y0
tau1 = LinearRegression().fit(X1, D1)
tau0 = LinearRegression().fit(X0, D0)
ps = LogisticRegression(max_iter=1000).fit(X_all, df.opt_in_agent_mode.values)
e_x = ps.predict_proba(X_all)[:, 1]
# Kunzel formula: tau(x) = g(x)*tau_1(x) + (1 - g(x))*tau_0(x)
cate_x = e_x * tau1.predict(X_all) + (1 - e_x) * tau0.predict(X_all)
df["cate_xlearner"] = cate_x

print(f"Mean CATE (X-learner): {cate_x.mean():+.4f}")
print("\\nT-learner vs X-learner per tier:")
print(df.groupby("engagement_tier")[["cate_tlearner", "cate_xlearner"]].mean().round(4))
"""))

# ── Qini curve ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""df_sorted = df.sort_values("cate_tlearner", ascending=False).copy()
n = len(df_sorted)
top_ks = np.arange(0.01, 1.01, 0.01)
qini_vals = []
for k in top_ks:
    top_n = max(1, int(k * n))
    sub = df_sorted.iloc[:top_n]
    t_sub = sub[sub.opt_in_agent_mode == 1]
    c_sub = sub[sub.opt_in_agent_mode == 0]
    uplift = t_sub.task_completed.mean() - c_sub.task_completed.mean() if len(t_sub) > 0 and len(c_sub) > 0 else np.nan
    qini_vals.append(uplift)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(top_ks * 100, qini_vals, linewidth=2, label="T-learner Qini")
ax.axhline(naive_ate, color="gray", linestyle="--", label=f"Naive ATE = {naive_ate:.4f}")
ax.set_xlabel("Top-k% of users (sorted by predicted CATE)")
ax.set_ylabel("Observed uplift in top-k group")
ax.set_title("Qini curve: T-learner ranking vs. observed uplift")
ax.legend()
plt.tight_layout()
plt.savefig("qini_curve.png", dpi=140)
plt.show()
print("\\nQini values at selected cutoffs:")
for k in [10, 20, 30, 50, 70, 100]:
    print(f"  Top {k:3d}%: {qini_vals[k-1]:.4f}")
"""))

# ── Segmented rollout ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""threshold = 0.085
selected = df[df.cate_tlearner >= threshold]
suppressed = df[df.cate_tlearner < threshold]

print(f"Rollout threshold: CATE >= {threshold}")
print(f"Selected: {len(selected):,} ({100*len(selected)/len(df):.0f}%)")
print(f"Suppressed: {len(suppressed):,} ({100*len(suppressed)/len(df):.0f}%)")
print("\\nTier composition of selected group:")
print((selected.groupby("engagement_tier").size() / len(selected)).round(3))
print(f"\\nMean CATE selected:   {selected.cate_tlearner.mean():.4f}")
print(f"Mean CATE suppressed: {suppressed.cate_tlearner.mean():.4f}")
"""))

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""def bootstrap_cate_ci(df, X_all, n_reps=500, seed=7):
    rng = np.random.default_rng(seed)
    n = len(df)
    tier_reps = {"light": [], "medium": [], "heavy": []}
    mean_reps = []
    for _ in range(n_reps):
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        X_b = X_all[idx]
        t_b = df_b.opt_in_agent_mode == 1
        m1_b = LinearRegression().fit(X_b[t_b], df_b[t_b].task_completed.values)
        m0_b = LinearRegression().fit(X_b[~t_b], df_b[~t_b].task_completed.values)
        cate_b = m1_b.predict(X_b) - m0_b.predict(X_b)
        df_b["cate"] = cate_b
        for tier in tier_reps:
            tier_reps[tier].append(df_b[df_b.engagement_tier == tier].cate.mean())
        mean_reps.append(cate_b.mean())
    cis = {}
    for tier, vals in tier_reps.items():
        arr = np.array(vals)
        cis[tier] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    arr = np.array(mean_reps)
    cis["mean"] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return cis

print("Running bootstrap (500 replicates, seed=7)...")
cis = bootstrap_cate_ci(df, X_all)
print(f"Mean CATE   95% CI: [{cis['mean'][0]:+.4f}, {cis['mean'][1]:+.4f}]")
print(f"Light tier  95% CI: [{cis['light'][0]:+.4f}, {cis['light'][1]:+.4f}]")
print(f"Medium tier 95% CI: [{cis['medium'][0]:+.4f}, {cis['medium'][1]:+.4f}]")
print(f"Heavy tier  95% CI: [{cis['heavy'][0]:+.4f}, {cis['heavy'][1]:+.4f}]")
"""))

# ── Results summary ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Results summary

| Estimator | Mean CATE | Light tier | Medium tier | Heavy tier |
|-----------|-----------|------------|-------------|------------|
| Naive ATE | +0.2106 (confounded) | — | — | — |
| T-learner | +0.0847 | +0.0954 | +0.0744 | +0.0665 |
| X-learner | +0.0847 | +0.0954 | +0.0744 | +0.0665 |
| Ground truth | ~+0.08 | — | — | — |

**Bootstrap 95% CIs (T-learner, 500 reps, seed=7):**

| Estimate | Lower | Upper |
|----------|-------|-------|
| Mean CATE | +0.0744 | +0.0951 |
| Light tier | +0.0781 | +0.1125 |
| Medium tier | +0.0596 | +0.0892 |
| Heavy tier | +0.0483 | +0.0842 |

**Rollout rule (threshold = 0.085):** 27,203 users selected (54%, all light tier),
22,797 suppressed (46%, all medium and heavy). Mean CATE of selected group: +0.0955.
"""))

nb.cells = cells
nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH}")
