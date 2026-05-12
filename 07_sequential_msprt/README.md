# Article 7 — mSPRT and Sequential Testing

Stop an AI product experiment early without inflating your false positive rate. This folder implements the mixture Sequential Probability Ratio Test (mSPRT) using always-valid e-values and applies it to the shared 50,000-user synthetic LLM product dataset.

## What this teaches

- Why peeking at running p-values inflates the false positive rate to 30%
- How e-values differ from p-values and why they enable valid optional stopping
- How to implement the mSPRT Bayes factor for Bernoulli outcomes in Python
- The real power trade-off between mSPRT and a fixed-sample t-test
- When mSPRT fails: prior misspecification, non-stationarity, multiple metrics, minimum runtime

## Run

```bash
# From repo root
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
python 07_sequential_msprt/msprt_demo.py
```

## What you should see

All values below are from executed runs against `synthetic_llm_logs.csv` (50,000 users, seed=42).

| Metric | Value |
|--------|-------|
| Treated n (wave 1) | 24,937 |
| Control n (wave 2) | 25,063 |
| Treated mean (task_completed) | 0.6202 |
| Control mean (task_completed) | 0.5718 |
| Observed lift | 0.0485 (4.85pp) |
| True effect (data generator) | 5pp |
| Peeking false positive rate (30 days, 60 obs/arm/day) | 30.2% |
| Single-look false positive rate | 4.2% |
| Null sanity check: e-value at end (500 obs, null data) | 0.078 |
| Null sanity check: max e-value | 2.188 |
| mSPRT stopping day (real data, seed=42) | 25.9 |
| E-value at stopping | 20.9 |
| Final e-value at day 30 | 75.64 |
| mSPRT power (sim, n=1800/arm, 5pp lift) | 49.3% |
| Fixed-sample t-test power (sim, n=1800/arm) | 88.7% |
| Median mSPRT stop day | 30.0 / 30 |
| Fraction stopping early | 49.3% |
| Validate: could have stopped on day | 27.1 |
| Bootstrap 95% CI (500 replicates, seed=7) | [4.07pp, 5.81pp] |
| Ground-truth 5pp inside CI | yes |

## Files

| File | Description |
|------|-------------|
| `msprt_demo.py` | Main analysis script — all steps including bootstrap CIs |
| `msprt_demo.ipynb` | Executed Jupyter notebook with pre-saved outputs |
| `build_notebook.py` | Script that generates `msprt_demo.ipynb` from source |
| `generate_msprt_charts.py` | Produces Figure 1 (conceptual) and Figure 2 (real data) |
| `msprt_evalue_schematic.png` | Figure 1 — conceptual e-value trajectory schematic |
| `msprt_evalue_real_data.png` | Figure 2 — real dataset e-value trajectory |
| `README.md` | This file |
