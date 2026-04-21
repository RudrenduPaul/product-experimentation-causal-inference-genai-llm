# Causal Inference for GenAI and LLM Applications

Working Python code for measuring the causal impact of AI features, prompt changes, model routing decisions, agentic product launches, and staged LLM rollouts in production.

Every notebook runs on a single shared synthetic dataset (`data/synthetic_llm_logs.csv`) that simulates a realistic SaaS product with an AI assistant feature. You clone the repo, run `generate_data.py` once, and then every technique can be applied to the same 10,000-row dataset and compared directly.

## Why this repo exists

Standard A/B testing breaks for modern AI products. Rollouts happen in waves, users opt into AI features, model routing depends on confidence thresholds, and global model upgrades leave no control group. This repo shows how to use modern causal inference methods to measure real impact in those scenarios, with code that runs end to end.

## Companion articles (freeCodeCamp)

| # | Article | Folder | Status |
|---|---------|--------|--------|
| 1 | Why A/B Testing Breaks for Staged AI Feature Rollouts (and How to Use Difference-in-Differences in Python Instead) | `01_did_staged_rollouts/` | Draft ready |
| 2 | When Users Opt Into Your AI Feature: Propensity Score Methods for Causal Inference in Python | `02_propensity_opt_in/` | Draft ready |
| 3 | The Threshold Trick: Regression Discontinuity Design for Causal Inference in Confidence-Based LLM Routing | `03_rdd_confidence_threshold/` | Draft ready |
| 4 | No Control Group, No Problem: Synthetic Control Methods for Causal Inference in Global LLM Rollouts | `04_synthetic_control/` | Coming soon |
| 5 | Uplift Modeling for Causal Inference: Finding the Users Your LLM Feature Actually Helps | `05_uplift_modeling/` | Coming soon |
| 6 | Regression Models for Causal Inference: Estimating LLM Feature Impact with Python and statsmodels | `06_regression/` | Coming soon |
| 7 | What Would Have Happened If We Shipped a Different Prompt? A Counterfactual Causal Inference Walkthrough | `07_counterfactual/` | Coming soon |
| 8 | When Your LLM Routing Is Confounded: Instrumental Variable Analysis for Clean Causal Inference | `08_instrumental_variables/` | Coming soon |
| 9 | Doubly Robust Estimation: The Safety Net for Causal Inference When Your LLM Experiment Is Noisy | `09_doubly_robust/` | Coming soon |
| 10 | Causal Inference in Production: Lessons from How Airbnb, Netflix, Lyft, and Uber Measure AI Impact | `10_case_studies/` | Coming soon |
| 11 (recap) | From A/B Tests to Causal Inference: A Practitioner's Map of How to Measure AI Feature Impact in Production | `11_recap/` | Publishes after Articles 1-10 are live |

## Quick start

```bash
git clone https://github.com/RudrenduPaul/causal-inference-for-genai-llm-applications.git
cd causal-inference-for-genai-llm-applications
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_data.py --seed 42 --n-users 10000 --out data/synthetic_llm_logs.csv
python 01_did_staged_rollouts/did_demo.py
```

## Repo structure

```text
causal-inference-for-genai-llm-applications/
  README.md
  requirements.txt
  data/
    generate_data.py             # Reproducible synthetic LLM product dataset
    synthetic_llm_logs.csv       # 10,000 rows, 16 columns (generated, gitignored)
  01_did_staged_rollouts/        # Article 1 — difference-in-differences
  02_propensity_opt_in/          # Article 2 — propensity score matching / IPW
  03_rdd_confidence_threshold/   # Article 3 — regression discontinuity
  04_synthetic_control/          # Article 4 (coming soon)
  05_uplift_modeling/            # Article 5 (coming soon)
  06_regression/                 # Article 6 (coming soon)
  07_counterfactual/             # Article 7 (coming soon)
  08_instrumental_variables/     # Article 8 (coming soon)
  09_doubly_robust/              # Article 9 (coming soon)
  10_case_studies/               # Article 10 (coming soon)
  11_recap/                      # Article 11 — capstone recap (coming last)
```

## Dataset schema

Each row is one user of an AI-assisted SaaS product.

| Column | Type | Meaning |
|--------|------|---------|
| `user_id` | int | Unique user ID |
| `workspace_id` | int | Enterprise workspace (used for staged rollout waves) |
| `engagement_tier` | str | `light`, `medium`, `heavy` — pre-existing engagement |
| `signup_week` | int | Weeks since product launch when the user joined |
| `wave` | int | `1` or `2` — which staged rollout wave this workspace belongs to |
| `treatment_week` | int | Calendar week when this user's workspace got the new feature |
| `treated_post` | int | 1 if the user was active after their workspace's treatment week |
| `prompt_variant` | int | 0 = control prompt, 1 = new prompt (randomized) |
| `opt_in_agent_mode` | int | 1 if user opted into agent mode (endogenous: power users opt in more) |
| `query_confidence` | float | Model confidence score for this user's typical query |
| `routed_to_premium` | int | 1 if queries were routed to the premium model (threshold < 0.85) |
| `task_completed` | int | Primary outcome: did the AI complete the user's task |
| `thumbs_up` | int | Secondary outcome: did the user give thumbs-up feedback |
| `session_minutes` | float | Session length in minutes |
| `cost_usd` | float | Per-session cost in USD |
| `retained_7d` | int | 1 if the user returned within 7 days |

Ground-truth effects baked into the generator (for validating that estimators recover them):

- New prompt → +4 percentage points on `task_completed`
- Opting into agent mode → +8 pp
- Premium model routing → +6 pp
- Staged rollout post-treatment → +5 pp

## Author

Rudrendu Paul is an AI/ML practitioner with 15+ years of enterprise experience. He writes about LLM evaluation engineering, causal inference for AI products, and production ML systems. His book *LLM Evaluation Engineering* (CRC Press / Taylor & Francis) is forthcoming.

- Website: [rudrendupaul.medium.com](https://rudrendupaul.medium.com/)
- ORCID: [0009-0008-0141-4690](https://orcid.org/0009-0008-0141-4690)
- LinkedIn: [linkedin.com/in/rudrendupaul](https://www.linkedin.com/in/rudrendupaul/)
- freeCodeCamp: [freecodecamp.org/news/author/rudrendupaul](https://www.freecodecamp.org/news/author/rudrendupaul/)

## License

MIT
