# Causal Inference for GenAI and LLM Applications

Working Python code for measuring the causal impact of AI features, prompt changes, model routing decisions, agentic product launches, and staged LLM rollouts in production.

Every notebook runs on a single shared synthetic dataset (`data/synthetic_llm_logs.csv`) that simulates a realistic SaaS product with an AI assistant feature. You clone the repo, run `generate_data.py` once, and then every technique can be applied to the same 10,000-row dataset and compared directly.

## Why this repo exists

Standard A/B testing breaks for modern AI products. Rollouts happen in waves, users opt into AI features, model routing depends on confidence thresholds, and global model upgrades leave no control group. This repo shows how to use modern causal inference methods to measure real impact in those scenarios, with code that runs end to end.

## Companion articles (freeCodeCamp)

| # | Article | Status |
|---|---------|--------|
| 0 | From A/B Tests to Causal Inference: How to Measure What Your LLM Features Actually Do in Production | Publishing |
| 1 | Regression Models for Causal Inference: Estimating LLM Feature Impact with Python and statsmodels | Coming soon |
| 2 | What Would Have Happened If We Shipped a Different Prompt? A Counterfactual Causal Inference Walkthrough | Coming soon |
| 3 | When Users Opt Into Your AI Feature: Propensity Score Methods for Causal Inference in Python | Coming soon |
| 4 | Uplift Modeling for Causal Inference: Finding the Users Your LLM Feature Actually Helps | Coming soon |
| 5 | No Control Group, No Problem: Synthetic Control Methods for Causal Inference in Global LLM Rollouts | Coming soon |
| 6 | When Your LLM Routing Is Confounded: Instrumental Variable Analysis for Clean Causal Inference | Coming soon |
| 7 | The Threshold Trick: Regression Discontinuity Design for Causal Inference in Confidence-Based LLM Routing | Coming soon |
| 8 | Shipping AI Features in Waves? Difference-in-Differences for Causal Inference on Staged LLM Rollouts | Coming soon |
| 9 | Doubly Robust Estimation: The Safety Net for Causal Inference When Your LLM Experiment Is Noisy | Coming soon |
| 10 | Causal Inference in Production: Lessons from How Airbnb, Netflix, Lyft, and Uber Measure AI Impact | Coming soon |

## Quick start

```bash
git clone https://github.com/RudrenduPaul/causal-inference-for-genai-llm-applications.git
cd causal-inference-for-genai-llm-applications
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_data.py --seed 42 --n-users 10000 --out data/synthetic_llm_logs.csv
jupyter notebook 00_foundation/foundation_intro.ipynb
```

## Repo structure

```text
causal-inference-for-genai-llm-applications/
  README.md
  requirements.txt
  data/
    generate_data.py             # Reproducible synthetic LLM product dataset
    synthetic_llm_logs.csv       # 10,000 rows, 16 columns (generated)
  00_foundation/                 # Article 0 notebook (all techniques at a glance)
  01_regression/                 # Article 1
  02_counterfactual/             # Article 2
  03_propensity_score/           # Article 3
  04_uplift/                     # Article 4
  05_synthetic_control/          # Article 5
  06_instrumental_variable/      # Article 6
  07_rdd/                        # Article 7
  08_did/                        # Article 8
  09_doubly_robust/              # Article 9
  10_case_studies/               # Article 10
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
