# Product Experimentation and Causal Inference for GenAI and LLM Applications

Working Python notebooks for measuring the causal impact of AI features, prompt changes, model routing decisions, agentic product launches, and staged LLM rollouts in production.

Every notebook runs on a single shared synthetic dataset (`data/synthetic_llm_logs.csv`) that simulates a realistic SaaS product with an AI assistant feature. You clone the repo, run `generate_data.py` once, and every technique runs against the same 50,000-row dataset so you can compare methods directly.

## Why this repo exists

Standard A/B testing breaks for modern AI products. Rollouts happen in waves, users opt into AI features, model routing depends on confidence thresholds, and global model upgrades leave no clean control group. This repo shows how to use modern causal inference methods to measure real impact in those scenarios, with Python code that runs end to end.

## Quick start

```bash
git clone https://github.com/RudrenduPaul/product-experimentation-causal-inference-genai-llm.git
cd product-experimentation-causal-inference-genai-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_data.py --seed 42 --n-users 50000 --out data/synthetic_llm_logs.csv
jupyter notebook 01_did_staged_rollouts/did_demo.ipynb
```

## Notebooks

| # | Method | Folder | Notebook |
|---|--------|--------|---------|
| 1 | Difference-in-Differences | `01_did_staged_rollouts/` | [did_demo.ipynb](01_did_staged_rollouts/did_demo.ipynb) |
| 2 | Propensity Score Matching / IPW | `02_propensity_opt_in/` | `propensity_demo.ipynb` |
| 3 | Regression Discontinuity Design | `03_rdd_confidence_threshold/` | `rdd_demo.ipynb` |
| 4 | Synthetic Control | `04_synthetic_control/` | `synthetic_control_demo.ipynb` |
| 5 | Uplift Modeling | `05_uplift_modeling/` | `uplift_demo.ipynb` |
| 6 | Regression Adjustment | `06_regression/` | `regression_demo.ipynb` |
| 7 | Counterfactual Reasoning | `07_counterfactual/` | `counterfactual_demo.ipynb` |
| 8 | Instrumental Variables | `08_instrumental_variables/` | `iv_demo.ipynb` |
| 9 | Doubly Robust Estimation | `09_doubly_robust/` | `doubly_robust_demo.ipynb` |
| 10 | Production Case Studies | `10_case_studies/` | `case_studies.ipynb` |

## Repo structure

```text
product-experimentation-causal-inference-genai-llm/
  README.md
  requirements.txt
  data/
    generate_data.py             # Reproducible synthetic LLM product dataset
    synthetic_llm_logs.csv       # 50,000 rows, 16 columns (generated, gitignored)
  01_did_staged_rollouts/
  02_propensity_opt_in/
  03_rdd_confidence_threshold/
  04_synthetic_control/
  05_uplift_modeling/
  06_regression/
  07_counterfactual/
  08_instrumental_variables/
  09_doubly_robust/
  10_case_studies/
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

Rudrendu Paul is an AI/ML practitioner with 15+ years of enterprise experience. He writes about product experimentation, LLM evaluation engineering, causal inference for AI products, and production ML systems.

- LinkedIn: [linkedin.com/in/rudrendupaul](https://www.linkedin.com/in/rudrendupaul/)
- ORCID: [0009-0008-0141-4690](https://orcid.org/0009-0008-0141-4690)
- freeCodeCamp: [freecodecamp.org/news/author/rudrendupaul](https://www.freecodecamp.org/news/author/rudrendupaul/)
- Medium: [rudrendupaul.medium.com](https://rudrendupaul.medium.com/)

## License

MIT
