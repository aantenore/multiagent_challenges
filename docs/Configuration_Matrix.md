# 📔 The Book of Mirror: Volume VI — Configuration Matrix

Mirror is a study in **Extensibility through Abstraction**. This volume provides a complete reference for every tunable parameter in `settings.py` and `.env`.

## 1. Core Model Tiering

Mirror uses a triple-provider architecture. You can mix and match providers (e.g., Gemini for cheap layers, OpenAI for the brain).

| Parameter | Default | Tier | Purpose |
|-----------|---------|------|---------|
| `llm_provider` | "gemini" | Global | Primary API driver ("openai", "gemini", "anthropic"). |
| `nano_model_name` | "gpt-5-nano" | L0 | Low-latency, ultra-cheap anomaly filtering. |
| `cheap_model_name`| "gpt-5-mini" | L1 | Balanced reasoning for Swarm Experts. |
| `smart_model_name`| "gpt-5.4" | L2 | High-IQ arbitration and economic analysis. |

## 2. Economic & Anomaly Parameters

These parameters define the "Sensitivity" of the system.

- **`l0_lower_threshold`** (Default: 0.15): If L0 confidence is below this, it forces escalation to L1.
- **`l0_upper_threshold`** (Default: 0.85): If L0 confidence is above this AND prediction is 0, subsequent layers are skipped.
- **`fn_cost`** (Default: 5.0): The weight of a False Negative (missed case). Higher values make L2 more "paranoid."
- **`fp_cost`** (Default: 1.0): The weight of a False Positive (false alarm). Higher values increase filtering.

## 3. Swarm Dynamics

- **`swarm_min_agents`** (Default: 1): Minimum experts per role.
- **`swarm_max_agents`** (Default: 5): Maximum experts for complex cases.
- **`swarm_complexity_threshold`** (Default: 0.2): The point at which the swarm starts scaling linearly.
- **`swarm_temp_spread`** (Default: 0.5): The temperature offset between agents in a swarm to foster creative disagreement.

## 4. Feature Engineering Abstractions

- **`profile_role`**: The key in the manifest designated for static identity data.
- **`context_role`**: The key for unstructured grounding data.
- **`feature_ignore_columns`**: A list of strings (regex-compatible) that the engineering engine will skip (e.g., IDs, Timestamps, PII).

---
*Volume VII: Deep Troubleshooting Guide coming next.*
