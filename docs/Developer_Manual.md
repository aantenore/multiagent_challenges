# 📙 The Book of Mirror: Volume III — Developer Manual

A technical deep-dive into the Mirror codebase for engineers and architects.

## 1. The Core Loop (`pipeline.py`)
The pipeline follows a `Fit -> Sanity -> Evaluate` lifecycle.

### 1.1 The Sanity Check (predict_train)
Before predicting on unseen data, Mirror predicts on its own training set. This serves two purposes:
1.  **Baseline Validation**: Confirms L0 is learning.
2.  **RAG Priming**: Populates memory with actual examples of training outliers.

## 2. LLM Provider Tiering (`llm_provider.py`)
Mirror calculates its own budget using a triple-tier model:
- **Nano**: $0.01 per 1M tokens. Used for L0 and simple filtering.
- **Cheap**: $0.15 per 1M tokens. Used for L1 SwarmExperts.
- **Smart**: $15.00 per 1M tokens. Reserved for high-stakes L2 arbitration.

## 3. Parallelism Strategies
- **Entity Level**: Pipeline processes entities in chunks using `ThreadPoolExecutor`.
- **Agent Level**: `RoleCoordinator` executes N agents in parallel to reach consensus.
- **IO Level**: Asynchronous logging and Langfuse tracing ensure no bottlenecks during API calls.

---

## 4. Troubleshooting

| Symptom | Probable Cause | Fix |
|---------|----------------|-----|
| `ValidationError` | Incorrect manifest role | Use `csv`, `json`, or `md` for format. |
| `IndexError` | Insufficient ACF data | Increase training set size for the entity. |
| `RateLimitError` | API Quota reached | Change `max_workers` from 10 to 3 in `pipeline.py`. |

---
*Volume IV: Architectural Blueprints & Flowcharts coming next.*
