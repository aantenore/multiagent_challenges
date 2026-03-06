# 🪞 Mirror — Universal Adaptive Multi-Agent Framework

Mirror is a production-grade, domain-agnostic **N-stage multi-agent classification pipeline** designed for high-stakes anomaly detection (e.g., healthcare, financial fraud). It optimizes for the **Reply Mirror Challenge** metrics by balancing mathematical precision with contextual LLM reasoning.

The system processes data in **N discrete stages**, where each stage isolates its training context to build a unique "Class 0" (well-being/normality) baseline before executing high-concurrency evaluation.

---

## 🏗️ 3-Tier Multi-Agent Architecture

Mirror implements a hierarchical defense-in-depth strategy, using tiered LLM models to maximize performance while minimizing economic cost.

```
┌─────────────────────────────────────────────────────────┐
│                  manifest.json                          │
│   N stages: [train₁,eval₁] → [train₂,eval₂] → ...       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
          ┌─────────────────────────┐
          │  Per-Level Training     │  train_i only
          │  Dossier + IF Fit       │  (zero-cost math baseline)
          └────────────┬────────────┘
                       ▼
          ┌─────────────────────────┐
          │   Feature Engineering   │  sliding windows, dynamic lag (ACF)
          └────────────┬────────────┘
                       ▼
         ┌──────────────────────────┐
         │  Layer 0 — Nano Filter   │  GPT-5-Nano / Gemini-3.1-Lite
         │  (Stat + LLM Hybrid)     │  Configurable Engine (isolation/llm)
         └─────┬───────────┬────────┘
       baseline│           │anomaly (escalate to L1)
               ▼           ▼
            OUTPUT   ┌──────────────────────────────────┐
          (pred=0)   │  Layer 1 — Cheap Domain Swarm    │
                     │  Parallel Anti-FP Experts        │
                     │  (GPT-5-Mini / Gemini-3-Flash)   │
                     └──────────┬───────────────────────┘
                                ▼
                     ┌──────────────────────────┐
                     │  Layer 2 — Smart Orchestr     │
                     │  Final Economic Decision  │
                     │  (GPT-5.4 / Gemini-3.1-Pro)│
                     └──────────┬───────────────┘
                                ▼
                             OUTPUT → predictions_{stage}.txt
                                │
                     ┌──────────▼───────┐
                     │  RAG (per-level) │  ChromaDB: reset between levels
                     └──────────────────┘
```

---

## 🧬 Core Intelligence Components

### 1. Dynamic Time-Aware Feature Extraction (ACF)
To maximize **Agentic Efficiency**, Mirror adapts to the unique physiological or behavioral rhythms of each entity.
- **Auto-Lag Detection**: Uses `statsmodels` to compute the **Autocorrelation (ACF)** of historical series. The system automatically identifies the dominant rhythm (e.g., circadian or weekly) and sizes its sliding windows accordingly.
- **Rolling Windows**: Extracts `3D` and `7D` rolling statistics (mean, deviation, velocity, slope) using `pandas`, handling irregular sampling intervals natively.

### 2. Layer 0 — Hybrid Anomaly Engine
Layer 0 acts as a high-throughput, low-cost gatekeeper.
- **Fit phase (Single-Pass):** The Isolation Forest is trained exactly ONCE per level on the training set.
- **RAG Warm-up Phase:** L0 predicts on its own training set; identified outliers are evaluated by LLMs and resolutions are saved into **ChromaDB** as "Long-Term Memory" (few-shot examples).
- **Execution**: Engine is configurable via `L0_ENGINE` (`isolation` or `llm`). If `llm` is active, a **Nano-tier LLM** provides a secondary sanity check on outliers before escalation.

### 3. Layer 1 — Anti-False-Positive Swarm
L1 agents act as **contextual filters**, not data discoverers.
- **Swarm Intelligence**: Spawns role-specific agents (temporal, spatial, etc.) to justify anomalies.
- **RAG-Assisted**: Queries the memory bank for similar cases handled during the warm-up phase.
- **Dynamic Scaling**: The number of agents per role scales from 1 to 5 based on case complexity.

### 4. Layer 2 — Orchestrator (Economic Decider)
The final stage uses a **Smart-tier LLM** to perform an economic trade-off analysis. It reconciles conflicting swarm consensus with raw data to make the final submission-critical decision.

---

## 🔌 Multi-Tier LLM Model System

Mirror automatically resolves model names based on the required "tier" for the task:

| Tier | Role | OpenAI Model | Google Gemini Model |
|------|------|--------------|---------------------|
| **Nano** | L0 verification | `gpt-5-nano` | `gemini-3.1-flash-lite` |
| **Cheap** | L1 Swarm Experts | `gpt-5-mini` | `gemini-3-flash` |
| **Smart** | L2 Orchestration | `gpt-5.4` | `gemini-3.1-pro` |

---

## 📊 Observability & Traceability

Mirror integrates deeply with **Langfuse** for production monitoring, using a granular session ID strategy:
- **Naming Convention**: `{phase}_{run_id}_{stage_name}` (e.g., `predict_eval_20260306_stage_1`).
- **Phases**: `train` (warm-up), `predict_train` (sanity check), `predict_eval` (production).

---

## 📁 Project Structure & Tech Stack

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point with colored logging and run management. |
| `pipeline.py` | Core N-stage orchestrator. Handles sequential training and parallel eval. |
| `layer0_router.py` | Hybrid L0 engine (IsolationForest + Nano LLM). |
| `feature_engineer.py` | Advanced ACF-based rolling window feature extraction. |
| `domain_swarm.py` | Swarm intelligence layer with parallel Role Coordinators. |
| `orchestrator.py` | Final Layer 2 smart model for economic trade-off decisions. |
| `rag_store.py` | ChromaDB local RAG (per-level, reset between stages). |
| `llm_provider.py` | Modular LLM provider (OpenAI, Gemini, extensible). |
| `models.py` | Pydantic data models for dossiers and verdicts. |

---

## 🛠️ Configuration & Prompting

### Prompt Placeholders
All LLM prompts are externalized in `prompts/`. Key placeholders include:
- **L1 Agents**: `{role}`, `{semantic_section}`, `{data_slice}`, `{rag_section}`.
- **L2 Orchestrator**: `{verdict_lines}`, `{features_json}`, `{fp_cost}`, `{fn_cost}`.

### Environment Settings (.env)
- `L0_ENGINE`: `llm` or `isolation`.
- `SWARM_MAX_AGENTS`: Default `5`.
- `FN_COST` / `FP_COST`: Asymmetric cost weights (e.g., 5.0 vs 1.0).

---

## 🚀 Execution

```bash
# Run all stages defined in manifest.json
python main.py -m manifest.json

# Run a specific stage
python main.py -m manifest.json --stage level_1
```

Mirror generates `predictions_{stage}.txt` containing ONLY the ASCII IDs of entities flagged as `1`.

License: Challenge submission — Reply Mirror 2026.
