# 🪞 Mirror — Universal Adaptive Multi-Agent Framework

Mirror is a production-grade, domain-agnostic **N-stage multi-agent classification pipeline** designed for high-stakes anomaly detection (e.g., healthcare, financial fraud). It optimizes for the **Reply Mirror Challenge** metrics by balancing mathematical precision with contextual LLM reasoning.

The system processes data in **N discrete stages**, where each stage isolates its training context to build a unique "Class 0" (well-being/normality) baseline before executing high-concurrency evaluation.

---

## 🏗️ 3-Tier Hybrid Architecture

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

## 🚀 Key Innovations

### 1. Dynamic Time-Aware Feature Extraction (ACF)
To maximize **Agentic Efficiency**, Mirror adapts to the unique physiological or behavioral rhythms of each entity.
- **Auto-Lag Detection**: Uses `statsmodels` to compute the **Autocorrelation (ACF)** of historical series. The system automatically identifies the dominant rhythm (e.g., circadian or weekly) and sizes its sliding windows accordingly.
- **Rolling Windows**: Extracts `3D` and `7D` rolling statistics (mean, deviation, velocity, slope) using `pandas`, handling irregular sampling intervals natively.

### 2. Hybrid Layer 0 Anomaly Engine
Layer 0 acts as a high-throughput, low-cost gatekeeper.
- **Engine Selection**: Configurable via `L0_ENGINE` (`isolation` or `llm`).
- **Isolation Forest**: Fits a zero-cost mathematical model on the training set's Class 0 data.
- **Nano-LLM Verification**: If configured, it uses a **Nano-tier LLM** (GPT-5-Nano) to provide a secondary sanity check on outliers before escalation, reducing false positives at the earliest possible stage.

### 3. Modular 3-Tier Model System
Mirror automatically resolves model names based on the required "tier" for the task:

| Tier | Role | OpenAI | Google Gemini |
|------|------|--------|---------------|
| **Nano** | L0 verification | `gpt-5-nano` | `gemini-3.1-flash-lite` |
| **Cheap** | L1 Swarm Experts | `gpt-5-mini` | `gemini-3-flash` |
| **Smart** | L2 Orchestration | `gpt-5.4` | `gemini-3.1-pro` |

### 4. Parallel Swarm Intelligence
- **Role Coordinators**: Spawns specialized swarms (temporal, spatial, etc.) based on the incoming data roles defined in the manifest.
- **Dynamic Scaling**: The number of agents in a swarm scales linearly with the case's complexity (1 to 5 agents).
- **Consensus Voting**: Agents use staggered temperatures and confidence-weighted voting to reach a `SwarmConsensus`.

---

## 📊 Observability & Traceability

Mirror integrates deeply with **Langfuse** for production monitoring, using a granular session ID strategy to allow for precise debugging:

- **Naming Convention**: `{phase}_{run_id}_{stage_name}`
- **Phases**:
    - `train`: RAG population and memory warm-up.
    - `predict_train`: Sanity checks on the training set.
    - `predict_eval`: Official production evaluation.
- **Granularity**: This allows you to drill down into a specific stage (e.g., `stage_1`) of a specific run without mixing traces.

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point with colored logging and run management. |
| `pipeline.py` | Core N-stage orchestrator. Handles sequential training and parallel eval. |
| `layer0_router.py` | Hybrid L0 engine (IsolationForest + Nano LLM). |
| `feature_engineer.py` | Advanced ACF-based rolling window feature extraction. |
| `domain_swarm.py` | Swarm intelligence layer with parallel Role Coordinators. |
| `orchestrator.py` | Final Layer 2 smart model for economic trade-off decisions. |
| `rag_store.py` | ChromaDB vector store for few-shot learning and memory persistence. |
| `llm_provider.py` | Clean abstraction for multi-tier model resolution. |

---

## 🛠️ Setup & Execution

### 1. Installation
```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configuration
Edit `.env` to set your providers and cost weights.
```env
LLM_PROVIDER=gemini
L0_ENGINE=llm  # or isolation
FN_COST=5.0
FP_COST=1.0
```

### 3. Execution
```bash
# Run all stages
python main.py -m manifest.json

# Run a specific stage
python main.py -m manifest.json --stage level_1
```

---

## 🔬 Metrics & Compliance
Mirror generates `predictions_{stage}.txt` containing ONLY the IDs of entities flagged as `1`. You can upload these to the challenge validator to track **Value Recovery** and **F1-Score**.

License: Challenge submission — Reply Mirror 2026.
