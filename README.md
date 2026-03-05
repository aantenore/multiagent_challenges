# 🪞 Mirror — Universal Adaptive Multi-Agent Framework

A production-ready, domain-agnostic **N-stage multi-agent classification pipeline** built for the **Reply Mirror Challenge**.

The system ingests heterogeneous data described by a `manifest.json`, processes N stages of **per-level** training + evaluation, and classifies entities as:

| Label | Meaning |
|-------|---------|
| `0` | Standard monitoring (well-being — trajectory acceptable) |
| `1` | Preventive support needed (anomaly detected) |

**Objective:** Maximise `(F1-Score + Value Recovery) / 2` with asymmetric cost awareness (False Negatives ≫ False Positives).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  manifest.json                          │
│   N stages: [train₁,eval₁] → [train₂,eval₂] → ...     │
│   Each level is SELF-CONTAINED (no cross-level data)    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
          ┌─────────────────────────┐
          │  Per-Level Training     │  train_i only (class 0 = well-being)
          │  DossierBuilder         │
          └────────────┬────────────┘
                       ▼
         ┌──────────────────────────┐
         │  Feature Engineering    │  sliding windows, dynamic lag (ACF)
         └────────────┬────────────┘
                      ▼

### Dynamic Time-Aware Feature Extraction

The pipeline uses `pandas` to extract true time-aware rolling features, solving issues with irregular sampling intervals. 
Instead of relying on fixed record counts, it calculates features over continuous `3D` and `7D` rolling windows.

**Killer Feature — ACF Dynamic Sizing:**
To maximize Agentic Efficiency (a 40% weighted challenge metric), the engine adapts to the physiological rhythms of each individual citizen.
Using `statsmodels`, it computes the **Autocorrelation (ACF)** on the citizen's historical series (e.g., PhysicalActivityIndex). By identifying the dominant lag (peak autocorrelation), the system discovers the user's natural cycle length automatically. 
It then extracts dynamic features (e.g., `_dynamic_mean`, `_dynamic_deviation`) perfectly tailored to their unique circadian or weekly rhythms.
If the series is too short or irregular to find an ACF peak, it seamlessly falls back to the configured baseline window size.
         ┌──────────────────────────┐
         │  Layer 0 — IsolationForest│  One-Class engine
         │  + DetectionMetadata      │  Fit on class-0 only
         └─────┬───────────┬────────┘
       inlier  │           │outlier + DetectionMetadata
               ▼           ▼
            OUTPUT   ┌──────────────────────────────────┐
          (pred=0)   │  Layer 1 — Anti-FP Filter         │
                     │  RoleCoordinator("temporal")      │
                     │    ├─ assess_complexity → N       │
                     │    ├─ spawn 1..5 DomainAgents     │
                     │    ├─ read L0 math details        │
                     │    └─ aggregate → SwarmConsensus  │
                     │  RoleCoordinator("spatial")  → ⬤ │
                     │  RoleCoordinator("profile")  → ⬤ │
                     │  RoleCoordinator("context")  → ⬤ │
                     └──────────┬───────────────────────┘
                               ▼ list[SwarmConsensus]
                    ┌──────────────────────────┐
                    │  Layer 2 — Orchestrator   │
                    │  Smart model, economic    │
                    │  trade-off + consensus    │
                    └──────────┬───────────────┘
                               ▼
                            OUTPUT → predictions_{stage}.txt
                               │
                    ┌──────────▼───────┐
                    │  RAG (per-level) │  ChromaDB: reset between levels
                    └──────────────────┘
```


---

### Hybrid Knowledge Base — RAG Memory

The RAG store (**ChromaDB**) acts as the framework's "Long-Term Memory," but it follows a strict lifecycle:
- **Phase 1: Population (Training Only):** During the warm-up cycle, the RAG is populated using a **Hybrid Knowledge** approach:
    - **Baseline Normality (Branch A):** For mathematical inliers (L0=0), synthetic RAG entries are created at zero LLM cost ("The user is healthy").
    - **Expert Reasoning (Branch B):** For anomalies (L0=1), the full LLM Swarm evaluates the case and saves its detailed analytical reasoning.
- **Phase 2: Inference (Evaluation Only):** During the evaluation/prediction phase, the RAG is **locked (Read-Only)**. It provides the agents with historical benchmarks to reduce uncertainty and confirm decisions.

### Sequential RAG Warm-up

During the training phase, citizens are processed **sequentially**. This allows each subsequent evaluation to query and "learn" from the decisions (both synthetic and expert) made earlier in the same warm-up cycle. This strictly cumulative memory significantly reduces false positives by providing a rich context of what "normal" looks like across different profiles.

### Layer 1 — Anti-False-Positive Filter (LLM Agents)

L1 agents do NOT discover anomalies in numbers (L0 does that). They act as **contextual false-positive filters**:
- Read the persona, profile, and lifestyle context.
- **RAG-Assisted**: Query the memory bank for similar cases handled during the warm-up phase.
- If the mathematical deviation is **justified** by life context → output `0`.
- If NOT justified → confirm `1` (preventive support).

### Dynamic Swarm Scaling

Each `RoleCoordinator` **linearly scales** the number of agents based on case complexity:

| Complexity | Agents per Role | Description |
|-----------|-----------------|-------------|
| 0.0 - 0.3 | 1 (minimum) | Clear-cut case, single expert sufficient |
| 0.5 | 3 | Moderate ambiguity, multiple perspectives |
| 1.0 | 5 (maximum) | Maximum uncertainty, full committee vote |

Agents within a swarm use **staggered temperatures** for opinion diversity. Votes are aggregated via **confidence-weighted voting** into a `SwarmConsensus` with agreement ratio.

### Parallel Execution

Both **role coordinators** and **agents within each role** execute in parallel using `ThreadPoolExecutor`.

### Per-Level Training & Sanity Cycle

Each stage `i` undergoes the full Single-Pass logic. After the Fit, the pipeline runs a **Sanity Predict Cycle** on the training set to verify baseline stability (expected anomaly rate $\approx$ 0%).

---

## 🔌 Modular LLM Providers

The framework supports **switchable LLM backends** via a modular provider system.

| Provider | Env Setting | Models |
|----------|------------|--------|
| **OpenAI** | `LLM_PROVIDER=openai` | `gpt-5-mini` (cheap), `gpt-5.2` (smart) |
| **Google Gemini** | `LLM_PROVIDER=gemini` | `gemini-3-flash-preview` (cheap), `gemini-3.1-pro-preview` (smart) |

Switch providers by changing a single `.env` variable — no code changes needed.

### Adding a Custom Provider

```python
from llm_provider import BaseLLMProvider, register_provider

class MyProvider(BaseLLMProvider):
    def chat(self, system_message, user_message, *, model, temperature, json_mode):
        ...  # implement
    def resolve_model(self, role):
        ...  # "cheap" or "smart" → model name

register_provider("my_provider", MyProvider)
```

Then set `LLM_PROVIDER=my_provider` in `.env`.

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd multiagent_challenge

# Create virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

**Required keys:**
| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI provider) |
| `GOOGLE_API_KEY` | Google API key (if using Gemini provider) |
| `LLM_PROVIDER` | Provider to use: `openai` or `gemini` |

**Swarm configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SWARM_MIN_AGENTS` | `1` | Minimum agents per role coordinator |
| `SWARM_MAX_AGENTS` | `5` | Maximum agents per role coordinator |
| `SWARM_COMPLEXITY_THRESHOLD` | `0.3` | Complexity below this → min agents |
| `SWARM_TEMP_SPREAD` | `0.15` | Temperature variation between swarm agents |

**Pipeline hyperparameters:**
| Variable | Default | Description |
|----------|---------|-------------|
| `FP_COST` | `1.0` | False Positive cost weight |
| `FN_COST` | `5.0` | False Negative cost weight |
| `TOP_K_RAG` | `3` | Number of RAG few-shot examples |

**Optional observability:**
| Variable | Description |
|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host URL |

### 3. Configure manifest.json

The manifest defines **N stages**, each with its own training and evaluation data sources:

```json
{
    "stages": [
        {
            "name": "level_1",
            "output_file": "predictions_lev1.txt",
            "training_sources": [
                {
                    "path": "resources/.../status.csv",
                    "role": "temporal",
                    "id_column": "CitizenID",
                    "format": "csv",
                    "description": "Level 1 training health events.",
                    "columns": { ... }
                }
            ],
            "evaluation_sources": [...]
        }
    ]
}
```

> **Configurable stages:** Add more stages by appending to the `stages` array. Works with 1, 3, 6, or any number of stages.
>
> **Zero-Code Swarms:** If a new dataset with a new role (e.g., `"role": "financial"`) is added, a new `RoleCoordinator("financial")` spawns automatically. No code changes needed.

### 4. Run the pipeline

```bash
# Run all stages defined in manifest.json
python main.py -m manifest.json

# Verbose logging
python main.py -m manifest.json --log-level DEBUG
```

**What happens per stage:**
1. **Reset:** RAG cleared for level isolation
2. **Train:** This level's training data → build dossiers → engineer features → fit IsolationForest (Single-Pass)
3. **Evaluate:** Level `i` eval data → predict (L0 → L1 anti-FP → L2) → write `predictions_{stage}.txt`

### 5. Build submission archive

```bash
python build_submission.py
# → creates submission.zip (source code only, no .env or data)
```

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point |
| `settings.py` | Pydantic Settings (loads `.env`) |
| `llm_provider.py` | **Modular LLM provider** (OpenAI, Gemini, extensible) |
| `models.py` | Shared Pydantic models (`Stage`, `Manifest`, `EntityDossier`, `DetectionMetadata`, `AgentVerdict`, `SwarmConsensus`) |
| `manifest_manager.py` | Loads `manifest.json`, provides per-stage access |
| `data_loader.py` | Multi-format file ingestion (CSV, JSON, MD) |
| `dossier_builder.py` | Assembles unified entity dossiers from source entries |
| `feature_engineer.py` | Sliding-window feature extraction (MA3, MA7, slope, velocity) |
| `layer0_router.py` | Layer 0 — **IsolationForest One-Class engine** (fit on class 0 only) |
| `agent_base.py` | Abstract LLM agent with retries + JSON validation |
| `domain_swarm.py` | Layer 1 — **RoleCoordinator** + **Anti-FP Filter** agents |
| `orchestrator.py` | Layer 2 — Smart model economic decider |
| `rag_store.py` | ChromaDB local RAG (per-level, reset between stages) |
| `pipeline.py` | N-stage per-level pipeline orchestrator |
| `output_writer.py` | Generates challenge-compliant TXT output |
| `prompt_loader.py` | Loads prompt templates from `prompts/` |
| `build_submission.py` | Creates ZIP archive for submission |

---

## 🎯 Prompt Configuration

All LLM prompts are **externalized** in the `prompts/` directory:

| File | What it controls |
|------|---------------------|
| `prompts/system.txt` | System identity shared by all agents |
| `prompts/domain.txt` | **Domain context** — describes the operating domain |
| `prompts/domain_agent.txt` | Layer 1 anti-FP agent prompt template |
| `prompts/coordinator.txt` | **Role coordinator** assessment prompt template |
| `prompts/orchestrator.txt` | Layer 2 orchestrator prompt template |

### Available placeholders

**`domain_agent.txt`:**
`{role}`, `{entity_id}`, `{semantic_section}`, `{role_title}`, `{data_slice}`, `{profile_json}`, `{context}`, `{rag_section}`, `{l0_section}`

**`orchestrator.txt`:**
`{entity_id}`, `{verdict_lines}`, `{profile_json}`, `{context}`, `{features_json}`, `{rag_section}`, `{fp_cost}`, `{fn_cost}`, `{fn_ratio}`

### Changing domain

To adapt the framework to a different domain (e.g., financial fraud):
1. Edit `prompts/domain.txt` — describe the new domain
2. Edit `prompts/system.txt` — adjust agent identity
3. Update `manifest.json` — point to new data files

> **Fallback:** if any prompt file is missing, the system uses built-in defaults.

---

## 📊 Output Format

**`predictions_{stage}.txt`** — One entity ID per line (ASCII), containing only entities classified as `1` (preventive support):

```
QOHAQRQI
IGKMHDGI
GGGZNWZD
```

---

## 🔬 Metrics

The pipeline outputs predictions for each stage to `predictions_{stage}.txt`. You can use the `resources\case_1\track-your-submission\how-to-track-your-submission` utility to query your exact metrics via Langfuse.

---

## 📋 Requirements

- **Python** ≥ 3.10
- **OS:** Windows, macOS, Linux (all paths handled via `pathlib`)
- **Dependencies** (see `requirements.txt`):
  - `pydantic` + `pydantic-settings` — configuration & data validation
  - `pandas` + `numpy` — data manipulation
  - `statsmodels` — autocorrelation (ACF) for dynamic time-aware features
  - `scikit-learn` — anomaly detection (IsolationForest, Layer 0)
  - `openai` — OpenAI LLM API client
  - `google-genai` — Google Gemini LLM API client
  - `langfuse` — observability & tracing
  - `chromadb` — local vector database (RAG)
  - `rich` — terminal formatting & progress bars
  - `python-dotenv` — `.env` file loading

---

## 📜 License

Challenge submission — Reply Mirror 2026.
