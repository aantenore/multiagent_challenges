# 🪞 Mirror — Universal Adaptive Multi-Agent Framework

A production-ready, domain-agnostic **N-stage multi-agent classification pipeline** built for the **Reply Mirror Challenge**.

The system ingests heterogeneous data described by a `manifest.json`, processes N stages of cumulative training + evaluation, and classifies entities as:

| Label | Meaning |
|-------|---------|
| `0` | Standard monitoring (trajectory acceptable) |
| `1` | Preventive support needed (deviation detected) |

**Objective:** Maximise `(F1-Score + Value Recovery) / 2` with asymmetric cost awareness (False Negatives ≫ False Positives).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  manifest.json                          │
│   N stages: [train₁,eval₁] → [train₂,eval₂] → ...     │
└──────────────────────┬──────────────────────────────────┘
                       ▼
          ┌─────────────────────────┐
          │  Cumulative Training    │  stages 0..i accumulated
          │  DossierBuilder         │
          └────────────┬────────────┘
                       ▼
         ┌──────────────────────────┐
         │  Feature Engineering    │  sliding windows, spatial stats
         └────────────┬────────────┘
                      ▼
         ┌──────────────────────────┐
         │  Layer 0 — Anomaly Router │  Z-Score / IsolationForest
         │  + Complexity Score       │  High-confidence → instant verdict
         └─────┬───────────┬────────┘
       certain│           │uncertain + complexity_score
              ▼           ▼
           OUTPUT   ┌──────────────────────────────────┐
                    │  Layer 1 — Per-Role Coordinators  │
                    │  RoleCoordinator("temporal")      │
                    │    ├─ assess_complexity → N       │
                    │    ├─ spawn 1..5 DomainAgents     │
                    │    ├─ varied temperatures         │
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
                     │  Layer 3 — RAG   │  ChromaDB: stores all cases
                     │  for next stage  │  (self-supervised when no labels)
                    └──────────────────┘
```

### Dynamic Swarm Scaling

Each `RoleCoordinator` **linearly scales** the number of agents based on case complexity:

| Complexity | Agents per Role | Description |
|-----------|-----------------|-------------|
| 0.0 - 0.3 | 1 (minimum) | Clear-cut case, single expert sufficient |
| 0.5 | 3 | Moderate ambiguity, multiple perspectives |
| 1.0 | 5 (maximum) | Maximum uncertainty, full committee vote |

Agents within a swarm use **staggered temperatures** for opinion diversity. Votes are aggregated via **confidence-weighted voting** into a `SwarmConsensus` with agreement ratio.

### Parallel Execution

Both **role coordinators** and **agents within each role** execute in parallel using `ThreadPoolExecutor`, maximising throughput and minimising wall-clock time.

### Cumulative Training

Each stage `i` trains on **all data from stages 0..i**, then evaluates on stage `i`'s evaluation data. Errors are stored in RAG and used as few-shot examples in subsequent stages.

---

## 🔌 Modular LLM Providers

The framework supports **switchable LLM backends** via a modular provider system.

| Provider | Env Setting | Models |
|----------|------------|--------|
| **OpenAI** | `LLM_PROVIDER=openai` | `gpt-4o-mini` (cheap), `gpt-4o` (smart) |
| **Google Gemini** | `LLM_PROVIDER=gemini` | `gemini-2.0-flash` (cheap), `gemini-2.5-pro-exp-03-25` (smart) |

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

**Recommended Model Tuning:**
The architecture splits processing into high-volume data analysis (Layer 1 Swarms) and low-volume high-reasoning synthesis (Layer 2 Orchestrator). 

- **Layer 1 (Per-Role Swarm Agents - "Cheap")**: Needs high throughput, fast JSON parsing, and good instruction following.
  - **OpenAI:** `gpt-5-mini` (Very fast, cheap, good JSON support).
  - **Gemini:** `gemini-3-flash-preview` (Fast, large context window).
- **Layer 2 (Global Orchestrator - "Smart")**: Needs deep reasoning, consensus conflict resolution, and complex economic balancing.
  - **OpenAI:** `gpt-5.2` (New fast reasoning model supporting structured JSON outputs) or highly reliable instruction following.
  - **Gemini:** `gemini-3.1-pro-preview` (Excellent reasoning and synthesis capabilities across multiple contexts).

*Environment variables to set these:*
| Variable | Default Fallback |
|----------|-----------------|
| `CHEAP_MODEL_NAME` | `gpt-5-mini` |
| `SMART_MODEL_NAME` | `gpt-5.2` |
| `GEMINI_CHEAP_MODEL_NAME` | `gemini-3-flash-preview` |
| `GEMINI_SMART_MODEL_NAME` | `gemini-3.1-pro-preview` |

**Optional observability:**
| Variable | Description |
|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host URL |

**Model tuning:**
| Variable | Default | Description |
|----------|---------|-------------|
| `CHEAP_MODEL_NAME` | `gpt-4o-mini` | OpenAI model for Layer 1 swarm agents |
| `SMART_MODEL_NAME` | `gpt-4o` | OpenAI model for Layer 2 orchestrator |
| `GEMINI_CHEAP_MODEL_NAME` | `gemini-2.0-flash` | Gemini model for Layer 1 |
| `GEMINI_SMART_MODEL_NAME` | `gemini-2.5-pro-exp-03-25` | Gemini model for Layer 2 |
| `MODEL_TEMPERATURE` | `0.2` | Base temperature for LLM calls |

**Swarm configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SWARM_MIN_AGENTS` | `1` | Minimum agents per role coordinator |
| `SWARM_MAX_AGENTS` | `5` | Maximum agents per role coordinator |
| `SWARM_COMPLEXITY_THRESHOLD` | `0.3` | Complexity below this → min agents |
| `SWARM_TEMP_SPREAD` | `0.15` | Temperature variation between swarm agents |

**Layer 0 — Hybrid Ensemble Anomaly Detection:**
| Variable | Default | Description |
|----------|---------|-------------|
| `ANOMALY_THRESHOLD` | `2.5` | Z-score σ multiplier for univariate anomaly detection |
| `MIN_HISTORICAL_SAMPLES` | `3` | Min temporal records for valid baseline |
| `SIGMA_WEIGHT` | `0.6` | Weight for Z-Score detector in ensemble |
| `FOREST_WEIGHT` | `0.4` | Weight for IsolationForest detector in ensemble |
| `ANOMALY_THRESHOLD_FOREST` | `-0.1` | IsolationForest decision_function threshold |

**Pipeline hyperparameters:**
| Variable | Default | Description |
|----------|---------|-------------|
| `WINDOW_SIZE` | `3` | Sliding window size for features |
| `FP_COST` | `1.0` | False Positive cost weight |
| `FN_COST` | `5.0` | False Negative cost weight |
| `TOP_K_RAG` | `3` | Number of RAG few-shot examples |

### 3. Configure manifest.json

The manifest defines **N stages**, each with training and evaluation data sources:

```json
{
    "stages": [
        {
            "name": "level_1",
            "output_file": "predictions_lev1.txt",
            "ground_truth": "",
            "training_sources": [
                {
                    "path": "resources/case_1/training/public_lev_1/status 2.csv",
                    "role": "temporal",
                    "id_column": "CitizenID",
                    "format": "csv",
                    "description": "Level 1 training health events.",
                    "columns": {
                        "EventType": "Health event type",
                        "PhysicalActivityIndex": "0-100 activity level"
                    }
                }
            ],
            "evaluation_sources": [...]
        }
    ]
}
```

**Multi-case support:** Data is organized under `resources/case_N/`:

```
resources/
├── case_1/
│   ├── challenge.pdf
│   ├── training/
│   └── evaluation/
├── case_2/
│   ├── training/
│   └── evaluation/
```

Each case can have its own `manifest_caseN.json`.

> **Scaling / Dynamic Swarms:** Add more stages by appending to the `stages` array. Works with 1, 3, 6, or any number of stages. 
> 
> **Zero-Code Swarms:** If a new dataset arrives with a new role (e.g., `"role": "financial"`), simply adding it to the `manifest.json` will **automatically** spawn a new `RoleCoordinator("financial")` and a new dynamic swarm of financial expert agents. No code changes are required.

### 4. Run the pipeline

```bash
# Run all stages defined in manifest.json
python main.py -m manifest.json

# Verbose logging
python main.py -m manifest.json --log-level DEBUG
```

**What happens per stage:**
1. **Train:** Cumulative training data (stages 0..i) → build dossiers → engineer features → train L0
2. **Evaluate:** Stage `i` evaluation data → predict (L0 → L1 coordinators → L2) → write `predictions_{stage}.txt`
3. **Learn:** All evaluations stored in RAG for next stage's few-shot examples

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
| `models.py` | Shared Pydantic data models (`Stage`, `Manifest`, `EntityDossier`, `AgentVerdict`, `SwarmConsensus`) |
| `manifest_manager.py` | Loads `manifest.json`, provides cumulative stage access |
| `data_loader.py` | Multi-format file ingestion (CSV, JSON, MD) |
| `dossier_builder.py` | Assembles unified entity dossiers from source entries |
| `feature_engineer.py` | Sliding-window feature extraction |
| `layer0_router.py` | Layer 0 — **Unsupervised Anomaly Router** (Z-Score / IsolationForest) + complexity scoring |
| `agent_base.py` | Abstract LLM agent with retries + JSON validation (uses modular provider) |
| `domain_swarm.py` | Layer 1 — **RoleCoordinator** (dynamic swarm, weighted voting, consensus) |
| `orchestrator.py` | Layer 2 — Smart model economic decider (reads `SwarmConsensus`) |
| `rag_store.py` | Layer 3 — ChromaDB local RAG |
| `pipeline.py` | N-stage cumulative pipeline orchestrator |
| `metrics.py` | Evaluation (F1, Value Recovery, Mirror Score) |
| `output_writer.py` | Generates challenge-compliant TXT output |
| `prompt_loader.py` | Loads prompt templates from `prompts/` |
| `build_submission.py` | Creates ZIP archive for submission |

---

## 🎯 Prompt Configuration

All LLM prompts are **externalized** in the `prompts/` directory. Edit these plain-text files to tune agent behavior **without touching code**:

| File | What it controls |
|------|--------------------|
| `prompts/system.txt` | System identity shared by all agents |
| `prompts/domain.txt` | **Domain context** — describes the operating domain |
| `prompts/domain_agent.txt` | Layer 1 per-role agent prompt template |
| `prompts/coordinator.txt` | **Role coordinator** assessment prompt template |
| `prompts/orchestrator.txt` | Layer 2 orchestrator prompt template (reads consensus) |

### Available placeholders

**`domain_agent.txt` / `coordinator.txt`:**
`{role}`, `{entity_id}`, `{semantic_section}`, `{role_title}`, `{data_slice}`, `{profile_json}`, `{context}`, `{rag_section}`

**`orchestrator.txt`:**
`{entity_id}`, `{verdict_lines}`, `{profile_json}`, `{context}`, `{features_json}`, `{rag_section}`, `{fp_cost}`, `{fn_cost}`, `{fn_ratio}`

### Changing domain

To adapt the framework to a different domain (e.g., financial fraud):
1. Edit `prompts/domain.txt` — describe the new domain and key signals
2. Edit `prompts/system.txt` — adjust agent identity if needed
3. Update `manifest.json` — point to new data files with updated `description` and `columns`

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

At the end of each stage (if ground truth is provided), the pipeline prints:

- **Confusion Matrix** (TP, FP, FN, TN)
- **Precision, Recall, F1-Score**
- **Value Recovery** (asymmetric cost-adjusted metric)
- **Mirror Score** = `(F1 + Value Recovery) / 2`
- **Langfuse Session ID** (for trace inspection)

---

## 📋 Requirements

- **Python** ≥ 3.10
- **OS:** Windows, macOS, Linux (all paths handled via `pathlib`)
- **Dependencies** (see `requirements.txt`):
  - `pydantic` + `pydantic-settings` — configuration & data validation
  - `pandas` + `numpy` — data manipulation
  - `scikit-learn` + `scipy` — anomaly detection (Layer 0)
  - `openai` — OpenAI LLM API client
  - `google-genai` — Google Gemini LLM API client
  - `langfuse` — observability & tracing
  - `chromadb` — local vector database (Layer 3 RAG)
  - `rich` — terminal formatting & progress bars
  - `python-dotenv` — `.env` file loading

---

## 📜 License

Challenge submission — Reply Mirror 2026.
