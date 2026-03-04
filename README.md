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
        │  Layer 0 — ML Router    │  XGBoost / RandomForest
        │  (Deterministic)        │  High-confidence → instant verdict
        └─────┬───────────┬───────┘
       certain│           │uncertain
              ▼           ▼
           OUTPUT   ┌──────────────┐
                    │ Layer 1 —    │  Per-role LLM agents (cheap model)
                    │ Domain Swarm │  + RAG few-shot examples
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ Layer 2 —    │  Smart model, economic trade-off
                    │ Orchestrator │
                    └──────┬───────┘
                           ▼
                        OUTPUT → predictions_{stage}.txt
                           │
                    ┌──────▼───────┐
                    │ Layer 3 —    │  ChromaDB: stores errors
                    │ Local RAG    │  for next stage's few-shot
                    └──────────────┘
```

### Cumulative Training

Each stage `i` trains on **all data from stages 0..i**, then evaluates on stage `i`'s evaluation data. Errors are stored in RAG and used as few-shot examples in subsequent stages.

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
| `OPENAI_API_KEY` | OpenAI API key for LLM agents |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (observability) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host URL |

**Optional tuning:**
| Variable | Default | Description |
|----------|---------|-------------|
| `CHEAP_MODEL_NAME` | `gpt-4o-mini` | Model for Layer 1 swarm agents |
| `SMART_MODEL_NAME` | `gpt-4o` | Model for Layer 2 orchestrator |
| `L0_LOWER_THRESHOLD` | `0.15` | ML confidence below this → class 0 |
| `L0_UPPER_THRESHOLD` | `0.85` | ML confidence above this → class 1 |
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
                    "path": "resources/training/public_lev_1/status 2.csv",
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
            "evaluation_sources": [
                {
                    "path": "resources/evaluation/public_lev_1/status_new.csv",
                    "role": "temporal",
                    "id_column": "CitizenID",
                    "format": "csv",
                    "description": "Level 1 evaluation health events."
                }
            ]
        },
        {
            "name": "level_2",
            "output_file": "predictions_lev2.txt",
            "ground_truth": "",
            "training_sources": ["..."],
            "evaluation_sources": ["..."]
        }
    ]
}
```

**Each stage has:**
| Field | Description |
|-------|-------------|
| `name` | Human-readable stage name |
| `output_file` | Where predictions are written for this stage |
| `ground_truth` | Optional path to labels (JSON/CSV/TXT) for L0 training + metrics |
| `training_sources` | Data sources for cumulative training |
| `evaluation_sources` | Data sources for prediction/evaluation |

**Each source entry has:**
| Field | Required | Description |
|-------|----------|-------------|
| `path` | ✅ | Relative path to data file (forward slashes, works on all OS) |
| `role` | ✅ | One of `temporal`, `spatial`, `profile`, `context` |
| `id_column` | ✅ | Name of the entity-ID column |
| `format` | ✅ | One of `csv`, `json`, `md` |
| `description` | ❌ | Semantic description injected into agent prompts |
| `columns` | ❌ | `{"col_name": "what it means"}` — column definitions for agents |

> **Scaling:** Add more stages by appending to the `stages` array. Works with 1, 3, 6, or any number of stages.

### 4. Run the pipeline

```bash
# Run all stages defined in manifest.json
python main.py -m manifest.json

# Verbose logging
python main.py -m manifest.json --log-level DEBUG
```

**What happens per stage:**
1. **Train:** Cumulative training data (stages 0..i) → build dossiers → engineer features → train L0
2. **Evaluate:** Stage `i` evaluation data → predict (L0 → L1 → L2) → write `predictions_{stage}.txt`
3. **Learn:** Errors stored in RAG for next stage's few-shot examples

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
| `models.py` | Shared Pydantic data models (`Stage`, `Manifest`, `EntityDossier`, `AgentVerdict`) |
| `manifest_manager.py` | Loads `manifest.json`, provides cumulative stage access |
| `data_loader.py` | Multi-format file ingestion (CSV, JSON, MD) |
| `dossier_builder.py` | Assembles unified entity dossiers from source entries |
| `feature_engineer.py` | Sliding-window feature extraction |
| `layer0_router.py` | Layer 0 — XGBoost/RF deterministic router |
| `agent_base.py` | Abstract LLM agent with retries + JSON validation |
| `domain_swarm.py` | Layer 1 — Per-role domain agents with semantic metadata |
| `orchestrator.py` | Layer 2 — Smart model economic decider |
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
|------|-----------------|
| `prompts/system.txt` | System identity shared by all agents (role, output format, asymmetric cost bias) |
| `prompts/domain.txt` | **Domain context** — describes the operating domain. Change this to adapt the system to finance, safety, education, etc. |
| `prompts/domain_agent.txt` | Layer 1 per-role agent prompt template. Uses `{placeholders}` for data injection |
| `prompts/orchestrator.txt` | Layer 2 orchestrator prompt template. Uses `{placeholders}` for verdicts, features, costs |

### Available placeholders

**`domain_agent.txt`:**
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
  - `scikit-learn` + `xgboost` — ML models (Layer 0)
  - `openai` — LLM API client (Layer 1 & 2)
  - `langfuse` — observability & tracing
  - `chromadb` — local vector database (Layer 3 RAG)
  - `rich` — terminal formatting & progress bars
  - `python-dotenv` — `.env` file loading

---

## 📜 License

Challenge submission — Reply Mirror 2026.
