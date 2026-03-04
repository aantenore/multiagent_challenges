# 🪞 Mirror — Universal Adaptive Multi-Agent Framework

A production-ready, domain-agnostic **4-layer multi-agent classification pipeline** built for the **Reply Mirror Challenge**.

The system ingests heterogeneous data described by a `manifest.json`, classifies entities as:

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
│   (temporal / spatial / profile / context sources)      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │  DossierBuilder  │  ← merges all sources per entity
              └────────┬────────┘
                       ▼
         ┌─────────────────────────┐
         │  Feature Engineering    │  ← sliding windows, spatial stats
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
                        OUTPUT
                           │
                    ┌──────▼───────┐
                    │ Layer 3 —    │  ChromaDB: stores errors
                    │ Local RAG    │  for continuous learning
                    └──────────────┘
```

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

### 3. Prepare your manifest

Edit `manifest.json` to point to your data files. Each source includes semantic metadata:

```json
{
    "sources": [
        {
            "path": "status.csv",
            "role": "temporal",
            "id_column": "CitizenID",
            "format": "csv",
            "description": "Longitudinal health events recorded at clinical touchpoints.",
            "columns": {
                "CitizenID": "8-char alphanumeric citizen identifier (join key)",
                "EventType": "Event type: routine check-up, emergency visit, specialist consultation, etc.",
                "PhysicalActivityIndex": "0-100 activity level. Declining trends signal sedentary drift.",
                "SleepQualityIndex": "0-100 sleep quality. Drops correlate with stress or health issues.",
                "EnvironmentalExposureLevel": "0-100 environmental risk. Rising values = worsening conditions.",
                "Timestamp": "ISO-8601 datetime of the event."
            }
        },
        {
            "path": "locations.json",
            "role": "spatial",
            "id_column": "user_id",
            "format": "json",
            "description": "GPS pings capturing mobility. Reduced diversity may signal withdrawal.",
            "columns": {
                "user_id": "Citizen identifier (maps to CitizenID)",
                "lat": "Latitude (WGS84)", "lng": "Longitude (WGS84)",
                "city": "Resolved city name", "timestamp": "ISO-8601 datetime"
            }
        },
        {
            "path": "users.json",
            "role": "profile",
            "id_column": "user_id",
            "format": "json",
            "description": "Static demographic profile: age, job, residence.",
            "columns": {
                "user_id": "Citizen identifier",
                "birth_year": "Year of birth (compute age)",
                "job": "Occupation", "residence": "Object with city, lat, lng"
            }
        },
        {
            "path": "personas.md",
            "role": "context",
            "id_column": "entity_id",
            "format": "md",
            "description": "Rich narrative descriptions of citizen lifestyle and health behaviors."
        }
    ]
}
```

### 4. Run the pipeline

The pipeline has **two modes**, controlled by the `--ground-truth` flag:

#### Mode A: Train + Evaluate (development)

Pass a ground-truth file to **train Layer 0** and **print metrics** at the end:

```bash
# Ground truth as JSON: {"CITIZENID": 0_or_1, ...}
python main.py -m manifest.json -g ground_truth.json -o predictions.txt

# Ground truth as TXT: one flagged-ID per line (those are label=1, rest=0)
python main.py -m manifest.json -g flagged_ids.txt -o predictions.txt

# Ground truth as CSV: id,label
python main.py -m manifest.json -g labels.csv -o predictions.txt
```

**What happens:**
1. Data loaded via `manifest.json` → entity dossiers built
2. Features engineered (sliding windows, spatial stats)
3. **L0 trained** on ground-truth labels (XGBoost/RandomForest)
4. Each entity processed: L0 confident → instant verdict; uncertain → L1 swarm → L2 orchestrator
5. `predictions.txt` written (flagged IDs only)
6. **Evaluation report** printed (Confusion Matrix, F1, Value Recovery, Mirror Score)
7. Errors stored in RAG (ChromaDB) for next run's few-shot learning

#### Mode B: Inference only (submission)

Without ground truth, L0 is untrained so all entities go through L1→L2:

```bash
python main.py -m manifest.json -o predictions.txt
```

#### Run per level

```bash
# Level 1
python main.py -m manifest_lev1.json -o predictions_lev1.txt

# Level 2
python main.py -m manifest.json -g ground_truth_lev2.json -o predictions_lev2.txt

# Level 3
python main.py -m manifest_lev3.json -o predictions_lev3.txt
```

#### Extra flags

```bash
# Verbose logging
python main.py -m manifest.json --log-level DEBUG -o predictions.txt
```

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
| `models.py` | Shared Pydantic data models |
| `manifest_manager.py` | Loads and validates `manifest.json` |
| `data_loader.py` | Multi-format file ingestion (CSV, JSON, MD) |
| `dossier_builder.py` | Assembles unified entity dossiers |
| `feature_engineer.py` | Sliding-window feature extraction |
| `layer0_router.py` | Layer 0 — XGBoost/RF deterministic router |
| `agent_base.py` | Abstract LLM agent with retries |
| `domain_swarm.py` | Layer 1 — Per-role domain agents |
| `orchestrator.py` | Layer 2 — Smart model economic decider |
| `rag_store.py` | Layer 3 — ChromaDB local RAG |
| `pipeline.py` | Full pipeline orchestrator |
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

## 📝 Semantic Manifest

`manifest.json` supports optional **semantic metadata** for each data source:

```json
{
    "sources": [
        {
            "path": "status.csv",
            "role": "temporal",
            "id_column": "CitizenID",
            "format": "csv",
            "description": "Longitudinal health events...",
            "columns": {
                "PhysicalActivityIndex": "0-100 activity level, declining trends signal sedentary drift",
                "EventType": "Type of health event: emergency visit is a strong deviation indicator"
            }
        }
    ]
}
```

These descriptions are **injected into agent prompts** so the LLM understands what each column means — no more guessing.

---

## 📊 Output Format

**`predictions.txt`** — One entity ID per line (ASCII), containing only entities classified as `1` (preventive support):

```
QOHAQRQI
IGKMHDGI
GGGZNWZD
```

---

## 🔬 Metrics

At the end of each run (if ground truth is provided), the pipeline prints:

- **Confusion Matrix** (TP, FP, FN, TN)
- **Precision, Recall, F1-Score**
- **Value Recovery** (asymmetric cost-adjusted metric)
- **Mirror Score** = `(F1 + Value Recovery) / 2`
- **Langfuse Session ID** (for trace inspection)

---

## 📋 Requirements

- **Python** ≥ 3.10
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
