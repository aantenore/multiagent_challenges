# 🪞 Mirror: The Infinite Adaptive Triage Engine

> **"Identity is a variable. Context is the only constant."**

Mirror is a high-performance, **zero-hardcode**, multi-agent classification framework engineered for the Google Deepmind Agentic Challenge 2026. It is designed to be perfectly domain-agnostic, treating every industry—from healthcare to finance—as a set of abstract analytical roles.

---

## 🏛️ Architecture: The Hierarchical Sieve

Mirror minimizes **Critical Regret** (False Negatives) through an asymmetric three-layer filtering process.

### 1. Layer 0: The Sentry (Statistical Gateway)
- **Engine**: Hybrid IsolationForest (zero-cost ML) or Nano-LLM (cost-optimized generative).
- **Dynamic Feature Extraction**: Automatically processes *any* numeric column in *any* manifest role.
- **Rhythm Discovery**: Uses Autocorrelation (ACF) to synchronize rolling windows with entity-specific cycles, preventing normal fluctuations from triggering alerts.

### 2. Layer 1: The Swarm (Domain Experts)
- **Expert Coordination**: Dynamically instantiates parallel expert agents for every unique role defined in your manifest.
- **Consensus Logic**: Weighted voting among experts with distinct personas and temperatures to eliminate stochastic hallucinations.

### 3. Layer 2: The Auditor (Economic Arbitration)
- **Weighted Reasoning**: Reconciles expert transcripts using high-reasoning models (Smart-tier).
- **Cost-Awareness**: Decisions are biased by the `FN_COST` and `FP_COST` ratio, ensuring a rational balance between safety and operation.

---

## ⚙️ The Abstraction Pattern: Configuring for Any Domain

Mirror has **Zero Hardcoded Roles**. Every role name is an abstraction:

- **Descriptor Roles** (Defined in `settings.py`):
  - `profile`: Defines the "Identity" of the entity (e.g., patient record, user profile).
  - `context`: Defines the "Knowledge Base" (e.g., medical guidelines, fraud manuals).
- **Analytical Roles** (Defined in `manifest.json`):
  - Any role name (e.g., `heart_rate`, `tx_logs`, `gps_pings`) automatically triggers a dedicated Swarm Expert and Feature Analysis.

---

## 📚 The Book of Mirror: Legendary Documentation

| Volume | Title | Content Focus |
|--------|-------|---------------|
| **Vol. I** | [**Theory of Operation**](docs/Theory_of_Operation.md) | Hierarchical triage, ACF theory, and RAG PURGE cycles. |
| **Vol. II** | [**Domain Adaptation**](docs/Domain_Adaptation_Guide.md) | Morphing Mirror for Fraud, Maintenance, or Clinical use. |
| **Vol. III**| [**Developer Manual**](docs/Developer_Manual.md) | Pipeline logic, tiering costs, and parallelism. |
| **Vol. IV** | [**Architecture Deep Dive**](docs/Architecture_Deep_Dive.md) | Dossier construction, swarm mechanics, and state flow. |
| **Vol. V**  | [**Feature Engineering**](docs/Feature_Engineering_Reference.md) | ACF synchronization math and dynamic namespacing. |
| **Vol. VI** | [**Configuration Matrix**](docs/Configuration_Matrix.md) | Comprehensive param reference for settings.py and .env. |
| **Vol. VII**| [**Troubleshooting**](docs/Troubleshooting_Deep_Dive.md) | Langfuse tracing, error state resolution, and tuning. |

---

## 🚀 Quick Execution

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env

# 2. Production Run
python main.py -m manifest.json

# 3. Bypass Training (Cold Start)
python main.py -m bypass_manifest.json --log-level DEBUG
```

---
*Developed for the Google Deepmind Agentic Challenge 2026.*
