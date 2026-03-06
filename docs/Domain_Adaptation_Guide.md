# 📗 Project Antigravity: Volume II — Domain Adaptation Guide

> **Core Directive**: Project Antigravity is 100% domain-agnostic. **NO Python code changes are required** to switch domains.  
> Swap your data files, update `manifest.json` and `.env`, and the system works identically on Healthcare, Finance, Cybersecurity, or any other domain.

---

## 1. The "Drop-In" Checklist

To deploy Antigravity on a completely new domain, follow these 4 steps:

| Step | Action | Files Touched |
|:---|:---|:---|
| **1** | Prepare your data files (CSV, JSON, or Markdown). | Your data directory |
| **2** | Write a `manifest.json` defining sources, roles, and feature mappings. | `manifest.json` |
| **3** | Update `.env` with domain-specific column names and thresholds. | `.env` |
| **4** | (Optional) Customize squad prompts for your domain's language. | `prompts/*.txt` |

**That's it.** No Python files are touched.

---

## 2. The Mapping Strategy

### 2.1 Configurable Descriptor Roles
Every data source in the manifest is assigned a **role**. The system treats these roles generically:

| Role | Purpose | Example (Healthcare) | Example (Finance) |
|:---|:---|:---|:---|
| `profile` | Static entity metadata | Patient demographics | Customer KYC profile |
| `context` | Narrative/unstructured text | Doctor notes | Support ticket logs |
| `temporal` | High-frequency event data | Daily health readings | Transaction history |
| `spatial` | Location/graph data | GPS check-in logs | ATM location logs |

### 2.2 Feature Mapping (Manifest V2)
The `feature_mapping` block inside `manifest.json` tells the engine how to interpret your columns **without any hardcoded Python logic**.

```yaml
feature_mapping:
  # Columns to synthesize spatial features from (Haversine, Entropy)
  synthesize_features:
    Location_Entropy: ["lat_col", "lng_col"]

  # Categorical columns preserved for L1 Semantic Reading
  categorical_context: ["EventType", "DeviceType"]

  # Numeric columns processed by the L0 Math Track Z-Score
  target_metrics: ["MetricA", "MetricB", "Location_Entropy"]

  # L0 trigger strategy: ANY_METRIC or COMPOSITE_INDEX
  trigger_logic: "COMPOSITE_INDEX"

  # Collision-Proof Acronym Mapping for Token Compression
  acronym_map:
    MetricA: "MA"
    MetricB: "MB"
```

---

## 3. Side-by-Side Domain Examples

Below are two complete `manifest.json` examples proving domain flexibility. **The Python engine is identical for both.**

### 🏥 Healthcare Manifest
```json
{
  "stages": [
    {
      "name": "monthly_triage",
      "training_sources": [
        {"file": "data/health_readings_jan.csv", "role": "temporal", "id_column": "CitizenID"},
        {"file": "data/patient_profiles.json", "role": "profile", "id_column": "CitizenID"},
        {"file": "data/gps_checkins.csv", "role": "spatial", "id_column": "CitizenID"}
      ],
      "evaluation_sources": [
        {"file": "data/health_readings_feb.csv", "role": "temporal", "id_column": "CitizenID"}
      ],
      "feature_mapping": {
        "synthesize_features": {"Location_Entropy": ["lat", "lng"]},
        "categorical_context": ["EventType"],
        "target_metrics": ["PhysicalActivity", "SleepQuality", "Location_Entropy"],
        "trigger_logic": "COMPOSITE_INDEX",
        "acronym_map": {"PhysicalActivity": "ACT", "SleepQuality": "SLP"}
      }
    }
  ]
}
```

### 🏦 Financial Fraud Manifest
```json
{
  "stages": [
    {
      "name": "weekly_fraud_scan",
      "training_sources": [
        {"file": "data/transactions_week1.csv", "role": "temporal", "id_column": "AccountID"},
        {"file": "data/customer_kyc.json", "role": "profile", "id_column": "AccountID"},
        {"file": "data/atm_locations.csv", "role": "spatial", "id_column": "AccountID"}
      ],
      "evaluation_sources": [
        {"file": "data/transactions_week2.csv", "role": "temporal", "id_column": "AccountID"}
      ],
      "feature_mapping": {
        "synthesize_features": {"ATM_Entropy": ["atm_lat", "atm_lng"]},
        "categorical_context": ["TransactionType", "MerchantCategory"],
        "target_metrics": ["TransactionAmount", "TransactionFrequency", "ATM_Entropy"],
        "trigger_logic": "ANY_METRIC",
        "acronym_map": {"TransactionAmount": "TXA", "TransactionFrequency": "TXF"}
      }
    }
  ]
}
```

> [!IMPORTANT]
> Notice: `id_column`, `role`, file paths, column names, and feature mappings are all different. **The Python code is identical.**

---

## 4. Environment Variable Overrides (`.env`)

Key settings to update per domain:

```dotenv
# Primary numeric column for L0 Filter
FILTER_PRIMARY_COL=TransactionAmount
FILTER_SECONDARY_COL=TransactionFrequency

# Timestamp column name in your data
TIMESTAMP_COL=Timestamp

# Spatial coordinate columns (empty = no spatial features)
SPATIAL_COORDINATE_COLS=["atm_lat", "atm_lng"]

# Persona compression: which prefixes to keep
PERSONA_DENSE_PREFIXES=["Risk profile", "Account type", "Transaction pattern"]

# Acronym overrides (can also be set in manifest)
ACRONYM_MAP={"TransactionAmount": "TXA"}

# Cost asymmetry (domain-dependent)
FP_COST=3.0
FN_COST=10.0
```

---

## 5. Pillar 3: Squad Customization

Modify `squad_configs` in `.env` or `settings.py` to define analytical squads for your domain:

```json
{
  "fraud_detection": {
    "roles": ["temporal"],
    "prompt": "squad_fraud_detection",
    "base_agents": 2
  },
  "identity_verification": {
    "roles": ["profile", "context"],
    "prompt": "squad_identity",
    "base_agents": 1
  },
  "geo_anomaly": {
    "roles": ["spatial"],
    "prompt": "squad_geo_anomaly",
    "base_agents": 1
  }
}
```

Then create the corresponding prompt templates in `prompts/squad_fraud_detection.txt`, etc.

---

## 6. The Golden Rule

> **If you are writing Python code to adapt to a new domain, you are doing it wrong.**  
> Every domain-specific detail lives in `manifest.json`, `.env`, and `prompts/*.txt`.
