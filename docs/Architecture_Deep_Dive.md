# 📔 Project Antigravity: Volume IV — Architecture Deep Dive

This volume provides a low-level structural analysis of the Project Antigravity framework, detailing the data flow, internal state transitions, and multi-agent coordination mechanics.

## 1. Unified Entity Dossier Flow

The `EntityDossier` is the single source of truth for every analytical operation.

### 1.1 Assembly (`DossierBuilder`)
1.  **Ingestion**: Loads heterogeneous files (CSV, JSON, MD) defined in the manifest.
2.  **Dual-Track Ingestion**: 
    - **Math Track**: Strips all categorical and text data to perform pure numerical operations (ACF, Sliding Window Z-Score) on continuous metrics.
    - **Memory Track**: Retains the original raw dataset (including text, strings, and categorical tags).
3.  **Entity-Centric Join**: Every data point is matched against a unique `entity_id` using the `id_column` mapping.
4.  **Role Bucketing**: Data is partitioned by its logical role (configurable via `settings.py`). When the Math Track triggers an anomaly, it crops the data from the Memory Track using the incident timestamps, ensuring L1 Swarms receive context-heavy JSON.

## 2. Temporal Stage Isolation

Antigravity enforces strict boundaries between project stages to prevent **Hindsight Bias**.

- **Memory Purge**: At the start of each stage, the Memory Store (Pillar 2) can be reset to ensure isolation.
- **Self-Refining Knowledge**: Knowledge artifacts (prompts, RAG) can evolve across stages as the richness of the historical window increases.

## 3. The Squad Coordinator Pattern (Pillar 3)

Analytical tasks are assigned to specialized `SquadCoordinator` instances.

1.  **Complexity Estimation**: Before calling LLMs, the coordinator analyzes the variance and magnitude of the data slice.
2.  **Swarm Scaling**:
    - Low Complexity: 1-2 Agents.
    - High Complexity: Dynamically scales up to `SWARM_MAX_AGENTS` based on the `squad_scaling_factor`.
3.  **Consensus Aggregation**:
    - Weighted voting based on agent confidence.
    - Full trace log including individual monologues for Pillar 4 review.

## 4. Global Orchestration (Pillar 4)

Pillar 4 acts as the "Recovery Judge," making the final call.

- **Transcript Analysis**: Reads the full reasoning of every squad.
- **Contradiction Detection**: Explicitly identifies experts who missed details that others caught.
- **Risk-Adjusted Decision**: Thresholds for labels (0, 1) shift based on configurable `fp_cost` and `fn_cost`.
