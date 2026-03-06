# 📔 The Book of Mirror: Volume IV — Architecture Deep Dive

This volume provides a low-level structural analysis of the Mirror framework, detailing the data flow, the internal state transitions, and the multi-agent coordination mechanics.

## 1. Unified Entity Dossier Flow

The `EntityDossier` is the single source of truth for every analytical operation.

### 1.1 Assembly (`DossierBuilder`)
1.  **Ingestion**: Loads heterogeneous files (CSV, JSON, MD) defined in the manifest.
2.  **Role Bucketing**: Data is partitioned by its logical role.
    - `profile_role` (default: `profile`): Static metadata.
    - `context_role` (default: `context`): Semantic grounding.
    - `domain_data`: Everything else (dynamic experts).
3.  **Joining**: Every data point is matched against a unique `entity_id` using the `id_column` mapping.

## 2. Layered Isolation Strategy

Mirror enforces strict boundaries between stages to prevent **Hindsight Bias**.

- **RAG Purge**: At the start of each stage (e.g., Stage 1 -> Stage 2), the ChromaDB collection is wiped.
- **Memory Repopulation**: Memory is only seeded with data available *up to that stage's temporal horizon*.

## 3. The RoleCoordinator Pattern

Each domain (e.g., `heart_rate`, `location`) is assigned a `RoleCoordinator`.

1.  **Complexity Scoping**: Before calling LLMs, the coordinator analyzes the variance and sparsity of the data slice.
2.  **Swarm Scaling**:
    - Low Complexity: 1 Agent.
    - High Complexity: Up to `SWARM_MAX_AGENTS`.
3.  **Consensus Aggregation**:
    - Votes are weighted by the agent's reported confidence.
    - If a conflict arises (e.g., 50/50 split), the coordinator flags the case for mandatory Level 2 review.

## 4. Economic Arbitration (Level 2)

Level 2 doesn't just "vote"; it **arbitrates**.

- **Transcript Analysis**: It reads the "internal monologue" and specific reasoning of every expert.
- **Contradiction Detection**: It specifically looks for experts who missed a detail that another caught.
- **Risk-Adjusted Sigmoid**: The final 0/1 decision is passed through a threshold that shifts based on `Settings.fp_cost` and `Settings.fn_cost`.

---
*Volume V: Feature Engineering Reference coming next.*
