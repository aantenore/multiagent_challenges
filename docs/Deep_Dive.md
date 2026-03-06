# Project Antigravity: Technical Deep Dive

Project Antigravity is a domain-agnostic anomaly detection framework built on four distinct "Intelligence Pillars". This document explains the step-by-step technological process.

## The 4-Pillar Workflow

### 1. Pillar 1: Mathematical Filter (Deterministic Ingestion)
The entry point of the pipeline. Unlike simple thresholding, it uses **Autocorrelation (ACF)** to understand the natural frequency of an entity (e.g., "this user typically syncs data every 7 days").
- **Process**: Merges all domain data into a daily sampled time series.
- **Trigger**: Slides a window and calculates a **Dynamic Z-Score**. If a deviation exceeds the threshold, it triggers an "Incident".
- **Output**: An incident window (e.g., "Start: 2026-01-01, End: 2026-01-15") and a `complexity_score`.

### 2. Pillar 2: Knowledge Store (Hierarchical RAG)
Acts as the system's "long-term memory". It prevents redundant LLM calls and provides cross-entity context.

#### RAG Lifecycle & Isolation
- **Persistence**: The RAG is **persistent** by default. It is *not* wiped at every startup unless `RAGStore.reset()` is explicitly called (manual/scripted). 
- **Stage Isolation**: To prevent data leakage (e.g., Stage 1 historical data affecting Stage 2), we use **namespacing**. Each collection is prefixed with the stage name (e.g., `stage_1_global_knowledge`).
- **Identity Level**: Stores historical dossiers and past verdicts for the specific entity.
- **Global Level**: Stores significant anomalies detected across *all* entities.
- **Process**: During analysis, the system retrieves the Top-K most similar historical cases to provide "precedents" to the agents.

### 3. Pillar 3: Analytical Squads (Dynamic Swarm)
A specialized swarm of LLM agents coordinated by a `RoleCoordinator`.
- **Scaling**: The number of agents in the swarm scales automatically: `n = base + (complexity * scaling)`.
- **Specialization**:
    - **Health & Behavioral**: Analyzes clinical metrics and lifestyle patterns.
    - **Spatial Patterns**: Tracks GPS pings, residence stability, and significant locations.
    - **Temporal Routines**: Looks for broken schedules or unusual event sequences.
    - **Profile & Context**: Interprets the raw persona and demographic data.
- **Consensus**: Agents vote on the prediction (Anomaly vs. Stability) and provide reasoning.

### 4. Pillar 4: Global Orchestrator (Reconciliation)
The "Supreme Court" of the system.
- **Process**: Collects the `SwarmConsensus` from all Pillar 3 squads.
- **Decision Logic**: Reconciles conflicting views. For example, if the Spatial Squad sees an anomaly but the Health Squad explains it as a "scheduled travel", the Orchestrator can override a simple majority based on holistic reasoning.

---

## Historical Indexing vs. Live Inference

### The "Historical Indexing" Pass
Before performing live inference on the continuous data stream, the system indexes the historical window.
1.  **Memory Population**: It reads the historical window, runs Pillar 1, and populates the RAG with the initial "ground truth" reasoning.
2.  **Verification**: It runs the full 4-Pillar pipeline on the historical entities to verify that the RAG and LLM prompts correctly identify known patterns.
3.  **Logs**: Generates `audit_log_sanity_<stage>.json` with a unique `sanity_` Session ID in Langfuse.

### The "Live Inference" Pass
Once the system is "primed" with memory:
1.  **Fresh Retrieval**: It analyzes live entities from the current window, using the historical memory to find similar behaviors.
2.  **Output**: Generates compliance-ready predictions (`predictions_<stage>.txt`) and production audit logs.
