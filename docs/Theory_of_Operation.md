# 📖 Project Antigravity: Volume I — Theory of Operation

## 1. The Antigravity Philosophy: Asymmetric Hierarchical Intelligence

At its core, **Project Antigravity** is a multi-layered response system designed to minimize **Critical Regret** (False Negatives) while maintaining operational viability (filtering False Positives). 

The system treats every data point as an abstract signal and every analyst as a configurable role, ensuring perfect domain agnosticity.

### 1.1 The Four Pillars of Intelligence
1.  **Pillar 1: Mathematical Filter (Dual-Track Ingestion)**: Uses high-speed statistical analysis and Autocorrelation (ACF) to define the "Volume of Normality." It filters stable cases at near-zero cost using a dual-track system:
    - **Math Track**: Strips all categorical data (e.g., `EventType`, `job`) to perform pure numerical operations on continuous metrics.
    - **Memory Track**: Retains the original raw dataset (including strings and text). When an anomaly is triggered by the Math Track, the system uses the "Incident Window" timestamps to crop data from the **Memory Track**, ensuring L1 LLM Swarms receive rich, context-heavy JSON describing exact events instead of just raw numbers.
2.  **Pillar 2: Memory Store**: A hierarchical vector memory system that stores "Identity", "Context", and "Global" behaviors to provide grounding for decision making.
3.  **Pillar 3: Analytical Squads**: Specialized swarms of domain-expert agents that analyze specific data slices to verify or explain anomalies detected by Pillar 1.
4.  **Pillar 4: Global Orchestrator**: A high-reasoning judge that performs final behavioral reconciliation and recovery assessment based on economic risk (`FN_COST` vs `FP_COST`).

---

## 2. Advanced Feature Engineering: Temporal Rhythm Mastery

### 2.1 Autocorrelation (ACF) Discovery
Antigravity automatically discovers the "heartbeat" of each entity.
- **The Algorithm**: For every numeric signal, we compute the ACF at various lags to find the natural frequency.
- **Dynamic Lags**: If an entity has a 48h behavioral cycle, the system automatically uses matching lags for its gradient and velocity calculations.
- **Resilience**: This reduces noise by baseline-adjusting for periodic patterns.

---

## 3. The Hierarchical Memory Circuit (Pillar 2)

Antigravity implements a **Self-Purifying Memory Store** using a 3-tier vector hierarchy:
- **Identity Tier**: Individual history of the entity.
- **Context Tier**: Cross-entity signals within similar groupings (e.g. Cohort tags based on the manifest `cohort_key`).
- **Global Tier**: Universal patterns and architectural heuristics.

During Historical Indexing, the system seeds the memory with confirmed stable trajectories from the historical window, allowing analytical squads to use "few-shot" grounding during Live Inference.

---

## 4. Economic Orchestration: Rational Decision Theory

The final decision (Pillar 4) is a product of risk-adjusted reasoning:
$$Decision = \text{Judge}(Verdicts, Memory, Context, \frac{FN_{cost}}{FP_{cost}})$$

### 4.1 Unanimous Consensus & Threshold Skipping (Skip L2)
When the Level 1 (L1) Swarm achieves a unanimous consensus on an entity, the system evaluates the internal confidence score. If this score exceeds the configurable environmental variable `L1_SKIP_L2_CONFIDENCE_THRESHOLD` (defaulting to 0.98), the system short-circuits the Level 2 (L2) Global Orchestrator. This ensures expensive arbitration is reserved only for complex or contradictory edge cases.

If the cost of missing a case is significantly higher than a false alarm, the Orchestrator will flag even subtle deviations if the squad reasoning suggests a potential for harm.
