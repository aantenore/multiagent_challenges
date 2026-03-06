# 📖 The Book of Mirror: Volume I — Theory of Operation

## 1. The Mirror Philosophy: Asymmetric Hierarchical Intelligence

At its core, **Mirror** is a multi-layered response system designed to minimize **Critical Regret** (False Negatives) while maintaining operational viability (filtering False Positives).

### 1.1 The Inverted Triage Pyramid
1.  **Level 0 (Mathematical Baseline)**: Uses `IsolationForest` to define the "Volume of Normality." It filters >>90% of data at near-zero token cost.
2.  **Level 1 (Swarm Verification)**: Parallel domain experts (LLMs) verify outliers. They act as "Skeptics," looking for valid contextual reasons for a statistical anomaly.
3.  **Level 2 (Economic Arbitration)**: A high-reasoning orchestrator makes the final call based on the cost of error (`FN_COST` vs `FP_COST`).

---

## 2. Advanced Feature Engineering: Temporal Rhythm Mastery

### 2.1 Autocorrelation (ACF) Discovery
Unlike static window systems, Mirror discovers the "heartbeat" of each entity.
- **The Algorithm**: For every numeric signal, we compute the ACF at various lags.
- **Dynamic Lags**: If a person has a 48h exercise cycle, Mirror automatically uses a 48h lag for its gradient and velocity calculations.
- **Resilience**: This prevents a "workout day" from being flagged as a "crisis" simply because it differs from the average of the last 3 days.

---

## 3. The RAG Memory Circuit

Mirror implements a **Self-Purifying RAG** system using ChromaDB.
- **Training Distillation**: During the training phase, Layer 0 identifies its own mistakes (False Positives on training data).
- **Explanation Storage**: Layer 2 explains *why* those training cases are normal despite being outliers.
- **Few-Shot Retrieval**: During evaluation, Layer 1 agents retrieve these specific explanations to prevent repeating past misclassifications.

---

## 4. Economic Orchestration: Rational Decision Theory

The final decision is a product of:
$$Decision = \sigma( \sum w_i * Verdict_i ) \times \frac{FN_{cost}}{FP_{cost}}$$

If the cost of missing a case is 5x higher than a false alarm, the Orchestrator will flag even "Subtle" anomalies if the reasoning suggests a risk.

---
*Volume II: The Implementation Guide coming next.*
