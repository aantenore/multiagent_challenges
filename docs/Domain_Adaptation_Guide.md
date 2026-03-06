# 📗 The Book of Mirror: Volume II — Domain Adaptation Guide

Mirror is intentionally **domain-ignorant**. It treats all data as abstract analytical dimensions. This guide explains how to morph Mirror for any industry.

## 1. The Domain Shift Workflow

### Case A: Financial Fraud Detection
1.  **Manifest**: 
    - Change roles to `transactions` (temporal), `user_metadata` (profile), `fraud_handbook` (context).
2.  **Prompts**: 
    - Update `domain_agent.txt`: "You are a senior AML investigator."
    - Update `orchestrator.txt`: Mention detection patterns like "layering" or "smurfing."
3.  **Settings**:
    - Update `FN_COST`: The cost of a stolen $1M is much higher than a blocked card (FP).

### Case B: Industrial Predictive Maintenance
1.  **Manifest**:
    - Roles: `vibration_sensors` (temporal), `machine_specs` (profile), `maintenance_logs` (context).
2.  **Feature Ignore**:
    - Add `sensor_serial` to `feature_ignore_columns`.

---

## 2. Configuring the Abstraction Layer

Mirror doesn't hardcode "Medical" logic. It uses **Descriptor Roles**:
- **Profile Role**: Provides the steady-state context.
- **Context Role**: Provides the "unstructured wisdom" (handbooks, notes).
- **Analytical Roles**: Any other string (e.g., "spatial", "temporal", "logs") becomes a domain for a Swarm expert.

---

## 3. Multi-Stage Pipeline Logic

Mirror supports N-stages. Each stage can represent:
- **Increasing Complexity**: Level 1 data -> Level 2 data -> Level 3 data.
- **Time Window Shift**: Q1 data -> Q2 data.
- **Isolation**: Each stage resets RAG and L0 to prevent "Context Osmosis" (bleeding information from future to past).

---
*Volume III: The Developer Manual coming next.*
