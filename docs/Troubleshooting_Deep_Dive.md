# 📒 Project Antigravity: Volume VII — Deep Troubleshooting & Observability

This volume is for engineers and architects maintaining the Project Antigravity pipeline.

## 1. Observability Stack

The system integrates with **Langfuse** for trace-based debugging.
- **Trace Context**: Every decision is grouped by `session_id`.
- **Granularity**: Traces include Pillar 1 scores, Pillar 3 Swarm Expert monologues, and Pillar 4 final arbitration.

## 2. Common Scenarios

### 2.1 Low Recall (Too many False Negatives)
- **Fix**: Increase `fn_cost` in `settings.py` or `.env`. This forces the Pillar 4 judge to be more risk-averse.
- **Fix**: Check `analytical_squads.py` prompts to ensure they emphasize sensitivity over specificity.

### 2.2 Memory Drift
- **Fix**: If similar cases are being missed, ensure `pillar_memory_enabled` is True and check RAG query logs for relevant retrieval.

### 2.3 Performance Bottlenecks
- **Fix**: Reduce `swarm_max_agents` or adjust the `squad_scaling_factor` in `settings.py`.
- **Fix**: Use a faster model for Pillar 3 (e.g., `gemini-3-flash-preview`).

## 3. Terminal Diagnostics

Standard logs are saved per run:
- `./runs/run_<timestamp>/logs/actions.log`: High-level narrative of the pipeline execution.
- `./runs/run_<timestamp>/logs/troubleshooting.log`: Deep audit trail including JSON payloads and raw scores.
