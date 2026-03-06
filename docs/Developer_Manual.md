# 📙 Project Antigravity: Volume III — Developer Manual

A technical deep-dive into the Project Antigravity codebase.

## 1. Development Workflow

### 1.1 The Streaming Execution
1.  **Historical Indexing**: The pipeline first processes the historical window to seed the Memory Store (Pillar 2).
2.  **Live Inference**: The pipeline then evaluates the continuous data stream using the established memory and filters.

### 1.2 Model Tiering
Antigravity supports a triple-tier model architecture to optimize cost vs. intelligence:
- **Nano**: Used for Pillar 1 (Mathematical Filter) prompting.
- **Cheap**: Used for Pillar 3 (Analytical Squads) swarms.
- **Smart**: Used for Pillar 4 (Global Orchestrator) final judgment.

## 2. Code Standards

- **Absolute Agnosticity**: Never hardcode strings like "User", "Patient", or "Citizen" in core modules. Use `{entity_id}`.
- **Stateless Pillars**: Pillars should be self-contained and resettable between stages.
- **Pydantic Validation**: All configurations must be validated via `settings.py`.

## 3. Telemetry & Tracing

Project Antigravity leverages Langfuse for deep observability. Execution traces for LLM interactions and squad consensus must be tagged properly to distinguish between operations. 
- Execution traces are tagged dynamically as `['historical_indexing']` when processing the past window.
- Execution traces are tagged dynamically as `['live_inference']` when scoring the current continuous sliding window.
