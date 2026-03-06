# 📒 The Book of Mirror: Volume VII — Deep Troubleshooting & Observability

This volume is for Site Reliability Engineers (SREs) and Architects maintaining the Mirror pipeline.

## 1. Observability Stack: Langfuse

Mirror integrates deeply with **Langfuse** for trace-based debugging.

### 1.1 Trace ID Resolution
Every Run generates a unique `Run ID` (format: `run_YYYYMMDD_HHMMSS`).
- In Langfuse, search for `session_id == Run ID` to see the complete tree of calls.
- **Traces include**: L0 scores, Swarm Expert monologues, and the final Level 2 arbitration reasoning.

## 2. Common Error States

### 2.1 `NameError: name 'cfg' is not defined`
- **Cause**: Refactoring of a static method to a domain-agnostic pattern without injecting the settings singleton.
- **Fix**: Ensure `cfg = get_settings()` is called inside the method scope or passed as an argument.

### 2.2 `AttributeError: object has no attribute 'spatial_data'`
- **Cause**: Legacy code (Medical Domain) attempting to access hardcoded fields on the new `EntityDossier`.
- **Fix**: Use `dossier.domain_data.get('your_role')` or `get_settings().profile_role`.

### 2.3 RAG Connection Failures
- **Cause**: Concurrent access to ChromaDB or corrupted index.
- **Fix**: Wipe the `./chroma_db` directory. Mirror will automatically rebuild it from the current stage data.

## 3. Performance Tuning

If the pipeline is slow:
1.  **Reduce `max_workers`** in `pipeline.py` if hitting Rate Limits.
2.  **Increase `l0_upper_threshold`** to allow more cases to be filtered mathematically at Level 0.
3.  **Use `bypass_l0=True`** only for local unit testing of LLM logic.

---
*Developed for the Google Deepmind Agentic Challenge 2026. This concludes the primary documentation suite.*
