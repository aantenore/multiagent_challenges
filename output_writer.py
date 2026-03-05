"""
Output writer — generates the challenge-compliant TXT file
and prints the Langfuse session ID.
"""

from __future__ import annotations

import logging
from pathlib import Path

from models import PipelineResult

logger = logging.getLogger(__name__)


def write_predictions(
    results: list[PipelineResult],
    output_path: str | Path,
) -> Path:
    """Write entity IDs classified as 1 (preventive support) to a TXT file.

    Format: one CitizenID per line, ASCII, newline-separated.

    Parameters
    ----------
    results:
        Pipeline outputs for all entities.
    output_path:
        Destination file path.

    Returns
    -------
    Resolved Path to the written file.
    """
    out = Path(output_path)
    flagged = sorted(
        r.entity_id for r in results if r.final_prediction == 1
    )

    out.write_text(
        "\n".join(flagged) + ("\n" if flagged else ""),
        encoding="ascii",
    )
    logger.info(
        "Wrote %d flagged IDs to %s (total entities: %d)",
        len(flagged), out, len(results),
    )
    return out


def write_audit_log(
    results: list[PipelineResult],
    output_path: str | Path,
) -> Path:
    """Write detailed audit log of the pipeline execution to JSON."""
    out = Path(output_path)
    
    audit_data = []
    for r in results:
        # Serialize verdicts into basic dicts without complex object types
        verdicts_data = []
        for v in r.verdicts:
            verdicts_data.append({
                "agent_name": getattr(v, "agent_name", "Unknown"),
                "prediction": v.prediction,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
            })
            
        audit_data.append({
            "entity_id": r.entity_id,
            "final_prediction": r.final_prediction,
            "layer_decided": r.layer_decided,
            "verdicts": verdicts_data
        })
        
    import json
    out.write_text(
        json.dumps(audit_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    logger.info("Wrote audit log for %d entities to %s", len(results), out)
    return out
