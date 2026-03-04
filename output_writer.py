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
