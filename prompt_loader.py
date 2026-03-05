"""
Prompt loader utility.
Loads prompt templates from the prompts/ directory, falling back
to built-in defaults if files are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# ── Built-in defaults (used if template files are missing) ──────────────

_DEFAULT_SYSTEM = (
    "You are a health-trajectory analysis agent. "
    'Respond ONLY with a JSON object: '
    '{{"prediction": 0 or 1, "confidence": 0.0-1.0, "reasoning": "..."}}. '
    "prediction=0 means standard monitoring, "
    "prediction=1 means preventive support needed."
)


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without .txt extension).

    Looks in the ``prompts/`` directory. Falls back to a built-in default
    if the file doesn't exist.

    Parameters
    ----------
    name:
        Template name, e.g. ``"system"``, ``"domain_agent"``, ``"orchestrator"``.

    Returns
    -------
    Template string (may contain ``{placeholder}`` markers).
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        logger.debug("Loaded prompt template: %s", path)
        return text

    logger.debug("Prompt file not found (%s), using built-in default", path)
    if name == "system":
        return _DEFAULT_SYSTEM
    return ""
