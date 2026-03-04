"""
Multi-format data loader.
Dispatches on ManifestEntry.format to load CSV, JSON, and Markdown tables.
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import pandas as pd

from models import ManifestEntry

logger = logging.getLogger(__name__)


def load_file(entry: ManifestEntry, base_dir: Path) -> pd.DataFrame:
    """Load a single data file described by a ManifestEntry.

    Parameters
    ----------
    entry:
        Manifest descriptor with path, format, id_column.
    base_dir:
        Root directory from which ``entry.path`` is resolved.

    Returns
    -------
    pd.DataFrame with at least the ``entry.id_column`` column.
    """
    file_path = base_dir / entry.path
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info("Loading [%s] %s  (id=%s)", entry.format, file_path, entry.id_column)

    match entry.format:
        case "csv":
            df = _load_csv(file_path)
        case "json":
            df = _load_json(file_path)
        case "md":
            df = _load_markdown(file_path)
        case _:
            raise ValueError(f"Unsupported format: {entry.format!r}")

    if entry.id_column not in df.columns:
        raise KeyError(
            f"ID column {entry.id_column!r} not found in {file_path}. "
            f"Available: {list(df.columns)}"
        )

    logger.info("  → %d rows, %d cols", len(df), len(df.columns))
    return df


# ── Private helpers ─────────────────────────────────────────────────────


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def _load_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return pd.json_normalize(raw, sep="_")
    if isinstance(raw, dict):
        # Attempt to flatten single-level dict of lists
        return pd.json_normalize([raw], sep="_")
    raise ValueError(f"Unexpected JSON shape in {path}")


def _load_markdown(path: Path) -> pd.DataFrame:
    """Parse a Markdown file into a DataFrame.

    Strategy:
    - If the file contains pipe-delimited tables, extract the first one.
    - Otherwise, parse structured sections (## HEADING blocks) into rows
      with columns: ``entity_id`` and ``context_text``.
    """
    text = path.read_text(encoding="utf-8")

    # Try pipe-delimited table first
    table_lines = [
        line for line in text.splitlines()
        if "|" in line and not line.strip().startswith("|-")
    ]
    if len(table_lines) >= 2:
        return _parse_pipe_table(table_lines)

    # Fallback: section-based parsing (personas.md style)
    return _parse_sections(text)


def _parse_pipe_table(lines: list[str]) -> pd.DataFrame:
    """Convert | col1 | col2 | style lines into a DataFrame."""
    # Remove separator rows like |---|---|
    data_lines = [
        ln for ln in lines
        if not re.match(r"^\s*\|[\s\-:]+\|", ln)
    ]
    if not data_lines:
        return pd.DataFrame()

    header = [c.strip() for c in data_lines[0].split("|") if c.strip()]
    rows: list[list[str]] = []
    for ln in data_lines[1:]:
        cells = [c.strip() for c in ln.split("|") if c.strip()]
        if len(cells) == len(header):
            rows.append(cells)
    return pd.DataFrame(rows, columns=header)


def _parse_sections(text: str) -> pd.DataFrame:
    """Parse ``## ID - Name`` sections into entity_id + context_text rows."""
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    rows: list[dict[str, str]] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # First line is the heading: "ID - Name"
        first_line, _, body = sec.partition("\n")
        # Extract entity ID (first word, usually an 8-char code)
        entity_id_match = re.match(r"(\S+)", first_line)
        if not entity_id_match:
            continue
        entity_id = entity_id_match.group(1)
        rows.append({
            "entity_id": entity_id,
            "context_text": sec.strip(),
        })
    return pd.DataFrame(rows)
