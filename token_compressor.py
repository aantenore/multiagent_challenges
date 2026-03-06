"""
TokenCompressor — Collision-Proof Token Compression for L1 Swarm Prompts.

Implements 4 compression pillars to drastically reduce context window size:
  1. Syntactic Compression: JSON → Markdown Table
  2. Collision-Proof Key Minification: Acronym Mapping with legend
  3. Run-Length Encoding: Temporal Rollup for stable baselines
  4. Persona Compression: Regex-based fluff removal
"""

from __future__ import annotations

import re
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


# ── Pillar 2: Collision-Proof Acronym Mapping ────────────────────────────


class AcronymMapper:
    """Generates collision-proof acronyms for column names.

    Resolution order:
    1. Explicit map from manifest config (``acronym_map``).
    2. Auto-generated from uppercase letters in CamelCase names.
    3. First 3 characters for single-word / all-lowercase names.
    4. Collision suffix (``PA1``, ``PA2``) if the acronym is already taken.
    """

    def __init__(self, explicit_map: dict[str, str] | None = None) -> None:
        self._explicit: dict[str, str] = explicit_map or {}
        self._seen: dict[str, str] = {}          # acronym → original column
        self._col_to_acronym: dict[str, str] = {}  # column → acronym

        # Pre-register explicit mappings
        for col, acr in self._explicit.items():
            acr_upper = acr.upper()
            self._seen[acr_upper] = col
            self._col_to_acronym[col] = acr_upper

    # ── Public API ───────────────────────────────────────────────────

    def get_acronym(self, col_name: str) -> str:
        """Return a collision-proof acronym for *col_name*."""
        if col_name in self._col_to_acronym:
            return self._col_to_acronym[col_name]

        # Step 1: Check explicit map
        if col_name in self._explicit:
            acr = self._explicit[col_name].upper()
        else:
            # Step 2: Auto-generate
            acr = self._auto_generate(col_name)

        # Step 3: Collision resolution
        acr = self._resolve_collision(acr, col_name)

        self._seen[acr] = col_name
        self._col_to_acronym[col_name] = acr
        return acr

    def get_legend(self) -> str:
        """Return a compact legend string mapping acronyms to columns."""
        if not self._col_to_acronym:
            return ""
        pairs = [f"{acr}={col}" for col, acr in self._col_to_acronym.items()]
        return f"Legend: {', '.join(pairs)}"

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _auto_generate(col_name: str) -> str:
        """Extract uppercase letters from CamelCase, or first 3 chars."""
        uppers = re.findall(r"[A-Z]", col_name)
        if len(uppers) >= 2:
            return "".join(uppers).upper()
        # Single-word / lowercase fallback: first 3 chars
        clean = re.sub(r"[^a-zA-Z0-9]", "", col_name)
        return clean[:3].upper()

    def _resolve_collision(self, acr: str, col_name: str) -> str:
        """Append incremental integer if acronym already taken."""
        if acr not in self._seen or self._seen[acr] == col_name:
            return acr
        counter = 1
        while f"{acr}{counter}" in self._seen:
            counter += 1
        return f"{acr}{counter}"


# ── Pillar 1: Syntactic Compression (JSON → Markdown Table) ─────────────


def _rows_to_markdown(rows: list[dict[str, Any]], mapper: AcronymMapper) -> str:
    """Convert a list of row-dicts into a compact Markdown table with acronym headers."""
    if not rows:
        return "_No data._"

    # Collect all keys in stable order
    all_keys: list[str] = list(OrderedDict.fromkeys(k for row in rows for k in row.keys()))

    # Build header with acronyms
    header_acronyms = [mapper.get_acronym(k) for k in all_keys]
    header_line = " | ".join(header_acronyms)
    separator = " | ".join("---" for _ in header_acronyms)

    # Build data rows
    data_lines: list[str] = []
    for row in rows:
        cells = []
        for k in all_keys:
            val = row.get(k, "")
            # Compact formatting for floats
            if isinstance(val, float):
                cells.append(f"{val:.1f}")
            else:
                cells.append(str(val))
        data_lines.append(" | ".join(cells))

    return f"{header_line}\n{separator}\n" + "\n".join(data_lines)


# ── Pillar 3: Run-Length Encoding (Temporal Rollup) ──────────────────────


def _temporal_rollup(
    rows: list[dict[str, Any]],
    date_key: str = "Timestamp",
    variance_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Collapse consecutive rows with identical categoricals and low numeric variance.

    Returns a new list of rows where stable baselines are merged into
    summary rows with date ranges and averaged numeric values.
    """
    if len(rows) <= 1:
        return rows

    def _categoricals(row: dict) -> dict:
        return {k: v for k, v in row.items() if isinstance(v, str) and k != date_key}

    def _numerics(row: dict) -> dict:
        return {k: v for k, v in row.items() if isinstance(v, (int, float))}

    result: list[dict] = []
    group: list[dict] = [rows[0]]

    for row in rows[1:]:
        prev_cats = _categoricals(group[-1])
        curr_cats = _categoricals(row)

        if prev_cats == curr_cats:
            group.append(row)
        else:
            result.extend(_flush_group(group, date_key, variance_threshold))
            group = [row]

    result.extend(_flush_group(group, date_key, variance_threshold))
    return result


def _flush_group(
    group: list[dict],
    date_key: str,
    variance_threshold: float,
) -> list[dict]:
    """Flush a group of similar rows. Collapse if stable, else return as-is."""
    if len(group) <= 2:
        return group

    # Check numeric variance within the group
    numeric_keys = [k for k in group[0] if isinstance(group[0].get(k), (int, float))]
    if numeric_keys:
        for nk in numeric_keys:
            values = [r.get(nk, 0) for r in group if isinstance(r.get(nk), (int, float))]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            if variance ** 0.5 > variance_threshold:
                # High variance — don't collapse
                return group

    # Collapse: build a summary row
    first_date = group[0].get(date_key, "?")
    last_date = group[-1].get(date_key, "?")
    summary = {date_key: f"{first_date} to {last_date}"}

    # Copy categoricals from the first row
    for k, v in group[0].items():
        if isinstance(v, str) and k != date_key:
            summary[k] = v

    # Average numerics
    for nk in numeric_keys:
        values = [r[nk] for r in group if isinstance(r.get(nk), (int, float))]
        if values:
            summary[nk] = round(sum(values) / len(values), 1)

    return [summary]


# ── Pillar 4: Persona Compression (Fluff Removal) ───────────────────────

# Matches lines starting with bold markdown (**Key:**) or configured pattern prefixes
_BOLD_PATTERN = re.compile(r"^\*\*[^*]+\*\*:?.*", re.MULTILINE)


def compress_persona(text: str, extra_prefixes: list[str] | None = None) -> str:
    """Extract only the analytically dense lines from a persona narrative.

    Keeps lines matching bold markdown patterns or configured behavioural
    """
    if not text:
        return "N/A"

    # Start with bold-markdown matches
    matches = _BOLD_PATTERN.findall(text)

    # Also match lines starting with any configured prefix
    from settings import get_settings
    prefixes = get_settings().persona_dense_prefixes
    if extra_prefixes:
        prefixes = list(set(prefixes + extra_prefixes))

    if prefixes:
        escaped = [re.escape(p) for p in prefixes]
        prefix_pattern = re.compile(
            r"^(?:" + "|".join(escaped) + r")[:\s].*",
            re.IGNORECASE | re.MULTILINE,
        )
        matches.extend(prefix_pattern.findall(text))

    if matches:
        return "\n".join(m.strip() for m in matches)

    # Fallback: if no structured bullets found, return truncated original
    return text[:500]


# ── Public Facade ────────────────────────────────────────────────────────


def compress_domain_data(
    data: list[dict[str, Any]],
    mapper: AcronymMapper,
    date_key: str = "Timestamp",
) -> str:
    """Full compression pipeline for a domain data slice.

    1. Temporal Rollup (RLE)
    2. JSON → Markdown Table with acronym headers
    """
    if not data:
        return "_No data._"

    rolled = _temporal_rollup(data, date_key=date_key)
    return _rows_to_markdown(rolled, mapper)
