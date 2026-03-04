"""
build_submission.py — creates a challenge-ready ZIP archive.

Includes only source code and metadata.
Excludes: .env, data files, vector DB, __pycache__, .git, resources/.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
OUTPUT_NAME = "submission.zip"

# Extensions to include
INCLUDE_EXTENSIONS = {".py", ".toml", ".json", ".md"}

# Top-level files/dirs to always exclude
EXCLUDE_NAMES = {
    ".env",
    ".git",
    "__pycache__",
    "chroma_db",
    "resources",
    "submission.zip",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    ".agents",
}

# Patterns in paths to exclude
EXCLUDE_PATTERNS = {"__pycache__", ".git", "chroma_db", "resources", ".env"}


def should_include(path: Path) -> bool:
    """Decide whether a file belongs in the submission."""
    # Must be a file
    if not path.is_file():
        return False

    # Check extension
    if path.suffix not in INCLUDE_EXTENSIONS:
        return False

    # Check excluded names
    for part in path.parts:
        if part in EXCLUDE_NAMES:
            return False
        for pattern in EXCLUDE_PATTERNS:
            if pattern in part:
                return False

    # Never include .env even if renamed
    if path.name.startswith(".env"):
        return False

    return True


def build_submission() -> Path:
    """Create the submission.zip archive."""
    output_path = PROJECT_ROOT / OUTPUT_NAME

    included: list[Path] = []
    for root, _dirs, files in os.walk(PROJECT_ROOT):
        for fname in files:
            fpath = Path(root) / fname
            rel = fpath.relative_to(PROJECT_ROOT)
            if should_include(fpath):
                included.append(rel)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(included):
            zf.write(PROJECT_ROOT / rel, arcname=str(rel))
            print(f"  + {rel}")

    print(f"\n✓ Created {output_path} with {len(included)} files")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

    # Verify no sensitive files leaked
    with zipfile.ZipFile(output_path, "r") as zf:
        names = zf.namelist()
        for name in names:
            assert ".env" not in name, f"SECURITY: .env found in archive: {name}"
        print("  ✓ No .env files in archive")

    return output_path


if __name__ == "__main__":
    build_submission()
