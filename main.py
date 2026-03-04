"""
CLI entry point for the Mirror Pipeline.

Usage:
    python main.py --manifest manifest.json --output predictions.txt
    python main.py --manifest manifest.json --ground-truth labels.json --output predictions.txt
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Adaptive Multi-Agent Pipeline — Reply Mirror",
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default="manifest.json",
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        default=None,
        help="Optional ground-truth file (JSON, CSV, or TXT) for training L0 and evaluation",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="predictions.txt",
        help="Output file for flagged entity IDs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    args = parser.parse_args()

    # ── Logging setup ───────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=Console(stderr=True))],
    )

    # ── Run pipeline ────────────────────────────────────────────────
    from pipeline import AdaptivePipeline

    pipeline = AdaptivePipeline()
    try:
        results = pipeline.run(
            manifest_path=args.manifest,
            ground_truth_path=args.ground_truth,
            output_path=args.output,
        )
        flagged = sum(1 for r in results if r.final_prediction == 1)
        Console().print(
            f"\n[bold]Done.[/] {flagged}/{len(results)} entities flagged for "
            f"preventive support → [cyan]{args.output}[/]"
        )
    except Exception as exc:
        Console().print(f"[bold red]Pipeline failed:[/] {exc}")
        logging.exception("Pipeline failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
