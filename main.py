"""
CLI entry point for the Mirror Pipeline.

Usage:
    python main.py -m manifest.json
    python main.py -m manifest.json --log-level DEBUG
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
        help="Path to manifest.json (contains N stages with train/eval sources)",
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
        stage_results = pipeline.run(manifest_path=args.manifest)

        c = Console()
        for stage_name, results in stage_results.items():
            flagged = sum(1 for r in results if r.final_prediction == 1)
            c.print(
                f"  [bold]{stage_name}:[/] {flagged}/{len(results)} flagged"
            )
        c.print("\n[bold green]All stages complete.[/]")

    except Exception as exc:
        Console().print(f"[bold red]Pipeline failed:[/] {exc}")
        logging.exception("Pipeline failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
