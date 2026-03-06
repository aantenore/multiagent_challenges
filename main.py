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

from dotenv import load_dotenv
load_dotenv()

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
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Specify a single stage/level to run (e.g., 'level_1')",
    )

    args = parser.parse_args()

    # ── Logging setup ───────────────────────────────────────────────
    from datetime import datetime
    import os
    from pathlib import Path

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"run_{run_timestamp}"
    log_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    actions_file = log_dir / "actions.log"
    troubleshoot_file = log_dir / "troubleshooting.log"

    # Console handler respects --log-level
    console_handler = RichHandler(rich_tracebacks=True, console=Console(stderr=True))
    console_handler.setLevel(getattr(logging, args.log_level))

    # [Expressive Flow] Actions file captures high-level narrative
    actions_handler = logging.FileHandler(actions_file, encoding="utf-8")
    actions_handler.setLevel(logging.INFO)
    actions_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)7s] %(message)s")
    )

    # [Deep Audit] Troubleshoot file captures the full trace with session IDs
    troubleshoot_handler = logging.FileHandler(troubleshoot_file, encoding="utf-8")
    troubleshoot_handler.setLevel(logging.DEBUG)
    troubleshoot_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)-7s][%(name)s] [%(process)d] %(message)s")
    )

    # Root Logger Configuration
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.DEBUG,  # Feed all handlers; filters happen at handler level
        handlers=[console_handler, actions_handler, troubleshoot_handler],
    )
    
    logger = logging.getLogger("MirrorMain")
    logger.info("  [System] Run ID: %s", run_timestamp)
    logger.info("  [Output] Logs: %s (Standard) | %s (Deep Debug)", actions_file, troubleshoot_file)
    logger.info("  [Output] Results will be saved in %s", results_dir)

    # ── Run pipeline ────────────────────────────────────────────────
    from pipeline import AdaptivePipeline

    logger.info("  [Pipeline] Initializing Adaptive Multilevel Pipeline...")
    pipeline = AdaptivePipeline()
    try:
        stage_results = pipeline.run(
            manifest_path=args.manifest, 
            results_dir=results_dir,
            target_level=args.level
        )

        c = Console()
        c.print("\n[bold green]─ Stage Summary ──────────────────────────────────[/]")
        for stage_name, results in stage_results.items():
            flagged = sum(1 for r in results if r.final_prediction == 1)
            secure = len(results) - flagged
            c.print(
                f"  ● [bold]{stage_name:15}:[/] Total {len(results):2} | [green]Secure: {secure:2}[/] | [red]Flagged: {flagged:2}[/]"
            )
        c.print("[bold green]──────────────────────────────────────────────────[/]")
        c.print(f"[bold green]Mission accomplished. Check {results_dir} for details.[/]")

    except Exception as exc:
        Console().print(f"[bold red]Critical Pipeline Failure:[/] {exc}")
        logging.exception("Mirror Pipeline crashed during execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
