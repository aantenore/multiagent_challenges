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

    # Actions file is INFO for a clean, expressive flow
    actions_handler = logging.FileHandler(actions_file, encoding="utf-8")
    actions_handler.setLevel(logging.INFO)
    actions_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    )

    # Troubleshoot file is ALWAYS DEBUG for deep inspection
    troubleshoot_handler = logging.FileHandler(troubleshoot_file, encoding="utf-8")
    troubleshoot_handler.setLevel(logging.DEBUG)
    troubleshoot_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s [%(levelname)s] %(message)s")
    )

    # Re-apply BasicConfig
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.DEBUG,  # Root logger must be DEBUG to feed the troubleshoot handler
        handlers=[console_handler, actions_handler, troubleshoot_handler],
    )
    logging.info("Logs saved: %s (flow) and %s (debug)", actions_file.name, troubleshoot_file.name)

    # ── Run pipeline ────────────────────────────────────────────────
    from pipeline import AdaptivePipeline

    pipeline = AdaptivePipeline()
    try:
        stage_results = pipeline.run(
            manifest_path=args.manifest, 
            results_dir=results_dir,
            target_level=args.level
        )

        c = Console()
        for stage_name, results in stage_results.items():
            flagged = sum(1 for r in results if r.final_prediction == 1)
            c.print(
                f"  [bold]{stage_name}:[/] {flagged}/{len(results)} flagged"
            )
        c.print("\n[bold green]All stages complete.[/]")
        
        # No flush needed per user request
        pass

    except Exception as exc:
        Console().print(f"[bold red]Pipeline failed:[/] {exc}")
        logging.exception("Pipeline failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
