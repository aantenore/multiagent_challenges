"""
AdaptivePipeline — full 4-layer orchestrator.

Flow per entity:
  L0 (ML Router)
    → if confident → emit verdict
    → else → L1 (Domain Swarm) → L2 (Global Orchestrator)
  L3 (RAG) feeds few-shot examples into L1 and stores errors post-hoc.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from domain_swarm import SwarmFactory
from dossier_builder import DossierBuilder
from feature_engineer import SlidingWindowExtractor
from layer0_router import DeterministicRouter
from manifest_manager import ManifestManager
from metrics import print_report
from models import EntityDossier, PipelineResult
from orchestrator import GlobalOrchestrator
from output_writer import write_predictions
from rag_store import RAGStore
from settings import get_settings

logger = logging.getLogger(__name__)
console = Console()


class AdaptivePipeline:
    """End-to-end adaptive multi-agent classification pipeline."""

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._router = DeterministicRouter()
        self._rag = RAGStore()
        self._extractor = SlidingWindowExtractor()
        self._orchestrator = GlobalOrchestrator()
        self._session_id = str(uuid.uuid4())

    # ── Main entry point ────────────────────────────────────────────────

    @observe(name="pipeline_run")
    def run(
        self,
        manifest_path: str | Path,
        ground_truth_path: str | Path | None = None,
        output_path: str | Path = "predictions.txt",
    ) -> list[PipelineResult]:
        """Execute the full pipeline.

        Parameters
        ----------
        manifest_path:
            Path to manifest.json.
        ground_truth_path:
            Optional JSON/CSV with {entity_id: label} for L0 training & metrics.
        output_path:
            Where to write the flagged-ID TXT file.
        """
        # Langfuse session
        langfuse_context.update_current_trace(
            session_id=self._session_id,
            name="mirror_pipeline",
        )

        console.rule("[bold cyan]Mirror Pipeline — Starting")
        console.print(f"Session ID: [bold green]{self._session_id}[/]")

        # ── 1. Load data ────────────────────────────────────────────
        manager = ManifestManager(manifest_path)
        manager.load()
        builder = DossierBuilder(manager)
        dossiers = builder.build_all()
        console.print(f"Loaded [bold]{len(dossiers)}[/] entity dossiers")

        # ── 2. Feature engineering ──────────────────────────────────
        for eid, dossier in dossiers.items():
            dossier.features = self._extractor.extract(dossier)

        # ── 3. Load ground truth (optional) ─────────────────────────
        labels: dict[str, int] = {}
        if ground_truth_path:
            labels = self._load_ground_truth(ground_truth_path)
            console.print(f"Ground truth loaded: {len(labels)} labels")

        # ── 4. Train L0 if we have labels ───────────────────────────
        if labels:
            self._train_l0(dossiers, labels)

        # ── 5. Create L1 swarm ──────────────────────────────────────
        swarm_agents = SwarmFactory.create_swarm(
            manager.roles,
            manifest_entries=manager.manifest.sources,
        )

        # ── 6. Process each entity ──────────────────────────────────
        results: list[PipelineResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Processing entities…", total=len(dossiers))

            for eid, dossier in dossiers.items():
                result = self._process_entity(
                    dossier, swarm_agents, labels.get(eid),
                )
                results.append(result)
                progress.advance(task_id)

        # ── 7. Write output ─────────────────────────────────────────
        write_predictions(results, output_path)

        # ── 8. Metrics (if ground truth available) ──────────────────
        if labels:
            y_true = [labels[r.entity_id] for r in results if r.entity_id in labels]
            y_pred = [r.final_prediction for r in results if r.entity_id in labels]
            if y_true:
                console.rule("[bold cyan]Evaluation Report")
                print_report(y_true, y_pred)

        # ── 9. Store errors in RAG ──────────────────────────────────
        if labels and self._rag.is_enabled:
            for r in results:
                true_label = labels.get(r.entity_id)
                if true_label is not None and r.final_prediction != true_label:
                    summary = self._rag.summarise_dossier(dossiers[r.entity_id])
                    self._rag.add_error_case(
                        r.entity_id, summary, true_label, r.final_prediction,
                    )

        # ── 10. Print session ID ────────────────────────────────────
        console.rule("[bold cyan]Pipeline Complete")
        console.print(f"\n[bold green]Langfuse Session ID: {self._session_id}[/]\n")

        return results

    # ── Entity processing ───────────────────────────────────────────────

    def _process_entity(
        self,
        dossier: EntityDossier,
        swarm_agents,
        true_label: int | None,
    ) -> PipelineResult:
        """Process a single entity through L0 → L1 → L2."""
        eid = dossier.entity_id
        verdicts = []

        # ── Layer 0 ────────────────────────────────────────────────
        l0_verdict = self._router.to_verdict(eid, dossier.features)
        if l0_verdict is not None:
            verdicts.append(l0_verdict)
            return PipelineResult(
                entity_id=eid,
                final_prediction=l0_verdict.prediction,
                layer_decided="L0_DeterministicRouter",
                verdicts=verdicts,
            )

        # ── Layer 1 — Domain Swarm ─────────────────────────────────
        rag_examples = []
        if self._rag.is_enabled:
            summary = self._rag.summarise_dossier(dossier)
            rag_examples = self._rag.query_similar(summary)

        swarm_verdicts = SwarmFactory.run_swarm(
            swarm_agents, dossier, rag_examples,
        )
        verdicts.extend(swarm_verdicts)

        # ── Layer 2 — Global Orchestrator ──────────────────────────
        final_verdict = self._orchestrator.decide(
            dossier, swarm_verdicts, rag_examples,
        )
        verdicts.append(final_verdict)

        return PipelineResult(
            entity_id=eid,
            final_prediction=final_verdict.prediction,
            layer_decided="L2_GlobalOrchestrator",
            verdicts=verdicts,
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _train_l0(
        self,
        dossiers: dict[str, EntityDossier],
        labels: dict[str, int],
    ) -> None:
        """Train the L0 router on labelled entities."""
        X: list[dict[str, float]] = []
        y: list[int] = []
        for eid, label in labels.items():
            if eid in dossiers:
                X.append(dossiers[eid].features)
                y.append(label)
        if len(set(y)) < 2:
            console.print("[yellow]⚠ Skipping L0 training: need at least 2 classes[/]")
            return
        self._router.fit(X, y)
        console.print(f"[green]✓ L0 trained on {len(X)} samples[/]")

    @staticmethod
    def _load_ground_truth(path: str | Path) -> dict[str, int]:
        """Load ground truth labels from JSON or CSV.

        Expected formats:
        - JSON: {"entity_id": label, ...} or [{"id": ..., "label": ...}]
        - CSV: id,label
        """
        p = Path(path)
        if not p.exists():
            logger.warning("Ground truth file not found: %s", p)
            return {}

        if p.suffix == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {str(k): int(v) for k, v in raw.items()}
            if isinstance(raw, list):
                return {
                    str(item.get("id", item.get("entity_id", ""))): int(
                        item.get("label", item.get("prediction", 0))
                    )
                    for item in raw
                }
        elif p.suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(p)
            id_col = df.columns[0]
            label_col = df.columns[1]
            return dict(zip(df[id_col].astype(str), df[label_col].astype(int)))
        elif p.suffix == ".txt":
            # TXT: one flagged ID per line → those are label=1
            ids = {
                line.strip()
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            return {eid: 1 for eid in ids}

        logger.warning("Unsupported ground truth format: %s", p.suffix)
        return {}
