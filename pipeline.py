"""
AdaptivePipeline — N-stage per-level multi-agent orchestrator.

For each stage i (0..N-1):
  1. Build dossiers from THIS LEVEL's training data only
  2. Engineer features and train L0 IsolationForest (class 0 = well-being)
  3. Build dossiers from evaluation data (stage i only)
  4. Predict: L0 → L1 coordinators (anti-FP filter) → L2 orchestrator
  5. Write predictions to stage-specific output file
  6. RAG is reset between levels (each level is self-contained)
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

try:
    from langfuse.decorators import langfuse_context, observe
except ImportError:
    # Fallback if langfuse is not installed
    class DummyLangfuseContext:
        def update_current_trace(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            # Allow accessing any attribute without error
            return lambda *args, **kwargs: None

    langfuse_context = DummyLangfuseContext()

    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from domain_swarm import RoleCoordinator, SwarmFactory
from dossier_builder import DossierBuilder
from feature_engineer import SlidingWindowExtractor
from layer0_router import OneClassRouter
from manifest_manager import ManifestManager
from models import EntityDossier, ManifestEntry, PipelineResult, Stage, SwarmConsensus
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
        self._router = OneClassRouter()
        self._rag = RAGStore()
        self._extractor = SlidingWindowExtractor()
        self._orchestrator = GlobalOrchestrator()
        self._session_id = str(uuid.uuid4())

    # ── Main entry point ────────────────────────────────────────────────

    def run(
        self,
        manifest_path: str | Path,
        results_dir: Path | None = None,
    ) -> dict[str, list[PipelineResult]]:
        """Execute the full N-stage pipeline.

        Returns a dict mapping stage_name → list[PipelineResult].
        """

        console.rule("[bold cyan]Mirror Pipeline — Starting")
        console.print(f"Session ID: [bold green]{self._session_id}[/]")
        console.print(f"LLM Provider: [bold]{self._cfg.llm_provider}[/]")
        console.print(
            f"Swarm Config: min={self._cfg.swarm_min_agents}, "
            f"max={self._cfg.swarm_max_agents}, "
            f"threshold={self._cfg.swarm_complexity_threshold}"
        )

        manager = ManifestManager(manifest_path)
        manager.load()
        stages = manager.stages

        console.print(f"Stages: [bold]{len(stages)}[/] — {[s.name for s in stages]}")

        all_results: dict[str, list[PipelineResult]] = {}

        for idx, stage in enumerate(stages):
            console.rule(f"[bold yellow]Stage {idx + 1}/{len(stages)}: {stage.name}")
            results = self._run_stage(manager, idx, stage, results_dir=results_dir)
            all_results[stage.name] = results

        console.rule("[bold cyan]Pipeline Complete")
        console.print("\n[bold green]Pipeline finished successfully.[/]\n")
        return all_results

    # ── Stage processing ────────────────────────────────────────────────

    @observe(name="stage_run")
    def _run_stage(
        self,
        manager: ManifestManager,
        stage_idx: int,
        stage: Stage,
        results_dir: Path | None = None,
    ) -> list[PipelineResult]:
        """Process one stage: train on THIS level's data only, then evaluate."""
        stage_session_id = str(uuid.uuid4())
        langfuse_context.update_current_trace(
            session_id=stage_session_id,
            name=f"mirror_stage_{stage.name}",
        )
        
        # ── Write session ID to separate result file ────────────────
        if results_dir:
            session_file = results_dir / f"langfuse_session_{stage.name}.json"
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump({"stage": stage.name, "session_id": stage_session_id}, f, indent=2)
        # ── 0. Reset RAG for this level (each level is self-contained) ──
        if self._rag.is_enabled:
            self._rag.reset()
            logger.info("  RAG reset for stage '%s'", stage.name)

        # ── 1. Per-level training ──────────────────────────────────
        train_sources = stage.training_sources
        if train_sources:
            console.print(
                f"  Training: [bold]{len(train_sources)}[/] sources "
                f"(level: {stage.name})"
            )
            train_builder = DossierBuilder.from_entries(train_sources, manager.base_dir)
            train_dossiers = train_builder.build_all()

            # Engineer features
            for dossier in train_dossiers.values():
                dossier.features = self._extractor.extract(dossier)

            # Build L0 baselines (training = well-being baseline)
            self._router.build_baselines(train_dossiers)
            console.print(f"  [green]✓ L0 baselines built on {len(train_dossiers)} entities[/]")

            # Store training entities in RAG with pred=0 (well-being)
            if self._rag.is_enabled:
                for eid, dossier in train_dossiers.items():
                    summary = self._rag.summarise_dossier(dossier)
                    self._rag.add_case(eid, summary, 0)  # pred=0

        # ── 2. Evaluation ──────────────────────────────────────────
        eval_sources = stage.evaluation_sources
        if not eval_sources:
            console.print("  [yellow]⚠ No evaluation sources — skipping[/]")
            return []

        console.print(
            f"  Evaluating: [bold]{len(eval_sources)}[/] sources"
        )
        eval_builder = DossierBuilder.from_entries(eval_sources, manager.base_dir)
        eval_dossiers = eval_builder.build_all()

        # Engineer features
        for dossier in eval_dossiers.values():
            dossier.features = self._extractor.extract(dossier)

        # Create coordinators with semantic metadata from this level's sources
        all_entries = train_sources + eval_sources if train_sources else eval_sources
        coordinators = SwarmFactory.create_coordinators(
            roles={e.role for e in all_entries},
            manifest_entries=all_entries,
        )

        # ── 3. Predict each entity ─────────────────────────────────
        results: list[PipelineResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"  {stage.name}: predicting…", total=len(eval_dossiers)
            )
            for dossier in eval_dossiers.values():
                result = self._process_entity(dossier, coordinators)
                results.append(result)
                progress.advance(task_id)

        # ── 4. Write output ────────────────────────────────────────
        out_file = stage.output_file or f"predictions_{stage.name}.txt"
        out_path = results_dir / out_file if results_dir else Path(out_file)
        write_predictions(results, out_path)

        # ── 5. Store all cases in RAG (self-supervised) ────────────────
        if self._rag.is_enabled:
            for r in results:
                summary = self._rag.summarise_dossier(eval_dossiers[r.entity_id])
                self._rag.add_case(
                    r.entity_id, summary, r.final_prediction,
                )

        return results

    # ── Entity processing ───────────────────────────────────────────────

    def _process_entity(
        self,
        dossier: EntityDossier,
        coordinators: list[RoleCoordinator],
    ) -> PipelineResult:
        """Process a single entity through L0 → L1 coordinators → L2."""
        eid = dossier.entity_id
        verdicts = []

        # Layer 0 — One-Class Anomaly Engine (SVM + IsolationForest)
        l0_verdict, complexity, detection_meta = self._router.to_verdict(eid, dossier)
        if l0_verdict is not None:
            verdicts.append(l0_verdict)
            return PipelineResult(
                entity_id=eid,
                final_prediction=l0_verdict.prediction,
                layer_decided="L0_OneClassRouter",
                verdicts=verdicts,
            )

        # Layer 1 — Dynamic Swarm via Coordinators
        rag_examples = []
        if self._rag.is_enabled:
            summary = self._rag.summarise_dossier(dossier)
            rag_examples = self._rag.query_similar(summary)

        swarm_consensus_list = SwarmFactory.run_coordinators(
            coordinators, dossier, rag_examples,
            l0_complexity=complexity,
            detection_metadata=detection_meta,
        )
        verdicts.extend(swarm_consensus_list)



        # Layer 2 — Global Orchestrator
        final_verdict = self._orchestrator.decide(
            dossier, swarm_consensus_list, rag_examples,
        )
        verdicts.append(final_verdict)

        return PipelineResult(
            entity_id=eid,
            final_prediction=final_verdict.prediction,
            layer_decided="L2_GlobalOrchestrator",
            verdicts=verdicts,
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_l0_baselines(
        self,
        dossiers: dict[str, EntityDossier],
    ) -> None:
        self._router.build_baselines(dossiers)
        console.print(f"  [green]✓ L0 baselines built on {len(dossiers)} entities[/]")
