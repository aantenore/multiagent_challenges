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
import os
from pathlib import Path

try:
    import ulid as _ulid
except ImportError:
    _ulid = None

from langfuse_utils import (
    langfuse_client, 
    generate_session_id, 
    set_current_session_id,
    get_current_session_id
)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from domain_swarm import RoleCoordinator, SwarmFactory
from dossier_builder import DossierBuilder
from feature_engineer import SlidingWindowExtractor
from layer0_router import OneClassRouter
from manifest_manager import ManifestManager
from models import EntityDossier, ManifestEntry, PipelineResult, Stage, SwarmConsensus
from orchestrator import GlobalOrchestrator
from output_writer import write_predictions, write_audit_log, write_audit_log_jsonl
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
        self._session_id = get_current_session_id()

    # ── Main entry point ────────────────────────────────────────────────

    def run(
        self,
        manifest_path: str | Path,
        results_dir: Path | None = None,
        target_level: str | None = None,
    ) -> dict[str, list[PipelineResult]]:
        """Execute the full N-stage pipeline or a specific stage.

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

        if target_level:
            stages = [s for s in stages if s.name == target_level]
            if not stages:
                console.print(f"[bold red]Target level '{target_level}' not found in manifest.[/]")
                return {}

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

    def _run_stage(
        self,
        manager: ManifestManager,
        stage_idx: int,
        stage: Stage,
        results_dir: Path | None = None,
    ) -> list[PipelineResult]:
        """Process one stage: train on THIS level's data only, then evaluate.
        
        LA REGOLA D'ORO PER IL LAYER 0:
        Per ogni stage, l'Isolation Forest deve essere istanziato a nuovo e il suo 
        metodo .fit() deve essere chiamato ESATTAMENTE UNA VOLTA sul Training Set.
        """
        # ── 1. Reset & Setup ────────────────────────────────────────────────
        if self._rag.is_enabled:
            self._rag.reset()
            logger.info("  RAG reset for stage '%s'", stage.name)

        self._router = OneClassRouter()
        logger.info("  Layer 0 instance reset for stage '%s'", stage.name)

        # ── 2. Training Phase (LA FASE DI FIT - Single-Pass) ────────────────
        train_sources = stage.training_sources
        coordinators = []
        
        if train_sources:
            console.print(f"  Training stage: [bold]{len(train_sources)}[/] sources")
            train_builder = DossierBuilder.from_entries(train_sources, manager.base_dir)
            train_dossiers = train_builder.build_all()

            for dossier in train_dossiers.values():
                dossier.features = self._extractor.extract(dossier)

            # >>> MOMENTO CRITICO: Esegui L0.build_baselines (FIT) <<<
            # The Isolation Forest is instantiated and trained exactly ONCE per level.
            console.print("  [blue]ℹ Layer 0: Building Baselines (Single-Pass)...[/]")
            self._router.build_baselines(train_dossiers)

            # ── RAG Warm-up Phase ───────────────────────────────────────────
            # L0 predicts on its own training set to identify edge cases for RAG.
            all_entries = train_sources + (stage.evaluation_sources or [])
            coordinators = SwarmFactory.create_coordinators(
                roles={e.role for e in all_entries},
                manifest_entries=all_entries,
            )

            train_session_id = generate_session_id()
            set_current_session_id(train_session_id)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                warmup_task = progress.add_task(
                    f"  {stage.name}: RAG Warm-up Phase…", total=len(train_dossiers)
                )

                for eid, dossier in train_dossiers.items():
                    # A. Predict L0
                    l0_v, complexity, meta = self._router.to_verdict(eid, dossier)
                    
                    if l0_v is not None: 
                        # Branch A - Inlier (Healthy)
                        # ACTION: Create synthetic normality entry in RAG
                        rag_content = f"BASELINE NORMALITY: Citizen {eid} shows standard physiological patterns. Verdict: 0 (Healthy)."
                        if self._rag.is_enabled:
                            self._rag.add_case(eid, rag_content, label=0)
                    else: 
                        # Branch B - Outlier (Anomalous / Edge Case)
                        # ACTION: Full escalation with RAG context from the same warm-up cycle
                        logger.info("  [Sequential Warm-up] Inspecting Expert Case: %s", eid)
                        
                        # Note: _process_entity already performs the RAG query internally, 
                        # picking up any entries added in previous iterations of this loop.
                        result = self._process_entity(dossier, coordinators, force_escalation=True, session_id=train_session_id)
                        
                        if self._rag.is_enabled:
                            summary = self._rag.summarise_dossier(dossier)
                            reasoning = result.verdicts[-1].reasoning if result.verdicts else "No reasoning."
                            rag_content = f"[EXPERT_REASONING][Decision={result.final_prediction}] {summary}\nReasoning: {reasoning}"
                            self._rag.add_case(eid, rag_content, label=result.final_prediction)
                    
                    progress.advance(warmup_task)

            # SANITY CHECK: Predict on Train Set
            train_results: list[PipelineResult] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                sanity_task = progress.add_task(
                    f"  {stage.name}: Sanity Predict (Train Set)…", total=len(train_dossiers)
                )
                for dossier in train_dossiers.values():
                    result = self._process_entity(dossier, coordinators, session_id=train_session_id)
                    train_results.append(result)
                    progress.advance(sanity_task)

            train_out_file = f"train_predictions_{stage.name}.txt"
            train_out_path = results_dir / train_out_file if results_dir else Path(train_out_file)
            write_predictions(train_results, train_out_path)
            
            audit_out_file = f"train_audit_log_{stage.name}.json"
            audit_out_path = results_dir / audit_out_file if results_dir else Path(audit_out_file)
            write_audit_log(train_results, audit_out_path)
            
            console.print(f"  [blue]ℹ Train stage complete. RAG warmed up and L0 model established.[/]")

        # ── 3. Evaluation Phase (ASSOLUTAMENTE NO FIT) ──────────────────────
        eval_sources = stage.evaluation_sources
        if not eval_sources:
            console.print("  [yellow]⚠ No evaluation sources — skipping[/]")
            return []

        console.print(f"  Evaluation phase: [bold]{len(eval_sources)}[/] sources")
        logger.info("INFO: Layer 0 PREDICTING on Evaluation Set (Model Locked)")
        
        eval_builder = DossierBuilder.from_entries(eval_sources, manager.base_dir)
        eval_dossiers = eval_builder.build_all()

        for dossier in eval_dossiers.values():
            dossier.features = self._extractor.extract(dossier)

        if not coordinators:
            all_entries = eval_sources
            coordinators = SwarmFactory.create_coordinators(
                roles={e.role for e in all_entries},
                manifest_entries=all_entries,
            )

        eval_session_id = generate_session_id()
        set_current_session_id(eval_session_id)
        
        eval_results: list[PipelineResult] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            eval_task = progress.add_task(
                f"  {stage.name}: Predicting Evaluation Set…", total=len(eval_dossiers)
            )
            for dossier in eval_dossiers.values():
                result = self._process_entity(dossier, coordinators, session_id=eval_session_id)
                eval_results.append(result)
                progress.advance(eval_task)

        # ── 4. Write eval output & audit ───────────────────────────────────
        if stage.name == "level_1":
            stage_results_dir = Path("runs") / "level_1"
            stage_results_dir.mkdir(parents=True, exist_ok=True)
            out_file = "predictions_level_1.txt"
            audit_file = "audit_log.jsonl"
            
            out_path = stage_results_dir / out_file
            audit_path = stage_results_dir / audit_file
            
            write_predictions(eval_results, out_path)
            write_audit_log_jsonl(eval_results, audit_path)
        else:
            out_file = stage.output_file or f"predictions_{stage.name}.txt"
            out_path = results_dir / out_file if results_dir else Path(out_file)
            write_predictions(eval_results, out_path)
            
            audit_file = f"audit_log_{stage.name}.json"
            audit_path = results_dir / audit_file if results_dir else Path(audit_file)
            write_audit_log(eval_results, audit_path)

        return eval_results

    # ── Entity processing ───────────────────────────────────────────────

    def _process_entity(
        self,
        dossier: EntityDossier,
        coordinators: list[RoleCoordinator],
        force_escalation: bool = False,
        session_id: str | None = None,
    ) -> PipelineResult:
        """Process a single entity through L0 → L1 coordinators → L2."""
        eid = dossier.entity_id
        verdicts = []

        # Layer 0 — One-Class Anomaly Engine (SVM + IsolationForest)
        l0_verdict, complexity, detection_meta = self._router.to_verdict(eid, dossier)
        if l0_verdict is not None and not force_escalation:
            verdicts.append(l0_verdict)
            return PipelineResult(
                entity_id=eid,
                session_id=session_id,
                final_prediction=l0_verdict.prediction,
                layer_decided="L0_OneClassRouter",
                verdicts=verdicts,
            )
        
        # In case of force_escalation, we might still want to record L0's opinion but proceed
        if l0_verdict and force_escalation:
            verdicts.append(l0_verdict)

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
            session_id=session_id,
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
