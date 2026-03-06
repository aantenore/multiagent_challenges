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
from output_writer import write_predictions, write_audit_log
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
        manager: ManifestManager,
        results_dir: Path | None = None,
        run_id: str | None = None,
        target_stage: str | None = None,
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

        manager.load()
        stages = manager.stages

        console.print(f"Stages: [bold]{len(stages)}[/] — {[s.name for s in stages]}")

        # Early validation for target_stage
        if target_stage and not any(s.name == target_stage for s in stages):
            console.print(f"[bold red]Target stage '{target_stage}' not found in manifest.[/]")
            return {}

        all_results: dict[str, list[PipelineResult]] = {}
        for idx, stage in enumerate(stages):
            if target_stage and stage.name != target_stage:
                continue
            
            console.rule(f"[bold yellow]Stage {idx + 1}/{len(stages)}: {stage.name}")
            results = self._run_stage(manager, idx, stage, results_dir=results_dir, run_id=run_id)
            all_results[stage.name] = results

        return all_results

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
        run_id: str | None = None,
    ) -> list[PipelineResult]:
        """
        Execute a full pipeline stage: 
        1. Context Reset: Clean RAG and L0 models for isolation between levels.
        2. Training/Baseline FIT: Train the Anomaly Engine on training dossiers.
        3. RAG Population: Seed memory with 'Normality' examples labeled by L0.
        4. Parallel Evaluation: Process appraisal entities concurrently across swarms.
        """
        import time
        stage_start_time = time.time()
        
        # ── 1. Reset & Setup ────────────────────────────────────────────────
        if self._rag.is_enabled:
            self._rag.reset()
            logger.info("  [RAG] Memory reset for Stage %d (%s)", stage_idx + 1, stage.name)

        self._router = OneClassRouter()
        logger.info("  [L0] Anomaly Engine (IsolationForest) reset for Stage %d", stage_idx + 1)

        # ── 2. Training Phase (FIT) ─────────────────────────────────────────
        train_sources = stage.training_sources
        coordinators = []
        
        if train_sources:
            logger.info("  [Data] Loading %d training sources for baseline creation...", len(train_sources))
            train_builder = DossierBuilder.from_entries(train_sources, manager.base_dir)
            train_dossiers = train_builder.build_all()

            for dossier in train_dossiers.values():
                dossier.features = self._extractor.extract(dossier)

            # Execution of L0 Training
            logger.info("  [L0] Training Anomaly Engine on %d entities...", len(train_dossiers))
            self._router.build_baselines(train_dossiers)

            # ── RAG Baseline Population Phase ───────────────────────────────
            # Dynamically labeled population: inliers vs outliers within the training set.
            all_entries = train_sources + (stage.evaluation_sources or [])
            roles_to_swarm = {e.role for e in all_entries if e.role in {"temporal", "spatial"}}
            coordinators = SwarmFactory.create_coordinators(
                roles=roles_to_swarm,
                manifest_entries=all_entries,
            )

            # Specific ID for RAG Population Phase
            rag_session_id = generate_session_id(prefix=f"train_{run_id or 'default'}_{stage.name}")
            set_current_session_id(rag_session_id)
            
            if self._rag.is_enabled:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    rag_task = progress.add_task(
                        f"  [RAG] Populating Stage {stage_idx + 1} Memory…", total=len(train_dossiers)
                    )

                    for eid, dossier in train_dossiers.items():
                        # L0 acts as the first filter to decide how to store the case
                        l0_v, _, meta = self._router.to_verdict(eid, dossier)
                        is_outlier = meta.is_anomalous
                        prediction = 1 if is_outlier else 0
                        
                        summary = self._rag.summarise_dossier(dossier)
                        if prediction == 0:
                            rag_content = f"[BASELINE][Inlier] {summary}\nStatus: Verified stability via L0."
                        else:
                            rag_content = f"[TRICKY_BASELINE][Outlier] {summary}\nReasoning: L0 flagged as internal anomaly. {meta.report}"
                        
                        self._rag.add_case(eid, rag_content, label=prediction)
                        progress.advance(rag_task)
            
            # ── Sanity Check (Parallel) ─────────────────────────────────────
            logger.info("  [Pipeline] Running Sanity Check on Training Set...")
            train_results: list[PipelineResult] = []
            import concurrent.futures

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                sanity_task = progress.add_task(
                    f"  [Sanity] Stage {stage_idx + 1} Self-Test…", total=len(train_dossiers)
                )
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    # Specific ID for Sanity Check Phase (predict_train)
                    sanity_session_id = generate_session_id(prefix=f"predict_train_{run_id or 'default'}_{stage.name}")
                    future_to_eid = {
                        executor.submit(self._process_entity, dossier, coordinators, session_id=sanity_session_id): eid 
                        for eid, dossier in train_dossiers.items()
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_eid):
                        try:
                            result = future.result()
                            train_results.append(result)
                            progress.advance(sanity_task)
                        except Exception as exc:
                            eid = future_to_eid[future]
                            raise RuntimeError(f"Critical Sanity test failure for {eid}: {exc}") from exc

            # Reporting for Sanity phase
            train_out_file = f"train_predictions_{stage.name}.txt"
            train_out_path = results_dir / train_out_file if results_dir else Path(train_out_file)
            write_predictions(train_results, train_out_path)
            
            audit_out_file = f"train_audit_log_{stage.name}.json"
            audit_out_path = results_dir / audit_out_file if results_dir else Path(audit_out_file)
            write_audit_log(train_results, audit_out_path)
            logger.info("  [Output] Sanity report saved to %s", train_out_path)

        # ── 3. Evaluation Phase (Parallel) ──────────────────────────────────
        eval_sources = stage.evaluation_sources
        if not eval_sources:
            logger.warning("  [Pipeline] No evaluation sources for stage '%s'", stage.name)
            return []

        logger.info("  [Data] Building evaluation dossiers for %d entities...", len(eval_sources))
        eval_builder = DossierBuilder.from_entries(eval_sources, manager.base_dir)
        eval_dossiers = eval_builder.build_all()

        for dossier in eval_dossiers.values():
            dossier.features = self._extractor.extract(dossier)

        if not coordinators:
            roles_to_swarm = {e.role for e in eval_sources if e.role in {"temporal", "spatial"}}
            coordinators = SwarmFactory.create_coordinators(
                roles=roles_to_swarm,
                manifest_entries=eval_sources,
            )

        # Specific ID for Evaluation Phase (predict_eval)
        eval_session_id = generate_session_id(prefix=f"predict_eval_{run_id or 'default'}_{stage.name}")
        set_current_session_id(eval_session_id)
        
        eval_results: list[PipelineResult] = []
        import concurrent.futures

        logger.info("  [Orchestration] Processing Evaluation Set in PARALLEL (5 workers)...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            eval_task = progress.add_task(
                f"  [Eval] Stage {stage_idx + 1} Production Appraisal…", total=len(eval_dossiers)
            )
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Map dossiers to the process function
                future_to_eid = {
                    executor.submit(self._process_entity, dossier, coordinators, session_id=eval_session_id): eid 
                    for eid, dossier in eval_dossiers.items()
                }
                
                for future in concurrent.futures.as_completed(future_to_eid):
                    try:
                        result = future.result()
                        eval_results.append(result)
                        progress.advance(eval_task)
                    except Exception as exc:
                        eid = future_to_eid[future]
                        raise RuntimeError(f"Critical error processing {eid}: {exc}") from exc

        # ── 4. Finalisation & Metrics ──────────────────────────────────────
        stage_duration = time.time() - stage_start_time
        logger.info("  [Metric] Stage '%s' completed in %.2fs (avg: %.2fs/entity)", 
                    stage.name, stage_duration, stage_duration / max(len(eval_dossiers), 1))

        out_file = stage.output_file or f"predictions_{stage.name}.txt"
        out_path = results_dir / out_file if results_dir else Path(out_file)
        write_predictions(eval_results, out_path)
        
        audit_file = f"audit_log_{stage.name}.json"
        audit_path = results_dir / audit_file if results_dir else Path(audit_file)
        write_audit_log(eval_results, audit_path)
        logger.info("  [Output] Production results saved to %s", out_path)

        return eval_results

    # ── Entity processing ───────────────────────────────────────────────

    def _process_entity(
        self,
        dossier: EntityDossier,
        coordinators: list[RoleCoordinator],
        force_escalation: bool = False,
        session_id: str | None = None,
    ) -> PipelineResult:
        """
        Main decision path for a single entity:
        1. Layer 0 (IsolationForest): Fast-path for confirmed inliers (Decision 0).
        2. Swarm RAG (Memory): Retrieve similar historical cases.
        3. Layer 1 (Coordinators): Parallel domain-specific swarms for Temporal/Spatial.
        4. Layer 2 (Orchestrator): Holistic review of swarm verdicts + Profile + Context.
        """
        import time
        if session_id:
            set_current_session_id(session_id)
        start_t = time.time()
        eid = dossier.entity_id
        verdicts = []

        # Layer 0 — Anomaly Engine
        l0_verdict, complexity, detection_meta = self._router.to_verdict(eid, dossier)
        
        # Expressive reasoning for L0 in Audit
        if not detection_meta.is_anomalous:
            l0_verdict.reasoning = f"Baseline verified. Anomaly score {detection_meta.confidence:.3f} is within normal thresholds."

        if l0_verdict is not None and not force_escalation:
            verdicts.append(l0_verdict)
            return PipelineResult(
                entity_id=eid,
                session_id=session_id,
                final_prediction=l0_verdict.prediction,
                layer_decided="Layer_0_FastPath",
                verdicts=verdicts,
                metadata={"total_time_ms": int((time.time() - start_t) * 1000)}
            )
        
        if l0_verdict and force_escalation:
            verdicts.append(l0_verdict)

        # Layer 1 — Domain Experts
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
            layer_decided="Layer_2_FullConsensus",
            verdicts=verdicts,
            metadata={
                "total_time_ms": int((time.time() - start_t) * 1000),
                "l0_diagnostics": detection_meta.report if detection_meta else "N/A"
            }
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_l0_baselines(
        self,
        dossiers: dict[str, EntityDossier],
    ) -> None:
        self._router.build_baselines(dossiers)
        console.print(f"  [green]✓ L0 baselines built on {len(dossiers)} entities[/]")
