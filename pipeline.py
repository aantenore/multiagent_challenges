"""
AdaptivePipeline — N-stage functional multi-agent orchestrator (Project Antigravity).

Architecture:
1. Pillar 1: Mathematical Filter (Trigger/Anomaly detection)
2. Pillar 2: Memory/Knowledge Store (RAG)
3. Pillar 3: Analytical Squads (Specialized Swarms)
4. Pillar 4: Global Orchestrator (Final Judge / Recovery Assessment)

The system is fully domain-agnostic and configuration-driven.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path

from langfuse_utils import (
    generate_session_id, 
    set_current_session_id,
    get_current_session_id
)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from analytical_squads import AntigravitySwarmFactory
from dossier_builder import DossierBuilder
from feature_engineer import SlidingWindowExtractor
from mathematical_filter import MathematicalFilter
from manifest_manager import ManifestManager
from models import EntityDossier, PipelineResult, Stage
from orchestrator import GlobalOrchestrator
from output_writer import write_predictions, write_audit_log
from rag_store import RAGStore
from settings import get_settings

logger = logging.getLogger(__name__)
console = Console()


class AdaptivePipeline:
    """End-to-end Project Antigravity pipeline."""

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._filter = MathematicalFilter()
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
        """Execute the full N-stage pipeline."""

        console.rule("[bold cyan]Project Antigravity — Pipeline Start")
        console.print(f"Session ID: [bold blue]{self._session_id}[/]")
        console.print(f"LLM Provider: [bold]{self._cfg.llm_provider}[/]")
        
        manager.load()
        stages = manager.stages

        # Early validation
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

        console.rule("[bold cyan]Pipeline Complete")
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
        """Execute a full pipeline stage."""
        stage_start_time = time.time()
        
        # ── 1. Initialization ───────────────────────────────────────────────
        # Per-stage RAG isolation (Pillar 2)
        self._rag = RAGStore(namespace=stage.name)
        self._rag.reset()
        logger.info("  [Memory] Pillar 2 isolated and reset for Stage: %s", stage.name)

        self._filter = MathematicalFilter()
        logger.info("  [Filter] Pillar 1 reset for Stage: %s", stage.name)

        squads = AntigravitySwarmFactory.create_squads()
        
        # ── 2. Training / Memory Population ──────────────────
        train_sources = stage.training_sources
        if train_sources:
            logger.info("  [Data] Loading %d training sources...", len(train_sources))
            train_builder = DossierBuilder.from_entries(train_sources, manager.base_dir)
            train_dossiers = train_builder.build_all()

            # Seed RAG with inliers/stable cases from training set
            if self._rag.is_enabled:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    rag_task = progress.add_task(
                        f"  [Memory] Populating Pillar 2 Knowledge…", total=len(train_dossiers)
                    )
                    anomalies_found = 0
                    for eid, dossier in train_dossiers.items():
                        # Use Filter to identify stable baselines
                        _, _, meta = self._filter.analyze_incident(eid, dossier)
                        summary = self._rag.summarise_dossier(dossier)
                        
                        label = 1 if meta.is_anomalous else 0
                        status_str = "Anomaly" if label == 1 else "Stability"
                        
                        if label == 1:
                            anomalies_found += 1
                            logger.info("  [Memory] [SEEDING] Identified Anomaly in training set: %s (Reason: %s)", eid, meta.report[:100] + "...")
                        
                        content = f"[Baseline][{status_str}] {summary}\nReport: {meta.report}"
                        self._rag.add_case(eid, content, label=label, scope="global")
                        progress.advance(rag_task)
                    
                    logger.info("  [Memory] Seeding complete. %d anomalies and %d stable cases added to RAG.", anomalies_found, len(train_dossiers) - anomalies_found)
            
            # ── Sanity Self-Test ─────────────────────────────────────────────
            logger.info("  [Pipeline] Running Sanity Check on Training Set...")
            sanity_session_id = generate_session_id(prefix=f"sanity_{run_id or 'run'}_{stage.name}")
            logger.info("  [Session] Sanity Check Session ID: %s", sanity_session_id)
            train_results = self._process_parallel(train_dossiers, squads, sanity_session_id, dataset_size=len(train_dossiers))
            
            train_out_file = f"sanity_check_{stage.name}.txt"
            train_out_path = results_dir / train_out_file if results_dir else Path(train_out_file)
            write_predictions(train_results, train_out_path)
            
            audit_train_file = f"audit_log_sanity_{stage.name}.json"
            audit_train_path = results_dir / audit_train_file if results_dir else Path(audit_train_file)
            write_audit_log(train_results, audit_train_path)

        # ── 3. Evaluation / Production Phase ────────────────────────────────
        eval_sources = stage.evaluation_sources
        if not eval_sources: return []

        logger.info("  [Data] Building evaluation dossiers...")
        eval_builder = DossierBuilder.from_entries(eval_sources, manager.base_dir)
        eval_dossiers = eval_builder.build_all()

        predict_session_id = generate_session_id(prefix=f"eval_{run_id or 'run'}_{stage.name}")
        set_current_session_id(predict_session_id)
        
        logger.info("  [Orchestration] Processing Evaluation via Pillar Workflow...")
        eval_results = self._process_parallel(eval_dossiers, squads, predict_session_id, dataset_size=len(eval_dossiers))

        # ── 4. Finalization ────────────────────────────────────────────────
        stage_duration = time.time() - stage_start_time
        logger.info("  [Metric] Stage '%s' completed in %.2fs", stage.name, stage_duration)

        out_file = stage.output_file or f"predictions_{stage.name}.txt"
        out_path = results_dir / out_file if results_dir else Path(out_file)
        write_predictions(eval_results, out_path)
        
        audit_file = f"audit_log_{stage.name}.json"
        audit_path = results_dir / audit_file if results_dir else Path(audit_file)
        write_audit_log(eval_results, audit_path)
        
        return eval_results

    # ── Parallel execution engine ───────────────────────────────────────

    def _process_parallel(
        self, 
        dossiers: dict[str, EntityDossier], 
        squads: list, 
        session_id: str,
        dataset_size: int = 1
    ) -> list[PipelineResult]:
        """Process multiple entities in parallel using the core pillar workflow."""
        results: list[PipelineResult] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("  [Execution] Processing entities…", total=len(dossiers))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._cfg.pipeline_max_workers) as executor:
                future_to_eid = {
                    executor.submit(self._process_entity, dossier, squads, session_id=session_id, dataset_size=dataset_size): eid 
                    for eid, dossier in dossiers.items()
                }
                for future in concurrent.futures.as_completed(future_to_eid):
                    results.append(future.result())
                    progress.advance(task)
        return results

    # ── Core Pillar Workflow ───────────────────────────────────────────

    def _process_entity(
        self,
        dossier: EntityDossier,
        squads: list,
        session_id: str | None = None,
        dataset_size: int = 1
    ) -> PipelineResult:
        """Sequential Pillar execution for a single entity."""
        if session_id: set_current_session_id(session_id)
        eid = dossier.entity_id
        verdicts = []

        # 1. Pillar 1: Mathematical Filtering (Trigger)
        filter_verdict, complexity, meta = self._filter.analyze_incident(eid, dossier)
        verdicts.append(filter_verdict)
        
        status = "ANOMALOUS" if meta.is_anomalous else "STABLE"
        logger.info("  [Pillar 1] Entity %s classified as %s (Z=%.2f)", eid, status, meta.confidence * 5.0)

        # Early exit if stable and confident
        high_conf_stable = (not meta.is_anomalous) and (filter_verdict.confidence > self._cfg.filter_upper_threshold)
        if high_conf_stable and self._cfg.filter_skip_enabled:
            logger.info("  [Decision] %s is Securely Stable. Skipping expensive Pillar 3/4 analysis.", eid)
            return PipelineResult(
                entity_id=eid, session_id=session_id,
                final_prediction=0, component_decided="Pillar1_Filter_Skip",
                verdicts=verdicts
            )

        # 2. Pillar 2: Hierarchical Memory Retrieval
        rag_examples = []
        if self._rag.is_enabled and self._cfg.pillar_memory_enabled:
            summary = self._rag.summarise_dossier(dossier)
            # Query all scopes to get a rich context
            rag_examples = self._rag.query_similar(summary, scope="all")
            if rag_examples:
                logger.info("  [Pillar 2] Found %d similar historical cases in memory. Injecting as context.", len(rag_examples))
            else:
                logger.info("  [Pillar 2] No relevant historical precedents found for this pattern.")

        # 3. Pillar 3: Analytical Squads (Verification)
        swarm_verdicts = AntigravitySwarmFactory.run_all(
            squads, dossier, rag_examples,
            filter_report=meta.report,
            dataset_size=dataset_size
        )
        verdicts.extend(swarm_verdicts)

        # 4. Pillar 4: Global Orchestration (Judge)
        # OPTIMIZATION: Skip Orchestrator if Pillar 3 has unanimous high-confidence consensus
        skip_orchestrator = False
        if self._cfg.swarm_skip_enabled and len(swarm_verdicts) > 0:
            preds = [v.prediction for v in swarm_verdicts]
            # Must be unanimous across all squads
            if all(p == preds[0] for p in preds):
                avg_conf = sum(v.confidence for v in swarm_verdicts) / len(swarm_verdicts)
                if avg_conf >= self._cfg.swarm_skip_threshold:
                    skip_orchestrator = True
                    logger.info(
                        "  [Decision] %s: Unanimous Squad Consensus (%d) with high confidence (%.2f). Skipping Orchestrator.",
                        eid, preds[0], avg_conf
                    )

        if skip_orchestrator:
            final_prediction = swarm_verdicts[0].prediction
            component = "Pillar3_Swarm_Skip"
        else:
            final_verdict = self._orchestrator.decide(
                dossier, swarm_verdicts, rag_examples
            )
            verdicts.append(final_verdict)
            final_prediction = final_verdict.prediction
            component = "Pillar4_Orchestrator_Judge"

        return PipelineResult(
            entity_id=eid, session_id=session_id,
            final_prediction=final_prediction,
            component_decided=component,
            verdicts=verdicts
        )
