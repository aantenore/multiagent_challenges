"""
AdaptivePipeline — N-stage cumulative multi-agent orchestrator.

For each stage i (0..N-1):
  1. Build dossiers from CUMULATIVE training data (stages 0..i)
  2. Engineer features and train L0 (if ground truth available)
  3. Build dossiers from evaluation data (stage i only)
  4. Predict: L0 → L1 coordinators (dynamic swarm) → L2 orchestrator
  5. Write predictions to stage-specific output file
  6. Store errors in RAG for next stage
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
from layer0_router import DeterministicRouter
from manifest_manager import ManifestManager
from metrics import print_report
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
        self._router = DeterministicRouter()
        self._rag = RAGStore()
        self._extractor = SlidingWindowExtractor()
        self._orchestrator = GlobalOrchestrator()
        self._session_id = str(uuid.uuid4())

    # ── Main entry point ────────────────────────────────────────────────

    @observe(name="pipeline_run")
    def run(self, manifest_path: str | Path, results_dir: Path | None = None) -> dict[str, list[PipelineResult]]:
        """Execute the full N-stage pipeline.

        Returns a dict mapping stage_name → list[PipelineResult].
        """
        langfuse_context.update_current_trace(
            session_id=self._session_id,
            name="mirror_pipeline",
        )

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
        console.print(f"\n[bold green]Langfuse Session ID: {self._session_id}[/]\n")
        return all_results

    # ── Stage processing ────────────────────────────────────────────────

    def _run_stage(
        self,
        manager: ManifestManager,
        stage_idx: int,
        stage: Stage,
        results_dir: Path | None = None,
    ) -> list[PipelineResult]:
        """Process one stage: train cumulatively, then evaluate."""
        # ── 1. Cumulative training ─────────────────────────────────
        cum_train_sources = manager.cumulative_training_sources(stage_idx)
        if cum_train_sources:
            console.print(
                f"  Training: [bold]{len(cum_train_sources)}[/] cumulative sources "
                f"(stages 0..{stage_idx})"
            )
            train_builder = DossierBuilder.from_entries(cum_train_sources, manager.base_dir)
            train_dossiers = train_builder.build_all()

            # Engineer features
            for dossier in train_dossiers.values():
                dossier.features = self._extractor.extract(dossier)

            # Load ground truth and train L0
            labels = self._load_ground_truth(stage, manager.base_dir)
            if labels:
                self._train_l0(train_dossiers, labels)
            else:
                console.print("  [dim]No ground truth for this stage — L0 untrained[/]")

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

        # Create coordinators with semantic metadata from all sources
        all_entries = cum_train_sources + eval_sources if cum_train_sources else eval_sources
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

        # ── 5. Metrics (if ground truth) ───────────────────────────
        labels = self._load_ground_truth(stage, manager.base_dir)
        if labels:
            y_true = [labels[r.entity_id] for r in results if r.entity_id in labels]
            y_pred = [r.final_prediction for r in results if r.entity_id in labels]
            if y_true:
                console.print(f"\n  [bold cyan]{stage.name} — Evaluation Report[/]")
                print_report(y_true, y_pred)

        # ── 6. Store all cases in RAG (Balanced Learning) ──────────
        if labels and self._rag.is_enabled:
            for r in results:
                true_label = labels.get(r.entity_id)
                if true_label is not None:
                    summary = self._rag.summarise_dossier(eval_dossiers[r.entity_id])
                    self._rag.add_case(
                        r.entity_id, summary, true_label, r.final_prediction,
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

        # Layer 0
        l0_verdict, complexity = self._router.to_verdict(eid, dossier.features)
        if l0_verdict is not None:
            verdicts.append(l0_verdict)
            return PipelineResult(
                entity_id=eid,
                final_prediction=l0_verdict.prediction,
                layer_decided="L0_DeterministicRouter",
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

    def _train_l0(
        self,
        dossiers: dict[str, EntityDossier],
        labels: dict[str, int],
    ) -> None:
        X: list[dict[str, float]] = []
        y: list[int] = []
        for eid, label in labels.items():
            if eid in dossiers:
                X.append(dossiers[eid].features)
                y.append(label)
        if len(set(y)) < 2:
            console.print("  [yellow]⚠ Skipping L0 training: need ≥2 classes[/]")
            return
        self._router.fit(X, y)
        console.print(f"  [green]✓ L0 trained on {len(X)} cumulative samples[/]")

    @staticmethod
    def _load_ground_truth(stage: Stage, base_dir: Path) -> dict[str, int]:
        """Load ground truth from the stage's ground_truth path."""
        if not stage.ground_truth:
            return {}

        p = base_dir / stage.ground_truth if not Path(stage.ground_truth).is_absolute() else Path(stage.ground_truth)
        if not p.exists():
            logger.warning("Ground truth not found: %s", p)
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
            return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(int)))
        elif p.suffix == ".txt":
            ids = {
                line.strip() for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            return {eid: 1 for eid in ids}

        return {}
