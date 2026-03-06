"""
Pillar 2 — Modular Dynamic Memory (Continuous Learning).
Uses ChromaDB as a local vector store to memorize case details
and provide few-shot examples to Analytical Squads.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal

from settings import get_settings

logger = logging.getLogger(__name__)

try:
    import chromadb  # type: ignore[import-untyped]
    from chromadb.config import Settings as ChromaSettings

    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False
    logger.info("ChromaDB not installed — Memory Pillar will be a no-op")


class RAGStore:
    """3-Tier Hierarchical Vector Store (Project Antigravity).

    Tiers:
    1. Identity History: Individual entity's past similar data points.
    2. Contextual Neighbors: Cross-entity signals within similar groupings (e.g. Geo/Social).
    3. Global Rules: Universal patterns and architectural heuristics.
    """

    def __init__(self, namespace: str = "default") -> None:
        cfg = get_settings()
        self._enabled = _HAS_CHROMADB
        self._top_k = cfg.top_k_rag
        self._namespace = namespace
        
        self._coll_indiv = None
        self._coll_geo = None
        self._coll_global = None

        if self._enabled:
            try:
                client = chromadb.PersistentClient(
                    path=str(cfg.rag_db_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._client = client
                
                # 1. Individual Collection
                self._coll_indiv = client.get_or_create_collection(
                    name=f"{self._namespace}_{cfg.rag_individual_collection}",
                    metadata={"hnsw:space": "cosine"},
                )
                # 2. Geo-Local/Contextual Collection
                self._coll_geo = client.get_or_create_collection(
                    name=f"{self._namespace}_{cfg.rag_geo_collection}",
                    metadata={"hnsw:space": "cosine"},
                )
                # 3. Global Knowledge Collection
                self._coll_global = client.get_or_create_collection(
                    name=f"{self._namespace}_{cfg.rag_global_collection}",
                    metadata={"hnsw:space": "cosine"},
                )
                
                logger.info(
                    "Hierarchical Knowledge Store ready: Identity(%d), Context(%d), Global(%d)",
                    self._coll_indiv.count(),
                    self._coll_geo.count(),
                    self._coll_global.count(),
                )
            except Exception as exc:
                logger.warning("Failed to init Hierarchical Knowledge Store: %s — disabling memory", exc)
                self._enabled = False

    def reset(self) -> None:
        """Clear all memory collections for a fresh run."""
        if not self._enabled:
            return
        cfg = get_settings()
        collections = [
            (f"{self._namespace}_{cfg.rag_individual_collection}", "_coll_indiv"),
            (f"{self._namespace}_{cfg.rag_geo_collection}", "_coll_geo"),
            (f"{self._namespace}_{cfg.rag_global_collection}", "_coll_global")
        ]
        for full_name, attr in collections:
            try:
                self._client.delete_collection(full_name)
                setattr(self, attr, self._client.get_or_create_collection(
                    name=full_name, metadata={"hnsw:space": "cosine"}
                ))
            except Exception as exc:
                logger.warning("Memory reset failed for %s: %s", full_name, exc)
        logger.info("Knowledge Store [%s] reset", self._namespace)

    # ── Write ───────────────────────────────────────────────────────────

    def add_case(
        self,
        entity_id: str,
        dossier_summary: str,
        label: int,
        scope: Literal["individual", "geo", "global"] = "individual",
        meta_tag: str | None = None
    ) -> None:
        """Store an evaluated record in the identity, context, or global memory tier."""
        cfg = get_settings()
        if not self._enabled or not cfg.pillar_memory_enabled:
            return
            
        collection = self._coll_indiv
        if scope == "geo": collection = self._coll_geo
        elif scope == "global": collection = self._coll_global
        
        if collection is None: return

        doc_id = hashlib.sha256(
            f"{entity_id}:{label}:{meta_tag or ''}".encode()
        ).hexdigest()[:16]

        collection.upsert(
            ids=[doc_id],
            documents=[dossier_summary],
            metadatas=[
                {
                    "entity_id": entity_id,
                    "label": label,
                    "meta_tag": meta_tag or "N/A",
                    "scope": scope,
                    "summary": dossier_summary[:500],
                }
            ],
        )
        logger.info("Memory [%s]: stored record for %s", scope, entity_id)

    # ── Read ────────────────────────────────────────────────────────────

    def query_similar(
        self,
        dossier_summary: str,
        top_k: int | None = None,
        scope: Literal["all", "individual", "geo", "global"] = "all",
        meta_tag: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve most similar historical records from the requested memory tier(s)."""
        cfg = get_settings()
        if not self._enabled or not cfg.pillar_memory_enabled:
            return []

        k = top_k or self._top_k
        collections = []
        if scope == "individual" or scope == "all": collections.append(self._coll_indiv)
        if scope == "geo" or scope == "all": collections.append(self._coll_geo)
        if scope == "global" or scope == "all": collections.append(self._coll_global)

        examples: list[dict[str, Any]] = []
        for coll in collections:
            if coll is None or coll.count() == 0: continue
            try:
                results = coll.query(
                    query_texts=[dossier_summary],
                    n_results=min(k, coll.count()),
                )
                metadatas = results.get("metadatas", [[]])
                for meta in metadatas[0]:
                    examples.append(meta)
            except Exception as exc:
                logger.warning("Memory query failed in knowledge store: %s", exc)

        return examples[:k]

    # ── Helpers ─────────────────────────────────────────────────────────

    def summarise_dossier(self, dossier) -> str:
        """Create a compact string representation of an entity dossier for vectorization."""
        parts = [f"ENTITY: {dossier.entity_id}"]
        parts.append(f"Properties: {dossier.get_compact_profile()}")
        
        feats = dossier.get_filtered_features(top_n=8, threshold=0.1)
        if feats:
            feat_str = ", ".join(f"{k}={v:.1f}" for k, v in sorted(feats.items()))
            parts.append(f"Numeric Signals: {feat_str}")
        
        if dossier.context_data:
            parts.append(f"Narrative: {dossier.context_data[:150]}...")
            
        return " | ".join(parts)

    @property
    def is_enabled(self) -> bool:
        """Return whether the RAG store (ChromaDB capability) is currently enabled."""
        return self._enabled

    @property
    def count(self) -> int:
        """Return the combined total count of records across all initialized memory collections."""
        count = 0
        if self._enabled:
            if self._coll_indiv: count += self._coll_indiv.count()
            if self._coll_geo: count += self._coll_geo.count()
            if self._coll_global: count += self._coll_global.count()
        return count
