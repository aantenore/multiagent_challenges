"""
Layer 3 — Lightweight Local RAG (Continuous Learning).
Uses ChromaDB as a local vector store to memorise error cases
and provide few-shot examples to Layer 1 agents.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from settings import get_settings

logger = logging.getLogger(__name__)

try:
    import chromadb  # type: ignore[import-untyped]
    from chromadb.config import Settings as ChromaSettings

    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False
    logger.info("ChromaDB not installed — RAG layer will be a no-op")


class RAGStore:
    """Local vector database for continuous learning via error cases.

    Stores dossier summaries of misclassified entities so that
    subsequent runs can retrieve them as few-shot examples.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._enabled = _HAS_CHROMADB
        self._collection_name = cfg.rag_collection_name
        self._top_k = cfg.top_k_rag
        self._collection = None

        if self._enabled:
            try:
                client = chromadb.PersistentClient(
                    path=str(cfg.rag_db_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._client = client
                self._collection = client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "RAG store ready: collection=%s, existing=%d docs",
                    self._collection_name,
                    self._collection.count(),
                )
            except Exception as exc:
                logger.warning("Failed to init ChromaDB: %s — disabling RAG", exc)
                self._enabled = False

    def reset(self) -> None:
        """Clear the collection for a new level (per-level isolation)."""
        if not self._enabled or self._collection is None:
            return
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("RAG: collection reset for new level")
        except Exception as exc:
            logger.warning("RAG reset failed: %s", exc)

    # ── Write ───────────────────────────────────────────────────────────

    def add_case(
        self,
        entity_id: str,
        dossier_summary: str,
        predicted_label: int,
    ) -> None:
        """Store an evaluated case for future few-shot retrieval.

        Stores the pipeline's decision in self-supervised mode.
        """
        if not self._enabled or self._collection is None:
            return

        doc_id = hashlib.sha256(
            f"{entity_id}:{predicted_label}".encode()
        ).hexdigest()[:16]

        self._collection.upsert(
            ids=[doc_id],
            documents=[dossier_summary],
            metadatas=[
                {
                    "entity_id": entity_id,
                    "predicted_label": predicted_label,
                    "summary": dossier_summary[:500],
                }
            ],
        )
        logger.info(
            "RAG: stored case for %s (pred=%d) — collection count: %d",
            entity_id,
            predicted_label,
            self._collection.count(),
        )
        logger.debug(
            "RAG Payload for %s:\n%s", entity_id, dossier_summary
        )

    # ── Read ────────────────────────────────────────────────────────────

    def query_similar(
        self,
        dossier_summary: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most similar past error cases."""
        if not self._enabled or self._collection is None:
            return []

        k = top_k or self._top_k

        try:
            results = self._collection.query(
                query_texts=[dossier_summary],
                n_results=min(k, max(self._collection.count(), 1)),
            )
        except Exception as exc:
            logger.warning("RAG query failed: %s", exc)
            return []

        examples: list[dict[str, Any]] = []
        metadatas = results.get("metadatas", [[]])
        for meta in metadatas[0]:
            examples.append(meta)

        logger.debug(
            "RAG Query returned %d examples. Top case: %s",
            len(examples),
            examples[0].get("entity_id") if examples else "None"
        )
        return examples

    # ── Helpers ─────────────────────────────────────────────────────────

    def summarise_dossier(self, dossier) -> str:
        """Create a compact text summary of a dossier for vectorisation."""
        parts = [f"Entity: {dossier.entity_id}"]
        if dossier.profile_data:
            profile = dossier.profile_data
            parts.append(
                f"Profile: {profile.get('first_name', '')} {profile.get('last_name', '')}, "
                f"job={profile.get('job', 'N/A')}, "
                f"city={profile.get('residence_city', profile.get('residence', {}).get('city', 'N/A') if isinstance(profile.get('residence'), dict) else 'N/A')}"
            )
        if dossier.features:
            feat_str = ", ".join(
                f"{k}={v:.1f}" for k, v in sorted(dossier.features.items())
            )
            parts.append(f"Features: {feat_str}")
        if dossier.context_data:
            parts.append(f"Context: {dossier.context_data[:300]}")
        return " | ".join(parts)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def count(self) -> int:
        if self._enabled and self._collection is not None:
            return self._collection.count()
        return 0
