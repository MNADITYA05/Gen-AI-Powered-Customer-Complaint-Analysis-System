"""
core/rag_engine.py
-------------------
Local RAG (Retrieval-Augmented Generation) engine for similar case retrieval.

Architecture:
  1. sentence-transformers  — embeds complaint text into dense vectors
  2. FAISS                  — fast approximate nearest-neighbour search
  3. MongoDB                — fetches full case details for returned IDs

No external API is called.  Everything runs locally.

Index is built lazily on first query and persisted to disk so restarts
are fast.  Call rebuild_index() after bulk imports to keep it fresh.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_DIR  = Path(os.getenv("MODEL_DIR", "models")) / "rag_index"
_INDEX_FILE = _INDEX_DIR / "faiss.index"
_META_FILE  = _INDEX_DIR / "meta.json"

_EMBED_MODEL = "all-MiniLM-L6-v2"   # 22 MB — fast, accurate, no GPU needed
_TOP_K       = 5


# ── Lazy imports ──────────────────────────────────────────────────────────────

def _faiss():
    try:
        import faiss
        return faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed.  Run: pip install faiss-cpu"
        ) from exc


def _sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed.  Run: pip install sentence-transformers"
        ) from exc


# ── Engine ────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Singleton-friendly retrieval engine.

    Usage:
        engine = RAGEngine()
        engine.load_or_build(db_session)         # call once at startup
        results = engine.find_similar(text, k=3) # call per request
    """

    def __init__(self):
        self._index = None          # faiss.IndexFlatIP
        self._meta:  list[dict] = []  # [{id, category, severity, emotion, status, snippet}]
        self._embedder = None
        self.is_ready  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load_or_build(self, db) -> None:
        """
        Try loading a persisted index from disk; rebuild from DB if absent.
        `db` is a pymongo.database.Database instance.
        """
        self._embedder = _sentence_transformers()(_EMBED_MODEL)

        if _INDEX_FILE.exists() and _META_FILE.exists():
            try:
                faiss = _faiss()
                self._index = faiss.read_index(str(_INDEX_FILE))
                with open(_META_FILE) as f:
                    self._meta = json.load(f)
                self.is_ready = True
                logger.info("RAG index loaded from disk (%d vectors)", self._index.ntotal)
                return
            except Exception as exc:
                logger.warning("Could not load RAG index: %s — rebuilding.", exc)

        self.rebuild_index(db)

    def rebuild_index(self, db) -> int:
        """
        Embed all complaints in the DB and rebuild the FAISS index.
        Returns the number of vectors indexed.
        `db` is a pymongo.database.Database instance.
        """
        if self._embedder is None:
            self._embedder = _sentence_transformers()(_EMBED_MODEL)

        rows = list(
            db.complaints.find(
                {"complaint_text": {"$exists": True, "$ne": None}},
                {"_id": 1, "complaint_text": 1, "category": 1,
                 "severity": 1, "emotion": 1, "status": 1},
            )
            .sort("created_at", -1)
            .limit(10_000)   # cap to keep index fast
        )

        if not rows:
            logger.warning("RAG: no complaints in DB to index.")
            self.is_ready = False
            return 0

        texts = [r["complaint_text"][:512] for r in rows]
        meta  = [
            {
                "id":       r["_id"],
                "category": r.get("category") or "",
                "severity": r.get("severity") or "",
                "emotion":  r.get("emotion")  or "",
                "status":   r.get("status")   or "open",
                "snippet":  r["complaint_text"][:200],
            }
            for r in rows
        ]

        logger.info("RAG: embedding %d complaints …", len(texts))
        embeddings = self._embedder.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        ).astype("float32")

        faiss = _faiss()
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # inner-product on normalised vecs = cosine sim
        index.add(embeddings)

        # Persist
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(_INDEX_FILE))
        with open(_META_FILE, "w") as f:
            json.dump(meta, f)

        self._index   = index
        self._meta    = meta
        self.is_ready = True
        logger.info("RAG: index built with %d vectors (dim=%d)", index.ntotal, dim)
        return index.ntotal

    def find_similar(self, text: str, k: int = _TOP_K, exclude_id: Optional[str] = None) -> list[dict]:
        """
        Return the k most semantically similar complaints to `text`.
        Each result dict contains id, category, severity, emotion, status,
        snippet, and similarity_score.
        """
        if not self.is_ready or self._index is None:
            return []

        vec = self._embedder.encode(
            [text[:512]], normalize_embeddings=True
        ).astype("float32")

        k_search = min(k + 5, self._index.ntotal)   # oversample to allow exclusions
        scores, indices = self._index.search(vec, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            m = self._meta[idx]
            if exclude_id and str(m["id"]) == str(exclude_id):
                continue
            results.append({**m, "similarity_score": round(float(score), 4)})
            if len(results) >= k:
                break

        return results

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "is_ready":    self.is_ready,
            "embed_model": _EMBED_MODEL,
            "index_size":  self._index.ntotal if self._index else 0,
            "index_path":  str(_INDEX_FILE),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
