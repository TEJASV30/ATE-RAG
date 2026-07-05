from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ate_rag.config import settings
from ate_rag.retrieval.vector_store import VectorStore


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./%$]*")
NUMERIC_QUERY_PATTERN = re.compile(r"\b(total|amount|price|cost|revenue|profit|loss|rate|percent|percentage|date|value|balance|table|row|column|number|sum)\b", re.I)


@dataclass
class RetrievalCandidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self.index = BM25Okapi(self.tokens) if self.tokens else None

    @classmethod
    def build_from_vector_store(cls, vector_store: VectorStore) -> "BM25Index":
        return cls(vector_store.get_all_chunks())

    def save(self, path: Path | None = None) -> None:
        target = path or settings.bm25_index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Index":
        target = path or settings.bm25_index_path
        with target.open("rb") as handle:
            return pickle.load(handle)

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if self.index is None or not self.chunks:
            return []
        query_tokens = tokenize(query)
        raw_scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(range(len(raw_scores)), key=lambda idx: raw_scores[idx], reverse=True)[:top_k]
        max_score = max([raw_scores[idx] for idx in ranked_indices] or [0.0])
        hits: list[dict[str, Any]] = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            raw = float(raw_scores[idx])
            normalized = raw / max_score if max_score > 0 else 0.0
            hits.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "bm25_score": normalized,
                    "bm25_raw_score": raw,
                }
            )
        return hits


def exact_phrase_boost(query: str, text: str) -> float:
    query = query.strip().lower()
    text_lower = text.lower()
    boost = 0.0
    quoted = re.findall(r'"([^"]+)"', query)
    for phrase in quoted:
        if phrase.lower() in text_lower:
            boost += 0.08

    terms = [token for token in tokenize(query) if len(token) >= 4]
    if terms:
        matches = sum(1 for term in terms if term in text_lower)
        boost += min(0.12, matches / len(terms) * 0.12)
    return boost


def metadata_boost(query: str, candidate: RetrievalCandidate) -> float:
    metadata = candidate.metadata
    text = candidate.text
    boost = exact_phrase_boost(query, text)

    is_table = str(metadata.get("is_table")).lower() == "true" or metadata.get("is_table") is True
    chunk_strategy = str(metadata.get("chunk_strategy", ""))
    section_title = str(metadata.get("section_title") or "").lower()

    if NUMERIC_QUERY_PATTERN.search(query) and is_table:
        boost += 0.16
    if chunk_strategy in {"table_row", "small"}:
        boost += 0.04
    if section_title and any(term in section_title for term in tokenize(query)):
        boost += 0.08
    if metadata.get("extraction_method") == "ocr" and candidate.vector_score < 0.2 and candidate.bm25_score < 0.2:
        boost -= 0.03
    if chunk_strategy == "page" and len(text) > 2500 and "page" not in query.lower():
        boost -= 0.05
    return boost


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        if bm25_index is not None:
            self.bm25_index = bm25_index
        elif settings.bm25_index_path.exists():
            self.bm25_index = BM25Index.load()
        else:
            self.bm25_index = BM25Index.build_from_vector_store(self.vector_store)

    @staticmethod
    def _merge(vector_hits: list[dict[str, Any]], bm25_hits: list[dict[str, Any]]) -> dict[str, RetrievalCandidate]:
        merged: dict[str, RetrievalCandidate] = {}
        for hit in vector_hits:
            candidate = merged.setdefault(
                hit["chunk_id"],
                RetrievalCandidate(
                    chunk_id=hit["chunk_id"],
                    text=hit["text"],
                    metadata=hit.get("metadata") or {},
                ),
            )
            candidate.vector_score = max(candidate.vector_score, float(hit.get("vector_score", 0.0)))

        for hit in bm25_hits:
            candidate = merged.setdefault(
                hit["chunk_id"],
                RetrievalCandidate(
                    chunk_id=hit["chunk_id"],
                    text=hit["text"],
                    metadata=hit.get("metadata") or {},
                ),
            )
            candidate.bm25_score = max(candidate.bm25_score, float(hit.get("bm25_score", 0.0)))
        return merged

    def retrieve(
        self,
        query: str,
        *,
        top_k_vector: int | None = None,
        top_k_bm25: int | None = None,
        top_k_hybrid: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalCandidate]:
        vector_hits = self.vector_store.query(query, top_k=top_k_vector or settings.top_k_vector, where=where)
        bm25_hits = self.bm25_index.search(query, top_k=top_k_bm25 or settings.top_k_bm25)
        merged = self._merge(vector_hits, bm25_hits)

        for candidate in merged.values():
            candidate.hybrid_score = (
                settings.dense_weight * candidate.vector_score
                + settings.bm25_weight * candidate.bm25_score
                + metadata_boost(query, candidate)
            )
            candidate.hybrid_score = max(0.0, min(1.0, candidate.hybrid_score))

        ranked = sorted(
            merged.values(),
            key=lambda item: (item.hybrid_score, item.vector_score, item.bm25_score),
            reverse=True,
        )
        return ranked[: top_k_hybrid or settings.top_k_hybrid]


def retrieval_confidence(candidates: list[RetrievalCandidate]) -> float:
    if not candidates:
        return 0.0
    best = candidates[0]
    rerank = best.rerank_score if best.rerank_score is not None else 0.0
    return max(best.hybrid_score, rerank, math.sqrt(max(best.vector_score, 0.0) * max(best.bm25_score, 0.0)))
