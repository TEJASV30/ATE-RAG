from __future__ import annotations

from functools import lru_cache
import math

from sentence_transformers import CrossEncoder

from ate_rag.config import settings
from ate_rag.retrieval.retriever import RetrievalCandidate


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None = None) -> CrossEncoder:
    name = model_name or settings.reranker_model
    kwargs = {}
    if settings.reranker_device:
        kwargs["device"] = settings.reranker_device
    return CrossEncoder(name, **kwargs)


def rerank(query: str, candidates: list[RetrievalCandidate], *, top_k: int | None = None) -> list[RetrievalCandidate]:
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, candidate.text) for candidate in candidates]
    scores = model.predict(pairs)
    for candidate, score in zip(candidates, scores):
        raw_score = float(score)
        if raw_score >= 0:
            normalized = 1.0 / (1.0 + math.exp(-min(raw_score, 50.0)))
        else:
            normalized = math.exp(max(raw_score, -50.0)) / (1.0 + math.exp(max(raw_score, -50.0)))
        candidate.rerank_score = normalized

    def blended_score(candidate: RetrievalCandidate) -> float:
        rerank_score = candidate.rerank_score if candidate.rerank_score is not None else 0.0
        return 0.65 * rerank_score + 0.35 * candidate.hybrid_score

    ranked = sorted(candidates, key=lambda item: (blended_score(item), item.hybrid_score), reverse=True)
    return ranked[: top_k or settings.top_k_final]
