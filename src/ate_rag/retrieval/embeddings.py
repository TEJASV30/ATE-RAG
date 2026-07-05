from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from ate_rag.config import settings


DOCUMENT_PREFIX = "Represent this document passage for retrieval: "
QUERY_PREFIX = "Represent this query for retrieving relevant document passages: "
TABLE_PREFIX = "Represent this table or table row for retrieval: "


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or settings.embedding_model
    kwargs = {}
    if settings.embedding_device:
        kwargs["device"] = settings.embedding_device
    model = SentenceTransformer(name, **kwargs)
    if settings.embedding_max_seq_length > 0:
        try:
            model.max_seq_length = settings.embedding_max_seq_length
        except Exception:
            pass
    return model


def compact_for_embedding(text: str) -> str:
    compact = " ".join(text.split())
    if settings.embedding_max_chars <= 0 or len(compact) <= settings.embedding_max_chars:
        return compact
    return compact[: settings.embedding_max_chars]


def embed_documents(texts: list[str], *, is_table: list[bool] | None = None) -> list[list[float]]:
    model = get_embedding_model()
    prefixed: list[str] = []
    table_flags = is_table or [False] * len(texts)
    for text, table_flag in zip(texts, table_flags):
        prefix = TABLE_PREFIX if table_flag else DOCUMENT_PREFIX
        prefixed.append(prefix + compact_for_embedding(text))
    embeddings = model.encode(
        prefixed,
        normalize_embeddings=settings.normalize_embeddings,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=len(prefixed) > 64,
    )
    return np.asarray(embeddings, dtype=np.float32).tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(
        QUERY_PREFIX + compact_for_embedding(query),
        normalize_embeddings=settings.normalize_embeddings,
        show_progress_bar=False,
    )
    return np.asarray(embedding, dtype=np.float32).tolist()
