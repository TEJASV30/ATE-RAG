from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from tqdm import tqdm

from chunking import Chunk
from config import settings
from embeddings import embed_documents, embed_query


class VectorStore:
    def __init__(self, path: Path | None = None, collection_name: str | None = None) -> None:
        self.path = path or settings.vector_db_path
        self.collection_name = collection_name or settings.collection_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.embedding_model,
            },
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.embedding_model,
            },
        )

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
        return sanitized

    def add_chunks(self, chunks: list[Chunk], batch_size: int | None = None) -> None:
        upsert_batch_size = batch_size or settings.vector_upsert_batch_size
        starts = range(0, len(chunks), upsert_batch_size)
        iterator = tqdm(starts, desc="Embedding chunks", leave=False) if len(chunks) > upsert_batch_size else starts
        for start in iterator:
            batch = chunks[start : start + upsert_batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = embed_documents(texts, is_table=[bool(chunk.metadata.get("is_table")) for chunk in batch])
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[self._sanitize_metadata(chunk.metadata) for chunk in batch],
            )

    def query(
        self,
        query: str,
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[embed_query(query)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, text, metadata, distance in zip(ids, docs, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                    "vector_score": score,
                }
            )
        return hits

    def get_all_chunks(self) -> list[dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"], limit=count)
        chunks: list[dict[str, Any]] = []
        for chunk_id, text, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
            chunks.append({"chunk_id": chunk_id, "text": text, "metadata": metadata or {}})
        return chunks


def reset_vector_store_files() -> None:
    if settings.vector_db_path.exists():
        shutil.rmtree(settings.vector_db_path)
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
