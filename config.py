from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    document_dir: Path = Path(os.getenv("DOCUMENT_DIR", "./data/documents"))
    vector_db_path: Path = Path(os.getenv("VECTOR_DB_PATH", "./vector_store"))
    collection_name: str = os.getenv("COLLECTION_NAME", "enterprise_rag")
    ingestion_cache_path: Path = Path(
        os.getenv("INGESTION_CACHE_PATH", "./vector_store/ingestion_manifest.json")
    )
    bm25_index_path: Path = Path(os.getenv("BM25_INDEX_PATH", "./vector_store/bm25_index.pkl"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    embedding_device: str | None = os.getenv("EMBEDDING_DEVICE", "cpu") or None
    embedding_batch_size: int = _get_int("EMBEDDING_BATCH_SIZE", 4)
    embedding_max_seq_length: int = _get_int("EMBEDDING_MAX_SEQ_LENGTH", 512)
    embedding_max_chars: int = _get_int("EMBEDDING_MAX_CHARS", 2200)
    vector_upsert_batch_size: int = _get_int("VECTOR_UPSERT_BATCH_SIZE", 32)
    normalize_embeddings: bool = _get_bool("NORMALIZE_EMBEDDINGS", True)

    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_device: str | None = os.getenv("RERANKER_DEVICE", "cpu") or None

    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    llm_temperature: float = _get_float("LLM_TEMPERATURE", 0.0)
    max_context_chars: int = _get_int("MAX_CONTEXT_CHARS", 14000)

    chunk_size: int = _get_int("CHUNK_SIZE", 900)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 180)
    small_chunk_size: int = _get_int("SMALL_CHUNK_SIZE", 350)
    small_chunk_overlap: int = _get_int("SMALL_CHUNK_OVERLAP", 70)
    min_native_text_chars: int = _get_int("MIN_NATIVE_TEXT_CHARS", 80)
    ocr_dpi: int = _get_int("OCR_DPI", 220)

    top_k_vector: int = _get_int("TOP_K_VECTOR", 30)
    top_k_bm25: int = _get_int("TOP_K_BM25", 30)
    top_k_hybrid: int = _get_int("TOP_K_HYBRID", 25)
    top_k_final: int = _get_int("TOP_K_FINAL", 5)
    min_retrieval_confidence: float = _get_float("MIN_RETRIEVAL_CONFIDENCE", 0.12)

    dense_weight: float = _get_float("DENSE_WEIGHT", 0.58)
    bm25_weight: float = _get_float("BM25_WEIGHT", 0.42)

    supported_extensions: tuple[str, ...] = (
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    )

    @property
    def index_fingerprint(self) -> str:
        parts = [
            self.embedding_model,
            str(self.chunk_size),
            str(self.chunk_overlap),
            str(self.small_chunk_size),
            str(self.small_chunk_overlap),
            str(self.embedding_max_seq_length),
            str(self.embedding_max_chars),
            str(self.min_native_text_chars),
        ]
        return "|".join(parts)


settings = Settings()


def ensure_directories(paths: Iterable[Path] | None = None) -> None:
    for path in paths or [settings.document_dir, settings.vector_db_path]:
        path.mkdir(parents=True, exist_ok=True)
