from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ate_rag.config import ensure_directories, settings
from ate_rag.ingestion.chunking import chunk_document
from ate_rag.ingestion.document_loaders import file_sha256, iter_document_paths, load_document
from ate_rag.retrieval.retriever import BM25Index
from ate_rag.retrieval.vector_store import VectorStore, reset_vector_store_files


def load_manifest() -> dict[str, str]:
    if not settings.ingestion_cache_path.exists():
        return {}
    with settings.ingestion_cache_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest: dict[str, str]) -> None:
    settings.ingestion_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.ingestion_cache_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def ingest_documents(*, reset: bool = False, document_dir: Path | None = None) -> int:
    ensure_directories()
    manifest = {} if reset else load_manifest()
    document_paths = iter_document_paths(document_dir)
    if not document_paths:
        print(f"No supported documents found in {document_dir or settings.document_dir}")
        return 0

    if manifest.get("__index_fingerprint__") != settings.index_fingerprint:
        reset = True
        manifest = {}

    current_manifest_values = {
        str(path.resolve()): f"{file_sha256(path)}:{settings.index_fingerprint}"
        for path in document_paths
    }
    tracked_files = {key for key in manifest if not key.startswith("__")}
    if not reset and (
        any(manifest.get(key) != value for key, value in current_manifest_values.items())
        or any(key not in current_manifest_values for key in tracked_files)
    ):
        print("Document set changed. Rebuilding indexes to avoid stale chunks.")
        reset = True
        manifest = {}

    if reset:
        reset_vector_store_files()

    vector_store = VectorStore()
    if reset:
        vector_store.reset()

    total_chunks = 0
    updated_manifest: dict[str, str] = dict(manifest)
    updated_manifest["__index_fingerprint__"] = settings.index_fingerprint

    for path in tqdm(document_paths, desc="Ingesting documents"):
        manifest_key = str(path.resolve())
        manifest_value = current_manifest_values[manifest_key]
        if not reset and manifest.get(manifest_key) == manifest_value:
            continue

        document = load_document(path)
        chunks = chunk_document(document)
        if chunks:
            vector_store.add_chunks(chunks)
            total_chunks += len(chunks)
        updated_manifest[manifest_key] = manifest_value

    bm25 = BM25Index.build_from_vector_store(vector_store)
    bm25.save()
    save_manifest(updated_manifest)
    print(f"Ingestion complete. Added or refreshed {total_chunks} chunks.")
    print(f"Vector store: {settings.vector_db_path}")
    print(f"BM25 index: {settings.bm25_index_path}")
    return total_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs and images into the enterprise RAG index.")
    parser.add_argument("--reset", action="store_true", help="Clear the vector store and rebuild indexes.")
    parser.add_argument(
        "--document-dir",
        type=Path,
        default=settings.document_dir,
        help="Directory containing PDFs and image documents.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_documents(reset=args.reset, document_dir=args.document_dir)


if __name__ == "__main__":
    main()
