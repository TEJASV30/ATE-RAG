# Enterprise RAG Pipeline

This project implements an open-source-first Retrieval-Augmented Generation pipeline for enterprise PDFs, scanned PDFs, image documents, mixed text/table documents, financial reports, legal documents, contracts, manuals, policies, forms, and research papers.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install OCR system dependency:

```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr
```

Create local configuration:

```bash
cp .env.example .env
mkdir -p data/documents
```

On MacBooks, the default config uses CPU plus `sentence-transformers/all-mpnet-base-v2` for embeddings and `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking. This avoids Apple MPS out-of-memory failures and keeps local indexing practical. For a stronger but slower index, set `EMBEDDING_MODEL=BAAI/bge-m3` and `RERANKER_MODEL=BAAI/bge-reranker-base` in `.env`.

Place PDFs and image files in `data/documents`, then ingest:

```bash
python3 ingest.py --reset
```

Ask questions:

```bash
python3 rag_user_app.py
python3 rag_user_app.py "What does the contract say about termination?"
```

## How It Works

- `document_loaders.py` detects PDFs and image files, extracts native PDF text with PyMuPDF, falls back to OCR when text is weak, preserves source filename and page number, and gathers layout blocks.
- `ocr_utils.py` preprocesses images with OpenCV using grayscale conversion, denoising, thresholding, and deskewing before Tesseract OCR. OCR chunks are marked with `extraction_method="ocr"` and confidence when available.
- `table_extractor.py` extracts native PDF tables with pdfplumber, normalizes them with Pandas, and creates OCR table candidates from line-heavy scanned images. Tables are stored as full-table, row-level, and column-aware retrieval chunks.
- `chunking.py` creates page, section, semantic, small precise, OCR, table, row-level, and column-level chunks with citation metadata.
- `embeddings.py` uses SentenceTransformers with `BAAI/bge-m3` by default and query/document/table prefixes for stronger retrieval behavior.
- `vector_store.py` persists embeddings, text, and metadata in ChromaDB using cosine distance.
- `retriever.py` combines dense vector search and BM25 keyword search, then applies metadata boosts for tables, sections, exact terms, and numerical questions.
- `reranker.py` applies a CrossEncoder reranker to improve precision before answer generation.
- `answer_generator.py` uses Groq when `GROQ_API_KEY` is present, otherwise Ollama if available, and falls back to an extractive answer. It uses strict context-only prompting and validation to reduce hallucination.
- `evaluate.py` evaluates expected Q&A pairs for retrieval hit rate, top-k accuracy, answer match, source page match, and confidence.

## Debugging Retrieval

The CLI prints source file, page, chunk ID, section title, chunk strategy, extraction method, OCR/table flags, vector score, BM25 score, hybrid score, rerank score, and a context preview for every cited chunk. Use those fields to diagnose whether the issue is extraction, chunking, dense retrieval, keyword retrieval, reranking, or answer generation.

For stale or changed indexes, run:

```bash
python3 ingest.py --reset
```

The ingestion manifest includes the embedding model and chunking settings, so changing those settings automatically forces a rebuild.
