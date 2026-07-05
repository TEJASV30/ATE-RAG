from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

from answer_generator import answer_result_to_json, generate_answer
from config import settings
from context_expander import expand_section_context
from reranker import rerank
from retriever import HybridRetriever, RetrievalCandidate


def context_preview(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def print_candidate(candidate: RetrievalCandidate, idx: int) -> None:
    metadata = candidate.metadata
    print(f"\nSource {idx}")
    print(f"  Source file: {metadata.get('source_file')}")
    print(f"  Page number: {metadata.get('page')}")
    print(f"  Chunk ID: {candidate.chunk_id}")
    print(f"  Section title: {metadata.get('section_title') or ''}")
    print(f"  Chunk strategy: {metadata.get('chunk_strategy')}")
    print(f"  Extraction method: {metadata.get('extraction_method')}")
    print(f"  OCR source: {metadata.get('extraction_method') in {'ocr', 'ocr_table'}}")
    print(f"  Table source: {metadata.get('is_table')}")
    print(f"  Table ID: {metadata.get('table_id') or ''}")
    print(f"  Vector score: {candidate.vector_score:.4f}")
    print(f"  BM25 score: {candidate.bm25_score:.4f}")
    print(f"  Hybrid score: {candidate.hybrid_score:.4f}")
    print(f"  Rerank score: {candidate.rerank_score if candidate.rerank_score is not None else ''}")
    print(f"  Context preview: {context_preview(candidate.text)}")


def answer_question(question: str, *, json_output: bool = False) -> None:
    retriever = HybridRetriever()
    candidates = retriever.retrieve(
        question,
        top_k_vector=settings.top_k_vector,
        top_k_bm25=settings.top_k_bm25,
        top_k_hybrid=settings.top_k_hybrid,
    )
    ranked = rerank(question, candidates, top_k=settings.top_k_final)
    expanded = expand_section_context(question, ranked, vector_store=retriever.vector_store)
    result = generate_answer(question, expanded)

    if json_output:
        print(answer_result_to_json(result))
        return

    print("\nAnswer")
    print(result.answer)
    print(f"\nConfidence: {result.confidence:.4f}")
    print(f"LLM: {result.used_llm}")
    print(f"Validation passed: {result.validation_passed}")
    for idx, candidate in enumerate(result.context, start=1):
        print_candidate(candidate, idx)


def interactive_loop() -> None:
    print("Enterprise RAG CLI. Ask a question, or type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        answer_question(question)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against the ingested document index.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable answer JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    question = " ".join(args.question).strip()
    if question:
        answer_question(question, json_output=args.json)
    else:
        interactive_loop()
