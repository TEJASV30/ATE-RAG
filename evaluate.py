from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from answer_generator import generate_answer
from config import settings
from reranker import rerank
from retriever import HybridRetriever


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def answer_contains_expected(answer: str, expected: str) -> bool:
    if not expected:
        return False
    return normalize(expected) in normalize(answer)


def source_page_match(citations: list[Any], expected_source: str | None, expected_page: str | int | None) -> bool:
    if not expected_source and not expected_page:
        return False
    for citation in citations:
        source_ok = not expected_source or expected_source in citation.source_file
        page_ok = not expected_page or str(citation.page) == str(expected_page)
        if source_ok and page_ok:
            return True
    return False


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON evaluation file must contain a list of examples.")
        return data

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(dataset_path: Path, *, top_k: int | None = None) -> dict[str, Any]:
    examples = load_dataset(dataset_path)
    retriever = HybridRetriever()
    rows: list[dict[str, Any]] = []

    for example in examples:
        question = example["question"]
        expected_answer = example.get("expected_answer", "")
        expected_source = example.get("expected_source_file") or example.get("source_file")
        expected_page = example.get("expected_page") or example.get("page")

        candidates = retriever.retrieve(question)
        hit_ids = [candidate.chunk_id for candidate in candidates]
        ranked = rerank(question, candidates, top_k=top_k or settings.top_k_final)
        result = generate_answer(question, ranked)

        exact_match = answer_contains_expected(result.answer, expected_answer)
        page_match = source_page_match(result.citations, expected_source, expected_page)
        top_k_hit = any(
            (not expected_source or expected_source in str(candidate.metadata.get("source_file", "")))
            and (not expected_page or str(candidate.metadata.get("page")) == str(expected_page))
            for candidate in ranked
        )

        rows.append(
            {
                "question": question,
                "answer": result.answer,
                "expected_answer": expected_answer,
                "exact_answer_match": exact_match,
                "source_page_match": page_match,
                "top_k_source_hit": top_k_hit,
                "confidence": result.confidence,
                "retrieved_chunk_ids": hit_ids[: settings.top_k_final],
                "citations": [asdict(citation) for citation in result.citations],
            }
        )

    summary = {
        "examples": len(rows),
        "retrieval_hit_rate": mean([1.0 if row["top_k_source_hit"] else 0.0 for row in rows]) if rows else 0.0,
        "top_k_accuracy": mean([1.0 if row["top_k_source_hit"] else 0.0 for row in rows]) if rows else 0.0,
        "exact_answer_match": mean([1.0 if row["exact_answer_match"] else 0.0 for row in rows]) if rows else 0.0,
        "source_page_match": mean([1.0 if row["source_page_match"] else 0.0 for row in rows]) if rows else 0.0,
        "mean_confidence": mean([row["confidence"] for row in rows]) if rows else 0.0,
        "rows": rows,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality with expected Q&A pairs.")
    parser.add_argument("dataset", type=Path, help="CSV or JSON file with question and expected_answer columns.")
    parser.add_argument("--top-k", type=int, default=settings.top_k_final, help="Final reranked context size.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = evaluate(args.dataset, top_k=args.top_k)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
