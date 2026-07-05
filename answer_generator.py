from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import requests

from config import settings
from retriever import RetrievalCandidate, retrieval_confidence


UNAVAILABLE = "The answer is not available in the provided context."


@dataclass
class Citation:
    source_file: str
    page: int | str
    chunk_id: str
    section_title: str | None
    chunk_strategy: str | None
    extraction_method: str | None
    is_ocr: bool
    is_table: bool
    table_id: str | None = None


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    confidence: float
    used_llm: str
    validation_passed: bool
    context: list[RetrievalCandidate]


def candidate_to_citation(candidate: RetrievalCandidate) -> Citation:
    metadata = candidate.metadata
    extraction_method = metadata.get("extraction_method")
    is_table = str(metadata.get("is_table")).lower() == "true" or metadata.get("is_table") is True
    return Citation(
        source_file=str(metadata.get("source_file", "")),
        page=metadata.get("page", ""),
        chunk_id=candidate.chunk_id,
        section_title=metadata.get("section_title") or None,
        chunk_strategy=metadata.get("chunk_strategy") or None,
        extraction_method=extraction_method,
        is_ocr=extraction_method == "ocr" or extraction_method == "ocr_table",
        is_table=is_table,
        table_id=metadata.get("table_id") or None,
    )


def build_context(candidates: list[RetrievalCandidate]) -> str:
    blocks: list[str] = []
    total = 0
    for idx, candidate in enumerate(candidates, start=1):
        citation = candidate_to_citation(candidate)
        header = (
            f"[Context {idx}] source_file={citation.source_file}; page={citation.page}; "
            f"chunk_id={citation.chunk_id}; section={citation.section_title or ''}; "
            f"strategy={citation.chunk_strategy or ''}; extraction={citation.extraction_method or ''}; "
            f"table_id={citation.table_id or ''}"
        )
        block = f"{header}\n{candidate.text.strip()}"
        if total + len(block) > settings.max_context_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def answer_prompt(question: str, context: str) -> str:
    return f"""You are a strict retrieval-augmented answer engine.

Rules:
- Use only the provided context.
- Do not use outside knowledge.
- Do not guess or infer beyond the context.
- If the answer is present, answer directly and preserve exact names, numbers, units, dates, and formatting.
- If the answer comes from a table, include the exact relevant row/value.
- If the answer is not available in the context, say exactly: "{UNAVAILABLE}"
- Include no citations in the answer body; citations are handled separately.

Question:
{question}

Context:
{context}

Answer:"""


def call_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.llm_temperature,
    )
    return completion.choices[0].message.content.strip()


def call_ollama(prompt: str) -> str:
    response = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": settings.llm_temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("response", "")).strip()


def extractive_fallback(question: str, candidates: list[RetrievalCandidate]) -> str:
    query_terms = {term for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-./%$]*", question.lower()) if len(term) > 2}
    best_sentence = ""
    best_score = 0
    for candidate in candidates:
        pieces = re.split(r"(?<=[.!?])\s+|\n+", candidate.text)
        for piece in pieces:
            clean = piece.strip()
            if not clean:
                continue
            terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-./%$]*", clean.lower()))
            score = len(query_terms & terms)
            if score > best_score:
                best_score = score
                best_sentence = clean
    if best_score == 0:
        return UNAVAILABLE
    return best_sentence


def validate_answer(answer: str, context: str) -> bool:
    if answer.strip() == UNAVAILABLE:
        return True
    answer_terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-./%$]*", answer)
        if len(term) > 3 or re.search(r"\d", term)
    ]
    if not answer_terms:
        return False
    context_lower = context.lower()
    matches = sum(1 for term in answer_terms if term in context_lower)
    return matches / len(answer_terms) >= 0.65


def generate_answer(question: str, candidates: list[RetrievalCandidate]) -> AnswerResult:
    confidence = retrieval_confidence(candidates)
    citations = [candidate_to_citation(candidate) for candidate in candidates]
    context = build_context(candidates)

    if confidence < settings.min_retrieval_confidence or not context.strip():
        return AnswerResult(
            answer=UNAVAILABLE,
            citations=[],
            confidence=confidence,
            used_llm="none_low_retrieval_confidence",
            validation_passed=True,
            context=candidates,
        )

    prompt = answer_prompt(question, context)
    used_llm = "extractive_fallback"
    try:
        if settings.groq_api_key:
            answer = call_groq(prompt)
            used_llm = f"groq:{settings.groq_model}"
        else:
            answer = call_ollama(prompt)
            used_llm = f"ollama:{settings.ollama_model}"
    except Exception:
        answer = extractive_fallback(question, candidates)

    validation_passed = validate_answer(answer, context)
    if not validation_passed:
        answer = UNAVAILABLE

    return AnswerResult(
        answer=answer,
        citations=citations,
        confidence=confidence,
        used_llm=used_llm,
        validation_passed=validation_passed,
        context=candidates,
    )


def answer_result_to_json(result: AnswerResult) -> str:
    payload = {
        "answer": result.answer,
        "citations": [asdict(citation) for citation in result.citations],
        "confidence": result.confidence,
        "used_llm": result.used_llm,
        "validation_passed": result.validation_passed,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
