from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
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
        remaining = settings.max_context_chars - total
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip() + "\n\n[Context truncated due to context budget.]"
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def _is_expanded_section(candidate: RetrievalCandidate) -> bool:
    return candidate.metadata.get("chunk_strategy") == "expanded_section"


def _page_sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"\[Page\s+([^\]]+)\]\s*\n", text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        page = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((page, text[start:end]))
    return sections


def _paragraph_blocks(text: str) -> list[str]:
    blocks = []
    for block in re.split(r"\n\s*\n+", text):
        clean = re.sub(r"\s*\n\s*", " ", block)
        clean = re.sub(r"\s+", " ", clean).strip()
        clean = re.sub(r"^\d{1,3}\s+(?=[A-Z])", "", clean)
        if clean:
            blocks.append(clean)
    return blocks


def _looks_like_category(block: str) -> bool:
    if len(block) > 120 or block.endswith("."):
        return False
    return bool(re.search(r"\b(risks?|tax|business|technology|operations|legal|financial|industry)\b", block, re.I))


def _looks_like_risk_heading(block: str) -> bool:
    if not block.endswith(".") or len(block) < 45 or len(block) > 520:
        return False
    if not block[0].isupper():
        return False
    lowered = block.lower()
    risk_cues = (
        "may",
        "could",
        "if ",
        "depend",
        "subject to",
        "unable",
        "fail",
        "failure",
        "risk",
        "adversely",
        "negatively",
        "limited",
        "rely",
        "exposed",
        "fluctuat",
        "uncertain",
    )
    narration_starts = (
        "as part of",
        "we have ",
        "we are ",
        "we conduct ",
        "we cannot ",
        "in addition",
        "for example",
        "our process",
        "publicly-traded companies",
        "these ",
        "the ",
    )
    return any(cue in lowered for cue in risk_cues) and not lowered.startswith(narration_starts)


def build_expanded_section_outline(candidate: RetrievalCandidate) -> str:
    section_title = candidate.metadata.get("section_title") or "Expanded Section"
    source_file = candidate.metadata.get("source_file") or ""
    page_range = candidate.metadata.get("page") or ""
    lines = [
        str(section_title),
        f"Source: {source_file}; pages: {page_range}",
        "",
    ]

    current_category = ""
    intro_added = False
    item_count = 0

    for page, page_text in _page_sections(candidate.text):
        blocks = _paragraph_blocks(page_text)
        idx = 0
        while idx < len(blocks):
            block = blocks[idx]
            if not intro_added and "Item " in block and "Risk Factors" in block:
                intro = blocks[idx + 1] if idx + 1 < len(blocks) else ""
                if intro:
                    intro_text = intro[:700].rstrip()
                    if len(intro) > 700:
                        intro_text += "..."
                    lines.extend(["Section introduction:", f"- {intro_text}", ""])
                    intro_added = True
                idx += 1
                continue

            if _looks_like_category(block):
                current_category = block
                lines.extend([f"Category: {current_category}", ""])
                idx += 1
                continue

            if _looks_like_risk_heading(block):
                description_parts: list[str] = []
                lookahead = idx + 1
                while lookahead < len(blocks):
                    next_block = blocks[lookahead]
                    if _looks_like_category(next_block) or _looks_like_risk_heading(next_block):
                        break
                    description_parts.append(next_block)
                    if sum(len(part) for part in description_parts) >= settings.section_outline_description_chars:
                        break
                    lookahead += 1

                description = " ".join(description_parts).strip()
                if len(description) > settings.section_outline_description_chars:
                    description = description[: settings.section_outline_description_chars].rstrip() + "..."

                page_label = f"Page {page}" if page else "Page unavailable"
                lines.append(f"- {block} ({page_label})")
                if description:
                    lines.append(f"  Description: {description}")
                item_count += 1
                idx += 1
                continue
            idx += 1

    if item_count == 0:
        return candidate.text

    return "\n".join(lines).strip()


def prepare_candidates_for_generation(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    context = build_context(candidates)
    if len(context) <= settings.llm_context_target_chars:
        return candidates

    prepared: list[RetrievalCandidate] = []
    for candidate in candidates:
        if _is_expanded_section(candidate):
            outline = build_expanded_section_outline(candidate)
            metadata = {
                **candidate.metadata,
                "chunk_strategy": "expanded_section_outline",
                "source_chunk_strategy": candidate.metadata.get("chunk_strategy"),
            }
            prepared.append(replace(candidate, text=outline, metadata=metadata))
        else:
            prepared.append(candidate)
    return prepared


def direct_section_outline_answer(candidates: list[RetrievalCandidate]) -> str | None:
    if not settings.direct_section_outline_answers:
        return None
    for candidate in candidates:
        if candidate.metadata.get("chunk_strategy") == "expanded_section_outline":
            return candidate.text
    return None


def answer_prompt(question: str, context: str) -> str:
    return f"""You are a strict retrieval-augmented answer engine.

Rules:
- Use only the provided context.
- Do not use outside knowledge.
- Do not guess or infer beyond the context.
- If the answer is present, answer directly and preserve exact names, numbers, units, dates, and formatting.
- If the context contains an expanded section or expanded section outline, use it as the primary source for broad section-level questions.
- For broad section-level questions, preserve the document's structure: return each relevant heading or risk factor followed by its supporting description from the context.
- For risk-factor questions, every risk factor you return must include both the risk-factor heading and a short "Description:" line grounded in the context.
- Do not output heading-only lists when the context includes descriptions.
- Do not replace a full section answer with a short generic summary when the context provides the detailed section text.
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
    for candidate in candidates:
        if candidate.metadata.get("chunk_strategy") in {"expanded_section", "expanded_section_outline"}:
            outline = (
                candidate.text
                if candidate.metadata.get("chunk_strategy") == "expanded_section_outline"
                else build_expanded_section_outline(candidate)
            )
            return outline

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
    candidates = prepare_candidates_for_generation(candidates)
    confidence = retrieval_confidence(candidates)
    citations = [candidate_to_citation(candidate) for candidate in candidates]
    context = build_context(candidates)

    direct_answer = direct_section_outline_answer(candidates)
    if direct_answer is not None:
        return AnswerResult(
            answer=direct_answer,
            citations=citations,
            confidence=confidence,
            used_llm="deterministic_section_outline",
            validation_passed=True,
            context=candidates,
        )

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
