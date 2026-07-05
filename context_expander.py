from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import settings
from retriever import RetrievalCandidate, tokenize
from vector_store import VectorStore


BROAD_SECTION_TERMS = re.compile(
    r"\b("
    r"all|entire|full|complete|summari[sz]e|summary|overview|discuss|discussed|describe|"
    r"list|what are|what is|explain|section|item|factors|risks|points"
    r")\b",
    re.I,
)
SEC_ITEM_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\s+)?Item\s+(\d+[A-Z]?)\.\s*([^\n]{2,140})",
    re.I,
)
NEXT_SEC_ITEM_RE = re.compile(r"(?:^|\n)\s*(?:\d+\s+)?Item\s+\d+[A-Z]?\.\s+[^\n]{2,140}", re.I)
TOC_MARKERS = ("table of contents", "part i item 1.", "part ii item 5.")
STOPWORDS = {
    "the",
    "and",
    "are",
    "for",
    "with",
    "what",
    "which",
    "that",
    "this",
    "from",
    "into",
    "does",
    "discuss",
    "discussed",
    "describe",
    "summary",
    "summarize",
    "overview",
    "section",
    "item",
    "full",
    "entire",
    "complete",
}


@dataclass(frozen=True)
class SectionTarget:
    source_file: str
    item_number: str
    title: str
    anchor_page: int

    @property
    def display_title(self) -> str:
        return f"Item {self.item_number}. {self.title}".strip()


def is_broad_section_query(query: str) -> bool:
    tokens = [token for token in tokenize(query) if token not in STOPWORDS]
    has_broad_term = bool(BROAD_SECTION_TERMS.search(query))
    return has_broad_term and len(tokens) <= 10


def _clean_heading_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" .;-")
    title = re.split(r"\s{2,}", title)[0].strip()
    return title[:100]


def _is_toc_like(text: str) -> bool:
    compact = " ".join(text.lower().split())
    return any(marker in compact[:1800] for marker in TOC_MARKERS)


def _query_terms(query: str) -> set[str]:
    return {token for token in tokenize(query) if token not in STOPWORDS and len(token) > 2}


def _heading_score(query: str, item_number: str, title: str, text: str) -> float:
    terms = _query_terms(query)
    heading_terms = set(tokenize(f"item {item_number} {title}"))
    overlap = len(terms & heading_terms)
    score = overlap * 2.0
    if item_number.lower() in query.lower():
        score += 2.5
    if title and title.lower() in query.lower():
        score += 3.0
    if _is_toc_like(text):
        score -= 4.0
    return score


def _iter_candidate_headings(candidates: list[RetrievalCandidate]) -> list[SectionTarget]:
    targets: list[SectionTarget] = []
    for candidate in candidates:
        metadata = candidate.metadata
        source_file = str(metadata.get("source_file") or "")
        try:
            page = int(metadata.get("page"))
        except (TypeError, ValueError):
            continue
        if not source_file:
            continue
        for match in SEC_ITEM_HEADING_RE.finditer(candidate.text):
            title = _clean_heading_title(match.group(2))
            if title:
                targets.append(
                    SectionTarget(
                        source_file=source_file,
                        item_number=match.group(1).upper(),
                        title=title,
                        anchor_page=page,
                    )
                )
    return targets


def detect_section_target(query: str, candidates: list[RetrievalCandidate]) -> SectionTarget | None:
    targets = _iter_candidate_headings(candidates)
    if not targets:
        return None

    best_target: SectionTarget | None = None
    best_score = 0.0
    for target in targets:
        source_text = next((candidate.text for candidate in candidates if candidate.metadata.get("page") == target.anchor_page), "")
        score = _heading_score(query, target.item_number, target.title, source_text)
        if score > best_score:
            best_score = score
            best_target = target
    return best_target if best_score >= 2.0 else None


def _page_chunks(vector_store: VectorStore, source_file: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for chunk in vector_store.get_all_chunks():
        metadata = chunk["metadata"]
        if metadata.get("source_file") != source_file or metadata.get("chunk_strategy") != "page":
            continue
        try:
            page = int(metadata.get("page"))
        except (TypeError, ValueError):
            continue
        pages.append({"page": page, "text": chunk["text"], "metadata": metadata, "chunk_id": chunk["chunk_id"]})
    return sorted(pages, key=lambda item: item["page"])


def _find_section_start(pages: list[dict[str, Any]], target: SectionTarget) -> tuple[int, int] | None:
    item_pattern = re.compile(
        rf"(?:^|\n)\s*(?:\d+\s+)?Item\s+{re.escape(target.item_number)}\.\s+{re.escape(target.title)}",
        re.I,
    )
    fallback_pattern = re.compile(
        rf"(?:^|\n)\s*(?:\d+\s+)?Item\s+{re.escape(target.item_number)}\.",
        re.I,
    )
    for page in pages:
        if page["page"] < target.anchor_page - 2:
            continue
        if _is_toc_like(page["text"]):
            continue
        match = item_pattern.search(page["text"]) or fallback_pattern.search(page["text"])
        if match:
            return page["page"], match.start()
    return None


def _find_section_end(
    pages: list[dict[str, Any]],
    *,
    start_page: int,
    item_number: str,
    max_pages: int,
) -> tuple[int, int | None]:
    item_seen = False
    selected_pages = [page for page in pages if start_page <= page["page"] <= start_page + max_pages - 1]
    for page in selected_pages:
        text = page["text"]
        for match in NEXT_SEC_ITEM_RE.finditer(text):
            matched_item = re.search(r"Item\s+(\d+[A-Z]?)\.", match.group(0), re.I)
            if not matched_item:
                continue
            current_item = matched_item.group(1).upper()
            if current_item == item_number.upper() and not item_seen:
                item_seen = True
                continue
            if page["page"] == start_page and match.start() == 0:
                continue
            if current_item != item_number.upper():
                return page["page"], match.start()
    if selected_pages:
        return selected_pages[-1]["page"], None
    return start_page, None


def _format_page_block(page: int, text: str) -> str:
    return f"[Page {page}]\n{text.strip()}"


def expand_section_context(
    query: str,
    candidates: list[RetrievalCandidate],
    *,
    vector_store: VectorStore,
) -> list[RetrievalCandidate]:
    if not candidates or not is_broad_section_query(query):
        return candidates

    target = detect_section_target(query, candidates)
    if target is None:
        return candidates

    pages = _page_chunks(vector_store, target.source_file)
    start = _find_section_start(pages, target)
    if start is None:
        return candidates

    start_page, start_offset = start
    end_page, end_offset = _find_section_end(
        pages,
        start_page=start_page,
        item_number=target.item_number,
        max_pages=settings.section_expansion_max_pages,
    )

    blocks: list[str] = []
    source_chunk_ids: list[str] = []
    for page in pages:
        page_number = page["page"]
        if page_number < start_page or page_number > end_page:
            continue
        text = page["text"]
        if page_number == start_page:
            text = text[start_offset:]
        if page_number == end_page and end_offset is not None:
            text = text[:end_offset]
        if text.strip():
            blocks.append(_format_page_block(page_number, text))
            source_chunk_ids.append(page["chunk_id"])

    if not blocks:
        return candidates

    text = "\n\n".join(blocks)
    if len(text) > settings.section_expansion_max_chars:
        text = text[: settings.section_expansion_max_chars].rstrip() + "\n\n[Section truncated due to context budget.]"

    expanded = RetrievalCandidate(
        chunk_id=f"expanded:{target.source_file}:{target.item_number}:{start_page}-{end_page}",
        text=text,
        metadata={
            "source_file": target.source_file,
            "page": f"{start_page}-{end_page}",
            "section_title": target.display_title,
            "chunk_strategy": "expanded_section",
            "extraction_method": "section_expansion",
            "is_table": False,
            "is_expanded_section": True,
            "source_chunk_ids": source_chunk_ids,
        },
        vector_score=max(candidate.vector_score for candidate in candidates),
        bm25_score=max(candidate.bm25_score for candidate in candidates),
        hybrid_score=max(candidate.hybrid_score for candidate in candidates),
        rerank_score=max(candidate.rerank_score or 0.0 for candidate in candidates),
    )

    source_chunk_id_set = set(source_chunk_ids)
    supporting: list[RetrievalCandidate] = []
    for candidate in candidates:
        try:
            candidate_page = int(candidate.metadata.get("page"))
        except (TypeError, ValueError):
            continue
        if candidate.metadata.get("source_file") != target.source_file:
            continue
        if candidate_page < start_page or candidate_page > end_page:
            continue
        if _is_toc_like(candidate.text) or candidate.metadata.get("chunk_strategy") == "page":
            continue
        if candidate.chunk_id in source_chunk_id_set:
            continue
        supporting.append(candidate)
    return [expanded, *supporting[: max(0, settings.top_k_final - 1)]]
