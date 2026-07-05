from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import settings
from document_loaders import LoadedDocument, PageContent
from table_extractor import ExtractedTable


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*[\).]?\s+|[A-Z][\).]\s+)?[A-Z][A-Z0-9 ,:/&()'\-]{4,}$"
)


def stable_chunk_id(*parts: Any) -> str:
    raw = "::".join(str(part) for part in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_headings(text: str) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    cursor = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= 120 and (HEADING_PATTERN.match(stripped) or re.match(r"^\d+(\.\d+)+\s+\S+", stripped)):
            start = text.find(line, cursor)
            if start >= 0:
                headings.append((start, stripped))
                cursor = start + len(line)
    return headings


def split_by_section(text: str, default_title: str | None = None) -> list[tuple[str | None, str, int, int]]:
    headings = detect_headings(text)
    if not headings:
        return [(default_title, text, 0, len(text))]

    sections: list[tuple[str | None, str, int, int]] = []
    for idx, (start, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body, start, end))

    if headings[0][0] > 0:
        prefix = text[: headings[0][0]].strip()
        if prefix:
            sections.insert(0, (default_title, prefix, 0, headings[0][0]))
    return sections


def window_text(text: str, size: int, overlap: int) -> list[tuple[str, int, int]]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= size:
        return [(text, 0, len(text))]

    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + int(size * 0.55):
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, start, end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def base_metadata(page: PageContent, document_hash: str) -> dict[str, Any]:
    return {
        "source_file": page.source_file,
        "page": page.page,
        "section_title": page.section_title,
        "is_table": False,
        "extraction_method": page.extraction_method,
        "ocr_confidence": page.ocr_confidence,
        "document_hash": document_hash,
    }


def chunk_page(page: PageContent, document_hash: str) -> list[Chunk]:
    text = normalize_text(page.text)
    if not text:
        return []

    chunks: list[Chunk] = []
    metadata = base_metadata(page, document_hash)

    page_id = stable_chunk_id(page.source_file, page.page, "page")
    chunks.append(
        Chunk(
            chunk_id=page_id,
            text=text,
            metadata={
                **metadata,
                "chunk_id": page_id,
                "chunk_strategy": "page",
                "start_char": 0,
                "end_char": len(text),
            },
        )
    )

    for section_index, (section_title, section_text, section_start, _) in enumerate(
        split_by_section(text, page.section_title),
        start=1,
    ):
        section_id = stable_chunk_id(page.source_file, page.page, "section", section_index, section_title)
        chunks.append(
            Chunk(
                chunk_id=section_id,
                text=section_text,
                metadata={
                    **metadata,
                    "chunk_id": section_id,
                    "chunk_strategy": "section",
                    "section_title": section_title,
                    "start_char": section_start,
                    "end_char": section_start + len(section_text),
                },
            )
        )

        for idx, (piece, start, end) in enumerate(
            window_text(section_text, size=settings.chunk_size, overlap=settings.chunk_overlap),
            start=1,
        ):
            semantic_id = stable_chunk_id(page.source_file, page.page, "semantic", section_index, idx)
            chunks.append(
                Chunk(
                    chunk_id=semantic_id,
                    text=piece,
                    metadata={
                        **metadata,
                        "chunk_id": semantic_id,
                        "chunk_strategy": "semantic",
                        "section_title": section_title,
                        "start_char": section_start + start,
                        "end_char": section_start + end,
                    },
                )
            )

            for small_idx, (small, small_start, small_end) in enumerate(
                window_text(piece, size=settings.small_chunk_size, overlap=settings.small_chunk_overlap),
                start=1,
            ):
                small_id = stable_chunk_id(page.source_file, page.page, "small", section_index, idx, small_idx)
                chunks.append(
                    Chunk(
                        chunk_id=small_id,
                        text=small,
                        metadata={
                            **metadata,
                            "chunk_id": small_id,
                            "chunk_strategy": "small",
                            "section_title": section_title,
                            "start_char": section_start + start + small_start,
                            "end_char": section_start + start + small_end,
                        },
                    )
                )
    return chunks


def table_to_chunks(table: ExtractedTable, document_hash: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    common = {
        "source_file": table.source_file,
        "page": table.page,
        "section_title": None,
        "is_table": True,
        "table_id": table.table_id,
        "column_names": table.column_names,
        "extraction_method": table.extraction_method,
        "document_hash": document_hash,
        "ocr_confidence": None,
    }

    full_text = table.describe()
    full_id = stable_chunk_id(table.source_file, table.page, table.table_id, "full")
    chunks.append(
        Chunk(
            chunk_id=full_id,
            text=full_text,
            metadata={
                **common,
                "chunk_id": full_id,
                "chunk_strategy": "table",
                "row_id": None,
                "table_csv": table.to_csv_text(),
            },
        )
    )

    records = table.to_records()
    for row_index, row in enumerate(records, start=1):
        row_text = "; ".join(f"{column}: {value}" for column, value in row.items() if str(value).strip())
        if not row_text:
            continue
        row_id = stable_chunk_id(table.source_file, table.page, table.table_id, "row", row_index)
        chunks.append(
            Chunk(
                chunk_id=row_id,
                text=f"Table {table.table_id}, row {row_index}. {row_text}",
                metadata={
                    **common,
                    "chunk_id": row_id,
                    "chunk_strategy": "table_row",
                    "row_id": row_index,
                    "row_json": row,
                },
            )
        )

    for column in table.column_names:
        values = [str(value) for value in table.dataframe[column].fillna("").tolist() if str(value).strip()]
        if not values:
            continue
        column_id = stable_chunk_id(table.source_file, table.page, table.table_id, "column", column)
        chunks.append(
            Chunk(
                chunk_id=column_id,
                text=f"Table {table.table_id}, column {column}. Values: {', '.join(values[:80])}",
                metadata={
                    **common,
                    "chunk_id": column_id,
                    "chunk_strategy": "table_column",
                    "row_id": None,
                    "column_name": column,
                },
            )
        )
    return chunks


def chunk_document(document: LoadedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        chunks.extend(chunk_page(page, document.document_hash))
    for table in document.tables:
        chunks.extend(table_to_chunks(table, document.document_hash))

    deduped: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.text.strip():
            deduped[chunk.chunk_id] = chunk
    return list(deduped.values())
