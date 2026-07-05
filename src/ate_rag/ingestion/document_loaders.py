from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from ate_rag.config import settings
from ate_rag.ingestion.ocr_utils import OCRResult, run_ocr
from ate_rag.ingestion.table_extractor import ExtractedTable, extract_ocr_table_candidates, extract_pdf_tables


@dataclass
class PageContent:
    source_file: str
    page: int
    text: str
    extraction_method: str
    ocr_confidence: float | None = None
    layout_blocks: list[dict[str, Any]] = field(default_factory=list)
    section_title: str | None = None


@dataclass
class LoadedDocument:
    source_path: Path
    source_file: str
    pages: list[PageContent]
    tables: list[ExtractedTable]
    document_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _block_to_dict(block: tuple[Any, ...]) -> dict[str, Any]:
    x0, y0, x1, y1, text, block_no, block_type = block[:7]
    return {
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "text": clean_extracted_text(str(text)),
        "block_no": int(block_no),
        "block_type": int(block_type),
    }


def _render_page_to_image(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


def _is_native_text_weak(text: str) -> bool:
    stripped = re.sub(r"\s+", "", text)
    if len(stripped) < settings.min_native_text_chars:
        return True
    suspicious = len(re.findall(r"[^\w\s.,;:!?()$%&/\-]", text))
    return suspicious / max(len(text), 1) > 0.12


def _extract_section_title(text: str) -> str | None:
    for line in text.splitlines()[:12]:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) <= 120 and (
            stripped.isupper()
            or re.match(r"^(\d+(\.\d+)*|[A-Z])[\).]?\s+[A-Z][\w ,:/&()'-]+$", stripped)
        ):
            return stripped
    return None


def load_pdf(path: Path) -> LoadedDocument:
    pages: list[PageContent] = []
    document = fitz.open(str(path))
    try:
        for page_index, page in enumerate(document, start=1):
            native_text = clean_extracted_text(page.get_text("text") or "")
            blocks = [_block_to_dict(block) for block in page.get_text("blocks") or [] if len(block) >= 7]
            extraction_method = "native_pdf"
            ocr_confidence: float | None = None
            text = native_text

            if _is_native_text_weak(native_text):
                image = _render_page_to_image(page, settings.ocr_dpi)
                ocr_result: OCRResult = run_ocr(image)
                if ocr_result.text:
                    text = ocr_result.text
                    extraction_method = "ocr"
                    ocr_confidence = ocr_result.confidence

            pages.append(
                PageContent(
                    source_file=path.name,
                    page=page_index,
                    text=text,
                    extraction_method=extraction_method,
                    ocr_confidence=ocr_confidence,
                    layout_blocks=blocks,
                    section_title=_extract_section_title(text),
                )
            )
    finally:
        document.close()

    tables = extract_pdf_tables(path)

    # Add a coarse OCR table candidate when a scanned page appears table-like.
    if any(page.extraction_method == "ocr" for page in pages):
        doc = fitz.open(str(path))
        try:
            for page_content, fitz_page in zip(pages, doc):
                if page_content.extraction_method != "ocr":
                    continue
                image = _render_page_to_image(fitz_page, settings.ocr_dpi)
                tables.extend(
                    extract_ocr_table_candidates(
                        image,
                        source_file=path.name,
                        page=page_content.page,
                        ocr_text=page_content.text,
                    )
                )
        finally:
            doc.close()

    return LoadedDocument(
        source_path=path,
        source_file=path.name,
        pages=pages,
        tables=tables,
        document_hash=file_sha256(path),
        metadata={"document_type": "pdf", "page_count": len(pages)},
    )


def load_image(path: Path) -> LoadedDocument:
    image = Image.open(path)
    ocr_result = run_ocr(image)
    page = PageContent(
        source_file=path.name,
        page=1,
        text=ocr_result.text,
        extraction_method="ocr",
        ocr_confidence=ocr_result.confidence,
        section_title=_extract_section_title(ocr_result.text),
    )
    tables = extract_ocr_table_candidates(
        image,
        source_file=path.name,
        page=1,
        ocr_text=ocr_result.text,
    )
    return LoadedDocument(
        source_path=path,
        source_file=path.name,
        pages=[page],
        tables=tables,
        document_hash=file_sha256(path),
        metadata={"document_type": "image", "page_count": 1},
    )


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        return load_image(path)
    raise ValueError(f"Unsupported document type: {path}")


def iter_document_paths(document_dir: Path | None = None) -> list[Path]:
    root = document_dir or settings.document_dir
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in settings.supported_extensions
    )
