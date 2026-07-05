from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pdfplumber
from PIL import Image

from ate_rag.ingestion.ocr_utils import preprocess_for_ocr


@dataclass
class ExtractedTable:
    source_file: str
    page: int
    table_id: str
    extraction_method: str
    dataframe: pd.DataFrame
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def column_names(self) -> list[str]:
        return [str(col) for col in self.dataframe.columns]

    def to_records(self) -> list[dict[str, Any]]:
        return self.dataframe.fillna("").astype(str).to_dict(orient="records")

    def to_markdown(self, max_rows: int = 30) -> str:
        df = self.dataframe.fillna("").astype(str)
        if len(df) > max_rows:
            df = df.head(max_rows)
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_csv(index=False)

    def to_csv_text(self) -> str:
        buffer = io.StringIO()
        self.dataframe.fillna("").astype(str).to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL)
        return buffer.getvalue().strip()

    def describe(self) -> str:
        rows, cols = self.dataframe.shape
        columns = ", ".join(self.column_names)
        sample = self.to_markdown(max_rows=8)
        return (
            f"Table {self.table_id} on page {self.page} of {self.source_file}. "
            f"It has {rows} rows and {cols} columns. Columns: {columns}.\n{sample}"
        )


def _normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_table_rows(rows: list[list[Any]]) -> pd.DataFrame | None:
    cleaned = [[_normalize_cell(cell) for cell in row] for row in rows if row and any(_normalize_cell(c) for c in row)]
    if not cleaned:
        return None

    width = max(len(row) for row in cleaned)
    normalized = [row + [""] * (width - len(row)) for row in cleaned]
    header = normalized[0]
    header_is_valid = len([cell for cell in header if cell]) >= max(1, width // 2)

    if header_is_valid:
        columns = []
        seen: dict[str, int] = {}
        for idx, col in enumerate(header):
            name = col or f"column_{idx + 1}"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
            columns.append(name)
        data = normalized[1:]
    else:
        columns = [f"column_{idx + 1}" for idx in range(width)]
        data = normalized

    if not data:
        return None
    return pd.DataFrame(data, columns=columns)


def extract_pdf_tables(pdf_path: Path) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                found = page.find_tables()
            except Exception:
                found = []
            for table_index, table in enumerate(found, start=1):
                try:
                    rows = table.extract()
                except Exception:
                    continue
                dataframe = normalize_table_rows(rows)
                if dataframe is None or dataframe.empty:
                    continue
                table_id = f"{pdf_path.stem}_p{page_index}_t{table_index}"
                tables.append(
                    ExtractedTable(
                        source_file=pdf_path.name,
                        page=page_index,
                        table_id=table_id,
                        extraction_method="pdfplumber",
                        dataframe=dataframe,
                        bbox=tuple(table.bbox) if table.bbox else None,
                    )
                )
    return tables


def _line_count(binary: np.ndarray, horizontal: bool) -> int:
    height, width = binary.shape
    if horizontal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 30), 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 30)))
    detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def likely_contains_table(image: Image.Image) -> bool:
    processed = preprocess_for_ocr(image, do_deskew=False)
    gray = np.array(processed.convert("L"))
    binary = cv2.bitwise_not(gray)
    horizontal = _line_count(binary, horizontal=True)
    vertical = _line_count(binary, horizontal=False)
    return horizontal >= 2 and vertical >= 2


def extract_ocr_table_candidates(
    image: Image.Image,
    *,
    source_file: str,
    page: int,
    ocr_text: str,
) -> list[ExtractedTable]:
    if not ocr_text.strip() or not likely_contains_table(image):
        return []

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    candidate_rows: list[list[str]] = []
    for line in lines:
        if "\t" in line:
            cells = [cell.strip() for cell in line.split("\t")]
        elif re.search(r"\s{2,}", line):
            cells = [cell.strip() for cell in re.split(r"\s{2,}", line)]
        elif "|" in line:
            cells = [cell.strip() for cell in line.split("|")]
        else:
            continue
        if len([cell for cell in cells if cell]) >= 2:
            candidate_rows.append(cells)

    dataframe = normalize_table_rows(candidate_rows)
    if dataframe is None or dataframe.empty:
        return []

    table_id = f"{Path(source_file).stem}_p{page}_ocr_t1"
    return [
        ExtractedTable(
            source_file=source_file,
            page=page,
            table_id=table_id,
            extraction_method="ocr_table",
            dataframe=dataframe,
            metadata={"table_detection": "opencv_lines_plus_ocr_text"},
        )
    ]
