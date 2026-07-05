from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    text: str
    confidence: float | None
    method: str = "ocr"


def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=gray.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines[:100]:
        coordinates = np.asarray(line).reshape(-1)
        if coordinates.size < 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in coordinates[:4]]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -20 <= angle <= 20:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(gray: np.ndarray) -> np.ndarray:
    try:
        angle = estimate_skew_angle(gray)
    except cv2.error:
        return gray
    if abs(angle) < 0.5:
        return gray

    height, width = gray.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_for_ocr(image: Image.Image, *, do_deskew: bool = True) -> Image.Image:
    cv_image = pil_to_cv(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=12)
    if do_deskew:
        gray = deskew(gray)
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return cv_to_pil(thresholded)


def _mean_confidence(data: dict[str, list[Any]]) -> float | None:
    confidences: list[float] = []
    for raw in data.get("conf", []):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confidences.append(value)
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 2)


def _text_from_data_with_lines(data: dict[str, list[Any]]) -> str:
    rows: list[tuple[int, int, int, int, str]] = []
    total = len(data.get("text", []))
    for idx in range(total):
        word = str(data["text"][idx]).strip()
        if not word:
            continue
        block = int(data.get("block_num", [0] * total)[idx])
        paragraph = int(data.get("par_num", [0] * total)[idx])
        line = int(data.get("line_num", [0] * total)[idx])
        left = int(data.get("left", [0] * total)[idx])
        rows.append((block, paragraph, line, left, word))

    if not rows:
        return ""

    grouped: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    for block, paragraph, line, left, word in rows:
        grouped.setdefault((block, paragraph, line), []).append((left, word))

    lines: list[str] = []
    for key in sorted(grouped):
        words = [word for _, word in sorted(grouped[key])]
        lines.append(" ".join(words))
    return "\n".join(lines)


def clean_ocr_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run_ocr(image: Image.Image, *, lang: str = "eng", config: str = "--psm 6") -> OCRResult:
    preprocessed = preprocess_for_ocr(image)
    data = pytesseract.image_to_data(
        preprocessed,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    text = clean_ocr_text(_text_from_data_with_lines(data))
    confidence = _mean_confidence(data)

    if not text:
        text = clean_ocr_text(pytesseract.image_to_string(preprocessed, lang=lang, config=config))
    return OCRResult(text=text, confidence=confidence)
