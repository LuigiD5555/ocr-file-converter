"""Contour-based text block detection and OCR for multi-block images."""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.logging import get_logger

log = get_logger(__name__)

try:
    import pytesseract
except Exception:
    pytesseract = None


class BoundingBox:
    """Simple bounding box container."""

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.x2 = x + width
        self.y2 = y + height

    def overlaps(self, other: "BoundingBox", min_overlap: float = 0.1) -> bool:
        """Check if this box overlaps with another."""
        horizontal = not (self.x2 < other.x or self.x > other.x2)
        vertical = not (self.y2 < other.y or self.y > other.y2)
        return horizontal and vertical

    def merge(self, other: "BoundingBox") -> "BoundingBox":
        """Merge with another box."""
        new_x = min(self.x, other.x)
        new_y = min(self.y, other.y)
        new_x2 = max(self.x2, other.x2)
        new_y2 = max(self.y2, other.y2)
        return BoundingBox(new_x, new_y, new_x2 - new_x, new_y2 - new_y)

    def area(self) -> int:
        """Calculate bounding box area."""
        return self.width * self.height


def extract_text_by_blocks(image_path: str) -> str:
    """
    Extracts structured text from an image by detecting visual text blocks,
    performing OCR per block, and reconstructing reading order.

    This function uses contour detection to identify separate text regions
    (e.g., in infographics, multi-column layouts, or UI screenshots). If
    fewer than 3 blocks are detected, returns empty string to signal the
    caller to use alternative OCR strategies. The caller (normalizer) will
    then fall back to the existing ImageLayoutReconstructor approach.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Structured text output with block separation via "\\n\\n",
             or empty string if block detection fails or fewer than 3 blocks found.
    """
    if pytesseract is None:
        log.warning("pytesseract not available, cannot extract text by blocks")
        return ""

    try:
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            log.error("Image not found: %s", image_path)
            return ""

        image = cv2.imread(image_path)
        if image is None:
            log.error("Failed to load image: %s", image_path)
            return ""

        height, width = image.shape[:2]
        log.debug("Image loaded: %dx%d", width, height)

        blocks = _detect_text_blocks(image)
        if not blocks:
            log.debug("No text blocks detected in %s", image_path)
            return ""

        if len(blocks) < 3:
            log.debug("Only %d blocks detected (threshold: 3), may be too sparse", len(blocks))

        block_texts = []
        for idx, block in enumerate(blocks):
            block_image = image[block.y : block.y2, block.x : block.x2]
            block_text = _ocr_block(block_image)

            if block_text:
                block_text = _clean_block_text(block_text)
                if len(block_text) >= 5:
                    block_texts.append(block_text)
                    log.debug("Block %d: %d chars", idx, len(block_text))

        if not block_texts:
            log.debug("No valid blocks with text >= 5 chars")
            return ""

        result = "\n\n".join(block_texts)
        log.info("Extracted %d blocks from %s", len(block_texts), image_path)
        return result

    except Exception as exc:
        log.error("Error extracting text by blocks from %s: %s", image_path, exc)
        return ""


def _detect_text_blocks(image) -> List[BoundingBox]:
    """Detect text blocks using horizontal/vertical projection analysis."""
    height, width = image.shape[:2]

    scale_factor = 1
    if width < 1200:
        scale_factor = 2
        image = cv2.resize(image, (width * scale_factor, height * scale_factor))
        height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    horizontal_projection = np.sum(thresh, axis=1)
    vertical_projection = np.sum(thresh, axis=0)

    horizontal_threshold = np.mean(horizontal_projection) * 0.05
    vertical_threshold = np.mean(vertical_projection) * 0.05

    h_breaks = _find_breaks(horizontal_projection, horizontal_threshold)
    v_breaks = _find_breaks(vertical_projection, vertical_threshold)

    log.debug("Horizontal breaks: %d, Vertical breaks: %d", len(h_breaks), len(v_breaks))

    boxes = []
    for h_start, h_end in h_breaks:
        for v_start, v_end in v_breaks:
            region = thresh[h_start:h_end, v_start:v_end]
            if np.sum(region) > 200:
                if scale_factor > 1:
                    h_start = h_start // scale_factor
                    h_end = h_end // scale_factor
                    v_start = v_start // scale_factor
                    v_end = v_end // scale_factor

                box = BoundingBox(v_start, h_start, v_end - v_start, h_end - h_start)
                boxes.append(box)

    log.debug("Found %d blocks via projection", len(boxes))

    merged_boxes = _merge_nearby_boxes(boxes)
    log.debug("Merged to %d boxes", len(merged_boxes))

    if len(merged_boxes) < 2:
        log.debug("Too few blocks found, falling back to original image")
        return []

    sorted_boxes = _sort_reading_order(merged_boxes)
    return sorted_boxes


def _find_breaks(projection, threshold: float) -> List[tuple]:
    """Find continuous regions (breaks between white space) in projection."""
    in_region = False
    start = 0
    regions = []

    for idx, value in enumerate(projection):
        if value > threshold:
            if not in_region:
                start = idx
                in_region = True
        else:
            if in_region:
                regions.append((start, idx))
                in_region = False

    if in_region:
        regions.append((start, len(projection)))

    return regions


def _merge_nearby_boxes(boxes: List[BoundingBox]) -> List[BoundingBox]:
    """Merge only overlapping boxes, preserve nearby boxes as separate."""
    if not boxes:
        return []

    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = set()

        for idx, box1 in enumerate(merged):
            if idx in used:
                continue

            current_box = box1
            for jdx, box2 in enumerate(merged[idx + 1:], start=idx + 1):
                if jdx in used:
                    continue

                if current_box.overlaps(box2):
                    current_box = current_box.merge(box2)
                    used.add(jdx)
                    changed = True

            new_merged.append(current_box)

        merged = new_merged

    return merged


def _sort_reading_order(boxes: List[BoundingBox]) -> List[BoundingBox]:
    """Sort boxes by reading order (top-to-bottom, left-to-right)."""
    if not boxes:
        return []

    row_threshold = 30
    sorted_boxes = sorted(boxes, key=lambda b: (b.y, b.x))

    rows = []
    current_row = [sorted_boxes[0]]
    current_row_y = sorted_boxes[0].y

    for box in sorted_boxes[1:]:
        if abs(box.y - current_row_y) <= row_threshold:
            current_row.append(box)
        else:
            current_row.sort(key=lambda b: b.x)
            rows.append(current_row)
            current_row = [box]
            current_row_y = box.y

    if current_row:
        current_row.sort(key=lambda b: b.x)
        rows.append(current_row)

    result = []
    for row in rows:
        result.extend(row)

    return result


def _ocr_block(block_image) -> str:
    """Perform OCR on a single block."""
    try:
        text = pytesseract.image_to_string(block_image, config="--psm 6")
        return text.strip()
    except Exception as exc:
        log.debug("OCR failed for block: %s", exc)
        return ""


def _clean_block_text(text: str) -> str:
    """Clean OCR output from a block."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line = " ".join(line.split())
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
