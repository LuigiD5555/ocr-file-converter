"""Layout-aware OCR reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
import difflib
import re
import statistics

from src.logging import get_logger

log = get_logger(__name__)

try:
    from PIL import Image
    import pytesseract
except Exception:  # pragma: no cover
    Image = None
    pytesseract = None


@dataclass
class OCRWord:
    """A single OCR word with geometry."""

    text: str
    left: int
    top: int
    width: int
    height: int
    right: int
    bottom: int
    confidence: float

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class OCRSegment:
    """A row segment made of nearby OCR words."""

    words: List[OCRWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in sorted(self.words, key=lambda item: item.left))

    @property
    def left(self) -> int:
        return min(word.left for word in self.words)

    @property
    def right(self) -> int:
        return max(word.right for word in self.words)

    @property
    def top(self) -> int:
        return min(word.top for word in self.words)

    @property
    def bottom(self) -> int:
        return max(word.bottom for word in self.words)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class OCRBlock:
    """A vertical block of related OCR segments."""

    header: OCRSegment
    segments: List[OCRSegment] = field(default_factory=list)

    def render(self) -> str:
        """Render a block into readable text."""
        rendered_lines = [self.header.text]
        body_segments = [segment for segment in self.segments if segment is not self.header]
        if body_segments:
            rendered_lines.extend(segment.text for segment in sorted(body_segments, key=lambda item: (item.top, item.left)))
        return "\n".join(line.strip() for line in rendered_lines if line.strip())


class ImageLayoutReconstructor:
    """Recover lightweight block structure from infographic-like images."""

    def __init__(self, *, min_confidence: float = 40.0) -> None:
        self.min_confidence = min_confidence

    def reconstruct(self, image_path: Path, anchors: Optional[Sequence[str]] = None) -> Optional[str]:
        """Rebuild readable text blocks from an OCR image.

        Args:
            image_path: Source image path.

        Returns:
            A readable block-based text representation, or ``None`` on failure.
        """
        if Image is None or pytesseract is None:
            return None

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                words = self._extract_words(image)
                if not words:
                    return None

                segments = self._build_segments(words, width)
                if not segments:
                    return None

                title_text = self._render_title(segments, height)
                blocks = self._build_blocks(segments, width, height, anchors=anchors)
                if not blocks:
                    return title_text or None

                rendered_blocks = [block.render() for block in blocks if block.render().strip()]
                parts = []
                if title_text:
                    parts.append(title_text)
                parts.extend(rendered_blocks)
                return "\n\n".join(part for part in parts if part.strip())
        except Exception as exc:  # pragma: no cover
            log.debug("Layout reconstruction failed for %s: %s", image_path, exc)
            return None

    def _extract_words(self, image: "Image.Image") -> List[OCRWord]:
        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )

        words: List[OCRWord] = []
        total = len(data.get("text", []))
        for index in range(total):
            text = (data["text"][index] or "").strip()
            if not text:
                continue

            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1.0

            if confidence < self.min_confidence:
                continue

            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
            words.append(
                OCRWord(
                    text=text,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    right=left + width,
                    bottom=top + height,
                    confidence=confidence,
                )
            )

        return words

    def _build_segments(self, words: List[OCRWord], image_width: int) -> List[OCRSegment]:
        row_tolerance = 12
        segment_gap = max(35, int(image_width * 0.08))
        rows: List[List[OCRWord]] = []
        row_centers: List[float] = []

        for word in sorted(words, key=lambda item: (item.top, item.left)):
            target_index: Optional[int] = None
            for index, center in enumerate(row_centers):
                if abs(word.center_y - center) <= row_tolerance:
                    target_index = index
                    break

            if target_index is None:
                rows.append([word])
                row_centers.append(word.center_y)
            else:
                rows[target_index].append(word)
                row_centers[target_index] = sum(item.center_y for item in rows[target_index]) / len(rows[target_index])

        segments: List[OCRSegment] = []
        ordered_rows = sorted(rows, key=lambda items: min(word.top for word in items))
        for row in ordered_rows:
            ordered_words = sorted(row, key=lambda item: item.left)
            current_words = [ordered_words[0]]
            for word in ordered_words[1:]:
                if word.left - current_words[-1].right > segment_gap:
                    segments.append(OCRSegment(words=current_words))
                    current_words = [word]
                else:
                    current_words.append(word)
            segments.append(OCRSegment(words=current_words))

        return sorted(segments, key=lambda item: (item.top, item.left))

    def _render_title(self, segments: List[OCRSegment], image_height: int) -> str:
        title_cutoff = image_height * 0.16
        title_segments = [segment for segment in segments if segment.top < title_cutoff]
        if not title_segments:
            return ""
        return "\n".join(segment.text for segment in title_segments)

    def _build_blocks(self, segments: List[OCRSegment], image_width: int, image_height: int, anchors: Optional[Sequence[str]] = None) -> List[OCRBlock]:
        title_cutoff = image_height * 0.16
        body_segments = [segment for segment in segments if segment.top >= title_cutoff]
        if not body_segments:
            return []

        median_height = statistics.median(segment.height for segment in body_segments)
        headers = self._select_headers(body_segments, image_width, median_height, anchors)
        if len(headers) < 3:
            return []

        column_threshold = max(80, int(image_width * 0.14))
        columns: List[List[OCRSegment]] = []
        column_centers: List[float] = []
        for header in sorted(headers, key=lambda item: item.center_x):
            target_index: Optional[int] = None
            for index, center in enumerate(column_centers):
                if abs(header.center_x - center) <= column_threshold:
                    target_index = index
                    break

            if target_index is None:
                columns.append([header])
                column_centers.append(header.center_x)
            else:
                columns[target_index].append(header)
                column_centers[target_index] = sum(item.center_x for item in columns[target_index]) / len(columns[target_index])

        blocks: List[OCRBlock] = []
        for center, column_headers in sorted(zip(column_centers, columns), key=lambda item: item[0]):
            ordered_headers = sorted(column_headers, key=lambda item: item.top)
            for index, header in enumerate(ordered_headers):
                next_header_top = ordered_headers[index + 1].top if index + 1 < len(ordered_headers) else image_height + 1
                block_segments = [header]
                for segment in body_segments:
                    if segment is header:
                        continue
                    if segment.top < header.bottom - 5 or segment.top >= next_header_top:
                        continue
                    if abs(segment.center_x - center) > column_threshold:
                        continue
                    block_segments.append(segment)

                cleaned_segments = self._dedupe_segments(sorted(block_segments, key=lambda item: (item.top, item.left)))
                blocks.append(OCRBlock(header=header, segments=cleaned_segments))

        return sorted(blocks, key=lambda item: (item.header.top, item.header.left))


    def _select_headers(
        self,
        body_segments: List[OCRSegment],
        image_width: int,
        median_height: float,
        anchors: Optional[Sequence[str]],
    ) -> List[OCRSegment]:
        """Choose header segments, optionally using domain anchors."""
        candidate_headers = [segment for segment in body_segments if self._is_header_candidate(segment, image_width, median_height)]
        if not anchors:
            return candidate_headers

        matched_headers: List[OCRSegment] = []
        used_segments = set()
        for anchor in anchors:
            normalized_anchor = self._normalize_token(anchor)
            best_segment = None
            best_score = 0.0
            for segment in candidate_headers:
                segment_key = (segment.top, segment.left, segment.text)
                if segment_key in used_segments:
                    continue
                normalized_segment = self._normalize_token(segment.text)
                if not normalized_segment:
                    continue
                score = difflib.SequenceMatcher(None, normalized_anchor, normalized_segment).ratio()
                if score > best_score:
                    best_score = score
                    best_segment = segment

            if best_segment is not None and best_score >= 0.45:
                used_segments.add((best_segment.top, best_segment.left, best_segment.text))
                best_segment.words = [OCRWord(
                    text=anchor,
                    left=best_segment.left,
                    top=best_segment.top,
                    width=best_segment.width,
                    height=best_segment.height,
                    right=best_segment.right,
                    bottom=best_segment.bottom,
                    confidence=best_segment.words[0].confidence if best_segment.words else 100.0,
                )]
                matched_headers.append(best_segment)

        return matched_headers or candidate_headers

    def _normalize_token(self, value: str) -> str:
        """Normalize OCR text for fuzzy matching."""
        return re.sub(r"[^a-z0-9]", "", value.lower())

    def _is_header_candidate(self, segment: OCRSegment, image_width: int, median_height: float) -> bool:
        text = segment.text.strip()
        tokens = text.split()
        alpha_text = "".join(character for character in text if character.isalpha())
        if not alpha_text or len(alpha_text) < 2:
            return False
        if len(tokens) > 3:
            return False
        if segment.width > image_width * 0.35:
            return False
        if segment.height < max(16, median_height * 1.2):
            return False
        return True

    def _dedupe_segments(self, segments: List[OCRSegment]) -> List[OCRSegment]:
        seen = set()
        unique_segments: List[OCRSegment] = []
        for segment in segments:
            key = (segment.top, segment.left, segment.text)
            if key in seen:
                continue
            seen.add(key)
            unique_segments.append(segment)
        return unique_segments
