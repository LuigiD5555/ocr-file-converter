"""Text normalization and serialization utilities."""

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.errors import SerializationError
from src.layout import ImageLayoutReconstructor
from src.logging import get_logger

log = get_logger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata about a processed document."""

    source: str = Field(..., description="Source filename or path")
    length: int = Field(..., description="Length of normalized content in characters")
    lines: int = Field(..., description="Number of lines in content")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration for examples."""

        json_schema_extra = {
            "example": {
                "source": "document.pdf",
                "length": 5000,
                "lines": 150,
                "extra": {},
            }
        }


class NormalizedDocument(BaseModel):
    """A normalized document with content and metadata."""

    content: str = Field(..., description="Normalized text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")

    class Config:
        """Pydantic configuration for examples."""

        json_schema_extra = {
            "example": {
                "content": "This is normalized text...",
                "metadata": {
                    "source": "document.pdf",
                    "length": 1000,
                    "lines": 50,
                    "extra": {},
                },
            }
        }


class DocumentChunk(BaseModel):
    """A chunk of a document for JSONL serialization."""

    id: str = Field(..., description="Unique chunk ID")
    source: str = Field(..., description="Source filename")
    chunk: int = Field(..., description="Chunk index")
    total_chunks: int = Field(..., description="Total number of chunks")
    content: str = Field(..., description="Chunk content")
    length: int = Field(..., description="Length of chunk in characters")

    class Config:
        """Pydantic configuration for examples."""

        json_schema_extra = {
            "example": {
                "id": "document_0",
                "source": "document.pdf",
                "chunk": 0,
                "total_chunks": 10,
                "content": "Chunk content...",
                "length": 1000,
            }
        }


class TextNormalizer:
    """Normalize extracted text while preserving readable structure.

    The previous implementation collapsed all whitespace using ``\s+``, which
    removed line breaks and converted the entire document into a single line.
    This implementation keeps paragraph structure, cleans OCR noise, and applies
    conservative heuristics so the output remains readable and useful for
    downstream systems.
    """

    def __init__(
        self,
        *,
        preserve_paragraphs: bool = True,
        max_consecutive_blank_lines: int = 2,
    ) -> None:
        """Initialize the normalizer.

        Args:
            preserve_paragraphs: Whether to preserve paragraph boundaries.
            max_consecutive_blank_lines: Maximum blank lines to keep together.
        """
        self.preserve_paragraphs = preserve_paragraphs
        self.max_consecutive_blank_lines = max_consecutive_blank_lines
        self.special_chars_pattern = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
        self.inline_space_pattern = re.compile(r"[ \t\u00A0\u2000-\u200B\u202F\u205F\u3000]+")
        self.multiple_blank_lines_pattern = re.compile(r"\n{%d,}" % (self.max_consecutive_blank_lines + 1))
        self.hyphenated_line_break_pattern = re.compile(r"(?<=\w)-\n(?=\w)")
        self.page_marker_pattern = re.compile(r"^<PARSED TEXT FOR PAGE: .*?>$", re.IGNORECASE)
        self.image_marker_pattern = re.compile(r"^<IMAGE FOR PAGE: .*?>$", re.IGNORECASE)
        self.standalone_page_number_pattern = re.compile(r"^\d{1,4}$")
        self.list_item_pattern = re.compile(r"^(?:[-*•▪◦‣]|\d+[.)]|[A-Za-z][.)])\s+")
        self.heading_pattern = re.compile(r"^(?:#{1,6}\s+.+|[A-ZÁÉÍÓÚÑ0-9][A-ZÁÉÍÓÚÑ0-9\s:/().,\-]{3,})$")
        self.figure_caption_pattern = re.compile(r"^(?:Figura|Figure|Fig\.)\s*\d+", re.IGNORECASE)
        self.layout_reconstructor = ImageLayoutReconstructor()

    def normalize(
        self,
        text: str,
        source_path: Optional[Union[str, Path]] = None,
        layout_anchors: Optional[List[str]] = None,
    ) -> str:
        """Normalize text while keeping paragraph structure.

        Args:
            text: Raw text to normalize.

        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""

        reconstructed_text = self._maybe_reconstruct_layout(text, source_path, layout_anchors)
        if reconstructed_text:
            text = reconstructed_text

        cleaned_text = self._normalize_unicode(text)
        cleaned_text = self._normalize_line_endings(cleaned_text)
        cleaned_text = self._remove_control_characters(cleaned_text)
        cleaned_text = self._replace_form_feeds(cleaned_text)
        cleaned_text = self._fix_hyphenated_line_breaks(cleaned_text)
        cleaned_text = self._recover_basic_structure(cleaned_text)

        raw_lines = cleaned_text.split("\n")
        sanitized_lines = [self._sanitize_line(line) for line in raw_lines]
        merged_lines = self._merge_lines(sanitized_lines)
        normalized_text = self._collapse_blank_lines("\n".join(merged_lines))
        normalized_text = self._fix_inline_punctuation_spacing(normalized_text)
        normalized_text = normalized_text.strip()

        return normalized_text

    def normalize_file(self, file_path: Path) -> str:
        """Read, normalize, and return text from a file.

        Args:
            file_path: Path to the text file.

        Returns:
            Normalized text content. Returns an empty string on read failure.
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            normalized = self.normalize(content)
            log.info(
                "Normalized text from %s (%d -> %d chars)",
                file_path.name,
                len(content),
                len(normalized),
            )
            return normalized
        except OSError as exc:
            log.error("Failed to normalize %s: %s", file_path.name, exc)
            return ""


    def _maybe_reconstruct_layout(
        self,
        text: str,
        source_path: Optional[Union[str, Path]],
        layout_anchors: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Try to rebuild structure from OCR images when text is flattened.

        Args:
            text: OCR text.
            source_path: Original source file.

        Returns:
            Reconstructed text, or ``None`` when no reconstruction should happen.
        """
        if not source_path:
            return None

        source = Path(source_path)
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            return None

        line_count = text.count("\n") + 1
        word_count = len(text.split())
        if line_count > 6 or word_count < 12:
            return None

        reconstructed = self.layout_reconstructor.reconstruct(source, anchors=layout_anchors)
        if not reconstructed:
            return None

        if reconstructed.count("\n") <= text.count("\n") + 2:
            return None

        return reconstructed

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters and common invisible spaces.

        Args:
            text: Raw text.

        Returns:
            Unicode-normalized text.
        """
        normalized_text = unicodedata.normalize("NFKC", text)
        normalized_text = normalized_text.replace("\u00ad", "")
        normalized_text = normalized_text.replace("\ufeff", "")
        return normalized_text

    def _normalize_line_endings(self, text: str) -> str:
        """Convert all line endings to UNIX style.

        Args:
            text: Input text.

        Returns:
            Text with ``\n`` line endings.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_control_characters(self, text: str) -> str:
        """Remove problematic control characters but keep line breaks.

        Args:
            text: Input text.

        Returns:
            Cleaned text.
        """
        return self.special_chars_pattern.sub("", text)

    def _replace_form_feeds(self, text: str) -> str:
        """Convert form feeds into paragraph separators.

        Args:
            text: Input text.

        Returns:
            Text with page breaks converted to blank lines.
        """
        return text.replace("\f", "\n\n")

    def _fix_hyphenated_line_breaks(self, text: str) -> str:
        """Join words broken by line-end hyphenation.

        Args:
            text: Input text.

        Returns:
            Text with joined hyphenated words.
        """
        return self.hyphenated_line_break_pattern.sub("", text)


    def _recover_basic_structure(self, text: str) -> str:
        """Recover minimal paragraph structure from flattened OCR output.

        Args:
            text: Input text.

        Returns:
            Text with lightweight paragraph boundaries when the input is mostly
            a single line.
        """
        if text.count("\n") > 2:
            return text

        recovered_text = text
        recovered_text = re.sub(r"\s+(?=(?:\d+\.|[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+:)\s)", "\n", recovered_text)
        recovered_text = re.sub(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])", "\n", recovered_text)
        recovered_text = re.sub(r"\s+(?=·\s)", "\n", recovered_text)
        return recovered_text

    def _sanitize_line(self, line: str) -> str:
        """Clean an individual line without destroying structure.

        Args:
            line: Raw line.

        Returns:
            Sanitized line.
        """
        stripped_line = line.strip()
        if not stripped_line:
            return ""

        if self.page_marker_pattern.match(stripped_line) or self.image_marker_pattern.match(stripped_line):
            return ""

        collapsed_spaces = self.inline_space_pattern.sub(" ", stripped_line)
        collapsed_spaces = collapsed_spaces.strip()

        if self.figure_caption_pattern.match(collapsed_spaces):
            return collapsed_spaces

        if self.standalone_page_number_pattern.match(collapsed_spaces):
            return ""

        return collapsed_spaces

    def _merge_lines(self, lines: List[str]) -> List[str]:
        """Merge wrapped OCR/PDF lines into readable paragraphs.

        Args:
            lines: Sanitized lines.

        Returns:
            A list of output lines with paragraph boundaries preserved.
        """
        if not self.preserve_paragraphs:
            return [line for line in lines if line]

        merged_lines: List[str] = []
        current_paragraph: List[str] = []

        for line in lines:
            if not line:
                self._flush_paragraph(current_paragraph, merged_lines)
                if merged_lines and merged_lines[-1] != "":
                    merged_lines.append("")
                continue

            if self._is_structural_line(line):
                self._flush_paragraph(current_paragraph, merged_lines)
                merged_lines.append(line)
                continue

            current_paragraph.append(line)

        self._flush_paragraph(current_paragraph, merged_lines)
        return merged_lines

    def _flush_paragraph(self, paragraph_lines: List[str], merged_lines: List[str]) -> None:
        """Flush the current paragraph into the merged output.

        Args:
            paragraph_lines: Collected paragraph lines.
            merged_lines: Final output lines.
        """
        if not paragraph_lines:
            return

        merged_lines.append(self._join_paragraph_lines(paragraph_lines))
        paragraph_lines.clear()

    def _join_paragraph_lines(self, lines: List[str]) -> str:
        """Join a group of paragraph lines into a readable paragraph.

        Args:
            lines: Paragraph lines.

        Returns:
            Joined paragraph text.
        """
        if len(lines) == 1:
            return lines[0]

        paragraph = ""
        for line in lines:
            if not paragraph:
                paragraph = line
                continue

            if self._should_concatenate_without_space(paragraph, line):
                paragraph += line
            else:
                paragraph += f" {line}"

        return paragraph

    def _should_concatenate_without_space(self, current_text: str, next_line: str) -> bool:
        """Determine whether a line should be appended without a space.

        Args:
            current_text: Current paragraph text.
            next_line: Incoming line.

        Returns:
            ``True`` when the line should be appended directly.
        """
        return current_text.endswith("-") or next_line.startswith(",") or next_line.startswith(".")

    def _is_structural_line(self, line: str) -> bool:
        """Check whether a line should remain isolated.

        Args:
            line: Candidate line.

        Returns:
            ``True`` for headings, list items, captions, or table-like rows.
        """
        if self.list_item_pattern.match(line):
            return True

        if self.figure_caption_pattern.match(line):
            return True

        if self.heading_pattern.match(line):
            return True

        if self._looks_like_table_row(line):
            return True

        return False

    def _looks_like_table_row(self, line: str) -> bool:
        """Heuristically detect table-like rows.

        Args:
            line: Candidate line.

        Returns:
            ``True`` if the line looks like a compact table row.
        """
        separators = line.count("|") + line.count("\t")
        if separators >= 2:
            return True

        tokens = line.split(" ")
        numeric_tokens = sum(token.replace(".", "", 1).isdigit() for token in tokens)
        short_tokens = sum(len(token) <= 3 for token in tokens)
        return len(tokens) >= 4 and numeric_tokens >= 2 and short_tokens >= 2

    def _collapse_blank_lines(self, text: str) -> str:
        """Limit repeated blank lines.

        Args:
            text: Input text.

        Returns:
            Text with bounded blank lines.
        """
        return self.multiple_blank_lines_pattern.sub("\n" * self.max_consecutive_blank_lines, text)

    def _fix_inline_punctuation_spacing(self, text: str) -> str:
        """Normalize punctuation spacing without touching paragraph breaks.

        Args:
            text: Input text.

        Returns:
            Text with cleaner spacing around punctuation.
        """
        text = re.sub(r"[ ]+([,.;:!?])", r"\1", text)
        text = re.sub(r"([\[({]) +", r"\1", text)
        text = re.sub(r" +([\])}])", r"\1", text)
        return text


class TextSerializer:
    """Serialize normalized text to structured formats using Pydantic."""

    def __init__(self, normalizer: Optional[TextNormalizer] = None) -> None:
        """Initialize the serializer.

        Args:
            normalizer: Optional injected normalizer instance.
        """
        self.normalizer = normalizer or TextNormalizer()

    def to_normalized_document(
        self,
        text: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NormalizedDocument:
        """Convert text to a ``NormalizedDocument`` model.

        Args:
            text: Text content.
            source: Source filename or path.
            metadata: Additional metadata.

        Returns:
            A validated ``NormalizedDocument`` instance.
        """
        normalized = self.normalizer.normalize(text)

        doc_metadata = DocumentMetadata(
            source=source,
            length=len(normalized),
            lines=len(normalized.split("\n")),
            extra=metadata or {},
        )

        return NormalizedDocument(content=normalized, metadata=doc_metadata)

    def to_chunks(
        self,
        text: str,
        source: str = "",
        chunk_size: int = 1000,
    ) -> List[DocumentChunk]:
        """Convert text to ``DocumentChunk`` models.

        Args:
            text: Text content.
            source: Source filename.
            chunk_size: Characters per chunk.

        Returns:
            A list of document chunks.
        """
        normalized = self.normalizer.normalize(text)
        chunks_data = [
            normalized[i:i + chunk_size]
            for i in range(0, len(normalized), chunk_size)
        ]

        chunks: List[DocumentChunk] = []
        for idx, chunk in enumerate(chunks_data):
            chunk_obj = DocumentChunk(
                id=f"{source}_{idx}",
                source=source,
                chunk=idx,
                total_chunks=len(chunks_data),
                content=chunk.strip(),
                length=len(chunk),
            )
            chunks.append(chunk_obj)

        return chunks

    def save_json(self, text: str, output_path: Path, source: str = "") -> Path:
        """Save normalized text as a JSON file.

        Args:
            text: Text content.
            output_path: Destination path.
            source: Source filename.

        Returns:
            The output path.

        Raises:
            SerializationError: If writing fails.
        """
        document = self.to_normalized_document(text, source=source)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write(document.model_dump_json(indent=2, exclude_none=True))
            log.info("Saved JSON to %s", output_path)
            return output_path
        except OSError as exc:
            log.error("Failed to save JSON to %s: %s", output_path, exc)
            raise SerializationError(f"Failed to save JSON to {output_path}", cause=exc)

    def save_jsonl(
        self,
        text: str,
        output_path: Path,
        source: str = "",
        chunk_size: int = 1000,
    ) -> Path:
        """Save normalized text as a JSONL file.

        Args:
            text: Text content.
            output_path: Destination path.
            source: Source filename.
            chunk_size: Characters per chunk.

        Returns:
            The output path.

        Raises:
            SerializationError: If writing fails.
        """
        chunks = self.to_chunks(text, source=source, chunk_size=chunk_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                for chunk in chunks:
                    output_file.write(chunk.model_dump_json(exclude_none=True) + "\n")
            log.info("Saved JSONL with %d chunks to %s", len(chunks), output_path)
            return output_path
        except OSError as exc:
            log.error("Failed to save JSONL to %s: %s", output_path, exc)
            raise SerializationError(f"Failed to save JSONL to {output_path}", cause=exc)
