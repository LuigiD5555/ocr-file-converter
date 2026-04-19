"""Text normalization and cleaning utilities with Pydantic schemas."""
import re
from typing import Dict, Any, List
from pathlib import Path

from pydantic import BaseModel, Field

from src.errors import NormalizationError, SerializationError
from src.logging import get_logger

log = get_logger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata about a processed document."""
    source: str = Field(..., description="Source filename or path")
    length: int = Field(..., description="Length of normalized content in characters")
    lines: int = Field(..., description="Number of lines in content")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
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
    """Normalize and clean extracted text from documents."""

    def __init__(self):
        self.patterns = {
            "extra_spaces": r"\s+",
            "multiple_newlines": r"\n{3,}",
            "special_chars": r"[\x00-\x08\x0b-\x0c\x0e-\x1f]",
        }

    def normalize(self, text: str) -> str:
        """Normalize text by cleaning whitespace and special characters.

        Args:
            text: Raw text to normalize

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        text = re.sub(self.patterns["special_chars"], "", text)
        text = re.sub(self.patterns["extra_spaces"], " ", text)
        text = re.sub(self.patterns["multiple_newlines"], "\n\n", text)
        text = text.strip()

        return text

    def normalize_file(self, file_path: Path) -> str:
        """Read, normalize, and return text from file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            normalized = self.normalize(content)
            log.info("Normalized text from %s (%d → %d chars)",
                     file_path.name, len(content), len(normalized))
            return normalized
        except Exception as exc:
            log.error("Failed to normalize %s: %s", file_path.name, exc)
            return ""


class TextSerializer:
    """Serialize normalized text to structured formats using Pydantic."""

    def __init__(self):
        self.normalizer = TextNormalizer()

    def to_normalized_document(
        self, text: str, source: str = "", metadata: Dict[str, Any] = None
    ) -> NormalizedDocument:
        """Convert text to NormalizedDocument (Pydantic model).

        Args:
            text: Text content
            source: Source filename or path
            metadata: Additional metadata

        Returns:
            NormalizedDocument instance
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
        self, text: str, source: str = "", chunk_size: int = 1000
    ) -> List[DocumentChunk]:
        """Convert text to DocumentChunks (Pydantic models).

        Args:
            text: Text content
            source: Source filename
            chunk_size: Characters per chunk

        Returns:
            List of DocumentChunk instances
        """
        normalized = self.normalizer.normalize(text)
        chunks_data = [
            normalized[i:i + chunk_size]
            for i in range(0, len(normalized), chunk_size)
        ]

        chunks = []
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
        """Save normalized text as JSON file.

        Args:
            text: Text content
            output_path: Path to save JSON file
            source: Source filename

        Returns:
            Path to saved file

        Raises:
            SerializationError: If saving fails
        """
        document = self.to_normalized_document(text, source=source)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path.open("w", encoding="utf-8") as f:
                f.write(document.model_dump_json(indent=2, exclude_none=True))
            log.info("Saved JSON to %s", output_path)
            return output_path
        except Exception as exc:
            log.error("Failed to save JSON to %s: %s", output_path, exc)
            raise SerializationError(f"Failed to save JSON to {output_path}", cause=exc)

    def save_jsonl(
        self, text: str, output_path: Path, source: str = "", chunk_size: int = 1000
    ) -> Path:
        """Save normalized text as JSONL file (line-delimited JSON).

        Args:
            text: Text content
            output_path: Path to save JSONL file
            source: Source filename
            chunk_size: Characters per chunk

        Returns:
            Path to saved file

        Raises:
            SerializationError: If saving fails
        """
        chunks = self.to_chunks(text, source=source, chunk_size=chunk_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(chunk.model_dump_json(exclude_none=True) + "\n")
            log.info("Saved JSONL with %d chunks to %s", len(chunks), output_path)
            return output_path
        except Exception as exc:
            log.error("Failed to save JSONL to %s: %s", output_path, exc)
            raise SerializationError(f"Failed to save JSONL to {output_path}", cause=exc)
