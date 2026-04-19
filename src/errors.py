"""Custom exceptions for OCR File Converter."""


class OCRError(Exception):
    """Base exception for OCR processing errors."""

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause

    def __str__(self):
        base = super().__str__()
        if self.cause:
            return f"{base} (caused by {type(self.cause).__name__}: {self.cause})"
        return base


class PreprocessingError(OCRError):
    """Raised when preprocessing (OCR, conversion, extraction) fails."""


class NormalizationError(OCRError):
    """Raised when text normalization fails."""


class SerializationError(OCRError):
    """Raised when serialization (JSON, JSONL) fails."""


class ToolError(OCRError):
    """Raised when a tool adapter (OCR, Office, Archive) fails."""


class ToolUnavailableError(ToolError):
    """Raised when a required tool service is unavailable."""

    def __init__(self, tool_name: str, url: str):
        super().__init__(f"Tool '{tool_name}' unavailable at {url}")
        self.tool_name = tool_name
        self.url = url
