"""File preprocessing using socket-activated tools."""
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess
import tempfile

import pypdf
import requests

from src import settings
from src.errors import PreprocessingError
from src.tool_adapters import (
    ToolAdapter,
    ToolAdapterConfig,
    ToolAdapterFactory,
)
from src.logging import get_logger

log = get_logger(__name__)


class FilePreprocessor:
    """Preprocess files using socket-activated tools.

    Converts Office documents, extracts archives, and runs OCR for images/scanned PDFs.
    """

    def __init__(
        self,
        office_url: Optional[str] = None,
        archive_url: Optional[str] = None,
        ocr_url: Optional[str] = None,
        enable_office: bool = True,
        enable_archive: bool = True,
        enable_ocr: bool = False,
        timeout: int = 120,
        work_dir: Optional[str] = None,
        tool_adapters: Optional[Dict[str, ToolAdapter]] = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            office_url: URL for Office conversion tool.
            archive_url: URL for archive extraction tool.
            ocr_url: URL for OCR tool.
            enable_office: Enable Office document conversion.
            enable_archive: Enable archive extraction.
            enable_ocr: Enable OCR for images/scanned PDFs.
            timeout: Tool request timeout in seconds.
            work_dir: Working directory for processed files.
            tool_adapters: Optional injected adapters (testing/custom tooling).
        """
        self.office_url = office_url or settings.TOOL_OFFICE_URL
        self.archive_url = archive_url or settings.TOOL_FILEEXTRACTOR_URL
        self.ocr_url = ocr_url or settings.TOOL_OCR_URL

        self.enable_office = enable_office and settings.ENABLE_OFFICE_CONVERSION
        self.enable_archive = enable_archive and settings.ENABLE_EXTRACTOR_EXTRACTION
        self.enable_ocr = enable_ocr or settings.ENABLE_OCR

        self.timeout = timeout
        self.work_dir = Path(work_dir or settings.PREPROCESSING_WORK_DIR)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        config = ToolAdapterConfig(
            office_url=self.office_url,
            archive_url=self.archive_url,
            ocr_url=self.ocr_url,
            timeout=self.timeout,
            retry_delay=settings.TOOL_CONNECT_RETRY_DELAY,
            retries=settings.TOOL_CONNECT_RETRIES,
            enable_office=self.enable_office,
            enable_archive=self.enable_archive,
            enable_ocr=self.enable_ocr,
            archive_max_size_mb=settings.ARCHIVE_MAX_SIZE_MB,
        )

        self.tool_adapters = tool_adapters or ToolAdapterFactory.build_default_adapters(config)
        self.office_adapter = self.tool_adapters.get("office")
        self.archive_adapter = self.tool_adapters.get("archive")
        self.ocr_adapter = self.tool_adapters.get("ocr")

        self.office_extensions = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt"}
        self.archive_extensions = {".zip", ".7z", ".tar", ".tar.gz", ".tar.xz", ".tar.bz2", ".tgz"}
        self.ocr_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"}
        self.text_extensions = {".txt", ".md", ".csv", ".log"}

        log.info(
            "FilePreprocessor initialized: office=%s, archive=%s, ocr=%s",
            self.enable_office,
            self.enable_archive,
            self.enable_ocr,
        )

    def should_preprocess(self, file_path: Path) -> bool:
        """Return True if the file should be preprocessed.

        Skips temporary/lock files from Office suites.
        """
        filename = file_path.name
        if filename.startswith(".~") or filename.startswith("~$"):
            return False

        suffix = file_path.suffix.lower()

        if self.office_adapter and self.office_adapter.enabled and suffix in self.office_extensions:
            return True

        if self.archive_adapter and self.archive_adapter.enabled and suffix in self.archive_extensions:
            return True

        if self.ocr_adapter and self.ocr_adapter.enabled and suffix in self.ocr_extensions:
            return True

        if suffix in self.text_extensions:
            return True

        return False

    def preprocess(self, file_path: Path) -> Optional[Path]:
        """Preprocess a file if needed.

        Args:
            file_path: Path to the input file.

        Returns:
            The processed file path, the original path if no preprocessing needed,
            or None if preprocessing failed.
        """
        if not self.should_preprocess(file_path):
            return file_path

        suffix = file_path.suffix.lower()

        try:
            if self.office_adapter and suffix in self.office_extensions:
                return self._convert_office_document(file_path)

            if self.archive_adapter and suffix in self.archive_extensions:
                return self._extract_archive(file_path)

            if self.ocr_adapter and suffix in self.ocr_extensions:
                # PDFs need special handling with strategy selection
                if suffix == ".pdf":
                    analysis = self._analyze_pdf(file_path)
                    strategy = analysis["strategy"]
                    log.info(
                        "PDF strategy: %s (text_quality=%s, pages=%d)",
                        strategy,
                        analysis["text_quality"],
                        analysis["page_count"],
                    )

                    if strategy == "text_extract":
                        return self._extract_text_with_pdftotext(file_path)
                    elif strategy == "hybrid":
                        return self._hybrid_pdf(file_path)
                    else:  # strategy == "ocr"
                        return self._ocr_pdf(file_path)

                return self._perform_ocr(file_path)

            if suffix in self.text_extensions:
                return self._handle_text_file(file_path)

            return file_path
        except PreprocessingError:
            raise
        except Exception as exc:
            log.error("Preprocessing failed for %s: %s", file_path, exc)
            raise PreprocessingError(f"Preprocessing failed for {file_path}", cause=exc)

    def _convert_office_document(self, file_path: Path) -> Optional[Path]:
        """Convert an Office document to text."""
        if not self.office_adapter:
            return file_path

        output = self.office_adapter.process(file_path)
        if output:
            return output

        log.info("Falling back to direct ingestion for %s", file_path.name)
        return file_path

    def convert_office_document(self, file_path: Path) -> Optional[Path]:
        """Public wrapper to convert an Office document to text."""
        return self._convert_office_document(file_path)

    def _extract_archive(self, file_path: Path) -> Optional[Path]:
        """Extract an archive."""
        if not self.archive_adapter:
            return None

        return self.archive_adapter.process(file_path)

    def extract_archive(self, file_path: Path) -> Optional[Path]:
        """Public wrapper to extract an archive."""
        return self._extract_archive(file_path)

    def _perform_ocr(self, file_path: Path) -> Optional[Path]:
        """Perform OCR on an image or PDF."""
        if not self.ocr_adapter:
            return None

        return self.ocr_adapter.process(file_path)

    def perform_ocr(self, file_path: Path) -> Optional[Path]:
        """Public wrapper to perform OCR."""
        return self._perform_ocr(file_path)

    def _analyze_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze PDF to determine best processing strategy.

        Returns dict with:
        - has_text: bool - whether PDF has extractable text
        - text_quality: str - 'good', 'partial', 'none'
        - page_count: int - number of pages
        - strategy: str - 'text_extract', 'ocr', 'hybrid'
        """
        try:
            reader = pypdf.PdfReader(pdf_path)
            page_count = len(reader.pages)

            # Analyze first 3 pages for text content
            text_pages = 0
            total_text_len = 0

            for page in reader.pages[:min(3, page_count)]:
                text = page.extract_text()
                total_text_len += len(text.strip())
                if len(text.strip()) > 100:
                    text_pages += 1

            avg_text = total_text_len / min(3, page_count)

            # Determine text quality
            if avg_text > 500:
                text_quality = "good"
                strategy = "text_extract"
            elif avg_text > 100:
                text_quality = "partial"
                strategy = "hybrid"
            else:
                text_quality = "none"
                strategy = "ocr"

            return {
                "has_text": total_text_len > 0,
                "text_quality": text_quality,
                "page_count": page_count,
                "strategy": strategy,
            }

        except Exception as exc:
            log.debug("Error analyzing PDF: %s", exc)
            return {
                "has_text": False,
                "text_quality": "unknown",
                "page_count": 0,
                "strategy": "ocr",
            }

    def _extract_text_with_pdftotext(self, pdf_path: Path) -> Optional[Path]:
        """Extract text from PDF using pdftotext (preserves structure/columns).

        This method is best for PDFs with good text content, as it preserves
        layout and column structure better than OCR.
        """
        try:
            output_path = self.work_dir / f"{pdf_path.stem}_text.txt"

            # pdftotext -layout preserves layout and column structure
            cmd = [
                "pdftotext",
                "-layout",  # Preserve layout
                str(pdf_path.absolute()),
                str(output_path.absolute()),
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if proc.returncode == 0 and output_path.exists():
                # Verify extracted text has content
                text = output_path.read_text(encoding="utf-8", errors="replace")
                if len(text.strip()) > 100:
                    log.info(
                        "Text extraction succeeded for %s -> %s",
                        pdf_path.name,
                        output_path.name,
                    )
                    return output_path
                else:
                    log.warning("Text extraction produced minimal content for %s", pdf_path.name)
                    output_path.unlink()
                    return None
            else:
                log.error(
                    "pdftotext failed for %s: %s",
                    pdf_path.name,
                    proc.stderr,
                )
                return None

        except subprocess.TimeoutExpired:
            log.error("pdftotext timeout for %s", pdf_path.name)
            return None
        except Exception as exc:
            log.error("Error extracting text from PDF %s: %s", pdf_path, exc)
            return None

    def _hybrid_pdf(self, pdf_path: Path) -> Optional[Path]:
        """Hybrid strategy: use pdftotext + OCR for missing sections.

        First tries to extract text with pdftotext, then uses OCR
        to fill in any gaps or missing pages.
        """
        # Try text extraction first
        text_result = self._extract_text_with_pdftotext(pdf_path)

        if text_result:
            log.info(
                "Hybrid PDF processing: using primary text extraction for %s",
                pdf_path.name,
            )
            return text_result

        # Fallback to OCR if text extraction fails
        log.info(
            "Hybrid PDF processing: text extraction failed, falling back to OCR for %s",
            pdf_path.name,
        )
        return self._ocr_pdf(pdf_path)

    def _ocr_pdf(self, pdf_path: Path) -> Optional[Path]:
        """OCR a PDF by converting to images then running OCR.

        Used for scanned PDFs or when text extraction isn't available.
        """
        if not self.ocr_adapter:
            return None

        try:
            # Convert PDF to images via the OCR service
            response = requests.post(
                f"{self.ocr_url}/pdf-to-images",
                json={"pdf_path": str(pdf_path.absolute())},
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                log.error("PDF to images conversion failed: %s", result.get("error"))
                return None

            image_paths = result.get("image_paths", [])
            if not image_paths:
                log.error("No images generated from PDF")
                return None

            # Combine all OCR'd text from all pages
            all_text = []
            for img_path in image_paths:
                img_result = self.ocr_adapter.process(Path(img_path))
                if img_result and img_result.exists():
                    try:
                        text = img_result.read_text(encoding="utf-8", errors="replace")
                        all_text.append(text)
                    except Exception as exc:
                        log.warning("Failed to read OCR result from %s: %s", img_path, exc)

            if not all_text:
                log.error("No text extracted from PDF pages")
                return None

            # Write combined text to output file
            output_path = self.work_dir / f"{pdf_path.stem}_ocr.txt"
            combined_text = "\n\n".join(all_text)
            output_path.write_text(combined_text, encoding="utf-8")
            log.info("OCR'd PDF %s -> %s", pdf_path.name, output_path.name)
            return output_path

        except requests.RequestException as exc:
            log.warning("PDF to images service unavailable: %s", exc)
            return None
        except Exception as exc:
            log.error("Error OCR'ing PDF %s: %s", pdf_path, exc)
            return None

    def _handle_text_file(self, file_path: Path) -> Optional[Path]:
        """Handle plain text files (TXT, MD, CSV, LOG).

        Text files are already in text format, so they're just validated
        and returned as-is for normalization in the next step.
        """
        if not file_path.exists():
            log.error("Text file not found: %s", file_path)
            return None

        try:
            file_path.read_text(encoding="utf-8", errors="replace")
            log.info("Text file ready for processing: %s", file_path.name)
            return file_path
        except Exception as exc:
            log.error("Failed to read text file %s: %s", file_path, exc)
            return None

    def handle_text_file(self, file_path: Path) -> Optional[Path]:
        """Public wrapper to handle text files."""
        return self._handle_text_file(file_path)

    def get_status(self) -> Dict[str, Any]:
        """Return preprocessor status and tool availability."""
        status: Dict[str, Any] = {
            "enabled": {
                "office": self.enable_office,
                "archive": self.enable_archive,
                "ocr": self.enable_ocr,
            },
            "tools_available": {},
        }

        for name, adapter in self.tool_adapters.items():
            status["tools_available"][name] = adapter.is_available()

        return status


_preprocessor_instance: Optional[FilePreprocessor] = None


def get_preprocessor(**kwargs: Any) -> FilePreprocessor:
    """Get or create the FilePreprocessor singleton."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = FilePreprocessor(**kwargs)
    return _preprocessor_instance
