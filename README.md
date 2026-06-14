# OCR File Converter

**Turn messy documents into clean, structured text ready for embeddings and ML pipelines.**

A modular Python library and CLI tool that extracts text from mixed-format documents (PDFs, images, Office files, archives), normalizes output, and serializes to TXT/JSON/JSONL with pluggable adapters for OCR, Office conversion, and archive extraction.

---

## The Problem

You have folders full of documents:
- 📄 PDFs (some scanned, some digital)
- 🖼️ Images (screenshots, photos)
- 📊 Office files (DOCX, XLSX, PPTX)
- 📦 Compressed archives

You need:
- ✅ Clean extracted text
- ✅ Consistent output format
- ✅ Metadata preservation
- ✅ Ready for embeddings or LLM ingestion
- ✅ Fault-tolerant (one bad file shouldn't break the batch)

**ocr-file-converter solves this.** It's the preprocessing step you need before RAG, before embeddings, before any ML pipeline.

---

## Quick Start

### As a CLI

```bash
# Process a PDF, get clean TXT
python main.py invoice.pdf
# → invoice.txt (normalized)

# Process a folder, get JSON with metadata
python main.py ./documents --format json -o ./output
# → output/doc1.json, output/doc2.json, ...

# Extract from scanned images with OCR
python main.py screenshot.png --format jsonl
# → screenshot.jsonl (OCR text + metadata)

# Batch process without OCR (just format conversion)
python main.py ./docs --no-ocr --format json -o ./results
```

### As a Python Library

```python
from src.preprocessor import FilePreprocessor
from src.normalizer import TextSerializer
from pathlib import Path

# Initialize preprocessor
preprocessor = FilePreprocessor(
    ocr_url="http://localhost:8000",
    enable_ocr=True,
)

# Process document
result_path = preprocessor.preprocess(Path("document.pdf"))

# Normalize and serialize
text = result_path.read_text()
serializer = TextSerializer()
serializer.save_json(text, Path("output.json"), source="document.pdf")
```

---

## Project Structure

```
ocr_file_converter/
├── src/
│   ├── cli.py                        # CLI: Command-line interface
│   ├── preprocessor.py               # Extractor: OCR, Office, Archives → TXT
│   ├── normalizer.py                 # Normalization and serialization (JSON, JSONL)
│   ├── tool_adapters.py              # HTTP adapters for external tools
│   ├── logging.py                    # Logging configuration
│   ├── settings.py                   # Centralized configuration
│   └── __init__.py                   # Package initialization
├── ocr_service/
│   ├── app.py                        # FastAPI OCR service with Tesseract
│   └── requirements.txt              # Service dependencies
├── main.py                           # Entry point (runs src.cli)
├── requirements.txt                  # CLI dependencies
├── Dockerfile                        # Container for OCR service
├── docker-compose.yml                # Service orchestration
├── .env.example                      # Example environment configuration
└── README.md                         # This file
```

---

## Architecture

```
Input Document
    ↓
[FilePreprocessor] ← orchestrates
    ├─→ OfficeAdapter (DOCX, XLSX, PPTX → TXT)
    ├─→ ArchiveAdapter (ZIP, TAR, 7Z → extract)
    ├─→ OcrAdapter (PNG, JPG, scanned PDF → OCR)
    └─→ PyPDF (native PDF text extraction)
    ↓
[TextNormalizer] ← clean & standardize
    • Remove extra whitespace
    • Fix encoding issues
    • Preserve structure signals
    ↓
[TextSerializer] ← format for ML
    • TXT (plain text)
    • JSON (with metadata)
    • JSONL (streaming, one doc per line)
    ↓
Output: TXT / JSON / JSONL
```

**Each component is pluggable.** Don't like Tesseract? Use a different OCR service. Need a custom extractor? Implement the `ToolAdapter` interface. The pipeline doesn't care—as long as it speaks HTTP.

---

## Supported Formats

| Input Type | Processing | Output |
|---|---|---|
| `.pdf` | OCR if scanned, text extraction if digital | ✓ TXT/JSON/JSONL |
| `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` | Tesseract OCR | ✓ TXT/JSON/JSONL |
| `.docx`, `.doc` | Office text extraction | ✓ TXT/JSON/JSONL |
| `.xlsx`, `.xls` | Office sheet extraction (as text) | ✓ TXT/JSON/JSONL |
| `.pptx`, `.ppt` | Office slide extraction | ✓ TXT/JSON/JSONL |
| `.zip`, `.tar`, `.7z`, `.tar.gz`, etc. | Recursive extraction + reprocess | ✓ TXT/JSON/JSONL |

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (recommended)
- Or: Tesseract OCR + LibreOffice (local setup)

### Step 1: Create Python Environment

```bash
cd ocr_file_converter
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

For conda environment:
```bash
conda activate rag
cd ocr_file_converter
pip install -r requirements.txt
```

### Step 3: Start OCR Service

#### Option A: Docker Compose (Recommended)

```bash
docker-compose up -d
```

#### Option B: Docker Direct

```bash
docker build -t ocr-service .
docker run -d \
  --name ocr-service \
  -p 8000:8000 \
  -v /tmp:/work \
  ocr-service
```

#### Option C: Python Direct

```bash
pip install -r ocr_service/requirements.txt
python ocr_service/app.py
```

Service will be available at `http://localhost:8000`.

### Step 4: Verify OCR Connectivity

```bash
curl http://localhost:8000/healthz
# Expected: {"status":"ok","service":"tool-ocr"}
```

---

## Usage

### Process Single File

```bash
# PDF → normalized TXT
python main.py documento.pdf

# Image → JSON with metadata
python main.py documento.pdf --format json

# JSONL output (useful for embeddings)
python main.py documento.pdf --format jsonl

# Custom OCR service URL
python main.py archivo.pdf --ocr-url http://ocr-service:8000 --format json --verbose

# Format conversion without OCR
python main.py documento.docx --no-ocr
```

### Process Directory

```bash
# Process all files in directory
python main.py /ruta/a/documentos

# Save to custom output directory
python main.py /ruta/a/documentos -o /ruta/salida

# Batch as JSON
python main.py /ruta/a/documentos --format json -o ./output

# Recursive, no OCR, JSONL output
python main.py /ruta/a/documentos --no-ocr --format jsonl -o ./results
```

### Programmatic Usage

```python
from src.preprocessor import FilePreprocessor
from src.normalizer import TextNormalizer, TextSerializer
from pathlib import Path

# Setup preprocessor
preprocessor = FilePreprocessor(
    ocr_url="http://localhost:8000",
    enable_ocr=True,
    enable_office=True,
    enable_archive=True,
)

# Process file
result_path = preprocessor.preprocess(Path("documento.pdf"))
if not result_path:
    print("Processing failed")
    exit(1)

# Read and normalize
text = result_path.read_text(encoding="utf-8", errors="replace")
normalizer = TextNormalizer()
normalized_text = normalizer.normalize(text, source_path=Path("documento.pdf"))

# Serialize to JSON
serializer = TextSerializer()
output_path = serializer.save_json(
    normalized_text,
    Path("output.json"),
    source="documento.pdf",
)
print(f"Saved to: {output_path}")
```

---

## Configuration

Edit `src/settings.py` or `.env`:

```python
# Service URLs
TOOL_OCR_URL = "http://localhost:8000"
TOOL_OFFICE_URL = "http://localhost:9000"
TOOL_FILEEXTRACTOR_URL = "http://localhost:9001"

# Feature flags
ENABLE_OCR = True
ENABLE_OFFICE_CONVERSION = True
ENABLE_EXTRACTOR_EXTRACTION = True

# OCR settings
OCR_DEFAULT_LANGUAGE = "eng"  # Tesseract language code

# Timeouts and retries
TOOL_REQUEST_TIMEOUT = 120  # seconds
TOOL_CONNECT_RETRIES = 3
TOOL_CONNECT_RETRY_DELAY = 2.0

# Working directories
PREPROCESSING_WORK_DIR = "/tmp/rag_preprocessing"
ARCHIVE_MAX_SIZE_MB = 500
```

Copy and customize `.env.example`:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Environment Variables

```bash
# Service configuration
export TOOL_OCR_URL="http://ocr-service:8000"
export ENABLE_OCR=true

# OCR service
export WORK_DIR="/var/ocr_tmp"
export PORT=8000

# Logging
export LOG_LEVEL="INFO"
```

---

## Components

### FilePreprocessor (`src/preprocessor.py`)

Central orchestrator managing:
- **OCR** for images and scanned PDFs (via `OcrToolAdapter`)
- **Office conversion** (.docx, .xlsx, .pptx → .txt)
- **Archive extraction** (.zip, .tar.gz, etc.)

**Public Methods:**
```python
preprocessor.preprocess(file_path)        # Main entry point, returns processed file path
preprocessor.perform_ocr(file_path)       # Force OCR on file
preprocessor.get_status()                 # Check tool availability
```

**Features:**
- Detects scanned vs digital PDFs automatically
- Recursive archive extraction
- Fault-tolerant (one bad file doesn't break the batch)
- Automatic retry with exponential backoff

### TextNormalizer & TextSerializer (`src/normalizer.py`)

Normalization and serialization of extracted text:

**TextNormalizer:**
- Removes extra whitespace and line breaks
- Fixes encoding issues
- Standardizes quotes and special characters
- Preserves section breaks and structure signals

**TextSerializer:**
- Pydantic-validated JSON/JSONL output
- Metadata preservation (source, length, language)
- Streaming format for large batches

**Pydantic Models:**
```python
NormalizedDocument      # Single document with metadata
DocumentChunk          # Text chunk for JSONL
DocumentMetadata       # Structured metadata
```

### ToolAdapters (`src/tool_adapters.py`)

HTTP adapters for external services (pluggable):

```python
class ToolAdapter(ABC):
    """Base class for tool adapters."""
    async def call(self, payload: dict) -> dict:
        """Send request to external tool, return result."""
        pass
```

**Included Adapters:**
- **OcrToolAdapter** — POST to `/ocr` endpoint
- **OfficeToolAdapter** — POST to `/convert` endpoint
- **ArchiveToolAdapter** — POST to `/extract` endpoint

**Custom Adapter Example:**
```python
class CustomOcrAdapter(ToolAdapter):
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

    async def call(self, payload: dict) -> dict:
        # Your implementation
        pass

# Inject into preprocessor
adapters = {"ocr": CustomOcrAdapter(...)}
preprocessor = FilePreprocessor(tool_adapters=adapters)
```

---

## Telemetry & Monitoring

Module `src/core/telemetry.py` provides error reporting hooks:

```python
from src.core.telemetry import emit_error, set_error_reporter

# Register custom error handler
def my_error_handler(error, component, operation, extra):
    print(f"{component}.{operation}: {error}")
    # Send to monitoring service, log to file, etc.

set_error_reporter(my_error_handler)

# Now all OCR, conversion, extraction errors route through your handler
```

**Error Types:**
- `OCRError` — OCR service failures
- `PreprocessingError` — File processing failures
- `ToolAdapterError` — External tool communication failures

---

## Performance & Scalability

### Typical Throughput
- **Simple PDFs** (text extraction): ~100 pages/min
- **Scanned PDFs** (OCR): ~5-10 pages/min (Tesseract on CPU)
- **Office files**: ~50-100 docs/min
- **Images**: ~2-5 per minute (Tesseract)

### Tuning
- **Increase OCR speed**: Use GPU Tesseract or swap OCR provider
- **Parallel processing**: Run multiple `main.py` instances on different files
- **Timeout issues**: Increase `TOOL_REQUEST_TIMEOUT` in `.env`

### Fault Tolerance
- One failed file doesn't break the batch
- Detailed error logging per file
- Retry logic for transient tool failures
- Resume capability (checkpoints work directory)

---

## Troubleshooting

### OCR Service Unavailable

```
⚠ Tool-ocr unavailable: Connection refused
```

**Solution:**
```bash
docker-compose up -d
# or
curl http://localhost:8000/healthz
```

### PDF Load Timeout

```
PDF load timeout after 120s
```

**Solution:** Increase timeout in `.env`:
```bash
TOOL_REQUEST_TIMEOUT=300
```

### OCR Returned No Text

Image quality too low or wrong language.

**Solution:**
```bash
# Try different language
OCR_DEFAULT_LANGUAGE=spa  # Spanish
python main.py image.png
```

### Import Error: No Module Named 'src'

```bash
pip install -r requirements.txt
```

---

## Development Notes

- Code extracted from **RAG-Agentic-Graphiti**, maintains compatibility
- `settings.py` imports via `from src.settings import settings`
- All dependencies in `src/` are read-only (no global state mutation)
- Logger obtained via `get_logger(__name__)` from `src.logging`
- Pydantic models provide input/output validation

### Design Patterns

- **Adapter Pattern** — `ToolAdapter` interface for swappable external services
- **Factory Pattern** — `ToolAdapterFactory.build_default_adapters()`
- **Pipeline Pattern** — Sequential processing stages (extract → normalize → serialize)
- **Strategy Pattern** — Different extraction strategies per file type

---

## Dependencies

**Core:**
- `pypdf` — PDF text extraction
- `python-docx` — DOCX reading
- `requests` — HTTP client for tool communication
- `pydantic` — Input/output validation
- `fastapi` — OCR service framework

**Optional:**
- `pymupdf` (fitz) — Faster PDF parsing for large files
- `pandas` — Enhanced Excel handling

See `requirements.txt` for pinned versions.

---

## Why This Matters

Before RAG. Before embeddings. Before any ML pipeline: **data cleaning.**

Most document tools are:
- **Too rigid** — only handle PDFs
- **Too heavy** — SaaS, expensive, cloud-only
- **Too simplistic** — extract text, skip normalization
- **Not designed for ML** — no structured output

This tool is:
- **Modular** — implement your own adapters
- **Local-first** — runs in your infrastructure
- **Production-ready** — Docker, logging, error handling
- **Built for scale** — batch processing, fault tolerance
- **ML-first** — JSON/JSONL output ready for embeddings

---

## Use Cases

### RAG Ingestion Pipeline

```
Raw Documents → [ocr-file-converter] → Clean JSON → Embeddings → Vector DB
```

### Compliance & Audit

```
Mixed document types → [ocr-file-converter] → Normalized archive
```

### Data Labeling Prep

```
Messy source files → [ocr-file-converter] → Clean JSON → Annotation tool
```

### Data Lake Processing

```
Heterogeneous sources → [ocr-file-converter] → Unified format → Warehouse
```

---

## Comparison with Similar Tools

| Feature | ocr-file-converter | Adobe SDK | Docling | pytesseract |
|---|:---:|:---:|:---:|:---:|
| Multi-format (PDF, Office, Images, Archives) | ✓ | ✗ | ✓ | ✗ |
| Local/self-hosted | ✓ | ✗ | ✓ | ✓ |
| Pluggable adapters | ✓ | ✗ | ✗ | ✗ |
| JSON/JSONL export | ✓ | ✗ | ✗ | ✗ |
| CLI + Library | ✓ | ✗ | ✓ | ✗ |
| Docker ready | ✓ | ✗ | ✓ | ✗ |
| Error recovery & logging | ✓ | ~ | ~ | ✗ |

---

## License

MIT

---

## Related Projects

- **RAG Agentic Graphiti** — Hybrid RAG engine that uses this for document ingestion
- **BPFW** — Architectural drift detection for AI-assisted development
- **pytest-readable** — Test specification reporting

---

## Contributing

PRs welcome. Areas for contribution:
- New tool adapters (different OCR engines, file extractors)
- Language & internationalization support
- Performance optimizations
- Additional output formats

Questions? Open a GitHub issue.
