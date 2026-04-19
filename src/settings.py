"""OCR File Converter - Configuración centralizada."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# === URLs DE SERVICIOS ===
TOOL_OCR_URL = os.getenv("TOOL_OCR_URL", "http://localhost:8000")
TOOL_OFFICE_URL = os.getenv("TOOL_OFFICE_URL", "http://localhost:9000")
TOOL_FILEEXTRACTOR_URL = os.getenv("TOOL_FILEEXTRACTOR_URL", "http://localhost:9001")

# === HABILITAR/DESHABILITAR FEATURES ===
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() in ("true", "1", "yes")
ENABLE_OFFICE_CONVERSION = os.getenv("ENABLE_OFFICE_CONVERSION", "true").lower() in ("true", "1", "yes")
ENABLE_EXTRACTOR_EXTRACTION = os.getenv("ENABLE_EXTRACTOR_EXTRACTION", "true").lower() in ("true", "1", "yes")

# === CONFIGURACIÓN OCR ===
OCR_DEFAULT_LANGUAGE = os.getenv("OCR_DEFAULT_LANGUAGE", "eng")

# === TIMEOUTS Y REINTENTOS ===
TOOL_REQUEST_TIMEOUT = int(os.getenv("TOOL_REQUEST_TIMEOUT", "120"))
TOOL_OFFICE_TIMEOUT = int(os.getenv("TOOL_OFFICE_TIMEOUT", "60"))
TOOL_CONNECT_RETRIES = int(os.getenv("TOOL_CONNECT_RETRIES", "3"))
TOOL_CONNECT_RETRY_DELAY = float(os.getenv("TOOL_CONNECT_RETRY_DELAY", "2.0"))

# === DIRECTORIOS ===
PREPROCESSING_WORK_DIR = os.getenv("PREPROCESSING_WORK_DIR", "/tmp/rag_preprocessing")
ARCHIVE_MAX_SIZE_MB = int(os.getenv("ARCHIVE_MAX_SIZE_MB", "500"))

# === LOGGING ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# === PDF CACHE ===
PDF_CACHE_ENABLED = os.getenv("RAG_PDF_CACHE_ENABLED", "false").lower() in ("true", "1", "yes")
PDF_CACHE_TTL = int(os.getenv("RAG_PDF_CACHE_TTL", "2592000"))  # 30 days
