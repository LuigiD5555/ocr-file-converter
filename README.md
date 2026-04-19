# OCR File Converter

Pipeline completo de procesamiento de documentos:
- **Extracción de texto**: OCR (Tesseract), PDFs, Office docs (DOCX, XLSX, PPTX)
- **Normalización**: Limpieza y estandarización de texto extraído
- **Serialización**: Exporta a TXT, JSON, o JSONL estructurado
- **Arquitectura modular**: Fácil de extender con nuevos procesadores o formatos

## Estructura

```
ocr_file_converter/
├── src/
│   ├── cli.py                        # CLI: Interfaz de línea de comandos
│   ├── preprocessor.py               # Extractor: OCR, Office, Archives → TXT
│   ├── normalizer.py                 # Normalización y serialización (JSON, JSONL)
│   ├── tool_adapters.py              # Adaptadores para servicios HTTP
│   ├── logging.py                    # Configuración de logging
│   ├── settings.py                   # Configuración centralizada
│   └── __init__.py                   # Inicialización del paquete
├── ocr_service/
│   ├── app.py                        # Servicio FastAPI OCR con Tesseract
│   └── requirements.txt              # Dependencias del servicio
├── main.py                           # Entry point (ejecuta src.cli)
├── requirements.txt                  # Dependencias del CLI
├── Dockerfile                        # Container para servicio OCR
├── docker-compose.yml                # Orquestador de servicios
└── README.md                         # Este archivo
```

## Instalación

### 1. Crear entorno Python

```bash
cd ocr_file_converter
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias principales

```bash
pip install -r requirements.txt
```

Si necesitas una version específica del conda env `rag`:
```bash
conda activate rag
cd ocr_file_converter
pip install -r requirements.txt
```

### 3. Ejecutar servicio OCR

El servicio OCR necesita estar corriendo en background. Hay dos opciones:

#### Opción A: Docker Compose (recomendado)

```bash
docker-compose up -d
```

#### Opción B: Docker directo

```bash
docker build -t ocr-service .
docker run -d \
  --name ocr-service \
  -p 8000:8000 \
  -v /tmp:/work \
  ocr-service
```

#### Opción C: Python directo

```bash
pip install -r ocr_service/requirements.txt
python ocr_service/app.py
```

El servicio estará disponible en `http://localhost:8000`.

### 4. Verificar conectividad OCR

```bash
curl http://localhost:8000/healthz
# Debe responder: {"status":"ok","service":"tool-ocr"}
```

## Uso

### Procesar un archivo individual

```bash
# Procesar documento y guardar como TXT normalizado
python main.py documento.pdf

# Guardar como JSON estructurado
python main.py documento.pdf --format json

# Guardar como JSONL (line-delimited JSON, útil para embeddings)
python main.py documento.pdf --format jsonl

# Con configuración personalizada
python main.py archivo.pdf --ocr-url http://ocr-service:8000 --format json --verbose

# Procesar sin OCR (solo conversión de formatos)
python main.py documento.docx --no-ocr
```

### Procesar un directorio completo

```bash
# Procesar todos los archivos en un directorio
python main.py /ruta/a/documentos

# Guardar resultados en otro directorio
python main.py /ruta/a/documentos -o /ruta/salida

# Procesar directorio como JSON
python main.py /ruta/a/documentos --format json -o ./output

# Procesar recursivamente sin OCR
python main.py /ruta/a/documentos --no-ocr --format jsonl -o ./results
```

### Programáticamente

```python
from src.preprocessor import FilePreprocessor
from src.normalizer import TextSerializer
from pathlib import Path

# Procesar documento
preprocessor = FilePreprocessor(
    ocr_url="http://localhost:8000",
    enable_ocr=True,
)
result_path = preprocessor.preprocess(Path("documento.pdf"))

# Normalizar y serializar a JSON
if result_path:
    text = result_path.read_text()
    serializer = TextSerializer()
    serializer.save_json(text, Path("output.json"), source="documento.pdf")
```

## Formatos soportados

El converter procesa automáticamente los siguientes formatos:

**Imágenes (OCR con Tesseract):**
- `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`

**Documentos Office:**
- `.docx`, `.doc` (Word)
- `.xlsx`, `.xls` (Excel)
- `.pptx`, `.ppt` (PowerPoint)

**PDFs:**
- `.pdf` (detecta automáticamente si son escaneados y aplica OCR)

## Componentes principales

### FilePreprocessor (`src/preprocessor.py`)

Orquestador central que gestiona:
- **OCR** para imágenes y PDFs escaneados (via `OcrToolAdapter`)
- **Conversión Office** (.docx, .xlsx, .pptx → .txt)
- **Extracción de archivos** (.zip, .tar.gz, etc.)

**Métodos públicos:**
```python
preprocessor.preprocess(file_path)        # Procesa y retorna ruta del resultado
preprocessor.perform_ocr(file_path)       # Fuerza OCR en archivo
preprocessor.get_status()                 # Status de herramientas
```

### TextNormalizer y TextSerializer (`src/normalizer.py`)

Normalización y serialización de texto extraído:
- **TextNormalizer**: Limpia espacios, caracteres especiales, formatos
- **TextSerializer**: Convierte a JSON/JSONL con validación Pydantic

Modelos Pydantic:
- `NormalizedDocument`: Documento con metadatos
- `DocumentChunk`: Chunk de documento para JSONL
- `DocumentMetadata`: Metadata estructurado

### ToolAdapters (`src/tool_adapters.py`)

Adaptadores que comunican con servicios externos via HTTP:
- **OcrToolAdapter**: Comunica con servicio OCR (POST `/ocr`)
- **OfficeToolAdapter**: Comunica con herramienta Office (POST `/convert`)
- **ArchiveToolAdapter**: Comunica con extractor (POST `/extract`)

## Configuración

Editar `src/settings.py` o `.env` para ajustar:

```python
# URLs de servicios
TOOL_OCR_URL = "http://localhost:8000"
TOOL_OFFICE_URL = "http://localhost:9000"
TOOL_FILEEXTRACTOR_URL = "http://localhost:9001"

# Habilitar/deshabilitar features
ENABLE_OCR = True
ENABLE_OFFICE_CONVERSION = True
ENABLE_EXTRACTOR_EXTRACTION = True

# OCR
OCR_DEFAULT_LANGUAGE = "eng"  # Tesseract language code

# Timeouts
TOOL_REQUEST_TIMEOUT = 120  # segundos
TOOL_CONNECT_RETRIES = 3
TOOL_CONNECT_RETRY_DELAY = 2.0

# Preprocessing
PREPROCESSING_WORK_DIR = "/tmp/rag_preprocessing"
ARCHIVE_MAX_SIZE_MB = 500
```

## Formatos soportados

### OCR (requiere servicio OCR)
- Imágenes: PNG, JPG, JPEG, TIFF, BMP
- PDFs escaneados (detectados automáticamente)

### Conversión Office (requiere servicio Office)
- `.docx`, `.doc` → texto
- `.xlsx`, `.xls` → texto
- `.pptx`, `.ppt` → texto

### Extracción de archivos (requiere servicio Extractor)
- `.zip`, `.7z`, `.tar`, `.tar.gz`, `.tar.xz`, `.tar.bz2`

## Telemetría/Monitoreo

El módulo `src/core/telemetry.py` proporciona hooks para monitoreo:

```python
from src.core.telemetry import emit_error, set_error_reporter

# Registrar handler personalizado
def my_error_handler(error, component, operation, extra):
    print(f"{component}.{operation}: {error}")

set_error_reporter(my_error_handler)

# Ahora los errores en OCR, conversión, etc. pasarán por tu handler
```

## Troubleshooting

### OCR service unavailable
```
⚠ Tool-ocr unavailable: ...
```
**Solución:** Verificar que el servicio OCR está corriendo:
```bash
curl http://localhost:8000/healthz
```

### PDF loading timeout
```
PDF load timeout after 300s
```
**Solución:** Aumentar timeout en `settings.py`:
```python
TOOL_REQUEST_TIMEOUT = 600
```

### PDF scanned but no OCR text
El PDF se detectó como escaneado pero OCR no extrajo texto significativo. Posibles causas:
- Imagen muy baja calidad
- OCR language incorrecto (ver `OCR_DEFAULT_LANGUAGE`)
- Archivo corrupto

### Importaciones faltantes
```bash
pip install -r requirements.txt
```

## Dependencias principales

- **pypdf** - Extracción de texto de PDFs
- **fitz (pymupdf)** - Parser rápido de PDFs (opcional, para >500 páginas)
- **python-docx** - Lectura de DOCX
- **requests** - HTTP client para comunicar con servicios
- **fastapi** - Framework para servicio OCR
- **pydantic** - Validación de request/response

## Variables de entorno

```bash
# Configuración base
export TOOL_OCR_URL="http://ocr-service:8000"
export ENABLE_OCR=true

# Para servicio OCR
export WORK_DIR="/var/ocr_tmp"
export PORT=8000

# Logging
export LOG_LEVEL="INFO"
```

## Notas de desarrollo

- El código está extraído del RAG-Agentic-Graphiti, mantiene estructura y compatibilidad
- `settings.py` se importa como `from src.conf import settings`
- Todas las dependencias en `src/core` son de lectura (no mutación de estado global)
- El logger se obtiene vía `get_logger(__name__)` desde `src.workflows.query.audit`

## Licencia

Mismo que RAG-Agentic-Graphiti
