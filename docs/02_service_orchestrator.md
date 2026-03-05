# FastAPI Orchestrator Service

## File: `app/routers/process.py` (endpoint) + `app/main.py` (app setup)

## Responsibility

Acts as the **workflow controller only**. Receives document uploads, orchestrates calls to each service in sequence, and returns the final result. Contains **zero business logic** — all logic lives in service files.

---

## Application Setup (`app/main.py`)

```python
from fastapi import FastAPI
from app.routers import process
from app.services.detector_service import DetectorService
from app.services.ocr_service import OCRService

app = FastAPI(
    title="Document Field Detection & Masking API",
    version="1.0.0",
    description="Hybrid document field detection using visual detection, OCR, and multimodal LLM reasoning"
)

# Register routers
app.include_router(process.router)

# Startup event: pre-load heavy models
@app.on_event("startup")
async def startup():
    DetectorService.load_model()
    OCRService.load_model()

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Run with: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

---

## Configuration (`app/config.py`)

All config values must be loaded from environment variables with sensible defaults:

```python
import os

class Settings:
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # File upload limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: list = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    # Image processing
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "1216"))

    # Detector
    DETECTOR_MODEL_PATH: str = os.getenv("DETECTOR_MODEL_PATH", "FFDNet-L (1).pt")
    DETECTOR_CONFIDENCE_THRESHOLD: float = float(os.getenv("DETECTOR_CONF_THRESHOLD", "0.3"))
    DETECTOR_NMS_IOU_THRESHOLD: float = float(os.getenv("DETECTOR_NMS_IOU", "0.1"))
    DETECTOR_AUGMENT: bool = os.getenv("DETECTOR_AUGMENT", "true").lower() == "true"
    DETECTOR_IMAGE_SIZE: int = int(os.getenv("DETECTOR_IMAGE_SIZE", "1216"))

    # OCR
    OCR_LANG: str = os.getenv("OCR_LANG", "en")
    OCR_USE_ANGLE_CLS: bool = os.getenv("OCR_USE_ANGLE_CLS", "true").lower() == "true"
    OCR_USE_GPU: bool = os.getenv("OCR_USE_GPU", "false").lower() == "true"

    # LLM (NVIDIA NIM)
    NIM_API_URL: str = os.getenv("NIM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    NIM_API_KEY: str = os.getenv("NIM_API_KEY", "")
    NIM_MODEL_NAME: str = os.getenv("NIM_MODEL_NAME", "qwen3.5-397b-a17b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "1"))

    # Validation
    VALIDATION_MAX_PAGE_COVERAGE: float = float(os.getenv("VALIDATION_MAX_PAGE_COVERAGE", "0.70"))
    VALIDATION_DUPLICATE_IOU_THRESHOLD: float = float(os.getenv("VALIDATION_DUPLICATE_IOU", "0.85"))
    VALIDATION_MIN_BOX_SIZE_PX: int = int(os.getenv("VALIDATION_MIN_BOX_SIZE_PX", "5"))

    # Masking
    MASK_COLOR_RGB: tuple = (0, 0, 0)  # Black redaction
    MASKED_OUTPUT_DIR: str = os.getenv("MASKED_OUTPUT_DIR", "output/masked")

settings = Settings()
```

---

## Primary Endpoint

### `POST /process-document`

**File: `app/routers/process.py`**

**Request:** Multipart file upload (PDF, PNG, JPG, JPEG, TIFF, BMP)

**Processing flow (pseudo-code):**

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf_service import PDFService
from app.services.detector_service import DetectorService
from app.services.ocr_service import OCRService
from app.services.llm_service import LLMService
from app.services.validation_service import ValidationService
from app.services.masking_service import MaskingService
from app.schemas.response_schema import DocumentResponse
from app.config import settings
import logging
import time
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/process-document", response_model=DocumentResponse)
async def process_document(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Processing file: {file.filename}")
    start_time = time.time()

    # 1. Validate upload
    validate_upload(file)

    # 2. Read file bytes
    file_bytes = await file.read()

    # 3. Convert to page images (handles both PDF and single images)
    page_images, pdf_document = PDFService.extract_pages(file_bytes, file.filename)
    # page_images: list of dict {"image": PIL.Image, "page_number": int, "original_size": (w, h)}

    # 4. Process each page
    all_page_results = []
    for page_info in page_images:
        page_result = process_single_page(request_id, page_info)
        all_page_results.append(page_result)

    # 5. Apply masking if input was PDF
    masked_pdf_path = None
    if file.filename.lower().endswith(".pdf") and pdf_document:
        masked_pdf_path = MaskingService.apply_masks(
            pdf_document, all_page_results, page_images
        )

    # 6. Return response
    total_time = time.time() - start_time
    logger.info(f"[{request_id}] Completed in {total_time:.2f}s")

    return DocumentResponse(
        request_id=request_id,
        filename=file.filename,
        total_pages=len(page_images),
        pages=all_page_results,
        masked_pdf_path=masked_pdf_path,
        processing_time_seconds=round(total_time, 3)
    )
```

**Per-page processing (inside router or as helper):**

```python
def process_single_page(request_id: str, page_info: dict) -> PageResult:
    image = page_info["image"]  # PIL.Image resized to max 1216px
    page_number = page_info["page_number"]

    # Step 1: Detect candidate fields
    proposals = DetectorService.detect(image)
    logger.info(f"[{request_id}] Page {page_number}: {len(proposals)} proposals")

    # Step 2: Extract OCR tokens
    ocr_result = OCRService.extract(image)

    # Step 3: Call LLM for reasoning
    try:
        llm_output = LLMService.reason(
            image=image,
            proposals=proposals,
            ocr_tokens=ocr_result.tokens,
            page_number=page_number
        )
    except Exception as e:
        logger.warning(f"[{request_id}] LLM failed for page {page_number}: {e}")
        llm_output = None

    # Step 4: Validate
    if llm_output:
        validated_fields = ValidationService.validate(
            fields=llm_output.fields,
            image_width=image.width,
            image_height=image.height
        )
    else:
        # Fallback: use detector proposals as-is (unclassified)
        validated_fields = ValidationService.fallback_from_proposals(
            proposals=proposals,
            image_width=image.width,
            image_height=image.height
        )

    return PageResult(
        page=page_number,
        fields=validated_fields,
        proposal_count=len(proposals),
        source="llm" if llm_output else "detector_fallback"
    )
```

---

## Pydantic Response Schema (`app/schemas/response_schema.py`)

```python
from pydantic import BaseModel
from typing import List, Optional

class FieldResult(BaseModel):
    type: str  # text_box | signature | checkbox | radio | initials
    bbox: List[float]  # [x1, y1, x2, y2] in resized image pixel coords
    confidence: float
    source: str  # detector | detector_corrected | llm_added

class PageResult(BaseModel):
    page: int
    fields: List[FieldResult]
    proposal_count: int
    source: str  # "llm" or "detector_fallback"

class DocumentResponse(BaseModel):
    request_id: str
    filename: str
    total_pages: int
    pages: List[PageResult]
    masked_pdf_path: Optional[str] = None
    processing_time_seconds: float
```

---

## Pydantic Request Schema (`app/schemas/request_schema.py`)

```python
from pydantic import BaseModel
from typing import List, Optional

class Proposal(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    label: str         # "TextBox", "ChoiceButton", "Signature"

class OCRToken(BaseModel):
    text: str
    bbox: List[float]  # [x1, y1, x2, y2]

class OCRResult(BaseModel):
    tokens: List[OCRToken]
    direction: str = "LTR"  # "LTR" or "RTL"

class LLMField(BaseModel):
    type: str
    bbox: List[float]
    confidence: float
    source: str

class LLMOutput(BaseModel):
    page: int
    fields: List[LLMField]
```

---

## Error Response Format

All errors must use this structure:

```python
class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None

# Usage:
raise HTTPException(
    status_code=400,
    detail={"error": "invalid_file", "detail": "File type .docx is not supported"}
)
```

---

## Upload Validation

```python
def validate_upload(file: UploadFile):
    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Unsupported file type: {ext}")

    # Check content type (basic)
    if file.content_type and not any(
        ct in file.content_type for ct in ["pdf", "image"]
    ):
        raise HTTPException(400, detail=f"Unsupported content type: {file.content_type}")
```

---

## Failure Handling Summary

| Stage | Failure Action |
|-------|----------------|
| File upload | Return 400 with error detail |
| PDF extraction | Return 500 with error detail |
| Detector | **Fail the request** (return 500) — detector is essential |
| OCR | Log warning, still call LLM with proposals only |
| LLM (1st attempt) | Retry once with correction prompt |
| LLM (2nd attempt) | Fall back to detector-only results |
| LLM invalid JSON | Retry once, then fall back to detector-only results |
| Validation | Always succeeds (filters bad data, returns valid subset) |
| Masking | Log error, return JSON without masked PDF path |