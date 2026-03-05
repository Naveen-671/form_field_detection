# AGENT BUILD INSTRUCTIONS
**Project:** Hybrid Document Field Detection & Masking System  
**Phase:** 1 – Local Testing Only  
**Runtime:** Python 3.10+ / FastAPI / Uvicorn  
**DO NOT skip any section. Read fully before generating code.**

---

# 1. PURPOSE

Build a modular FastAPI application that performs end-to-end document field detection and masking:

1. **PDF/Image ingestion** — accept multipart file upload (PDF, PNG, JPG, JPEG, TIFF, BMP)
2. **Page conversion** — extract pages from PDF as images (or handle single image input)
3. **Image normalization** — resize to max 1216px dimension, preserve aspect ratio
4. **Field proposal detection** — run FFDNet-L (YOLO11-based, fine-tuned for document forms) for high-recall candidate proposals
5. **OCR extraction** — run PaddleOCR to get text tokens with bounding boxes
6. **Multimodal LLM reasoning** — call NVIDIA NIM (Qwen3.5-397B-A17B) with image + proposals + OCR tokens to validate, classify, and recover fields
7. **Validation** — enforce strict rules on LLM output to prevent hallucination
8. **Deterministic masking** — apply black redaction rectangles on original PDF using PyMuPDF 
9. **Structured JSON response** — return per-page field data + optional masked PDF file path

This is a **Phase 1 local testing system**. Keep it clean, modular, and testable.

### DO NOT ADD:
- Docker / Docker Compose
- Kubernetes
- GPU optimization / CUDA tuning
- Distributed infrastructure
- Redis / Kafka / Celery / RabbitMQ
- Background workers / task queues
- Streaming responses
- WebSockets
- Async complexity beyond FastAPI's default

---

# 2. ARCHITECTURAL PRINCIPLES

1. **LLM must return STRICT JSON only.** No markdown, no code fences, no commentary.
2. **LLM never performs masking.** Masking is deterministic, coordinate-based, backend-only.
3. **Masking uses only validated output.** Never apply raw LLM output directly.
4. **Detector uses FFDNet-L defaults** (confidence 0.3, IoU 0.1, augment=True). Returns labelled proposals (TextBox / ChoiceButton / Signature); LLM refines types.
5. **Validation layer guards against hallucination.** Every LLM field passes 10 validation rules.
6. **Business logic belongs inside service files.** Routers only orchestrate.
7. **Modular separation.** Each service is independently replaceable without touching other files.
8. **No global mutable state** except model singletons loaded at application startup.
9. **All bounding boxes use pixel coordinates** on the resized image (max 1216px). Masking service maps back to original PDF coordinates.
10. **Fail gracefully.** If LLM fails → fallback to detector-only. If OCR fails → proceed without OCR. If detector fails → fail the request.

---

# 3. REQUIRED DIRECTORY STRUCTURE

Create **exactly** this structure. Do not add extra files, directories, or layers:

```
app/
├── __init__.py              # Empty
├── main.py                  # FastAPI app setup, startup events, health endpoint
├── config.py                # Settings class with all configuration (env vars + defaults)
├── routers/
│   ├── __init__.py          # Empty
│   └── process.py           # POST /process-document endpoint + upload validation
├── services/
│   ├── __init__.py          # Empty
│   ├── pdf_service.py       # PDF→images extraction, image resizing
│   ├── detector_service.py  # FFDNet-L detection (singleton model)
│   ├── ocr_service.py       # PaddleOCR text extraction (singleton engine)
│   ├── llm_service.py       # NVIDIA NIM API calls, prompt construction, response parsing
│   ├── validation_service.py # 10-rule validation of LLM output fields
│   └── masking_service.py   # PyMuPDF coordinate mapping + redaction application
├── schemas/
│   ├── __init__.py          # Empty
│   ├── request_schema.py    # Pydantic models for internal data flow
│   └── response_schema.py   # Pydantic models for API response
└── utils/
    ├── __init__.py          # Empty
    └── image_utils.py       # Image helper functions (if needed, otherwise empty)
```

**Additional project-level files to create:**

```
requirements.txt             # Python dependencies
FFDNet-L (1).pt              # FFDNet-L model weights (in project root, gitignored)
output/masked/               # Directory for masked PDF output (auto-created by masking_service)
.gitignore                   # Standard Python gitignore + models/ + output/
```

**Rules:**
- ALL `__init__.py` files must be empty (or just contain `pass` if preferred)
- Services contain all business logic
- Schemas contain Pydantic models only
- Routers contain endpoint definitions + upload validation only
- No business logic in routers
- No global state except model singletons in service classes

---

# 4. DEPENDENCIES (`requirements.txt`)

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
PyMuPDF>=1.23.0
Pillow>=10.0.0
paddleocr>=2.7.0
paddlepaddle>=2.5.0
ultralytics>=8.1.0            # YOLO11 support required for FFDNet-L
httpx>=0.25.0
python-multipart>=0.0.6
numpy>=1.24.0
```

---

# 5. CONFIGURATION (`app/config.py`)

All values must be loaded from environment variables with sensible defaults. Use a simple class (no Pydantic BaseSettings — keep it simple):

```python
import os

class Settings:
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # File upload
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: list = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    # Image processing
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "1216"))

    # Detector (FFDNet-L — YOLO11-based document field detector)
    DETECTOR_MODEL_PATH: str = os.getenv("DETECTOR_MODEL_PATH", "FFDNet-L (1).pt")
    DETECTOR_CONFIDENCE_THRESHOLD: float = float(os.getenv("DETECTOR_CONF_THRESHOLD", "0.3"))
    DETECTOR_NMS_IOU_THRESHOLD: float = float(os.getenv("DETECTOR_NMS_IOU", "0.1"))
    DETECTOR_AUGMENT: bool = os.getenv("DETECTOR_AUGMENT", "true").lower() == "true"
    DETECTOR_IMAGE_SIZE: int = int(os.getenv("DETECTOR_IMAGE_SIZE", "1216"))

    # OCR (PaddleOCR)
    OCR_LANG: str = os.getenv("OCR_LANG", "en")
    OCR_USE_ANGLE_CLS: bool = os.getenv("OCR_USE_ANGLE_CLS", "true").lower() == "true"
    OCR_USE_GPU: bool = os.getenv("OCR_USE_GPU", "false").lower() == "true"

    # LLM (NVIDIA NIM — OpenAI-compatible API)
    NIM_API_URL: str = os.getenv("NIM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    NIM_API_KEY: str = os.getenv("NIM_API_KEY", "")
    NIM_MODEL_NAME: str = os.getenv("NIM_MODEL_NAME", "qwen3.5-397b-a17b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "1"))

    # Validation thresholds
    VALIDATION_MAX_PAGE_COVERAGE: float = float(os.getenv("VALIDATION_MAX_PAGE_COVERAGE", "0.70"))
    VALIDATION_DUPLICATE_IOU_THRESHOLD: float = float(os.getenv("VALIDATION_DUPLICATE_IOU", "0.85"))
    VALIDATION_MIN_BOX_SIZE_PX: int = int(os.getenv("VALIDATION_MIN_BOX_SIZE_PX", "5"))

    # Masking
    MASK_COLOR_RGB: tuple = (0, 0, 0)  # Black
    MASKED_OUTPUT_DIR: str = os.getenv("MASKED_OUTPUT_DIR", "output/masked")

settings = Settings()
```

---

# 6. PYDANTIC SCHEMAS

## `app/schemas/request_schema.py` — Internal Data Flow Models

```python
from pydantic import BaseModel
from typing import List

class Proposal(BaseModel):
    """A detector proposal — candidate bounding box with class label."""
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords on resized image
    confidence: float       # 0.0–1.0
    label: str              # "TextBox", "ChoiceButton", or "Signature" (from FFDNet-L)

class OCRToken(BaseModel):
    """A single OCR text token with position."""
    text: str
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords on resized image

class OCRResult(BaseModel):
    """Full OCR extraction result for one page."""
    tokens: List[OCRToken]
    direction: str = "LTR"  # "LTR" or "RTL"

class LLMField(BaseModel):
    """A single field from LLM output (before validation)."""
    type: str               # text_box | signature | checkbox | radio | initials
    bbox: List[float]       # [x1, y1, x2, y2]
    confidence: float       # 0.0–1.0
    source: str             # detector | detector_corrected | llm_added

class LLMOutput(BaseModel):
    """Full LLM response for one page."""
    page: int
    fields: List[LLMField]
```

## `app/schemas/response_schema.py` — API Response Models

```python
from pydantic import BaseModel
from typing import List, Optional

class FieldResult(BaseModel):
    """A single validated field in the API response."""
    type: str               # text_box | signature | checkbox | radio | initials
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords on resized image
    confidence: float       # 0.0–1.0
    source: str             # detector | detector_corrected | llm_added

class PageResult(BaseModel):
    """Results for a single page."""
    page: int
    fields: List[FieldResult]
    proposal_count: int       # Number of detector proposals before LLM
    source: str               # "llm" or "detector_fallback"

class DocumentResponse(BaseModel):
    """Top-level API response."""
    request_id: str
    filename: str
    total_pages: int
    pages: List[PageResult]
    masked_pdf_path: Optional[str] = None
    processing_time_seconds: float
```

---

# 7. PROCESSING PIPELINE (`app/routers/process.py`)

When `POST /process-document` is called, execute these steps **in exact order**:

```
1. Validate upload (extension, size)
2. Read file bytes
3. PDFService.extract_pages(file_bytes, filename)
   → Returns: List[{"image": PIL.Image, "page_number": int, "original_size": (w,h)}]
   → Image is already resized to max 1216px
4. For each page:
   a. DetectorService.detect(image) → List[Proposal]
   b. OCRService.extract(image) → OCRResult
      (If OCR fails: log warning, set ocr_result = OCRResult(tokens=[], direction="LTR"))
   c. LLMService.reason(image, proposals, ocr_tokens, page_number) → Optional[LLMOutput]
      (If LLM fails: returns None)
   d. If LLMOutput exists:
        ValidationService.validate(fields, image_width, image_height) → List[FieldResult]
      Else:
        ValidationService.fallback_from_proposals(proposals, image_width, image_height) → List[FieldResult]
5. Aggregate all PageResult objects
6. If input was PDF:
   MaskingService.apply_masks(pdf_bytes, page_results, page_images) → Optional[str]
7. Return DocumentResponse
```

---

# 8. PDF SERVICE (`app/services/pdf_service.py`)

- Use **PyMuPDF** (`import fitz`) for PDF page extraction
- Render each PDF page to pixmap with `zoom=2.0` for quality
- Convert pixmap → PIL Image
- Resize to max dimension 1216px (preserve aspect ratio, use `Image.LANCZOS`)
- For single image input: wrap as 1-page document
- Return both resized image AND original page dimensions (needed for masking coordinate mapping)

**Key method signatures:**
```python
@classmethod
def extract_pages(cls, file_bytes: bytes, filename: str) -> Tuple[List[dict], Optional[bytes]]:
    """Returns (page_list, original_pdf_bytes_or_None)"""

@classmethod
def _resize_to_max_dimension(cls, image: Image.Image) -> Image.Image:
    """Resize so max(w,h) == MAX_IMAGE_DIMENSION. No-op if already smaller."""
```

---

# 9. DETECTOR SERVICE (`app/services/detector_service.py`)

- Use **FFDNet-L** — a YOLO11-based model fine-tuned for document form field detection
- Model source: `jbarrow/FFDNet-L` on HuggingFace (CommonForms dataset, 25M params)
- Model weights file: `FFDNet-L (1).pt` in the project root
- Load via `ultralytics.YOLO(path, task="detect")` — requires `ultralytics>=8.1.0`
- Load model once at startup via `load_model()` class method (called from `main.py` startup event)
- Keep as singleton — never reload per request

### FFDNet-L Class Mapping (hardcoded):
```python
CLASS_MAP = {0: "TextBox", 1: "ChoiceButton", 2: "Signature"}
```

### Inference Parameters (from config):
```python
results = cls._model.predict(
    source=image,
    conf=settings.DETECTOR_CONFIDENCE_THRESHOLD,   # default 0.3
    iou=settings.DETECTOR_NMS_IOU_THRESHOLD,        # default 0.1
    augment=settings.DETECTOR_AUGMENT,               # default True
    imgsz=settings.DETECTOR_IMAGE_SIZE,              # default 1216
    verbose=False,
)
```

### Box Extraction Pattern (from commonforms source):
```python
proposals = []
for result in results:
    for box in result.boxes.cpu().numpy():
        xyxy = box.xyxy[0].tolist()       # [x1, y1, x2, y2] pixel coords
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = CLASS_MAP.get(cls_id, "TextBox")
        proposals.append(Proposal(bbox=xyxy, confidence=conf, label=label))
```

- Output: `List[Proposal]` with `bbox=[x1,y1,x2,y2]` in pixel coords + `confidence` + `label`
- The `label` field carries the FFDNet-L class name through the pipeline to the LLM

**Key method signatures:**
```python
@classmethod
def load_model(cls): ...

@classmethod
def detect(cls, image: Image.Image) -> List[Proposal]: ...

@classmethod
def is_ready(cls) -> bool: ...
```

---

# 10. OCR SERVICE (`app/services/ocr_service.py`)

- Use **PaddleOCR** with `use_angle_cls=True`, `lang='en'`, `use_gpu=False`
- Load engine once at startup via `load_model()` class method
- Convert PIL Image → numpy array for PaddleOCR input
- PaddleOCR returns 4-point polygons → convert to axis-aligned bbox `[min_x, min_y, max_x, max_y]`
- **Do NOT clean or filter OCR output** — pass raw tokens to LLM
- Include simple reading direction heuristic (LTR vs RTL)

**Key method signatures:**
```python
@classmethod
def load_model(cls): ...

@classmethod
def extract(cls, image: Image.Image) -> OCRResult: ...

@classmethod
def is_ready(cls) -> bool: ...
```

---

# 11. LLM SERVICE (`app/services/llm_service.py`)

- Call NVIDIA NIM via **httpx** HTTP client (OpenAI-compatible chat completions API)
- Send: system prompt + multimodal user message (base64 JPEG image + text with proposals/tokens)
- Temperature MUST be 0.0
- Parse response JSON; strip code fences if LLM ignores instructions
- On invalid JSON: retry once with correction prompt
- On second failure: return `None` (orchestrator falls back to detector)

### SYSTEM PROMPT (use exactly this — do not modify):

```
You are a document layout reasoning engine.

You receive:
1. A full-page document image.
2. A list of detector proposals (candidate bounding boxes with confidence scores and class labels from FFDNet-L).
3. A list of OCR tokens (text strings with bounding boxes).

FFDNet-L detector class labels and their meanings:
- "TextBox" — any text input field. Map to "text_box".
- "ChoiceButton" — a checkbox OR radio button. Use visual shape to decide:
  - Square/rounded-square → "checkbox"
  - Circle/oval → "radio"
- "Signature" — a signature or initials region. Use size/label context to decide:
  - Large region or labelled "Signature" → "signature"
  - Small region or labelled "Initials" → "initials"

You must:
- Validate each detector proposal: confirm it is a real interactive form field, or reject it.
- Use the detector's class label as a strong hint, but refine the final type using visual context.
- Classify confirmed fields into exactly one of the allowed types.
- Add any interactive fields visible in the image that the detector missed.
- Remove decorative borders, logos, watermarks, and non-interactive visual elements.
- Output STRICT JSON only. No markdown. No code fences. No commentary. No explanations.
- If uncertain about a field, lower its confidence instead of omitting it or hallucinating.
- Every field must correspond to a visually identifiable interactive UI element in the image.

Allowed field types (use EXACTLY these strings):
- text_box
- signature
- checkbox
- radio
- initials

Required output JSON schema:
{
  "page": <page_number_integer>,
  "fields": [
    {
      "type": "<allowed_type>",
      "bbox": [x1, y1, x2, y2],
      "confidence": <float_0_to_1>,
      "source": "<source_tag>"
    }
  ]
}

Source tags:
- "detector" — proposal confirmed as-is
- "detector_corrected" — proposal confirmed but bbox or type was adjusted
- "llm_added" — new field the detector missed, identified by visual reasoning

Return ONLY the JSON object. Nothing else.
```

### NIM API Request Format:

```json
{
  "model": "qwen3.5-397b-a17b",
  "messages": [
    {"role": "system", "content": "<SYSTEM_PROMPT>"},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64>"}},
      {"type": "text", "text": "Page: 1\n\nDetector proposals (12 candidates):\n[{\"bbox\": [x1, y1, x2, y2], \"confidence\": 0.87, \"label\": \"TextBox\"}, ...]\n\nOCR tokens (45 tokens):\n[...]\n\nAnalyze the image and the data above. Return the validated fields as JSON."}
    ]}
  ],
  "temperature": 0.0,
  "max_tokens": 2048,
  "top_p": 0.8,
  "repetition_penalty": 1.1
}
```

**Key method signatures:**
```python
@classmethod
def reason(cls, image, proposals, ocr_tokens, page_number) -> Optional[LLMOutput]: ...

@classmethod
def _encode_image(cls, image: Image.Image) -> str: ...  # PIL → base64 JPEG

@classmethod
def _build_user_message(cls, image_b64, proposals, ocr_tokens, page_number) -> list: ...

@classmethod
def _call_nim(cls, user_content: list) -> Optional[str]: ...  # HTTP POST

@classmethod
def _parse_response(cls, response_text: str, page_number: int) -> Optional[LLMOutput]: ...
```

---

# 12. VALIDATION SERVICE (`app/services/validation_service.py`)

**LLM output is NEVER trusted blindly.** Apply ALL 10 rules in order:

| # | Rule | Action |
|---|------|--------|
| 1 | Field type must be in `{text_box, signature, checkbox, radio, initials}` | Reject |
| 2 | Source must be in `{detector, detector_corrected, llm_added}` | Fix to "detector" |
| 3 | Bbox must be exactly 4 numbers | Reject |
| 4 | x1 < x2 AND y1 < y2 (non-degenerate) | Reject |
| 5 | Bbox within image bounds (±20px tolerance) | Reject |
| 6 | Clamp bbox to exact image bounds | Clamp |
| 7 | Min dimension ≥ 5px in both width and height | Reject |
| 8 | Box area ≤ 70% of page area | Reject |
| 9 | Confidence clamped to [0.0, 1.0] | Clamp |
| 10 | Remove duplicates with IoU > 0.85 (keep higher confidence) | Suppress |

Also implement `fallback_from_proposals()` — converts raw detector proposals to FieldResult using label-to-type mapping, then runs through the same validation:

```python
LABEL_TO_TYPE = {
    "TextBox": "text_box",
    "ChoiceButton": "checkbox",   # default when LLM unavailable
    "Signature": "signature",
}
```

For each proposal: `type = LABEL_TO_TYPE.get(proposal.label, "text_box")`, `source = "detector"`.

**Key method signatures:**
```python
@classmethod
def validate(cls, fields, image_width, image_height) -> List[FieldResult]: ...

@classmethod
def fallback_from_proposals(cls, proposals, image_width, image_height) -> List[FieldResult]: ...

@classmethod
def _compute_iou(cls, box_a, box_b) -> float: ...

@classmethod
def _remove_duplicates(cls, fields) -> List[FieldResult]: ...
```

---

# 13. MASKING SERVICE (`app/services/masking_service.py`)

- Use **PyMuPDF** (`import fitz`)
- Map bounding boxes from resized image coordinates → original PDF coordinates using scale factors
- Apply `page.add_redact_annot(rect, fill=(0,0,0))` then `page.apply_redactions()`
- Save masked PDF to `output/masked/masked_<uuid>.pdf`
- Return absolute file path of saved PDF (or None on failure)

### Coordinate Mapping:
```python
scale_x = original_page_width / resized_image_width
scale_y = original_page_height / resized_image_height
pdf_rect = fitz.Rect(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
```

**Key method signatures:**
```python
@classmethod
def apply_masks(cls, pdf_bytes, page_results, page_images) -> Optional[str]: ...
```

---

# 14. APPLICATION SETUP (`app/main.py`)

```python
from fastapi import FastAPI
from app.routers import process
from app.services.detector_service import DetectorService
from app.services.ocr_service import OCRService
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI(
    title="Document Field Detection & Masking API",
    version="1.0.0",
    description="Hybrid document field detection using visual detection, OCR, and multimodal LLM reasoning"
)

app.include_router(process.router)

@app.on_event("startup")
async def startup():
    DetectorService.load_model()
    OCRService.load_model()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "detector_ready": DetectorService.is_ready(),
        "ocr_ready": OCRService.is_ready()
    }
```

Run with: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

---

# 15. ERROR HANDLING RULES

| Stage | On Failure | Action |
|-------|-----------|--------|
| File upload validation | Invalid type/size | Return HTTP 400 with error detail |
| PDF extraction | Corrupt PDF | Return HTTP 400 "Invalid or corrupt PDF" |
| Detector | Model not loaded / inference error | **Fail the request** — return HTTP 500 |
| OCR | Engine error | Log warning, proceed without OCR tokens |
| LLM (1st try) | Timeout / HTTP error / invalid JSON | Retry once with correction prompt |
| LLM (2nd try) | Still fails | Return `None` → orchestrator uses fallback |
| Validation | N/A (always succeeds) | Returns valid subset (may be empty) |
| Masking | PyMuPDF error | Log error, return JSON without `masked_pdf_path` |

---

# 16. LOGGING RULES

### DO Log:
- Request ID (UUID per request)
- Filename (original upload name)
- Time per stage (detector, OCR, LLM, validation, masking)
- Proposal count per page
- Validated field count per page
- LLM retry count
- Total processing time

### NEVER Log:
- Raw document content or file bytes
- Base64-encoded images
- OCR text content
- Full LLM request/response payloads
- User-identifiable information from documents

### Format:
```
2026-02-27 10:30:15 [INFO] app.routers.process: [abc123] Processing file: form.pdf
2026-02-27 10:30:16 [INFO] app.services.detector_service: Detected 15 proposals
2026-02-27 10:30:17 [INFO] app.services.ocr_service: OCR extracted 42 tokens, direction: LTR
2026-02-27 10:30:19 [WARNING] app.services.llm_service: LLM returned invalid JSON, retrying
2026-02-27 10:30:21 [INFO] app.services.validation_service: Validation: 18 input → 14 valid → 12 after dedup
2026-02-27 10:30:22 [INFO] app.routers.process: [abc123] Completed in 7.12s
```

---

# 17. SIMPLICITY CONSTRAINTS (MANDATORY — Phase 1)

**DO NOT add any of the following. Violations will require full rework:**

- ❌ Async workers / task queues (Celery, RQ, etc.)
- ❌ Message brokers (Redis, Kafka, RabbitMQ)
- ❌ Docker / Docker Compose files
- ❌ Kubernetes manifests
- ❌ Model parallelism / GPU tuning
- ❌ Streaming responses / WebSockets
- ❌ Database (SQLite, PostgreSQL, etc.)
- ❌ Authentication / OAuth middleware
- ❌ Rate limiting middleware
- ❌ Response caching layers
- ❌ Custom middleware stacks
- ❌ Abstract base classes or over-engineered patterns

**DO use:**
- ✅ Simple class methods with `@classmethod`
- ✅ Pydantic models for data validation
- ✅ `logging` module (stdlib)
- ✅ `uuid` for request IDs
- ✅ `time` for timing
- ✅ `os` for env vars and paths
- ✅ `httpx` for HTTP calls
- ✅ `json` for parsing

---

# 18. FINAL CHECKLIST

Before considering the implementation complete, verify:

- [ ] `uvicorn app.main:app` starts without errors
- [ ] `GET /health` returns `{"status": "ok", ...}`
- [ ] `GET /docs` shows Swagger UI with `/process-document` endpoint
- [ ] `POST /process-document` with a PDF returns valid `DocumentResponse` JSON
- [ ] `POST /process-document` with an image returns valid `DocumentResponse` JSON
- [ ] Detector model loads once at startup (check logs)
- [ ] OCR engine loads once at startup (check logs)
- [ ] LLM failure gracefully falls back to detector-only results
- [ ] Masked PDF is saved to `output/masked/` directory
- [ ] All response fields match the `DocumentResponse` Pydantic schema
- [ ] No raw document content appears in logs
- [ ] Invalid file types return HTTP 400
- [ ] Every file has proper imports and no circular dependencies
- [ ] All `__init__.py` files exist in every package directory

---

# 19. CROSS-REFERENCE

For detailed implementation specifics, refer to the companion architecture documents:

| Document | Content |
|----------|---------|
| `01_system_overview.md` | Architecture overview, dependencies, coordinate system, field types |
| `02_service_orchestrator.md` | FastAPI setup, endpoint flow, schema definitions, error handling |
| `03_detector_service.md` | FFDNet-L implementation, model loading, inference code |
| `04_ocr_service.md` | PaddleOCR implementation, polygon→bbox conversion |
| `05_llm_service_nim.md` | NIM API format, system prompt, image encoding, retry logic |
| `06_validation_layer.md` | 10 validation rules, IoU computation, deduplication algorithm |
| `07_masking_engine.md` | Coordinate mapping math, PyMuPDF redaction, PDF service |
| `08_infrastructure_and_scaling.md` | Future Phase 2 plans (DO NOT implement) |
| `09_security_and_guardrails.md` | Input validation, logging rules, guardrails |
| `10_operational_playbook.md` | Setup, startup, testing, troubleshooting |