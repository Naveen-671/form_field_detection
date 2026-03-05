# Security and Guardrails

## Scope

These rules apply to both Phase 1 (local) and Phase 2 (production). Implement all of these in Phase 1 code.

---

## 1. Input Validation (implement in `app/routers/process.py`)

### File Upload Restrictions

```python
# Maximum file size: 50 MB (configurable via MAX_FILE_SIZE_MB)
MAX_FILE_SIZE_MB = 50

# Allowed file extensions
ALLOWED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
```

### Validation Steps

1. **Check file extension** against allowed list → reject with 400 if invalid
2. **Check file size** → reject with 413 if exceeds limit
3. **Validate PDF structure** → try opening with PyMuPDF; if it fails, reject with 400 ("corrupt or invalid PDF")
4. **Sanitize filename** → strip path components, replace special characters; use only for logging (never for file system operations with user-provided names)

### Implementation

```python
import os
from fastapi import UploadFile, HTTPException
from app.config import settings

def validate_upload(file: UploadFile):
    """Validate uploaded file before processing."""
    # 1. Check extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )

    # 2. Check content type (basic sanity)
    if file.content_type:
        allowed_types = ["application/pdf", "image/png", "image/jpeg", "image/tiff", "image/bmp"]
        if not any(ct in file.content_type for ct in ["pdf", "image"]):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {file.content_type}"
            )

async def check_file_size(file: UploadFile):
    """Check file size after reading. Call after await file.read()."""
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB} MB"
        )
    await file.seek(0)  # Reset for subsequent reads
    return contents
```

---

## 2. LLM Safety (implement in `app/services/llm_service.py` and `app/services/validation_service.py`)

### Guardrails

| Rule | Enforcement Point |
|------|-------------------|
| JSON schema enforcement | `LLMService._parse_response()` + `ValidationService.validate()` |
| Reject non-JSON output | `LLMService._parse_response()` returns `None` |
| Timeout guard | `httpx.Client(timeout=30)` in `LLMService._call_nim()` |
| Strict temperature=0 | Hardcoded in `LLMService._call_nim()` payload |
| Retry limit | Max 1 retry in `LLMService.reason()` |
| Fallback on failure | Orchestrator falls back to detector-only output |

### JSON Schema Enforcement

The validation service enforces:
- Only allowed field types (`text_box`, `signature`, `checkbox`, `radio`, `initials`)
- Only allowed source tags (`detector`, `detector_corrected`, `llm_added`)
- Bbox must be exactly 4 numeric values
- Confidence must be 0.0–1.0
- No fields larger than 70% of page area
- No duplicate fields (IoU > 0.85)

---

## 3. Logging Rules (implement in all service files)

### What to Log

```python
import logging
logger = logging.getLogger(__name__)

# ✅ DO log:
logger.info(f"[{request_id}] Processing file: {filename}")
logger.info(f"[{request_id}] Page {page_num}: {proposal_count} proposals detected")
logger.info(f"[{request_id}] Page {page_num}: {field_count} validated fields")
logger.info(f"[{request_id}] LLM call took {elapsed:.2f}s")
logger.info(f"[{request_id}] Total processing time: {total:.2f}s")
logger.warning(f"[{request_id}] LLM returned invalid JSON, retrying")
logger.warning(f"[{request_id}] LLM failed after retry, falling back to detector")
logger.error(f"[{request_id}] Detector failed: {error}")
```

### What NEVER to Log

```python
# ❌ NEVER log in production:
# - Raw document content or file bytes
# - Page images or base64-encoded images
# - OCR token text content
# - Full LLM request/response payloads
# - User-identifiable information from documents
```

### Logging Configuration (in `app/main.py`)

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

---

## 4. Request Tracking

Every request must be assigned a UUID (`request_id`) at the start of processing. This ID must be:
- Included in all log messages
- Returned in the response body
- Used for correlating logs across pipeline stages