# OCR Service

## File: `app/services/ocr_service.py`

## Purpose

Extract **semantic anchors** (text labels and their positions) from document page images. These tokens provide contextual grounding for the LLM to reason about what each detected region represents (e.g., a label "Signature" near a box tells the LLM it's a signature field).

---

## Engine: PaddleOCR

- **Package**: `paddleocr` (pip install paddleocr) + `paddlepaddle` (pip install paddlepaddle)
- **Configuration**: `use_angle_cls=True`, `lang='en'`, `use_gpu=False` (Phase 1)
- **Model download**: PaddleOCR downloads models automatically on first run to `~/.paddleocr/`

---

## Configuration (from `app/config.py`)

```python
OCR_LANG = "en"
OCR_USE_ANGLE_CLS = True
OCR_USE_GPU = False
```

---

## Implementation Specification

```python
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from typing import List
from app.schemas.request_schema import OCRToken, OCRResult
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class OCRService:
    _engine: PaddleOCR = None

    @classmethod
    def load_model(cls):
        """Load PaddleOCR engine once at startup. Called from main.py startup event."""
        if cls._engine is None:
            logger.info("Loading PaddleOCR engine")
            cls._engine = PaddleOCR(
                use_angle_cls=settings.OCR_USE_ANGLE_CLS,
                lang=settings.OCR_LANG,
                use_gpu=settings.OCR_USE_GPU,
                show_log=False
            )
            logger.info("PaddleOCR engine loaded successfully")

    @classmethod
    def extract(cls, image: Image.Image) -> OCRResult:
        """
        Run OCR on a resized page image.

        Args:
            image: PIL.Image already resized to max 1216px dimension

        Returns:
            OCRResult with list of OCRToken objects and detected direction.
        """
        if cls._engine is None:
            raise RuntimeError("OCR engine not loaded. Call load_model() first.")

        # Convert PIL Image to numpy array (PaddleOCR expects numpy/path)
        image_np = np.array(image)

        # Run OCR
        results = cls._engine.ocr(image_np, cls=settings.OCR_USE_ANGLE_CLS)

        tokens = []
        if results and results[0]:
            for line in results[0]:
                # line format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                points = line[0]  # 4-point polygon
                text = line[1][0]
                # confidence = line[1][1]  # OCR confidence (not used downstream)

                # Convert 4-point polygon to axis-aligned bounding box
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]  # [x1, y1, x2, y2]

                tokens.append(OCRToken(
                    text=text,
                    bbox=bbox
                ))

        # Detect reading direction (heuristic: compare x-positions of first and last tokens)
        direction = cls._detect_direction(tokens)

        logger.info(f"OCR extracted {len(tokens)} tokens, direction: {direction}")
        return OCRResult(tokens=tokens, direction=direction)

    @classmethod
    def _detect_direction(cls, tokens: List[OCRToken]) -> str:
        """
        Simple heuristic for reading direction.
        If the majority of tokens have right-to-left flow, return 'RTL'.
        Default to 'LTR'.
        """
        if len(tokens) < 2:
            return "LTR"

        # Compare average x-position of first half vs second half of tokens
        mid = len(tokens) // 2
        first_half_avg_x = sum(t.bbox[0] for t in tokens[:mid]) / mid
        second_half_avg_x = sum(t.bbox[0] for t in tokens[mid:]) / (len(tokens) - mid)

        # If second half starts further left, likely RTL
        if second_half_avg_x < first_half_avg_x - 50:  # 50px threshold
            return "RTL"
        return "LTR"

    @classmethod
    def is_ready(cls) -> bool:
        return cls._engine is not None
```

---

## Input / Output Contract

### Input
- A PIL `Image.Image` object already resized to max dimension 1216px

### Output
```json
{
  "tokens": [
    {
      "text": "Full Name",
      "bbox": [45.0, 120.5, 180.3, 145.2]
    },
    {
      "text": "Date of Birth",
      "bbox": [45.0, 200.0, 210.5, 225.8]
    },
    {
      "text": "Signature",
      "bbox": [45.0, 500.0, 160.0, 525.0]
    }
  ],
  "direction": "LTR"
}
```

- `text`: raw OCR text (do NOT clean, normalize, or filter — LLM reasons over raw tokens)
- `bbox`: `[x1, y1, x2, y2]` in **pixel coordinates** on the resized image (same coordinate space as detector proposals)
- `direction`: `"LTR"` or `"RTL"`

---

## Critical Rules

1. **Do NOT pre-clean OCR output.** No spell-correction, no filtering, no deduplication. The LLM must reason over raw tokens to understand document layout.
2. **Singleton pattern**: Load engine once at startup. Never reload per-request.
3. **Coordinate consistency**: OCR bounding boxes must be in the same coordinate space as detector proposals (pixel coordinates on the resized image).
4. **PaddleOCR polygon → axis-aligned bbox**: PaddleOCR returns 4-point polygons. Convert to `[min_x, min_y, max_x, max_y]` axis-aligned bounding boxes.
5. **Empty results are valid**: If OCR finds no text (e.g., blank page), return `OCRResult(tokens=[], direction="LTR")`.
6. **Graceful failure**: If OCR throws an exception, the orchestrator should log it and still proceed with LLM using proposals only (OCR is helpful but not mandatory).

---

## Performance Target

- < 300ms per page on CPU
- OCR is CPU-bound and does not require GPU for Phase 1