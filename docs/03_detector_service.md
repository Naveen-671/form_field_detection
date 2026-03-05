# Detector Service

## File: `app/services/detector_service.py`

## Purpose

High recall detection of **candidate** form fields on document page images. FFDNet-L is specifically fine-tuned for form field detection and outputs 3 classes: **TextBox**, **ChoiceButton**, and **Signature**. The LLM reasoning layer downstream further refines these (e.g., splitting ChoiceButton into checkbox vs radio).

---

## Model: FFDNet-L (Form Field Detection Network — Large)

- **Source**: [jbarrow/FFDNet-L on Hugging Face](https://huggingface.co/jbarrow/FFDNet-L)
- **Architecture**: YOLO11-based object detector, 25M parameters
- **Training data**: [CommonForms dataset](https://arxiv.org/abs/2509.16506) — large diverse dataset of document form fields
- **Trained resolution**: 1216px
- **Package**: `ultralytics>=8.1.0` (loaded via `YOLO(path, task="detect")`)
- **Model file**: `FFDNet-L (1).pt` in the project root directory
- **3 detection classes**:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | `TextBox` | Text input fields, text areas, form fields |
| 1 | `ChoiceButton` | Checkboxes and radio buttons (LLM splits these) |
| 2 | `Signature` | Signature capture areas |

---

## Configuration (from `app/config.py`)

```python
DETECTOR_MODEL_PATH = "FFDNet-L (1).pt"  # Path to FFDNet-L weights
DETECTOR_CONFIDENCE_THRESHOLD = 0.3      # Default from commonforms (proven optimal)
DETECTOR_NMS_IOU_THRESHOLD = 0.1         # Aggressive NMS (from commonforms source)
DETECTOR_AUGMENT = True                  # Test-time augmentation for better recall
DETECTOR_IMAGE_SIZE = 1216               # Trained resolution
```

---

## Implementation Specification

```python
from ultralytics import YOLO
from PIL import Image
from typing import List
from app.schemas.request_schema import Proposal
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# FFDNet-L class ID to label mapping
CLASS_MAP = {0: "TextBox", 1: "ChoiceButton", 2: "Signature"}


class DetectorService:
    _model: YOLO = None

    @classmethod
    def load_model(cls):
        """Load FFDNet-L model once at application startup. Called from main.py startup event."""
        if cls._model is None:
            logger.info(f"Loading FFDNet-L model from {settings.DETECTOR_MODEL_PATH}")
            cls._model = YOLO(settings.DETECTOR_MODEL_PATH, task="detect")
            logger.info("FFDNet-L model loaded successfully")

    @classmethod
    def detect(cls, image: Image.Image) -> List[Proposal]:
        """
        Run FFDNet-L detection on a resized page image.

        Args:
            image: PIL.Image already resized to max 1216px dimension

        Returns:
            List of Proposal objects with bbox [x1, y1, x2, y2] in pixel coords,
            confidence score, and class label.
        """
        if cls._model is None:
            raise RuntimeError("FFDNet-L model not loaded. Call load_model() first.")

        # Run inference — matching commonforms inference.py settings
        results = cls._model.predict(
            source=image,
            conf=settings.DETECTOR_CONFIDENCE_THRESHOLD,
            iou=settings.DETECTOR_NMS_IOU_THRESHOLD,
            augment=settings.DETECTOR_AUGMENT,
            imgsz=settings.DETECTOR_IMAGE_SIZE,
            verbose=False,
            device=settings.DETECTOR_DEVICE if hasattr(settings, 'DETECTOR_DEVICE') else "cpu"
        )

        proposals = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.cpu().numpy():
                xyxy = box.xyxy[0].tolist()       # [x1, y1, x2, y2] pixel coords
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = CLASS_MAP.get(cls_id, "TextBox")  # Default to TextBox if unknown

                proposals.append(Proposal(
                    bbox=xyxy,
                    confidence=conf,
                    label=label
                ))

        logger.info(f"FFDNet-L detected {len(proposals)} proposals")
        return proposals

    @classmethod
    def is_ready(cls) -> bool:
        return cls._model is not None
```

---

## Input / Output Contract

### Input
- A PIL `Image.Image` object already resized to max dimension 1216px (resizing is done by `pdf_service.py` before calling detector)

### Output
```json
{
  "proposals": [
    {
      "bbox": [120.5, 340.2, 450.8, 375.1],
      "confidence": 0.87,
      "label": "TextBox"
    },
    {
      "bbox": [50.0, 500.0, 70.0, 520.0],
      "confidence": 0.72,
      "label": "ChoiceButton"
    },
    {
      "bbox": [100.0, 600.0, 500.0, 700.0],
      "confidence": 0.91,
      "label": "Signature"
    }
  ]
}
```

- `bbox`: `[x1, y1, x2, y2]` in **pixel coordinates** on the resized image
- `confidence`: float 0.0–1.0
- `label`: one of `"TextBox"`, `"ChoiceButton"`, `"Signature"` — the detector's class prediction

---

## Design Rules

1. **Singleton pattern**: Load model once at startup via `load_model()`. Never reload per-request.
2. **Keep model in memory**: Do not unload between requests.
3. **No per-request initialization**: Model must already be in memory when `detect()` is called.
4. **Include class labels**: FFDNet-L outputs class predictions — include them in proposals for the LLM to use as context.
5. **Use `.cpu().numpy()`**: Convert box tensors to CPU numpy before extracting values (matches commonforms source code pattern).
6. **Test-time augmentation**: `augment=True` by default for better recall (as used in commonforms).
7. **Thread safety**: YOLO predict is thread-safe for inference. No mutex needed for Phase 1.

---

## Performance Target

- < 200ms per page on GPU
- < 1s per page on CPU with augmentation (acceptable for Phase 1 local testing)

---

## Health Check

```python
@classmethod
def is_ready(cls) -> bool:
    return cls._model is not None
```