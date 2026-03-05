# Validation Layer

## File: `app/services/validation_service.py`

## Purpose

Prevent hallucination, enforce geometric and semantic correctness, and ensure only valid fields reach the masking engine. **LLM output is NEVER trusted blindly** — every field must pass all validation rules before being included in the final result.

---

## Configuration (from `app/config.py`)

```python
VALIDATION_MAX_PAGE_COVERAGE = 0.70     # Reject boxes covering >70% of page area
VALIDATION_DUPLICATE_IOU_THRESHOLD = 0.85  # Remove duplicate boxes with IoU > 0.85
VALIDATION_MIN_BOX_SIZE_PX = 5          # Reject boxes smaller than 5px in any dimension
```

---

## Allowed Field Types

```python
ALLOWED_FIELD_TYPES = {"text_box", "signature", "checkbox", "radio", "initials"}
ALLOWED_SOURCES = {"detector", "detector_corrected", "llm_added"}
```

---

## Implementation Specification

```python
import logging
from typing import List
from app.schemas.request_schema import LLMField, Proposal
from app.schemas.response_schema import FieldResult
from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_FIELD_TYPES = {"text_box", "signature", "checkbox", "radio", "initials"}
ALLOWED_SOURCES = {"detector", "detector_corrected", "llm_added"}


class ValidationService:

    @classmethod
    def validate(
        cls,
        fields: List[LLMField],
        image_width: int,
        image_height: int
    ) -> List[FieldResult]:
        """
        Apply all validation rules to LLM output fields.
        Returns only fields that pass every check.

        Args:
            fields: List of LLMField from LLM output
            image_width: width of resized image in pixels
            image_height: height of resized image in pixels

        Returns:
            List of validated FieldResult objects
        """
        original_count = len(fields)
        valid_fields = []

        for field in fields:
            # Rule 1: Validate field type
            if field.type not in ALLOWED_FIELD_TYPES:
                logger.debug(f"Rejected field: invalid type '{field.type}'")
                continue

            # Rule 2: Validate source tag
            if field.source not in ALLOWED_SOURCES:
                field.source = "detector"  # Fix silently rather than reject

            # Rule 3: Validate bbox format (must be 4 floats)
            if not cls._is_valid_bbox_format(field.bbox):
                logger.debug(f"Rejected field: invalid bbox format {field.bbox}")
                continue

            x1, y1, x2, y2 = field.bbox

            # Rule 4: Ensure x1 < x2 and y1 < y2
            if x1 >= x2 or y1 >= y2:
                logger.debug(f"Rejected field: degenerate bbox {field.bbox}")
                continue

            # Rule 5: Bbox must be inside image bounds (with small tolerance)
            if not cls._is_within_bounds(x1, y1, x2, y2, image_width, image_height):
                logger.debug(f"Rejected field: bbox outside image bounds {field.bbox}")
                continue

            # Rule 6: Clamp bbox to image bounds (for minor overflows)
            clamped_bbox = cls._clamp_bbox(x1, y1, x2, y2, image_width, image_height)

            # Rule 7: Minimum size threshold
            box_w = clamped_bbox[2] - clamped_bbox[0]
            box_h = clamped_bbox[3] - clamped_bbox[1]
            if box_w < settings.VALIDATION_MIN_BOX_SIZE_PX or box_h < settings.VALIDATION_MIN_BOX_SIZE_PX:
                logger.debug(f"Rejected field: too small ({box_w}x{box_h}px)")
                continue

            # Rule 8: Reject boxes covering >70% of page area
            page_area = image_width * image_height
            box_area = box_w * box_h
            if page_area > 0 and (box_area / page_area) > settings.VALIDATION_MAX_PAGE_COVERAGE:
                logger.debug(f"Rejected field: covers {box_area/page_area:.1%} of page")
                continue

            # Rule 9: Clamp confidence to [0.0, 1.0]
            confidence = max(0.0, min(1.0, field.confidence))

            valid_fields.append(FieldResult(
                type=field.type,
                bbox=clamped_bbox,
                confidence=confidence,
                source=field.source
            ))

        # Rule 10: Remove duplicates (IoU > threshold)
        deduplicated = cls._remove_duplicates(valid_fields)

        logger.info(
            f"Validation: {original_count} input → {len(valid_fields)} valid → {len(deduplicated)} after dedup"
        )
        return deduplicated

    @classmethod
    def fallback_from_proposals(
        cls,
        proposals: List[Proposal],
        image_width: int,
        image_height: int
    ) -> List[FieldResult]:
        """
        Convert raw FFDNet-L proposals to FieldResult when LLM is unavailable.
        Maps detector class labels to allowed field types:
          TextBox → text_box
          ChoiceButton → checkbox (default, since we can't distinguish without LLM)
          Signature → signature

        Args:
            proposals: List of Proposal from FFDNet-L detector (with label field)
            image_width: width of resized image in pixels
            image_height: height of resized image in pixels

        Returns:
            List of validated FieldResult objects
        """
        # Map FFDNet-L class labels to allowed field types
        LABEL_TO_TYPE = {
            "TextBox": "text_box",
            "ChoiceButton": "checkbox",  # Default to checkbox; LLM would split to radio
            "Signature": "signature",
        }

        fallback_fields = [
            LLMField(
                type=LABEL_TO_TYPE.get(p.label, "text_box"),
                bbox=p.bbox,
                confidence=p.confidence,
                source="detector"
            )
            for p in proposals
        ]
        return cls.validate(fallback_fields, image_width, image_height)

    # ─── Helper Methods ────────────────────────────────────────

    @classmethod
    def _is_valid_bbox_format(cls, bbox) -> bool:
        """Check bbox is a list/tuple of exactly 4 numbers."""
        if not isinstance(bbox, (list, tuple)):
            return False
        if len(bbox) != 4:
            return False
        return all(isinstance(v, (int, float)) for v in bbox)

    @classmethod
    def _is_within_bounds(
        cls, x1: float, y1: float, x2: float, y2: float,
        width: int, height: int, tolerance: float = 20.0
    ) -> bool:
        """
        Check if bbox is reasonably within image bounds.
        Allow small overflow (tolerance) for edge fields.
        Reject if completely outside.
        """
        if x2 < 0 or y2 < 0:
            return False
        if x1 > width + tolerance or y1 > height + tolerance:
            return False
        return True

    @classmethod
    def _clamp_bbox(
        cls, x1: float, y1: float, x2: float, y2: float,
        width: int, height: int
    ) -> List[float]:
        """Clamp bbox coordinates to image bounds."""
        return [
            max(0.0, x1),
            max(0.0, y1),
            min(float(width), x2),
            min(float(height), y2)
        ]

    @classmethod
    def _compute_iou(cls, box_a: List[float], box_b: List[float]) -> float:
        """
        Compute Intersection over Union between two bounding boxes.
        Both boxes in [x1, y1, x2, y2] format.
        """
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @classmethod
    def _remove_duplicates(cls, fields: List[FieldResult]) -> List[FieldResult]:
        """
        Remove duplicate fields based on IoU threshold.
        When two boxes overlap with IoU > threshold, keep the one with higher confidence.
        Uses greedy approach: sort by confidence descending, skip suppressed boxes.
        """
        if len(fields) <= 1:
            return fields

        # Sort by confidence descending
        sorted_fields = sorted(fields, key=lambda f: f.confidence, reverse=True)
        keep = []
        suppressed = set()

        for i, field_i in enumerate(sorted_fields):
            if i in suppressed:
                continue
            keep.append(field_i)

            for j in range(i + 1, len(sorted_fields)):
                if j in suppressed:
                    continue
                iou = cls._compute_iou(field_i.bbox, sorted_fields[j].bbox)
                if iou > settings.VALIDATION_DUPLICATE_IOU_THRESHOLD:
                    suppressed.add(j)

        return keep
```

---

## Validation Rules Summary

| # | Rule | Action |
|---|------|--------|
| 1 | Field type must be in allowed set | Reject |
| 2 | Source must be in allowed set | Fix silently to "detector" |
| 3 | Bbox must be [x1, y1, x2, y2] (4 numbers) | Reject |
| 4 | x1 < x2 AND y1 < y2 (non-degenerate) | Reject |
| 5 | Bbox must be within image bounds (±20px tolerance) | Reject |
| 6 | Clamp bbox to exact image bounds | Clamp |
| 7 | Minimum dimension ≥ 5px | Reject |
| 8 | Box area ≤ 70% of page area | Reject |
| 9 | Confidence clamped to [0.0, 1.0] | Clamp |
| 10 | Remove duplicate boxes (IoU > 0.85) | Keep higher confidence |

---

## Retry Logic (handled by orchestrator, not validation service)

The validation service itself **never calls the LLM**. Retry logic belongs in the orchestrator:

1. LLM returns invalid JSON → orchestrator retries with correction prompt
2. LLM still returns invalid JSON → orchestrator falls back to `fallback_from_proposals()`
3. LLM returns valid JSON but validation rejects all fields → result is empty fields list (not a failure)

---

## Critical Rule

**LLM output is never trusted blindly.** Every field from the LLM passes through ALL validation rules before reaching the masking engine or the final response.