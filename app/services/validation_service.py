import logging
from typing import List

from app.config import settings
from app.schemas.request_schema import LLMField, Proposal
from app.schemas.response_schema import FieldResult

logger = logging.getLogger(__name__)

ALLOWED_FIELD_TYPES = {"text_box", "signature", "checkbox", "radio", "initials", "date"}
ALLOWED_SOURCES = {"detector", "detector_corrected", "detector_merged", "llm_added", "ocr_pattern", "ocr_inferred"}

# FFDNet-L label → field type mapping used when LLM is unavailable
LABEL_TO_TYPE = {
    "TextBox": "text_box",
    "ChoiceButton": "checkbox",  # Default; LLM would split this to radio when available
    "Signature": "signature",
}


class ValidationService:

    @classmethod
    def validate(
        cls,
        fields: List[LLMField],
        image_width: int,
        image_height: int,
    ) -> List[FieldResult]:
        """
        Apply all 10 validation rules to LLM output fields.

        Rules applied in order:
         1. Field type in allowed set — Reject
         2. Source in allowed set — Fix to "detector"
         3. Bbox must be 4 numbers — Reject
         4. x1 < x2 AND y1 < y2 — Reject
         5. Bbox within image bounds (±20px tolerance) — Reject
         6. Clamp bbox to image bounds — Clamp
         7. Minimum dimension ≥ 5px — Reject
         8. Box area ≤ 70% of page — Reject
         9. Confidence clamped to [0.0, 1.0] — Clamp
        10. Remove duplicates with IoU > threshold — Suppress

        Returns:
            List of validated FieldResult objects
        """
        original_count = len(fields)
        valid_fields: List[FieldResult] = []

        for field in fields:

            # Rule 1: validate field type
            if field.type not in ALLOWED_FIELD_TYPES:
                logger.debug(f"Rejected field: invalid type '{field.type}'")
                continue

            # Rule 2: validate source tag (fix silently rather than reject)
            source = field.source if field.source in ALLOWED_SOURCES else "detector"

            # Rule 3: bbox must be exactly 4 numbers
            if not cls._is_valid_bbox_format(field.bbox):
                logger.debug(f"Rejected field: invalid bbox format {field.bbox}")
                continue

            x1, y1, x2, y2 = float(field.bbox[0]), float(field.bbox[1]), float(field.bbox[2]), float(field.bbox[3])

            # Rule 4: non-degenerate (x1 < x2, y1 < y2)
            if x1 >= x2 or y1 >= y2:
                logger.debug(f"Rejected field: degenerate bbox [{x1},{y1},{x2},{y2}]")
                continue

            # Rule 5: within image bounds with ±20px tolerance
            if not cls._is_within_bounds(x1, y1, x2, y2, image_width, image_height):
                logger.debug(f"Rejected field: bbox completely outside image bounds [{x1},{y1},{x2},{y2}]")
                continue

            # Rule 6: clamp to exact image bounds
            clamped = cls._clamp_bbox(x1, y1, x2, y2, image_width, image_height)
            cx1, cy1, cx2, cy2 = clamped

            # Rule 7: minimum size ≥ 5px in both dimensions
            box_w = cx2 - cx1
            box_h = cy2 - cy1
            if box_w < settings.VALIDATION_MIN_BOX_SIZE_PX or box_h < settings.VALIDATION_MIN_BOX_SIZE_PX:
                logger.debug(f"Rejected field: too small ({box_w:.1f}x{box_h:.1f}px)")
                continue

            # Rule 8: area ≤ 70% of page
            page_area = image_width * image_height
            box_area = box_w * box_h
            if page_area > 0 and (box_area / page_area) > settings.VALIDATION_MAX_PAGE_COVERAGE:
                logger.debug(f"Rejected field: covers {box_area/page_area:.1%} of page area")
                continue

            # Rule 9: clamp confidence to [0.0, 1.0]
            confidence = max(0.0, min(1.0, float(field.confidence)))

            valid_fields.append(FieldResult(
                type=field.type,
                bbox=clamped,
                confidence=confidence,
                source=source,
            ))

        # Rule 10: remove duplicates with IoU > threshold or containment
        deduplicated = cls._remove_duplicates(valid_fields)

        # Rule 11: merge fragmented character-cells into single text_box fields
        merged = cls._merge_character_cells(deduplicated)

        # Rule 12: final dedup after merging (merged boxes may overlap)
        final = cls._remove_duplicates(merged)

        logger.info(
            f"Validation: {original_count} input → {len(valid_fields)} valid "
            f"→ {len(deduplicated)} dedup → {len(merged)} merged → {len(final)} final"
        )
        return final

    @classmethod
    def fallback_from_proposals(
        cls,
        proposals: List[Proposal],
        image_width: int,
        image_height: int,
    ) -> List[FieldResult]:
        """
        Convert raw FFDNet-L detector proposals to FieldResult when LLM is unavailable.
        Maps: TextBox→text_box, ChoiceButton→checkbox, Signature→signature.

        Runs the same validation pipeline as validate().
        """
        fallback_fields = [
            LLMField(
                type=LABEL_TO_TYPE.get(p.label, "text_box"),
                bbox=p.bbox,
                confidence=p.confidence,
                source="detector",
            )
            for p in proposals
        ]
        return cls.validate(fallback_fields, image_width, image_height)

    # ─── Helper Methods ──────────────────────────────────────────────────────

    @classmethod
    def _is_valid_bbox_format(cls, bbox) -> bool:
        """Check bbox is a list/tuple of exactly 4 numeric values."""
        if not isinstance(bbox, (list, tuple)):
            return False
        if len(bbox) != 4:
            return False
        return all(isinstance(v, (int, float)) for v in bbox)

    @classmethod
    def _is_within_bounds(
        cls,
        x1: float, y1: float, x2: float, y2: float,
        width: int, height: int,
        tolerance: float = 20.0,
    ) -> bool:
        """
        Return True if the bbox is at least partially within the image (+tolerance).
        Rejects boxes that are completely outside the image.
        """
        if x2 < -tolerance or y2 < -tolerance:
            return False
        if x1 > width + tolerance or y1 > height + tolerance:
            return False
        return True

    @classmethod
    def _clamp_bbox(
        cls,
        x1: float, y1: float, x2: float, y2: float,
        width: int, height: int,
    ) -> List[float]:
        """Clamp bbox coordinates to exact image bounds."""
        return [
            max(0.0, x1),
            max(0.0, y1),
            min(float(width), x2),
            min(float(height), y2),
        ]

    @classmethod
    def _compute_iou(cls, box_a: List[float], box_b: List[float]) -> float:
        """Compute Intersection over Union of two [x1,y1,x2,y2] bounding boxes."""
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])

        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @classmethod
    def _remove_duplicates(cls, fields: List[FieldResult]) -> List[FieldResult]:
        """
        Greedy NMS-style deduplication:
        Sort by confidence descending, suppress lower-confidence boxes that:
          - have IoU > threshold with a kept box, OR
          - are fully contained inside a kept box (containment > 85%)
        """
        if len(fields) <= 1:
            return fields

        sorted_fields = sorted(fields, key=lambda f: f.confidence, reverse=True)
        keep: List[FieldResult] = []
        suppressed: set = set()

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
                    continue
                # Also suppress if one is largely contained inside the other
                contain_j_in_i = cls._containment_ratio(sorted_fields[j].bbox, field_i.bbox)
                contain_i_in_j = cls._containment_ratio(field_i.bbox, sorted_fields[j].bbox)
                if contain_j_in_i > 0.60 or contain_i_in_j > 0.60:
                    suppressed.add(j)

        return keep

    @classmethod
    def _containment_ratio(cls, inner: List[float], outer: List[float]) -> float:
        """What fraction of inner's area falls inside outer?"""
        ix1 = max(inner[0], outer[0])
        iy1 = max(inner[1], outer[1])
        ix2 = min(inner[2], outer[2])
        iy2 = min(inner[3], outer[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
        if inner_area <= 0:
            return 0.0
        return inter / inner_area

    @classmethod
    def _merge_character_cells(cls, fields: List[FieldResult]) -> List[FieldResult]:
        """
        Merge fragmented character-cell detections into single text_box fields.

        FFDNet-L detects individual character cells in form text boxes as
        separate ~20x20px text_box proposals. This method detects horizontal
        runs of small adjacent boxes at similar y-coordinates and merges
        them into one unified text_box.

        Strategy:
        1. Separate fields into "small" (width < CELL_MAX_W) and "large".
        2. Group small fields by y-band (rounded to BAND_SIZE px).
        3. Within each band, sort by x1 and find horizontal runs where
           the gap between consecutive boxes is < GAP_THRESHOLD.
        4. Runs of >= MIN_RUN_LENGTH boxes are merged into a single text_box.
        5. Remaining small boxes are kept as-is.
        """
        CELL_MAX_W   = 35   # max width to be considered a character cell
        CELL_MAX_H   = 35   # max height
        BAND_SIZE    = 25   # y-band grouping tolerance (px)
        GAP_THRESHOLD = 15  # max horizontal gap between adjacent cells
        MIN_RUN_LENGTH = 3  # minimum cells to trigger a merge

        large: List[FieldResult] = []
        small: List[FieldResult] = []

        for f in fields:
            w = f.bbox[2] - f.bbox[0]
            h = f.bbox[3] - f.bbox[1]
            if w <= CELL_MAX_W and h <= CELL_MAX_H and f.type == "text_box":
                small.append(f)
            else:
                large.append(f)

        if len(small) < MIN_RUN_LENGTH:
            return fields  # not enough small boxes to merge

        # Group by y-band
        bands: dict = {}
        for f in small:
            y_mid = (f.bbox[1] + f.bbox[3]) / 2
            band_key = round(y_mid / BAND_SIZE) * BAND_SIZE
            bands.setdefault(band_key, []).append(f)

        merged_fields: List[FieldResult] = list(large)
        total_merged_from = 0

        for band_key in sorted(bands.keys()):
            band = sorted(bands[band_key], key=lambda f: f.bbox[0])

            # Find horizontal runs
            runs: List[List[FieldResult]] = []
            current_run = [band[0]]
            for i in range(1, len(band)):
                prev = current_run[-1]
                curr = band[i]
                gap = curr.bbox[0] - prev.bbox[2]  # x-gap between boxes
                y_diff = abs((curr.bbox[1] + curr.bbox[3]) / 2 - (prev.bbox[1] + prev.bbox[3]) / 2)
                if gap < GAP_THRESHOLD and y_diff < BAND_SIZE:
                    current_run.append(curr)
                else:
                    runs.append(current_run)
                    current_run = [curr]
            runs.append(current_run)

            for run in runs:
                if len(run) >= MIN_RUN_LENGTH:
                    # Merge into a single text_box
                    x1 = min(f.bbox[0] for f in run)
                    y1 = min(f.bbox[1] for f in run)
                    x2 = max(f.bbox[2] for f in run)
                    y2 = max(f.bbox[3] for f in run)
                    avg_conf = sum(f.confidence for f in run) / len(run)
                    merged_fields.append(FieldResult(
                        type="text_box",
                        bbox=[x1, y1, x2, y2],
                        confidence=avg_conf,
                        source="detector_merged",
                    ))
                    total_merged_from += len(run)
                    logger.debug(
                        f"Merged {len(run)} cells → text_box "
                        f"[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]"
                    )
                else:
                    # Keep individual small boxes
                    merged_fields.extend(run)

        logger.info(
            f"Cell merger: {len(small)} small cells → "
            f"{total_merged_from} merged into groups, "
            f"{len(small) - total_merged_from} kept individual. "
            f"Total fields: {len(merged_fields)}"
        )
        return merged_fields
