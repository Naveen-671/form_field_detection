"""
Spatial Analysis Service  — 3-stage aware
==========================================

Responsibilities:

1. **OCR Gap-Fill (Stage 2)** — find OCR tokens NOT covered by Stage 1 fields.
   Detect both keyword indicators AND structural patterns ([], _____, [   ]).
   Every candidate is checked against *existing* fields before being added.

2. **Structural Pattern Detection** — look for checkbox-like ( [] ), empty
   text-box ( [     ] ), and underline ( _____ ) patterns in OCR text.

3. **Coverage Check** — `is_field_covered()` so that no stage produces
   fields that overlap with detections from a previous stage.

4. **Intermediate Image Rendering** — draw coloured overlays on a PIL image
   so the LLM can *see* what has already been masked (Stage 3 input).

5. **LLM Dedup** — after the LLM returns new fields, reject any that
   overlap with existing detections from Stages 1+2.
"""

import logging
import math
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from app.schemas.request_schema import LLMField, OCRToken, Proposal
from app.schemas.response_schema import FieldResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  FIELD-INDICATOR KEYWORDS
# ──────────────────────────────────────────────────────────────────────────────
FIELD_INDICATORS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bsignature\b", re.I),              "signature"),
    (re.compile(r"\binitial[s]?\b", re.I),             "initials"),
    (re.compile(r"\bdate\b", re.I),                    "date"),
    (re.compile(r"\bplace\b", re.I),                   "text_box"),
    (re.compile(r"\bmobile\b", re.I),                  "text_box"),
    (re.compile(r"\bphone\b", re.I),                   "text_box"),
    (re.compile(r"\btel\b", re.I),                     "text_box"),
    (re.compile(r"\be[-\s]?mail\b", re.I),             "text_box"),
    (re.compile(r"\bpan\b", re.I),                     "text_box"),
    (re.compile(r"\baadhaar\b", re.I),                 "text_box"),
    (re.compile(r"\baccount\s*no\b", re.I),            "text_box"),
    (re.compile(r"\baddress\b", re.I),                 "text_box"),
    (re.compile(r"\bname\b", re.I),                    "text_box"),
    (re.compile(r"\bpincode\b", re.I),                 "text_box"),
    (re.compile(r"\bcity\b", re.I),                    "text_box"),
    (re.compile(r"\bstate\b", re.I),                   "text_box"),
    (re.compile(r"\bcountry\b", re.I),                 "text_box"),
    (re.compile(r"\bnominee\b", re.I),                 "text_box"),
    (re.compile(r"\boccupation\b", re.I),              "text_box"),
    (re.compile(r"\bincome\b", re.I),                  "text_box"),
    (re.compile(r"\bfather\b", re.I),                  "text_box"),
    (re.compile(r"\bspouse\b", re.I),                  "text_box"),
    (re.compile(r"\bdob\b|birth\b", re.I),             "text_box"),
    (re.compile(r"\bgender\b", re.I),                  "checkbox"),
    (re.compile(r"\bmale\b", re.I),                    "checkbox"),
    (re.compile(r"\bfemale\b", re.I),                  "checkbox"),
    (re.compile(r"\bmarried\b", re.I),                 "checkbox"),
    (re.compile(r"\bsingle\b", re.I),                  "checkbox"),
    (re.compile(r"\byes\b", re.I),                     "checkbox"),
    (re.compile(r"\bno\b", re.I),                      "checkbox"),
]

# Structural patterns in OCR text that indicate form fields
_PATTERN_EMPTY_BRACKET  = re.compile(r"^\s*\[\s*\]\s*$")          # "[ ]" or "[]"
_PATTERN_WIDE_BRACKET   = re.compile(r"^\s*\[[\s_]{2,}\]\s*$")   # "[     ]" or "[___]"
_PATTERN_UNDERLINE      = re.compile(r"^[_]{3,}$")                # "______"
_PATTERN_DASHES         = re.compile(r"^[-]{4,}$")                # "------"
_PATTERN_DOTS           = re.compile(r"^[.]{4,}$")                # "......"

COVERAGE_MARGIN = 20.0
FILL_RIGHT_PX = 200
FILL_BELOW_PX = 30

# Coverage thresholds
COVERAGE_IOU_THRESHOLD = 0.30        # IoU above this = "already covered"
COVERAGE_CONTAINMENT_THRESHOLD = 0.6  # >60% of candidate inside existing = covered


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _bbox_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _center(b: List[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _center_dist(a: List[float], b: List[float]) -> float:
    ca, cb = _center(a), _center(b)
    return math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def _overlap_area(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _containment_of(candidate: List[float], existing: List[float]) -> float:
    """What fraction of candidate's area falls inside existing?"""
    cand_area = _bbox_area(candidate)
    if cand_area <= 0:
        return 0.0
    return _overlap_area(candidate, existing) / cand_area


def _token_covered_by_any(tok: OCRToken, boxes: List[List[float]], margin: float = COVERAGE_MARGIN) -> bool:
    tok_area = _bbox_area(tok.bbox)
    if tok_area <= 0:
        return False
    expanded = [
        [b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin]
        for b in boxes
    ]
    for eb in expanded:
        overlap = _overlap_area(tok.bbox, eb)
        if overlap / tok_area > 0.5:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# 3.  COVERAGE CHECK — prevent cross-stage duplicates
# ──────────────────────────────────────────────────────────────────────────────

def is_field_covered(
    candidate_bbox: List[float],
    existing_fields: List[FieldResult],
    iou_threshold: float = COVERAGE_IOU_THRESHOLD,
    containment_threshold: float = COVERAGE_CONTAINMENT_THRESHOLD,
) -> bool:
    """
    Return True if candidate_bbox is already covered by any existing field.
    Covered means either:
      - IoU >= iou_threshold, OR
      - >containment_threshold of candidate area falls inside an existing field
    """
    for f in existing_fields:
        if _iou(candidate_bbox, f.bbox) >= iou_threshold:
            return True
        if _containment_of(candidate_bbox, f.bbox) >= containment_threshold:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# 4.  OCR STRUCTURAL PATTERN DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_ocr_patterns(
    ocr_tokens: List[OCRToken],
    existing_fields: List[FieldResult],
) -> List[FieldResult]:
    """
    Scan OCR tokens for structural patterns that indicate form fields:
      - "[]" or "[ ]"       -> checkbox
      - "[     ]" or "[___]" -> text_box
      - "_____"             -> text_box (underline fill-in)
      - "------"            -> text_box (dash fill-in)

    Each detected field is checked against existing_fields before adding.
    Returns new FieldResult objects (source="ocr_pattern").
    """
    new_fields: List[FieldResult] = []

    for tok in ocr_tokens:
        text = tok.text.strip()
        if not text:
            continue

        field_type = None

        if _PATTERN_EMPTY_BRACKET.match(text):
            field_type = "checkbox"
        elif _PATTERN_WIDE_BRACKET.match(text):
            field_type = "text_box"
        elif _PATTERN_UNDERLINE.match(text):
            field_type = "text_box"
        elif _PATTERN_DASHES.match(text):
            field_type = "text_box"
        elif _PATTERN_DOTS.match(text):
            field_type = "text_box"

        if field_type is None:
            continue

        # Check if this region is already covered by Stage 1
        if is_field_covered(tok.bbox, existing_fields):
            logger.debug(
                f"OCR pattern '{text}' already covered at "
                f"[{tok.bbox[0]:.0f},{tok.bbox[1]:.0f},{tok.bbox[2]:.0f},{tok.bbox[3]:.0f}]"
            )
            continue

        # Also check against already-added pattern fields
        if is_field_covered(tok.bbox, new_fields):
            continue

        new_fields.append(FieldResult(
            type=field_type,
            bbox=list(tok.bbox),
            confidence=0.50,
            source="ocr_pattern",
        ))
        logger.debug(
            f"OCR pattern: '{text}' -> [{field_type}] at "
            f"[{tok.bbox[0]:.0f},{tok.bbox[1]:.0f},{tok.bbox[2]:.0f},{tok.bbox[3]:.0f}]"
        )

    # Also detect checkbox sequences: adjacent small bracket tokens in same y-band
    _merge_checkbox_sequences(new_fields, ocr_tokens, existing_fields)

    logger.info(f"OCR pattern detection: {len(new_fields)} new fields from structural patterns")
    return new_fields


def _merge_checkbox_sequences(
    new_fields: List[FieldResult],
    ocr_tokens: List[OCRToken],
    existing_fields: List[FieldResult],
) -> None:
    """
    Find sequences of bracket-like tokens at similar y-coordinates
    (e.g. "[]  []  []  []") and ensure each is detected as a checkbox.
    Modifies new_fields in-place.
    """
    bracket_tokens = []
    for tok in ocr_tokens:
        t = tok.text.strip()
        if t in ("[]", "[ ]", "[", "]", "\u2610", "\u2611", "\u2612", "\u25FB", "\u25FC", "\u25A0", "\u25A1"):
            bracket_tokens.append(tok)

    if len(bracket_tokens) < 2:
        return

    BAND = 15
    bands: Dict[int, List[OCRToken]] = {}
    for tok in bracket_tokens:
        y_mid = (tok.bbox[1] + tok.bbox[3]) / 2
        band_key = round(y_mid / BAND) * BAND
        bands.setdefault(band_key, []).append(tok)

    all_existing = list(existing_fields) + list(new_fields)

    for _band_key, tokens in bands.items():
        for tok in tokens:
            bbox = list(tok.bbox)
            if not is_field_covered(bbox, all_existing):
                field = FieldResult(
                    type="checkbox",
                    bbox=bbox,
                    confidence=0.45,
                    source="ocr_pattern",
                )
                new_fields.append(field)
                all_existing.append(field)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  OCR GAP-FILL — keyword-based inference (Stage 2)
# ──────────────────────────────────────────────────────────────────────────────

def find_ocr_gaps(
    proposals: List[Proposal],
    ocr_tokens: List[OCRToken],
    existing_fields: List[FieldResult],
    image_width: int,
    image_height: int,
) -> List[FieldResult]:
    """
    Identify OCR tokens NOT covered by any Stage 1 field.
    For tokens matching field-indicator patterns, create inferred fields.

    Every inferred field is checked against existing_fields before adding.
    Returns new FieldResult objects (source="ocr_inferred").
    """
    if not ocr_tokens:
        return []

    det_boxes = [p.bbox for p in proposals]
    uncovered: List[OCRToken] = []

    for tok in ocr_tokens:
        if not _token_covered_by_any(tok, det_boxes):
            uncovered.append(tok)

    logger.info(
        f"OCR gap-fill: {len(ocr_tokens)} total tokens, "
        f"{len(uncovered)} not covered by detector"
    )

    inferred: List[FieldResult] = []
    all_existing = list(existing_fields)

    for tok in uncovered:
        matched_type = _match_field_indicator(tok.text)
        if matched_type is None:
            continue

        candidate = _infer_fill_region(tok, matched_type, image_width, image_height)
        if candidate is None:
            continue

        # Check against ALL existing fields (Stage 1 + already-inferred)
        if is_field_covered(candidate, all_existing):
            logger.debug(f"OCR gap-fill: skipping already-covered region for '{tok.text}'")
            continue

        field = FieldResult(
            type=matched_type,
            bbox=candidate,
            confidence=0.45,
            source="ocr_inferred",
        )
        inferred.append(field)
        all_existing.append(field)
        logger.debug(
            f"OCR gap-fill: inferred [{matched_type}] from \"{tok.text}\" "
            f"-> [{candidate[0]:.0f},{candidate[1]:.0f},{candidate[2]:.0f},{candidate[3]:.0f}]"
        )

    logger.info(f"OCR gap-fill: created {len(inferred)} inferred fields")
    return inferred


def _match_field_indicator(text: str) -> Optional[str]:
    for pattern, field_type in FIELD_INDICATORS:
        if pattern.search(text):
            return field_type
    return None


def _infer_fill_region(
    tok: OCRToken,
    field_type: str,
    img_w: int,
    img_h: int,
) -> Optional[List[float]]:
    x1, y1, x2, y2 = tok.bbox
    w = x2 - x1
    h = y2 - y1

    if field_type == "signature":
        fx1 = x1
        fy1 = y2 + 2
        fx2 = min(x1 + max(w * 2.5, 250), img_w)
        fy2 = min(fy1 + max(h * 3, 50), img_h)
        return [fx1, fy1, fx2, fy2]

    elif field_type == "initials":
        fx1 = x1
        fy1 = y2 + 2
        fx2 = min(x1 + max(w * 1.5, 100), img_w)
        fy2 = min(fy1 + max(h * 2, 30), img_h)
        return [fx1, fy1, fx2, fy2]

    elif field_type == "date":
        # Date fill region to the right of the label
        fx1 = x2 + 5
        fy1 = y1
        fx2 = min(fx1 + max(FILL_RIGHT_PX, 180), img_w)
        fy2 = y1 + max(h, FILL_BELOW_PX)
        if fx2 - fx1 < 20:
            fx1 = x1
            fy1 = y2 + 2
            fx2 = min(x1 + max(w * 2, FILL_RIGHT_PX), img_w)
            fy2 = min(fy1 + max(h, FILL_BELOW_PX), img_h)
        return [fx1, fy1, fx2, fy2]

    elif field_type == "checkbox":
        box_size = max(h, 15)
        fx1 = max(x1 - box_size - 5, 0)
        fy1 = y1
        fx2 = x1 - 2
        fy2 = y1 + box_size
        if fx2 <= fx1 + 5:
            return None
        return [fx1, fy1, fx2, fy2]

    else:
        fx1 = x2 + 5
        fy1 = y1
        fx2 = min(fx1 + FILL_RIGHT_PX, img_w)
        fy2 = y1 + max(h, FILL_BELOW_PX)
        if fx2 - fx1 < 20:
            fx1 = x1
            fy1 = y2 + 2
            fx2 = min(x1 + max(w * 2, FILL_RIGHT_PX), img_w)
            fy2 = min(fy1 + max(h, FILL_BELOW_PX), img_h)
        return [fx1, fy1, fx2, fy2]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  STAGE 2 ENTRY POINT — combines pattern + keyword detection
# ──────────────────────────────────────────────────────────────────────────────

def run_stage2(
    proposals: List[Proposal],
    ocr_tokens: List[OCRToken],
    stage1_fields: List[FieldResult],
    image_width: int,
    image_height: int,
) -> List[FieldResult]:
    """
    Stage 2: OCR-based gap-fill.

    1. Detect structural patterns ([], _____, etc.) not covered by Stage 1
    2. Detect keyword-based fields not covered by Stage 1 or pattern fields
    3. Return ONLY the new fields (caller merges with Stage 1)
    """
    # Step A: structural pattern detection
    pattern_fields = detect_ocr_patterns(ocr_tokens, stage1_fields)

    # Step B: keyword-based gap-fill (now aware of Stage 1 + pattern fields)
    combined_existing = list(stage1_fields) + pattern_fields
    keyword_fields = find_ocr_gaps(
        proposals, ocr_tokens, combined_existing, image_width, image_height
    )

    all_new = pattern_fields + keyword_fields
    logger.info(
        f"Stage 2 total: {len(all_new)} new fields "
        f"({len(pattern_fields)} pattern + {len(keyword_fields)} keyword)"
    )
    return all_new


# ──────────────────────────────────────────────────────────────────────────────
# 6b.  RECLASSIFY FIELDS BY OCR CONTEXT
# ──────────────────────────────────────────────────────────────────────────────

# Patterns that trigger reclassification of nearby text_box fields
_RECLASSIFY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bsignature\b|\bsigned\b|\bsign\b|\bauthori[sz]ed\b", re.I), "signature"),
    (re.compile(
        r"\bprepared\s+by\b|\bapproved\s+by\b|\bverified\s+by\b"
        r"|\breviewed\s+by\b|\bwitnessed\s+by\b|\baccept(?:ed)?\s+by\b",
        re.I,
    ), "signature"),
    (re.compile(r"\bdate\b|^date", re.I), "date"),  # "Date", "Service Completion Date", "Datelz" (OCR noise)
]

# Adjacency parameters
_RECLASS_HORIZ_GAP = 250   # label up to 250 px left of field
_RECLASS_VERT_GAP = 50     # label up to 50 px above field
_RECLASS_BAND_TOL = 15     # vertical-band alignment tolerance
_COLUMN_MAX_X_DRIFT = 60   # max x-drift for column-header matching


def reclassify_by_ocr_context(
    fields: List[FieldResult],
    ocr_tokens: List[OCRToken],
) -> List[FieldResult]:
    """
    Post-process fields: use OCR label context to correct types.

    Two modes:
      - **Column header**: label is above 2+ fields in the same x-range →
        reclassify ALL fields in that column.
      - **Row label**: label is left of or above a single row → reclassify
        only the closest adjacent text_box.

    Only reclassifies text_box fields.
    """
    if not ocr_tokens:
        return fields

    # Collect every OCR token that matches a reclassification pattern
    label_hits: List[Tuple[OCRToken, str]] = []
    for tok in ocr_tokens:
        for pattern, target_type in _RECLASSIFY_PATTERNS:
            if pattern.search(tok.text):
                label_hits.append((tok, target_type))
                break  # first pattern wins per token

    if not label_hits:
        return fields

    logger.info(
        f"Reclassify: {len(label_hits)} OCR label tokens matched "
        f"reclassification patterns"
    )

    assignments: Dict[int, str] = {}   # field_index -> target_type
    claimed: set = set()

    for tok, target_type in label_hits:
        # Try column-header detection first
        col_fields = _find_column_fields(tok, fields)
        if len(col_fields) >= 2:
            for idx in col_fields:
                if fields[idx].type == "text_box" and idx not in assignments:
                    assignments[idx] = target_type
                    claimed.add(idx)
            logger.info(
                f"Reclassify (column header): \"{tok.text}\" → "
                f"{len(col_fields)} fields → {target_type}"
            )
            continue

        # Fallback: row-label mode — closest adjacent field
        best_idx: Optional[int] = None
        best_dist = float("inf")
        for i, field in enumerate(fields):
            if field.type != "text_box" or i in claimed:
                continue
            dist = _adjacency_score(tok, field)
            if dist is not None and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            assignments[best_idx] = target_type
            claimed.add(best_idx)
            logger.info(
                f"Reclassify (row label): \"{tok.text}\" → "
                f"[{fields[best_idx].bbox[0]:.0f},{fields[best_idx].bbox[1]:.0f},"
                f"{fields[best_idx].bbox[2]:.0f},{fields[best_idx].bbox[3]:.0f}] → {target_type}"
            )

    # ── Propagation: spread type to overlapping text_box neighbours ───────
    _propagate_to_overlapping(fields, assignments)

    # Build final list
    result: List[FieldResult] = []
    for i, field in enumerate(fields):
        if i in assignments:
            result.append(FieldResult(
                type=assignments[i],
                bbox=field.bbox,
                confidence=field.confidence,
                source=field.source,
            ))
        else:
            result.append(field)

    return result


def _find_column_fields(
    tok: OCRToken,
    fields: List[FieldResult],
) -> List[int]:
    """
    Find fields that are IN THE SAME COLUMN below the token.
    A column match requires: field's center-x is close to the token's
    center-x AND the field is below the token.
    Returns list of field indices.
    """
    tx1, ty1, tx2, ty2 = tok.bbox
    t_xmid = (tx1 + tx2) / 2

    col_indices: List[int] = []
    for i, f in enumerate(fields):
        if f.type != "text_box":
            continue
        fx1, fy1, fx2, fy2 = f.bbox
        f_xmid = (fx1 + fx2) / 2

        # Field must be below the token
        if fy1 < ty2 - 5:
            continue

        # Center-x must be close (same column)
        if abs(f_xmid - t_xmid) < _COLUMN_MAX_X_DRIFT:
            col_indices.append(i)

    return col_indices


def _adjacency_score(
    tok: OCRToken,
    field: FieldResult,
) -> Optional[float]:
    """
    Return the distance if the token is adjacent to the field, else None.
    "Adjacent" means the label is directly to the LEFT or ABOVE the field.
    """
    fx1, fy1, fx2, fy2 = field.bbox
    f_ymid = (fy1 + fy2) / 2
    f_xmid = (fx1 + fx2) / 2
    f_h = fy2 - fy1

    tx1, ty1, tx2, ty2 = tok.bbox
    t_ymid = (ty1 + ty2) / 2
    t_xmid = (tx1 + tx2) / 2

    # Case 1 — label is to the LEFT (same horizontal band)
    if tx2 <= fx1 + 20:
        horiz_gap = fx1 - tx2
        vert_diff = abs(t_ymid - f_ymid)
        if horiz_gap < _RECLASS_HORIZ_GAP and vert_diff < _RECLASS_BAND_TOL + f_h / 2:
            return math.sqrt(horiz_gap ** 2 + vert_diff ** 2)

    # Case 2 — label is ABOVE the field (similar x alignment)
    if ty2 <= fy1 + 10 and (fy1 - ty2) < _RECLASS_VERT_GAP:
        # Token's horizontal range should overlap with field's
        h_overlap = min(tx2, fx2) - max(tx1, fx1)
        if h_overlap > 0:
            vert_gap = fy1 - ty2
            return math.sqrt(abs(t_xmid - f_xmid) ** 2 + vert_gap ** 2)

    # Case 3 — label OVERLAPS with the field (label printed inside)
    tok_area = _bbox_area(tok.bbox)
    if tok_area > 0:
        overlap = _overlap_area(tok.bbox, field.bbox)
        if overlap / tok_area > 0.3:
            return math.sqrt((t_xmid - f_xmid) ** 2 + (t_ymid - f_ymid) ** 2)

    return None


def _propagate_to_overlapping(
    fields: List[FieldResult],
    assignments: Dict[int, str],
) -> None:
    """
    If a reclassified field overlaps with a text_box that was NOT
    reclassified, propagate the type.  Useful when both detector AND
    OCR-inferred fields cover the same region.
    Modifies assignments in-place.
    """
    new_assignments: Dict[int, str] = {}
    for src_idx, target_type in assignments.items():
        src_bbox = fields[src_idx].bbox
        src_area = _bbox_area(src_bbox)
        if src_area <= 0:
            continue
        for j, f in enumerate(fields):
            if j in assignments or j in new_assignments:
                continue
            if f.type != "text_box":
                continue
            overlap = _overlap_area(src_bbox, f.bbox)
            f_area = _bbox_area(f.bbox)
            # Propagate if >30% of either field overlaps
            if f_area > 0 and (overlap / f_area > 0.3 or overlap / src_area > 0.3):
                new_assignments[j] = target_type
                logger.debug(
                    f"Reclassify propagation: [{f.type}] → [{target_type}] at "
                    f"[{f.bbox[0]:.0f},{f.bbox[1]:.0f},{f.bbox[2]:.0f},{f.bbox[3]:.0f}]"
                )

    assignments.update(new_assignments)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  INTERMEDIATE IMAGE RENDERING — for Stage 3 (LLM) input
# ──────────────────────────────────────────────────────────────────────────────

# RGBA overlay colours per field type (for PIL)
_OVERLAY_COLORS = {
    "text_box":  (46, 140, 250, 60),   # blue
    "checkbox":  (33, 200, 90, 70),    # green
    "radio":     (250, 166, 26, 70),   # orange
    "signature": (232, 50, 77, 65),    # red
    "initials":  (166, 77, 217, 65),   # purple
    "date":      (255, 140, 0, 65),    # deep orange
}
_DEFAULT_OVERLAY = (128, 128, 128, 50)


def render_intermediate_image(
    image: Image.Image,
    existing_fields: List[FieldResult],
) -> Image.Image:
    """
    Draw semitransparent coloured overlays on a copy of the image so the
    LLM can SEE which regions are already detected.

    Returns a new PIL.Image.Image (RGBA -> converted back to RGB for JPEG).
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for f in existing_fields:
        x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
        color = _OVERLAY_COLORS.get(f.type, _DEFAULT_OVERLAY)
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (180,), width=2)
        # Small type label
        label = f.type
        try:
            draw.text((x1 + 2, y1 + 1), label, fill=(255, 255, 255, 220))
        except Exception:
            pass

    composite = Image.alpha_composite(base, overlay)
    return composite.convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  LLM DEDUP — reject LLM fields that overlap Stage 1+2
# ──────────────────────────────────────────────────────────────────────────────

def dedup_llm_fields(
    llm_fields: List[LLMField],
    existing_fields: List[FieldResult],
) -> List[LLMField]:
    """
    Filter LLM output: keep only fields NOT already covered by existing ones.
    This prevents the LLM from creating overlapping/duplicate detections.
    """
    kept: List[LLMField] = []
    for f in llm_fields:
        if is_field_covered(f.bbox, existing_fields):
            logger.debug(
                f"LLM dedup: suppressed [{f.type}] at "
                f"[{f.bbox[0]:.0f},{f.bbox[1]:.0f},{f.bbox[2]:.0f},{f.bbox[3]:.0f}] "
                f"-- already covered"
            )
            continue
        kept.append(f)

    suppressed = len(llm_fields) - len(kept)
    if suppressed > 0:
        logger.info(f"LLM dedup: {suppressed}/{len(llm_fields)} fields suppressed as duplicates")
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# 9.  BUILD INDEXED PROPOSALS (for LLM prompt — ONLY uncovered proposals)
# ──────────────────────────────────────────────────────────────────────────────

def build_indexed_proposals(
    detector_proposals: List[Proposal],
    existing_fields: List[FieldResult],
) -> List[Dict]:
    """
    Build indexed proposal list for LLM, but EXCLUDE proposals already
    covered by existing Stage 1+2 fields.  This prevents the LLM from
    re-classifying fields that are already finalised.

    Uses STRICTER thresholds than general coverage check so that the LLM
    still gets proposals in ambiguous regions where it can add value.
    """
    # Stricter thresholds: only filter proposals with very high overlap
    PROPOSAL_IOU_THRESHOLD = 0.50         # was 0.30 — need higher overlap to filter
    PROPOSAL_CONTAINMENT_THRESHOLD = 0.80  # was 0.60 — need 80% containment to filter

    merged = []
    idx = 0
    for p in detector_proposals:
        if is_field_covered(
            p.bbox, existing_fields,
            iou_threshold=PROPOSAL_IOU_THRESHOLD,
            containment_threshold=PROPOSAL_CONTAINMENT_THRESHOLD,
        ):
            continue
        merged.append({
            "idx": idx,
            "bbox": [round(v, 1) for v in p.bbox],
            "confidence": round(p.confidence, 3),
            "label": p.label,
            "origin": "detector",
        })
        idx += 1
    logger.info(
        f"Indexed proposals for LLM: {idx} uncovered "
        f"(from {len(detector_proposals)} total detector proposals)"
    )
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 10.  COORDINATE ANCHORING  (still needed for LLM's "added" fields)
# ──────────────────────────────────────────────────────────────────────────────

ANCHOR_IOU_ACCEPT = 0.30
ANCHOR_MAX_CENTER_DIST = 80.0


def anchor_llm_fields(
    llm_fields: List[LLMField],
    reference_proposals: List[Dict],
    image_width: int,
    image_height: int,
) -> List[LLMField]:
    """
    Snap LLM output fields to nearest reference proposal coordinates.

    DIFFERENCE from old version:
      - Does NOT add back unmatched reference proposals.  Stage 1+2 already
        cover those.  Adding them back caused dual-type overlaps.
    """
    if not reference_proposals:
        return llm_fields

    ref_bboxes = [r["bbox"] for r in reference_proposals]
    ref_used = set()
    anchored: List[LLMField] = []

    for field in llm_fields:
        best_idx, best_iou, best_dist = _find_best_anchor(field.bbox, ref_bboxes, ref_used)

        if best_idx is not None and (best_iou >= ANCHOR_IOU_ACCEPT or best_dist <= ANCHOR_MAX_CENTER_DIST):
            ref = reference_proposals[best_idx]
            ref_used.add(best_idx)
            anchored.append(LLMField(
                type=field.type,
                bbox=ref["bbox"],
                confidence=min(1.0, field.confidence + 0.05),
                source="llm_added",
            ))
            logger.debug(
                f"Anchored LLM field [{field.type}] to ref {best_idx} "
                f"(IoU={best_iou:.2f}, dist={best_dist:.0f}px)"
            )
        else:
            # Truly new — keep LLM bbox, penalise confidence
            anchored.append(LLMField(
                type=field.type,
                bbox=field.bbox,
                confidence=max(0.1, field.confidence - 0.15),
                source="llm_added",
            ))

    # Do NOT add back unmatched refs — Stage 1+2 already have them
    logger.info(
        f"Anchoring: {len(llm_fields)} LLM fields processed, "
        f"{len(ref_used)} anchored to refs"
    )
    return anchored


def _find_best_anchor(
    field_bbox: List[float],
    ref_bboxes: List[List[float]],
    used: set,
) -> Tuple[Optional[int], float, float]:
    best_idx = None
    best_iou = 0.0
    best_dist = float("inf")

    for i, rb in enumerate(ref_bboxes):
        if i in used:
            continue
        iou = _iou(field_bbox, rb)
        dist = _center_dist(field_bbox, rb)
        if iou > best_iou or (iou == best_iou and dist < best_dist):
            best_idx = i
            best_iou = iou
            best_dist = dist

    return best_idx, best_iou, best_dist


# Label -> type for fallback
_LABEL_MAP = {
    "TextBox": "text_box",
    "ChoiceButton": "checkbox",
    "Signature": "signature",
    "text_box": "text_box",
    "checkbox": "checkbox",
    "signature": "signature",
    "initials": "initials",
    "radio": "radio",
    "date": "date",
}


def _label_to_type(label: str) -> str:
    return _LABEL_MAP.get(label, "text_box")
