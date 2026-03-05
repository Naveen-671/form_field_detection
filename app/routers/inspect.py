"""
/inspect endpoint — returns an annotated image + structured JSON showing:
  - FFDNet-L raw detector boxes (red)
  - OCR tokens with bboxes (blue)
  - Spatial overlap: which OCR tokens are inside/near each detector field

Use this to visually compare FFDNet-L alone vs combined with OCR.
"""
import base64
import io
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, ImageDraw, ImageFont

from app.schemas.request_schema import OCRResult
from app.services.detector_service import DetectorService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService

router = APIRouter()
logger = logging.getLogger(__name__)

# Annotation colours
COLOR_DETECTOR = {
    "TextBox": "#FF4444",        # red
    "ChoiceButton": "#FF8C00",   # orange
    "Signature": "#9B59B6",      # purple
}
COLOR_OCR = "#2196F3"            # blue
COLOR_SPATIAL_HIT = "#00BCD4"   # cyan — OCR token inside a detector field


def _draw_annotations(
    image: Image.Image,
    proposals,
    ocr_tokens,
    show_ocr: bool = True,
    min_conf: float = 0.0,
) -> Image.Image:
    """Draw detector boxes and OCR token boxes on a copy of the image."""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Find all OCR token bboxes that are inside any detector field
    inside_ocr_bboxes = set()
    for p in proposals:
        if p.confidence < min_conf:
            continue
        for tok in ocr_tokens:
            ix1 = max(p.bbox[0], tok.bbox[0])
            iy1 = max(p.bbox[1], tok.bbox[1])
            ix2 = min(p.bbox[2], tok.bbox[2])
            iy2 = min(p.bbox[3], tok.bbox[3])
            if ix2 > ix1 and iy2 > iy1:
                inside_ocr_bboxes.add(id(tok))

    # Draw OCR tokens first (background layer)
    if show_ocr:
        for tok in ocr_tokens:
            x1, y1, x2, y2 = (int(v) for v in tok.bbox)
            colour = COLOR_SPATIAL_HIT if id(tok) in inside_ocr_bboxes else COLOR_OCR
            # Semi-transparent fill
            draw.rectangle([x1, y1, x2, y2], outline=colour, width=1,
                           fill=(*_hex_to_rgb(colour), 30))
            # Token text (small)
            try:
                draw.text((x1 + 2, y1), tok.text[:20], fill=(*_hex_to_rgb(colour), 220))
            except Exception:
                pass

    # Draw detector boxes on top
    for p in proposals:
        if p.confidence < min_conf:
            continue
        x1, y1, x2, y2 = (int(v) for v in p.bbox)
        colour = COLOR_DETECTOR.get(p.label, "#FF4444")
        rgb = _hex_to_rgb(colour)
        draw.rectangle([x1, y1, x2, y2], outline=(*rgb, 240), width=2,
                       fill=(*rgb, 40))
        label_text = f"{p.label[:3]} {p.confidence:.2f}"
        draw.rectangle([x1, y1 - 14, x1 + len(label_text) * 6 + 4, y1], fill=(*rgb, 200))
        draw.text((x1 + 2, y1 - 13), label_text, fill=(255, 255, 255, 255))

    composite = Image.alpha_composite(img, overlay)
    return composite.convert("RGB")


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _image_to_b64(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


@router.post("/inspect")
async def inspect_document(
    file: UploadFile = File(...),
    page: int = Query(default=1, ge=1, description="Page number to inspect (1-based)"),
    min_conf: float = Query(default=0.3, ge=0.0, le=1.0, description="Minimum detector confidence to show"),
    show_ocr: bool = Query(default=True, description="Overlay OCR token boxes on the image"),
):
    """
    Diagnostic endpoint — returns:

    - `annotated_image_b64`: JPEG image with FFDNet-L boxes (red/orange/purple) and
      OCR tokens (blue; cyan when inside a detector field)
    - `page_size`: pixel dimensions of the rendered page image
    - `proposals`: raw FFDNet-L proposals with label + confidence
    - `ocr_tokens`: OCR tokens with text + positions
    - `spatial_fields`: for each proposal, which OCR tokens are inside / nearby
    - `ocr_direction`: detected reading direction (LTR/RTL)
    - `summary`: quick stats

    Use `?show_ocr=false` to see just FFDNet-L detections without OCR clutter.
    """
    allowed_ext = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file: {ext}")

    file_bytes = await file.read()

    try:
        page_images, _ = PDFService.extract_pages(file_bytes, file.filename or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read file: {exc}")

    if page > len(page_images):
        raise HTTPException(
            status_code=400,
            detail=f"Page {page} out of range (document has {len(page_images)} pages)"
        )

    page_info = page_images[page - 1]
    image: Image.Image = page_info["image"]

    # ── Run FFDNet-L ────────────────────────────────────────────────────────
    proposals = DetectorService.detect(image)
    filtered = [p for p in proposals if p.confidence >= min_conf]
    logger.info(f"[inspect] Page {page}: {len(proposals)} total proposals, {len(filtered)} above conf={min_conf}")

    # ── Run OCR ─────────────────────────────────────────────────────────────
    ocr_result = OCRResult(tokens=[], direction="LTR")
    ocr_tokens_out = []
    ocr_error = None
    try:
        ocr_result = OCRService.extract(image)
        ocr_tokens_out = [
            {"text": t.text, "bbox": [round(v, 1) for v in t.bbox]}
            for t in ocr_result.tokens
        ]
        logger.info(f"[inspect] Page {page}: OCR returned {len(ocr_result.tokens)} tokens")
    except Exception as exc:
        ocr_error = str(exc)
        logger.warning(f"[inspect] OCR failed: {exc}")

    # ── Build spatial overlap ────────────────────────────────────────────────
    spatial_fields = []
    for p in filtered:
        inside_texts, nearby_texts = [], []
        for tok in ocr_result.tokens:
            ix1 = max(p.bbox[0], tok.bbox[0])
            iy1 = max(p.bbox[1], tok.bbox[1])
            ix2 = min(p.bbox[2], tok.bbox[2])
            iy2 = min(p.bbox[3], tok.bbox[3])
            if ix2 > ix1 and iy2 > iy1:
                inside_texts.append(tok.text)
            else:
                exp = [p.bbox[0] - 30, p.bbox[1] - 30, p.bbox[2] + 30, p.bbox[3] + 30]
                if (max(exp[0], tok.bbox[0]) < min(exp[2], tok.bbox[2]) and
                        max(exp[1], tok.bbox[1]) < min(exp[3], tok.bbox[3])):
                    nearby_texts.append(tok.text)
        spatial_fields.append({
            "label": p.label,
            "confidence": round(p.confidence, 3),
            "bbox": [round(v, 1) for v in p.bbox],
            "ocr_inside": inside_texts,
            "ocr_nearby": nearby_texts,
        })

    # ── Annotate image ───────────────────────────────────────────────────────
    annotated = _draw_annotations(image, filtered, ocr_result.tokens, show_ocr=show_ocr, min_conf=min_conf)
    b64 = _image_to_b64(annotated)

    # ── Summary stats ────────────────────────────────────────────────────────
    from collections import Counter
    label_counts = Counter(p.label for p in filtered)
    fields_with_ocr_inside = sum(1 for s in spatial_fields if s["ocr_inside"])
    fields_with_ocr_nearby = sum(1 for s in spatial_fields if s["ocr_nearby"])

    return {
        "page": page,
        "total_pages": len(page_images),
        "page_size": {"width": image.width, "height": image.height},
        "summary": {
            "detector_proposals_total": len(proposals),
            "detector_proposals_shown": len(filtered),
            "detector_by_class": dict(label_counts),
            "ocr_tokens": len(ocr_result.tokens),
            "ocr_direction": ocr_result.direction,
            "ocr_error": ocr_error,
            "fields_with_ocr_inside": fields_with_ocr_inside,
            "fields_with_ocr_nearby": fields_with_ocr_nearby,
        },
        "spatial_fields": spatial_fields,
        "ocr_tokens": ocr_tokens_out,
        "annotated_image_b64": b64,
    }
