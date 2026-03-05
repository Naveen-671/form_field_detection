import logging
import os
import time
import uuid
from typing import List, Literal, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import settings
from app.schemas.request_schema import OCRResult
from app.schemas.response_schema import (
    DocumentResponse,
    FieldResult,
    OcrTokenResult,
    PageResult,
    SpatialField,
)
from app.services.detector_service import DetectorService
from app.services.llm_service import LLMService
from app.services.masking_service import MaskingService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.validation_service import ValidationService

router = APIRouter()
logger = logging.getLogger(__name__)


def validate_upload(file: UploadFile, file_bytes: bytes) -> None:
    """Validate file extension and size. Raises HTTPException on failure."""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_file", "detail": f"Unsupported file type: '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}"}
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail={"error": "file_too_large", "detail": f"File size {size_mb:.1f}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB"}
        )


def _spatial_overlap(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU-like overlap area between two [x1,y1,x2,y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _token_is_nearby(field_bbox: List[float], token_bbox: List[float], margin: float = 30.0) -> bool:
    """Return True if token bbox is within `margin` pixels of field bbox (expanded)."""
    exp = [
        field_bbox[0] - margin,
        field_bbox[1] - margin,
        field_bbox[2] + margin,
        field_bbox[3] + margin,
    ]
    return _spatial_overlap(exp, token_bbox) > 0


def _build_spatial_fields(fields: List[FieldResult], ocr_tokens: List) -> List[SpatialField]:
    """
    For each detected field, find OCR tokens that:
      - overlap the field bbox (ocr_inside)
      - are within 30px margin (ocr_nearby, excluding inside)
    """
    spatial = []
    for f in fields:
        inside_texts, nearby_texts = [], []
        for tok in ocr_tokens:
            overlap = _spatial_overlap(f.bbox, tok.bbox)
            if overlap > 0:
                inside_texts.append(tok.text)
            elif _token_is_nearby(f.bbox, tok.bbox, margin=30.0):
                nearby_texts.append(tok.text)
        spatial.append(SpatialField(
            type=f.type,
            bbox=f.bbox,
            confidence=f.confidence,
            source=f.source,
            ocr_inside=inside_texts,
            ocr_nearby=nearby_texts,
        ))
    return spatial


def process_single_page(
    request_id: str,
    page_info: dict,
    pipeline: str = "full",
) -> PageResult:
    """
    Run the detection pipeline for one page image.

    pipeline options:
      "detector_only"  — FFDNet-L only, skip OCR + LLM
      "detector_ocr"   — FFDNet-L + OCR with spatial analysis, skip LLM
      "full"           — complete pipeline (detector → OCR → LLM → validate)
    """
    image = page_info["image"]
    page_number = page_info["page_number"]

    # ── Step 1: FFDNet-L detection (always runs) ────────────────────────────
    t0 = time.time()
    proposals = DetectorService.detect(image)
    det_time = time.time() - t0
    logger.info(
        f"[{request_id}] Page {page_number}: {len(proposals)} proposals "
        f"from FFDNet-L ({det_time:.2f}s)"
    )

    # Log breakdown by class
    from collections import Counter
    class_counts = Counter(p.label for p in proposals)
    logger.info(
        f"[{request_id}] Page {page_number}: detector breakdown — "
        + ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items()))
    )

    # ── Detector-only mode ──────────────────────────────────────────────────
    if pipeline == "detector_only":
        validated_fields = ValidationService.fallback_from_proposals(
            proposals=proposals,
            image_width=image.width,
            image_height=image.height
        )
        logger.info(f"[{request_id}] Page {page_number}: [detector_only] {len(validated_fields)} fields after validation")
        return PageResult(
            page=page_number,
            fields=validated_fields,
            proposal_count=len(proposals),
            source="detector_only",
        )

    # ── Step 2: OCR ─────────────────────────────────────────────────────────
    ocr_result = OCRResult(tokens=[], direction="LTR")
    ocr_tokens_out: List[OcrTokenResult] = []
    try:
        t0 = time.time()
        ocr_result = OCRService.extract(image)
        ocr_time = time.time() - t0
        logger.info(
            f"[{request_id}] Page {page_number}: OCR extracted {len(ocr_result.tokens)} tokens "
            f"| direction: {ocr_result.direction} ({ocr_time:.2f}s)"
        )
        # Log every OCR token with position for debugging
        for i, tok in enumerate(ocr_result.tokens):
            bbox_str = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*tok.bbox)
            logger.debug(
                f"[{request_id}] Page {page_number}: OCR token {i:03d}: "
                f'"{tok.text}" @ {bbox_str}'
            )
        ocr_tokens_out = [OcrTokenResult(text=t.text, bbox=t.bbox) for t in ocr_result.tokens]
    except Exception as exc:
        logger.warning(f"[{request_id}] Page {page_number}: OCR failed — {exc}")

    # ── Detector + OCR mode (spatial analysis, no LLM) ─────────────────────
    if pipeline == "detector_ocr":
        validated_fields = ValidationService.fallback_from_proposals(
            proposals=proposals,
            image_width=image.width,
            image_height=image.height
        )
        spatial = _build_spatial_fields(validated_fields, ocr_result.tokens)

        # Log spatial overview
        logger.info(
            f"[{request_id}] Page {page_number}: [detector_ocr] "
            f"{len(validated_fields)} fields | "
            f"{sum(1 for s in spatial if s.ocr_inside)} fields have inside OCR tokens | "
            f"{sum(1 for s in spatial if s.ocr_nearby)} fields have nearby OCR tokens"
        )
        for i, s in enumerate(spatial[:20]):   # log first 20 to avoid log flood
            if s.ocr_inside or s.ocr_nearby:
                bbox_str = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*s.bbox)
                logger.info(
                    f"[{request_id}] Page {page_number}: field {i:03d} [{s.type}] {bbox_str} "
                    f"| inside={s.ocr_inside} | nearby={s.ocr_nearby}"
                )

        return PageResult(
            page=page_number,
            fields=validated_fields,
            proposal_count=len(proposals),
            source="detector_ocr",
            ocr_tokens=ocr_tokens_out,
            spatial_fields=spatial,
            ocr_direction=ocr_result.direction,
        )

    # ── Full 3-stage pipeline: FFDNet + OCR Patterns -> LLM Reasoner ──────────
    from app.services.spatial_analysis import detect_ocr_patterns
    from app.schemas.request_schema import LLMField

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: FFDNet-L → baseline proposals
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    stage1_fields = ValidationService.fallback_from_proposals(
        proposals=proposals,
        image_width=image.width,
        image_height=image.height,
    )
    logger.info(
        f"[{request_id}] Page {page_number}: STAGE 1 (Detector) → "
        f"{len(stage1_fields)} proposals ({time.time() - t0:.2f}s)"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: OCR pattern detection → adds missed structural boxes (like [])
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    # We DO NOT run full keyword gap-fill to avoid duplicate dates or misaligned boxes.
    # Instead, we just detect OCR structural patterns (e.g. check boxes).
    stage2_new = detect_ocr_patterns(
        ocr_tokens=ocr_result.tokens,
        existing_fields=stage1_fields,
    )
    all_proposals = stage1_fields + stage2_new
    logger.info(
        f"[{request_id}] Page {page_number}: STAGE 2 (OCR Patterns) → "
        f"{len(stage2_new)} new pattern proposals, total={len(all_proposals)} ({time.time() - t0:.2f}s)"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: LLM Reasoner — receives the image with ALL proposals numbered
    #          and classifies them + dedups them visually.
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()

    # Build indexed proposals for EVERYTHING
    # Turn FieldResult into Proposal-like dicts
    from app.routers.process import _spatial_overlap

    indexed_proposals = []
    for idx, f in enumerate(all_proposals):
        # find nearby text specifically for this proposal
        nearby_texts = []
        # Expand box by 40px to capture labels near the edge
        margin = 40.0
        exp_box = [
            f.bbox[0] - margin,
            f.bbox[1] - margin,
            f.bbox[2] + margin,
            f.bbox[3] + margin,
        ]
        for tok in ocr_result.tokens:
            ox1 = max(exp_box[0], tok.bbox[0])
            oy1 = max(exp_box[1], tok.bbox[1])
            ox2 = min(exp_box[2], tok.bbox[2])
            oy2 = min(exp_box[3], tok.bbox[3])
            if ox2 > ox1 and oy2 > oy1:
                nearby_texts.append(tok.text)

        indexed_proposals.append({
            "idx": idx,
            "bbox": [round(v, 1) for v in f.bbox],
            "confidence": round(f.confidence, 3),
            "origin": f.source,
            "nearby_text": " | ".join(nearby_texts) if nearby_texts else ""
        })
        
    logger.info(
        f"[{request_id}] Page {page_number}: {len(indexed_proposals)} total proposals for LLM"
    )

    llm_output = None
    if len(indexed_proposals) == 0:
        logger.info(
            f"[{request_id}] Page {page_number}: STAGE 3 SKIPPED — "
            f"no proposals found"
        )
    else:
        try:
            llm_output = LLMService.reason(
                image=image,
                indexed_proposals=indexed_proposals,
                ocr_tokens=ocr_result.tokens,
                page_number=page_number,
            )
            if llm_output:
                logger.info(
                    f"[{request_id}] Page {page_number}: LLM returned {len(llm_output.fields)} fields "
                    f"({time.time() - t0:.2f}s)"
                )
            else:
                logger.warning(f"[{request_id}] Page {page_number}: LLM returned None")
        except Exception as exc:
            logger.warning(f"[{request_id}] Page {page_number}: LLM error: {exc}")

    # Process LLM output
    final_fields: List[FieldResult] = []
    if llm_output and llm_output.fields:
        # Validate the LLM additions and selected proposals
        final_fields = ValidationService.validate(
            fields=llm_output.fields,
            image_width=image.width,
            image_height=image.height,
        )
    else:
        # Fallback if LLM fails: just use the proposals
        if len(all_proposals) > 0:
            logger.warning(f"[{request_id}] Page {page_number}: LLM fallback used")
            final_fields = ValidationService.validate(
                fields=[LLMField(type=f.type, bbox=f.bbox, confidence=0.5, source=f.source) for f in all_proposals],
                image_width=image.width,
                image_height=image.height,
            )

    # Source breakdown for logging
    from collections import Counter as Ctr
    src_counts = Ctr(f.source for f in final_fields)
    logger.info(
        f"[{request_id}] Page {page_number}: FINAL {len(final_fields)} fields — "
        + ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items()))
    )

    # Build spatial overlay for diagnostics
    spatial = _build_spatial_fields(final_fields, ocr_result.tokens)

    return PageResult(
        page=page_number,
        fields=final_fields,
        proposal_count=len(proposals),
        source="3stage_pipeline",
        ocr_tokens=ocr_tokens_out,
        spatial_fields=spatial,
        ocr_direction=ocr_result.direction,
    )


@router.post("/process-document", response_model=DocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    pipeline: str = Query(
        default="full",
        description=(
            "Pipeline mode: "
            "'detector_only' (FFDNet-L only, fastest), "
            "'detector_ocr' (FFDNet-L + OCR spatial analysis, no LLM), "
            "'full' (complete pipeline with LLM reasoning)"
        )
    ),
):
    if pipeline not in ("detector_only", "detector_ocr", "full"):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_pipeline", "detail": "pipeline must be 'detector_only', 'detector_ocr', or 'full'"}
        )

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing file: {file.filename} | pipeline={pipeline}")
    start_time = time.time()

    file_bytes = await file.read()
    validate_upload(file, file_bytes)

    try:
        page_images, pdf_bytes = PDFService.extract_pages(file_bytes, file.filename or "")
    except Exception as exc:
        logger.error(f"[{request_id}] Page extraction failed: {exc}")
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_pdf", "detail": "Invalid or corrupt PDF/image file"}
        )

    all_page_results = []
    for page_info in page_images:
        try:
            page_result = process_single_page(request_id, page_info, pipeline=pipeline)
            all_page_results.append(page_result)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"[{request_id}] Fatal error on page {page_info.get('page_number', '?')}: {exc}")
            raise HTTPException(
                status_code=500,
                detail={"error": "processing_error", "detail": str(exc)}
            )

    masked_pdf_path = None
    is_pdf = (file.filename or "").lower().endswith(".pdf")
    if is_pdf and pdf_bytes:
        try:
            t0 = time.time()
            masked_pdf_path = MaskingService.apply_masks(pdf_bytes, all_page_results, page_images)
            logger.info(f"[{request_id}] Masking done in {time.time() - t0:.2f}s → {masked_pdf_path}")
        except Exception as exc:
            logger.error(f"[{request_id}] Masking failed (non-fatal): {exc}")

    total_time = time.time() - start_time
    logger.info(f"[{request_id}] Completed in {total_time:.2f}s")

    return DocumentResponse(
        request_id=request_id,
        filename=file.filename or "",
        total_pages=len(page_images),
        pages=all_page_results,
        masked_pdf_path=masked_pdf_path,
        processing_time_seconds=round(total_time, 3)
    )
