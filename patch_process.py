import re

with open("app/routers/process.py", "r", encoding="utf-8") as f:
    text = f.read()

start_marker = "# ── Full 3-stage pipeline: detector → OCR gap-fill → LLM visual ──────────"
end_marker = "@router.post(\"/process-document\", response_model=DocumentResponse)\nasync def process_document("

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Could not find markers!")
    print("Start:", start_idx)
    print("End:", end_idx)
    exit(1)

new_code = """# ── Full 3-stage pipeline: FFDNet + OCR Patterns -> LLM Reasoner ──────────
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
    indexed_proposals = []
    for idx, f in enumerate(all_proposals):
        indexed_proposals.append({
            "idx": idx,
            "bbox": [round(v, 1) for v in f.bbox],
            "confidence": round(f.confidence, 3),
            "label": f.type,
            "origin": f.source,
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


"""

new_file = text[:start_idx] + new_code + text[end_idx:]

with open("app/routers/process.py", "w", encoding="utf-8") as f:
    f.write(new_file)
print("done")
