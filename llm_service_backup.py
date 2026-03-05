import base64
import io
import json
import logging
from typing import List, Optional

import httpx
from PIL import Image

from app.config import settings
from app.schemas.request_schema import LLMField, LLMOutput, OCRToken, Proposal
from app.schemas.response_schema import FieldResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  — Stage 3: find ONLY unmasked fields
# The LLM receives an image with coloured overlays showing already-detected
# fields.  It must ONLY add fields that are NOT already covered by an overlay.
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a document layout classification engine — Stage 3.

You receive:
1. A document page image. **Coloured translucent overlays** mark regions
   that have ALREADY been detected as form fields by earlier stages:
     - Blue overlay   = text_box  (already detected)
     - Green overlay   = checkbox  (already detected)
     - Red overlay     = signature (already detected)
     - Orange overlay  = radio     (already detected)
     - Purple overlay  = initials  (already detected)
   These are FINAL — do NOT re-detect, reclassify, or duplicate them.

2. A list of REMAINING unmatched proposals (fields the earlier stages did
   NOT finalise). Each has:
     - idx, bbox, confidence, label, origin
   You may accept or reject these.

3. OCR tokens (text + bounding boxes) for additional context.

Your tasks — respond with STRICT JSON ONLY:

A. For each remaining proposal, decide:
   - "accept" if it is a real interactive form field
   - "reject" if it is NOT fillable (decorative, heading, logo, etc.)
   Use the EXACT bbox from the proposal — do NOT modify coordinates.

B. If you see interactive fields in the image that are:
   - NOT covered by any coloured overlay, AND
   - NOT in the proposal list
   then add them under "added" with bbox coordinates.
   DO NOT add fields that overlap with any coloured overlay.

CRITICAL RULES:
- NEVER duplicate a field that already has a coloured overlay.
- Allowed field types: text_box, signature, checkbox, radio, initials
- A checkbox is a small square (< 30×30px). Larger rectangles are text_box.
- Output STRICT JSON only. No markdown, no code fences, no commentary.

Required JSON schema:
{
  "page": <page_number>,
  "accepted": [
    { "idx": <int>, "type": "<type>", "confidence": <0.0-1.0> }
  ],
  "rejected": [ <int>, ... ],
  "added": [
    { "type": "<type>", "bbox": [x1,y1,x2,y2], "confidence": <0.0-1.0> }
  ]
}

Return ONLY the JSON object. Nothing else."""

# ──────────────────────────────────────────────────────────────────────────────
# CORRECTION PROMPT — appended on retry after invalid JSON response
# ──────────────────────────────────────────────────────────────────────────────
CORRECTION_PROMPT = (
    "Your previous response was not valid JSON. You MUST return ONLY a valid JSON object "
    "with no markdown, no code fences, no explanation, no text before or after the JSON. "
    "Follow the exact schema specified in the system prompt."
)


class LLMService:

    @classmethod
    def reason(
        cls,
        image: Image.Image,
        indexed_proposals: List[dict],
        ocr_tokens: List[OCRToken],
        page_number: int,
        existing_fields: Optional[List[FieldResult]] = None,
    ) -> Optional[LLMOutput]:
        """
        Call NVIDIA NIM with page image, indexed proposals, and OCR tokens.

        Stage 3 mode: if existing_fields is provided, render coloured overlays
        on the image so the LLM can see what is already detected and only
        add truly new fields.

        Args:
            image: PIL.Image (resized page, max 1216px)
            indexed_proposals: list of dicts — ONLY uncovered proposals
            ocr_tokens: List of OCRToken objects
            page_number: 1-based page number
            existing_fields: Stage 1+2 fields already finalised (drawn as overlays)

        Returns:
            LLMOutput with classified fields, or None if all attempts fail.
        """
        # If we have existing fields, render intermediate mask on the image
        if existing_fields:
            from app.services.spatial_analysis import render_intermediate_image
            image = render_intermediate_image(image, existing_fields)
            logger.info(
                f"LLM Stage 3: sending masked image with {len(existing_fields)} "
                f"existing overlays + {len(indexed_proposals)} uncovered proposals"
            )

        image_b64 = cls._encode_image(image)
        user_content = cls._build_user_message(image_b64, indexed_proposals, ocr_tokens, page_number)

        # First attempt
        response_text = cls._call_nim(user_content)
        if response_text:
            parsed = cls._parse_response(response_text, page_number, indexed_proposals)
            if parsed is not None:
                return parsed

        # Retry once with correction prompt
        logger.warning(f"LLM returned invalid JSON on page {page_number}, retrying with correction")
        retry_content = cls._build_retry_message(user_content)
        response_text = cls._call_nim(retry_content)
        if response_text:
            parsed = cls._parse_response(response_text, page_number, indexed_proposals)
            if parsed is not None:
                return parsed

        logger.error(f"LLM failed after retry on page {page_number}")
        return None

    @classmethod
    def _encode_image(cls, image: Image.Image) -> str:
        """Convert PIL Image to base64-encoded JPEG string. Converts RGBA→RGB first."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    def _build_user_message(
        cls,
        image_b64: str,
        indexed_proposals: List[dict],
        ocr_tokens: List[OCRToken],
        page_number: int,
    ) -> list:
        """Build multimodal user message content (image + indexed proposals).

        Caps proposals at top 150 by confidence to keep token count manageable.
        The LLM classifies by idx — coordinates are immutable.
        """
        # Sort by confidence desc, cap at 150
        top = sorted(indexed_proposals, key=lambda p: p["confidence"], reverse=True)[:150]
        ocr_data = [
            {"text": t.text, "bbox": [round(c, 1) for c in t.bbox]}
            for t in ocr_tokens
        ]

        text_context = (
            f"Page: {page_number}\n\n"
            f"Indexed proposals ({len(top)} candidates — DO NOT modify bbox values):\n"
            f"{json.dumps(top, separators=(',', ':'))}\n\n"
            f"OCR tokens ({len(ocr_tokens)} tokens):\n"
            f"{json.dumps(ocr_data, separators=(',', ':'))}\n\n"
            f"Classify each proposal by idx. Accept or reject. "
            f"Add any missed interactive fields. Return JSON only."
        )

        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
            {
                "type": "text",
                "text": text_context,
            },
        ]

    @classmethod
    def _build_retry_message(cls, original_content: list) -> list:
        """Append correction instruction to the original content list for retry."""
        retry_content = list(original_content)
        retry_content.append({"type": "text", "text": CORRECTION_PROMPT})
        return retry_content

    @classmethod
    def _call_nim(cls, user_content: list) -> Optional[str]:
        """
        POST to NVIDIA NIM using Server-Sent Events (stream=True).
        Streaming prevents httpx read-timeout on long generations (397B model).
        Each chunk resets the per-read timeout, so large outputs complete correctly.
        Returns the fully assembled response text, or None on failure.
        """
        headers = {"Content-Type": "application/json"}
        if settings.NIM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.NIM_API_KEY}"

        payload = {
            "model": settings.NIM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            full_text = ""
            # connect timeout 30s; read timeout per-chunk (resets with each SSE chunk)
            with httpx.Client(
                timeout=httpx.Timeout(connect=30.0, read=settings.LLM_TIMEOUT_SECONDS, write=30.0, pool=5.0)
            ) as client:
                with client.stream("POST", settings.NIM_API_URL, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        chunk_data = line[len("data: "):]
                        if chunk_data == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(chunk_data)
                            delta = chunk_json["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_text += delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
            return full_text if full_text else None

        except httpx.TimeoutException:
            logger.error("LLM request timed out")
            return None
        except httpx.HTTPStatusError as exc:
            logger.error(f"LLM HTTP error: {exc.response.status_code} — {exc.response.text[:200]}")
            return None
        except Exception as exc:
            logger.error(f"LLM unexpected error: {exc}")
            return None

    @classmethod
    def _parse_response(
        cls,
        response_text: str,
        page_number: int,
        indexed_proposals: List[dict],
    ) -> Optional[LLMOutput]:
        """
        Parse the new classify-only LLM response format.

        Expected JSON:
        {
          "page": N,
          "accepted": [ { "idx": int, "type": str, "confidence": float }, ... ],
          "rejected": [ int, ... ],
          "added":    [ { "type": str, "bbox": [...], "confidence": float }, ... ]
        }

        Converts back to a flat list of LLMField for downstream processing.
        Accepted fields get their bbox from the indexed_proposals (immutable).
        """
        text = response_text.strip()

        # Strip markdown code fences if LLM ignored instructions
        if text.startswith("```"):
            lines = text.split("\n", 1)
            text = lines[1] if len(lines) > 1 else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"LLM returned invalid JSON on page {page_number}: {exc}")
            return None

        if not isinstance(data, dict):
            logger.warning(f"LLM returned non-dict on page {page_number}")
            return None

        # Build idx → proposal lookup
        ref_map = {p["idx"]: p for p in indexed_proposals}

        fields: List[LLMField] = []

        # ── Accepted proposals (bbox from reference, type from LLM) ─────────
        for item in data.get("accepted", []):
            try:
                idx = int(item["idx"])
                ref = ref_map.get(idx)
                if ref is None:
                    logger.debug(f"LLM accepted unknown idx {idx}, skipping")
                    continue
                ftype = item.get("type", "text_box")
                conf = float(item.get("confidence", ref["confidence"]))
                fields.append(LLMField(
                    type=ftype,
                    bbox=ref["bbox"],    # IMMUTABLE — use reference coordinates
                    confidence=conf,
                    source="detector" if ref["origin"] == "detector" else "detector_corrected",
                ))
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug(f"Skipping malformed accepted item: {item} — {exc}")

        # ── Newly added fields by LLM (bbox from LLM) ──────────────────────
        for item in data.get("added", []):
            try:
                fields.append(LLMField(
                    type=item.get("type", "text_box"),
                    bbox=item.get("bbox", [0, 0, 0, 0]),
                    confidence=float(item.get("confidence", 0.4)),
                    source="llm_added",
                ))
            except (ValueError, TypeError) as exc:
                logger.debug(f"Skipping malformed added item: {item} — {exc}")

        # ── Fallback: handle old "fields" format for backward compatibility ─
        if not fields and "fields" in data:
            logger.info("LLM used legacy 'fields' format — falling back")
            for raw_field in data["fields"]:
                try:
                    fields.append(LLMField(
                        type=raw_field.get("type", "text_box"),
                        bbox=raw_field.get("bbox", [0, 0, 0, 0]),
                        confidence=float(raw_field.get("confidence", 0.5)),
                        source=raw_field.get("source", "detector"),
                    ))
                except (ValueError, TypeError) as exc:
                    logger.debug(f"Skipping malformed field: {raw_field} — {exc}")

        rejected = data.get("rejected", [])
        logger.info(
            f"LLM parse: {len(fields)} fields "
            f"({len(data.get('accepted', []))} accepted, "
            f"{len(data.get('added', []))} added, "
            f"{len(rejected)} rejected)"
        )

        return LLMOutput(
            page=data.get("page", page_number),
            fields=fields,
        )
