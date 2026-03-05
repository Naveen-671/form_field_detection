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
SYSTEM_PROMPT = """You are a highly precise document layout classification engine.

You receive:
1. A document page image. **Numbered red bounding boxes** have been drawn over potential form fields (proposals).
   Each box has an ID label like [0], [1], [2], etc.
2. A list of candidate proposals with their coordinates (idx, bbox) and crucially, their `nearby_text`, representing the OCR label visually adjacent to the field.

Your task is to REASON over all these proposals using the visual and provided `nearby_text` context, and classify EVERY valid proposal.
Many proposals might be duplicated, misaligned, or completely wrong. You must act as the final judge.

For EACH valid form field:
- "accept" the best proposal for that field.
- If multiple proposals cover the same logical field, ACCEPT ONLY ONE (the one with the best coordinates).
- Classify the accepted field into ONE of these types: `text_box`, `signature`, `checkbox`, `radio`, `initials`, `date`.
    -> EXTREMELY IMPORTANT: Do not rely on previous type guesses. If the `nearby_text` of a box contains "signature", "sign", "employee signature", "approved by", you MUST classify it as `signature`. If it contains "date", you MUST classify it as `date`.
- "reject" any proposal that is decorative, text-only, duplicate, or misaligned.
- If you physically see an essential field (checkbox, signature line) in the image but it is NOT in the proposals list, add it to the "added" list with your visually estimated bounding box [x1,y1,x2,y2].

CRITICAL RULES:
- `checkbox` is for small selection squares. `text_box` is for larger text areas.
- `date` should be strictly bounding the date input area. 
- `signature` should bound the area where signing occurs. Rely heavily on the `nearby_text` in the JSON metadata!
- DO NOT ACCEPT multiple bounding boxes for the exact same signature or date line! Deduplicate visually.
- Output STRICT JSON only. No markdown, no code fences, no commentary.

Required JSON schema:
{
  "page": <page_number>,
  "accepted": [
      { "idx": <int>, "reason": "<short reasoning about nearby text or visuals marking it as signature, date, checkbox, etc>", "type": "<type>", "confidence": <0.0-1.0> }
    ],
    "rejected": [ <int>, ... ],
    "added": [
      { "type": "<type>", "reason": "<short reasoning>", "bbox": [x1,y1,x2,y2], "confidence": <0.0-1.0> }
    ]
}
"""

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
    ) -> Optional[LLMOutput]:
        """
        Call NVIDIA NIM with page image, indexed proposals, and OCR tokens.

        Stage 3 mode: renders numbered boxes on the image so the LLM can see
        the proposals and classify them visually.
        """
        from app.services.spatial_analysis import render_numbered_boxes
        image = render_numbered_boxes(image, indexed_proposals)
        logger.info(
            f"LLM Stage 3: sending masked image with {len(indexed_proposals)} "
            f"numbered proposals"
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
        text_context = (
            f"Page: {page_number}\n\n"
            f"Indexed proposals ({len(top)} candidates — DO NOT modify bbox values):\n"
            f"{json.dumps(top, separators=(',', ':'))}\n\n"
            f"Classify each proposal by idx. Accept or reject.\n"
            f"Use the 'nearby_text' provided in the proposal to determine if a field is a 'signature', 'date', or 'text_box'.\n"
            f"Add any missed interactive fields natively visible on the image. Return JSON only."
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
        POST to NVIDIA NIM using Requests.
        """
        import requests
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
            logger.info("Sending request to NIM with requests...")
            full_text = ""
            with requests.post(
                settings.NIM_API_URL, 
                headers=headers, 
                json=payload, 
                stream=True, 
                timeout=120
            ) as response:
                logger.info(f"NIM POST status: {response.status_code}")
                response.raise_for_status()
                for i, line in enumerate(response.iter_lines()):
                    if i % 10 == 0:
                        logger.debug(f"Received chunk {i}")
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: "):
                            chunk_data = decoded_line[len("data: "):]
                            if chunk_data == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(chunk_data)
                                delta = chunk_json["choices"][0]["delta"].get("content", "")
                                if delta:
                                    full_text += delta
                                    # print(delta, end="", flush=True)  # noisy
                                    if len(full_text) > 15000:
                                        logger.warning("Generation exceeded 15k chars, truncating...")
                                        break
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
            
            logger.info(f"Completed NIM stream. Generated {len(full_text)} characters.")
            return full_text if full_text else None

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return None
        except requests.exceptions.RequestException as exc:
            logger.error(f"LLM request error: {exc}")
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
            logger.info(f"LLM generated JSON: {text}")
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
        from app.services.spatial_analysis import FIELD_INDICATORS
        for item in data.get("accepted", []):
            try:
                idx = int(item["idx"])
                ref = ref_map.get(idx)
                if ref is None:
                    logger.debug(f"LLM accepted unknown idx {idx}, skipping")
                    continue
                ftype = item.get("type", "text_box")
                
                # ── LOCAL OVERRIDE STRATEGY START ──
                # If LLM classified as text_box but nearby text clearly shows a signature/date,
                # forcefully override the LLM using classical regex rules.
                nearby_text = ref.get("nearby_text", "")
                if nearby_text and ftype == "text_box":
                    forced_type = None
                    for pattern, known_type in FIELD_INDICATORS:
                        if pattern.search(nearby_text):
                            forced_type = known_type
                            if known_type == "signature":  # Prioritize most critical fields
                                break
                    
                    if forced_type:
                        logger.info(f"Local rule override: LLM said {ftype}, local rule forces {forced_type} (Text: '{nearby_text}')")
                        ftype = forced_type
                # ── LOCAL OVERRIDE STRATEGY END ──

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
