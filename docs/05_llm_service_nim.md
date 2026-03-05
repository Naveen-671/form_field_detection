# Multimodal LLM Service (NVIDIA NIM)

## File: `app/services/llm_service.py`

## Model

**Qwen3.5-397B-A17B** served via **NVIDIA NIM** using an OpenAI-compatible chat completions API.

---

## Responsibility

The LLM is the **reasoning/correction layer** — it receives the full page image, detector proposals, and OCR tokens, then:

1. **Validates** detector proposals (confirm or reject each one)
2. **Classifies** field types (text_box, signature, checkbox, radio, initials)
3. **Adds missed fields** that the detector failed to catch (using visual + OCR context)
4. **Removes decorative false positives** (borders, logos, non-interactive elements)
5. **Outputs strict JSON only** — no markdown, no commentary, no explanation

---

## API Configuration (from `app/config.py`)

```python
NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"  # OpenAI-compatible endpoint
NIM_API_KEY = ""  # Set via environment variable if NIM requires auth
NIM_MODEL_NAME = "qwen3.5-397b-a17b"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 1

use this as reference code snippet
import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

def read_b64(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer $NVIDIA_API_KEY",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "qwen/qwen3.5-397b-a17b",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 16384,
  "temperature": 0.60,
  "top_p": 0.95,
  "top_k": 20,
  "presence_penalty": 0,
  "repetition_penalty": 1,
  "stream": stream,
  "chat_template_kwargs": {"enable_thinking":True},
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())

```

---

## Implementation Specification

```python
import httpx
import json
import base64
import io
import logging
from PIL import Image
from typing import List, Optional
from app.schemas.request_schema import Proposal, OCRToken, LLMOutput, LLMField
from app.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# SYSTEM PROMPT — DO NOT MODIFY WITHOUT ARCHITECTURAL REVIEW
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a document layout reasoning engine.

You receive:
1. A full-page document image.
2. A list of detector proposals from FFDNet-L (bounding boxes with confidence scores and class labels: TextBox, ChoiceButton, Signature).
3. A list of OCR tokens (text strings with bounding boxes).

You must:
- Validate each detector proposal: confirm it is a real interactive form field, or reject it.
- Refine detector class labels into the final allowed types:
  - TextBox → text_box
  - ChoiceButton → checkbox OR radio (decide based on visual shape: squares are checkboxes, circles are radio buttons)
  - Signature → signature OR initials (decide based on field size and nearby labels)
- Add any interactive fields visible in the image that the detector missed.
- Remove decorative borders, logos, watermarks, and non-interactive visual elements.
- Output STRICT JSON only. No markdown. No code fences. No commentary. No explanations.
- If uncertain about a field, lower its confidence instead of omitting it or hallucinating.
- Every field must correspond to a visually identifiable interactive UI element in the image.

Allowed field types (use EXACTLY these strings):
- text_box
- signature
- checkbox
- radio
- initials

Required output JSON schema:
{
  "page": <page_number_integer>,
  "fields": [
    {
      "type": "<allowed_type>",
      "bbox": [x1, y1, x2, y2],
      "confidence": <float_0_to_1>,
      "source": "<source_tag>"
    }
  ]
}

Source tags:
- "detector" — proposal confirmed as-is
- "detector_corrected" — proposal confirmed but bbox or type was adjusted
- "llm_added" — new field the detector missed, identified by visual reasoning

Return ONLY the JSON object. Nothing else."""

# ──────────────────────────────────────────────────────────────
# RETRY CORRECTION PROMPT — used when first attempt returns invalid JSON
# ──────────────────────────────────────────────────────────────
CORRECTION_PROMPT = """Your previous response was not valid JSON. You MUST return ONLY a valid JSON object with no markdown, no code fences, no explanation, no text before or after the JSON. Follow the exact schema specified in the system prompt."""


class LLMService:

    @classmethod
    def reason(
        cls,
        image: Image.Image,
        proposals: List[Proposal],
        ocr_tokens: List[OCRToken],
        page_number: int
    ) -> Optional[LLMOutput]:
        """
        Call NVIDIA NIM with page image, proposals, and OCR tokens.
        Returns validated LLMOutput or None if all attempts fail.

        Args:
            image: PIL.Image (resized page, max 1216px)
            proposals: List of detector Proposal objects
            ocr_tokens: List of OCRToken objects from PaddleOCR
            page_number: 1-based page number

        Returns:
            LLMOutput with classified fields, or None on failure
        """
        # Encode image to base64
        image_b64 = cls._encode_image(image)

        # Build user message with proposals and OCR context
        user_content = cls._build_user_message(
            image_b64=image_b64,
            proposals=proposals,
            ocr_tokens=ocr_tokens,
            page_number=page_number
        )

        # First attempt
        response_text = cls._call_nim(user_content)
        if response_text:
            parsed = cls._parse_response(response_text, page_number)
            if parsed:
                return parsed

        # Retry with correction prompt
        logger.warning(f"LLM first attempt failed for page {page_number}, retrying with correction")
        retry_content = cls._build_retry_message(user_content)
        response_text = cls._call_nim(retry_content)
        if response_text:
            parsed = cls._parse_response(response_text, page_number)
            if parsed:
                return parsed

        logger.error(f"LLM failed after retry for page {page_number}")
        return None

    @classmethod
    def _encode_image(cls, image: Image.Image) -> str:
        """Convert PIL Image to base64-encoded JPEG string."""
        buffer = io.BytesIO()
        # Convert to RGB if necessary (e.g., RGBA PNGs)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    def _build_user_message(
        cls,
        image_b64: str,
        proposals: List[Proposal],
        ocr_tokens: List[OCRToken],
        page_number: int
    ) -> list:
        """
        Build the multimodal user message content array.
        Uses OpenAI vision API format (supported by NVIDIA NIM).
        """
        # Format proposals as compact JSON (include FFDNet-L class label)
        proposals_data = [
            {"bbox": p.bbox, "confidence": round(p.confidence, 3), "label": p.label}
            for p in proposals
        ]

        # Format OCR tokens as compact JSON
        ocr_data = [
            {"text": t.text, "bbox": [round(c, 1) for c in t.bbox]}
            for t in ocr_tokens
        ]

        text_context = f"""Page: {page_number}

Detector proposals ({len(proposals)} candidates):
{json.dumps(proposals_data, separators=(',', ':'))}

OCR tokens ({len(ocr_tokens)} tokens):
{json.dumps(ocr_data, separators=(',', ':'))}

Analyze the image and the data above. Return the validated fields as JSON."""

        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }
            },
            {
                "type": "text",
                "text": text_context
            }
        ]

    @classmethod
    def _build_retry_message(cls, original_content: list) -> list:
        """Append correction instruction to the original message for retry."""
        retry_content = original_content.copy()
        retry_content.append({
            "type": "text",
            "text": CORRECTION_PROMPT
        })
        return retry_content

    @classmethod
    def _call_nim(cls, user_content: list) -> Optional[str]:
        """
        Make HTTP POST to NVIDIA NIM OpenAI-compatible endpoint.
        Returns raw response text or None on failure.
        """
        headers = {
            "Content-Type": "application/json"
        }
        if settings.NIM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.NIM_API_KEY}"

        payload = {
            "model": settings.NIM_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "top_p": 0.8,
            "repetition_penalty": 1.1
        }

        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                response = client.post(
                    settings.NIM_API_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

                # Extract text from OpenAI-compatible response
                return data["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            logger.error("LLM request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"LLM unexpected error: {e}")
            return None

    @classmethod
    def _parse_response(cls, response_text: str, page_number: int) -> Optional[LLMOutput]:
        """
        Parse raw LLM response text into LLMOutput.
        Handles common issues: code fences, extra whitespace, trailing commas.
        """
        text = response_text.strip()

        # Strip markdown code fences if present (LLM sometimes ignores instructions)
        if text.startswith("```"):
            # Remove opening fence (with optional language tag)
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}")
            return None

        # Validate basic structure
        if not isinstance(data, dict) or "fields" not in data:
            logger.warning("LLM JSON missing 'fields' key")
            return None

        # Parse fields
        fields = []
        for f in data.get("fields", []):
            try:
                field = LLMField(
                    type=f.get("type", "text_box"),
                    bbox=f.get("bbox", [0, 0, 0, 0]),
                    confidence=float(f.get("confidence", 0.5)),
                    source=f.get("source", "detector")
                )
                fields.append(field)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed field: {f}, error: {e}")
                continue

        return LLMOutput(
            page=data.get("page", page_number),
            fields=fields
        )
```

---

## API Request Format (NVIDIA NIM — OpenAI Compatible)

```json
POST {NIM_API_URL}
Content-Type: application/json
Authorization: Bearer {NIM_API_KEY}

{
  "model": "qwen3.5-397b-a17b",
  "messages": [
    {
      "role": "system",
      "content": "<SYSTEM_PROMPT>"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<BASE64_IMAGE>"
          }
        },
        {
          "type": "text",
          "text": "Page: 1\n\nDetector proposals (12 candidates):\n[{\"bbox\":[120,340,450,375],\"confidence\":0.87,\"label\":\"TextBox\"}, ...]\n\nOCR tokens (45 tokens):\n[{\"text\":\"Full Name\",\"bbox\":[45,120,180,145]}, ...]\n\nAnalyze the image and the data above. Return the validated fields as JSON."
        }
      ]
    }
  ],
  "temperature": 0.0,
  "max_tokens": 2048,
  "top_p": 0.8,
  "repetition_penalty": 1.1
}
```

---

## Expected LLM Response (raw text, no wrapper)

```json
{
  "page": 1,
  "fields": [
    {
      "type": "text_box",
      "bbox": [120.5, 340.2, 450.8, 375.1],
      "confidence": 0.92,
      "source": "detector"
    },
    {
      "type": "signature",
      "bbox": [100.0, 600.0, 500.0, 700.0],
      "confidence": 0.88,
      "source": "detector_corrected"
    },
    {
      "type": "checkbox",
      "bbox": [45.0, 450.0, 65.0, 470.0],
      "confidence": 0.75,
      "source": "llm_added"
    }
  ]
}
```

---

## Critical Rules

1. **Temperature MUST be 0.0** — deterministic output is essential for consistency.
2. **System prompt is sacred** — do not modify without architectural review.
3. **Image format**: Always JPEG, quality 85, base64-encoded. Convert RGBA→RGB first.
4. **Compact JSON in user message**: Use `separators=(',', ':')` to minimize token usage.
5. **Retry once only**: If first attempt returns invalid JSON, retry with `CORRECTION_PROMPT`. If retry also fails, return `None` (orchestrator will fall back to detector-only).
6. **Never trust raw output**: `_parse_response()` validates structure but the Validation Service applies full business rule validation.
7. **httpx over requests**: Use `httpx` for HTTP calls (better timeout handling, connection pooling).
8. **No streaming**: Use synchronous request/response for Phase 1.