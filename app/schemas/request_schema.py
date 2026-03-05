from pydantic import BaseModel
from typing import List


class Proposal(BaseModel):
    """A detector proposal — candidate bounding box with class label."""
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords on resized image
    confidence: float       # 0.0–1.0
    label: str              # "TextBox", "ChoiceButton", or "Signature" (from FFDNet-L)


class OCRToken(BaseModel):
    """A single OCR text token with position."""
    text: str
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords on resized image


class OCRResult(BaseModel):
    """Full OCR extraction result for one page."""
    tokens: List[OCRToken]
    direction: str = "LTR"  # "LTR" or "RTL"


class LLMField(BaseModel):
    """A single field from LLM output (before validation)."""
    type: str               # text_box | signature | checkbox | radio | initials
    bbox: List[float]       # [x1, y1, x2, y2]
    confidence: float       # 0.0–1.0
    source: str             # detector | detector_corrected | llm_added


class LLMOutput(BaseModel):
    """Full LLM response for one page."""
    page: int
    fields: List[LLMField]
