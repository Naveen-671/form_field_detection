from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class FieldResult(BaseModel):
    """A single validated field in the API response."""
    type: str               # text_box | signature | checkbox | radio | initials
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords on resized image
    confidence: float       # 0.0–1.0
    source: str             # detector | detector_corrected | llm_added


class OcrTokenResult(BaseModel):
    """A single OCR token (returned in diagnostic modes)."""
    text: str
    bbox: List[float]       # [x1, y1, x2, y2]


class SpatialField(BaseModel):
    """Detector field annotated with nearby/overlapping OCR tokens (diagnostic)."""
    type: str
    bbox: List[float]
    confidence: float
    source: str
    ocr_inside: List[str]   = []   # OCR tokens whose bbox overlaps this field
    ocr_nearby: List[str]   = []   # OCR tokens within 30px of this field


class PageResult(BaseModel):
    """Results for a single page."""
    page: int
    fields: List[FieldResult]
    proposal_count: int       # Number of detector proposals before LLM
    source: str               # "llm" or "detector_fallback"
    # Diagnostic fields — populated only when pipeline != "full"
    ocr_tokens: Optional[List[OcrTokenResult]] = None
    spatial_fields: Optional[List[SpatialField]] = None
    ocr_direction: Optional[str] = None


class DocumentResponse(BaseModel):
    """Top-level API response."""
    request_id: str
    filename: str
    total_pages: int
    pages: List[PageResult]
    masked_pdf_path: Optional[str] = None
    processing_time_seconds: float
