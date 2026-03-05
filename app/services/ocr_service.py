import logging
from typing import List, Optional

import numpy as np
from PIL import Image

from app.config import settings
from app.schemas.request_schema import OCRResult, OCRToken

logger = logging.getLogger(__name__)


class OCRService:
    _reader = None

    @classmethod
    def load_model(cls) -> None:
        if cls._reader is not None:
            return
        logger.info("Loading PaddleOCR reader")
        try:
            from paddleocr import PaddleOCR
            # Using use_angle_cls=True to handle rotated images, and lang based on settings
            cls._reader = PaddleOCR(use_angle_cls=True, lang=settings.OCR_LANG)
            logger.info("PaddleOCR reader loaded successfully")
        except Exception as exc:
            logger.error("Failed to load PaddleOCR reader: %s", exc)
            raise

    @classmethod
    def extract(cls, image: Image.Image) -> OCRResult:
        if cls._reader is None:
            raise RuntimeError("OCR reader not loaded.")
        image_np = np.array(image)
        try:
            results = cls._reader.ocr(image_np)
        except Exception as exc:
            logger.error("PaddleOCR inference failed: %s", exc)
            return OCRResult(tokens=[], direction="LTR")

        tokens: List[OCRToken] = []
        if results and len(results) > 0 and results[0]:
            for item in results[0]: # PaddleOCR returns a list of results per image/page
                try:
                    token = cls._parse_item(item)
                    if token is not None:
                        tokens.append(token)
                except Exception as exc:
                    logger.debug("Skipping OCR item: %s", exc)

        direction = cls._detect_direction(tokens)
        logger.info("OCR extracted %d tokens, direction: %s", len(tokens), direction)
        for i, tok in enumerate(tokens):
            logger.debug('OCR token %03d: "%s" @ [%.0f,%.0f,%.0f,%.0f]',
                         i, tok.text, *tok.bbox)
        return OCRResult(tokens=tokens, direction=direction)

    @staticmethod
    def _parse_item(item) -> "Optional[OCRToken]":
        if not (isinstance(item, (list, tuple)) and len(item) >= 2):
            return None
        polygon = item[0]
        text = str(item[1]).strip()
        confidence = float(item[2]) if len(item) >= 3 else 1.0
        if confidence < 0.3 or not text:
            return None
        if isinstance(polygon, (list, tuple)) and len(polygon) >= 2:
            try:
                xs = [float(p[0]) for p in polygon]
                ys = [float(p[1]) for p in polygon]
                return OCRToken(text=text, bbox=[min(xs), min(ys), max(xs), max(ys)])
            except (TypeError, IndexError, ValueError):
                return None
        return None

    @classmethod
    def _detect_direction(cls, tokens: List[OCRToken]) -> str:
        if len(tokens) < 4:
            return "LTR"
        mid = len(tokens) // 2
        first_x = sum(t.bbox[0] for t in tokens[:mid]) / mid
        second_x = sum(t.bbox[0] for t in tokens[mid:]) / (len(tokens) - mid)
        return "RTL" if second_x < first_x - 50 else "LTR"

    @classmethod
    def is_ready(cls) -> bool:
        return cls._reader is not None
