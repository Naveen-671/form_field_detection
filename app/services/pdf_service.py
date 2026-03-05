import io
import logging
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class PDFService:

    @classmethod
    def extract_pages(
        cls,
        file_bytes: bytes,
        filename: str,
    ) -> Tuple[List[dict], Optional[bytes]]:
        """
        Convert a PDF or image file into a list of page image dicts.

        Args:
            file_bytes: Raw file bytes (PDF or image)
            filename:   Original filename (extension determines handling)

        Returns:
            Tuple of:
            - List of page dicts:
                {
                    "image": PIL.Image (resized to max 1216px),
                    "page_number": int (1-based),
                    "original_size": (width_pts, height_pts)  ← PDF points for PDFs,
                                                                 pixels for images
                }
            - Original PDF bytes (for masking), or None if input was an image
        """
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if ext == "pdf":
            return cls._extract_from_pdf(file_bytes), file_bytes
        elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
            return cls._extract_from_image(file_bytes), None
        else:
            raise ValueError(f"Unsupported file type: '.{ext}'")

    @classmethod
    def _extract_from_pdf(cls, pdf_bytes: bytes) -> List[dict]:
        """Render each PDF page to a resized PIL Image."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: List[dict] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]

            # Original page dimensions in PDF points (used for masking coordinate mapping)
            original_width = page.rect.width
            original_height = page.rect.height

            # Render at 2× zoom for quality
            matrix = fitz.Matrix(2.0, 2.0)
            pixmap = page.get_pixmap(matrix=matrix)

            # Pixmap → PIL Image
            img_bytes = pixmap.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Resize to max 1216px (preserving aspect ratio)
            resized = cls._resize_to_max_dimension(image)

            pages.append({
                "image": resized,
                "page_number": page_idx + 1,
                "original_size": (original_width, original_height),
            })

        doc.close()
        logger.info(f"Extracted {len(pages)} pages from PDF")
        return pages

    @classmethod
    def _extract_from_image(cls, image_bytes: bytes) -> List[dict]:
        """Wrap a single image as a 1-page document."""
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = (float(image.width), float(image.height))
        resized = cls._resize_to_max_dimension(image)

        return [{
            "image": resized,
            "page_number": 1,
            "original_size": original_size,
        }]

    @classmethod
    def _resize_to_max_dimension(cls, image: Image.Image) -> Image.Image:
        """
        Resize image so max(w, h) == MAX_IMAGE_DIMENSION.
        Returns image unchanged if it is already within bounds.
        """
        max_dim = settings.MAX_IMAGE_DIMENSION
        w, h = image.size

        if max(w, h) <= max_dim:
            return image

        if w >= h:
            new_w = max_dim
            new_h = int(h * max_dim / w)
        else:
            new_h = max_dim
            new_w = int(w * max_dim / h)

        return image.resize((new_w, new_h), Image.LANCZOS)
