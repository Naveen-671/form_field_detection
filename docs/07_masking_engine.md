# Masking Engine

## File: `app/services/masking_service.py`

## Responsibility

Deterministically apply redaction overlays to the original PDF based on validated field coordinates. The masking engine is a **pure coordinate-to-PDF mapper** — it receives validated bounding boxes and applies black rectangles at the corresponding positions in the original PDF.

---

## Key Dependency

- **PyMuPDF** (`fitz`): `pip install PyMuPDF`

---

## Configuration (from `app/config.py`)

```python
MASK_COLOR_RGB = (0, 0, 0)         # Black redaction rectangles
MASKED_OUTPUT_DIR = "output/masked"  # Directory for saved masked PDFs
```

---

## Coordinate Mapping (CRITICAL)

The processing pipeline works on **resized images** (max 1216px). All bounding boxes from detection, OCR, and LLM are in the resized image coordinate space. The masking engine must **map these back to original PDF page coordinates** before applying redactions.

### Scale Factor Computation

```python
# For each page:
# original_size = (original_page_width, original_page_height) in PDF points
# resized_size  = (resized_image_width, resized_image_height) in pixels

scale_x = original_page_width / resized_image_width
scale_y = original_page_height / resized_image_height

# Map a bbox from resized coords → original PDF coords:
pdf_x1 = x1 * scale_x
pdf_y1 = y1 * scale_y
pdf_x2 = x2 * scale_x
pdf_y2 = y2 * scale_y
```

> **Note:** PyMuPDF uses a coordinate system where (0,0) is the **top-left** of the page, same as image coordinates. No Y-axis inversion needed.

---

## Implementation Specification

```python
import fitz  # PyMuPDF
import os
import uuid
import logging
from typing import List, Optional
from app.schemas.response_schema import PageResult
from app.config import settings

logger = logging.getLogger(__name__)


class MaskingService:

    @classmethod
    def apply_masks(
        cls,
        pdf_bytes: bytes,
        page_results: List[PageResult],
        page_images: List[dict]
    ) -> Optional[str]:
        """
        Apply redaction masks to original PDF and save the masked copy.

        Args:
            pdf_bytes: Original PDF file bytes
            page_results: List of PageResult with validated fields per page
            page_images: List of page info dicts with keys:
                - "page_number": int (1-based)
                - "image": PIL.Image (resized)
                - "original_size": (width, height) in PDF points

        Returns:
            Absolute file path of the masked PDF, or None on failure
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF for masking: {e}")
            return None

        total_redactions = 0

        for page_result in page_results:
            page_number = page_result.page  # 1-based
            page_idx = page_number - 1      # 0-based for PyMuPDF

            if page_idx >= len(doc):
                logger.warning(f"Page {page_number} out of range, skipping")
                continue

            # Find matching page_info for coordinate mapping
            page_info = None
            for pi in page_images:
                if pi["page_number"] == page_number:
                    page_info = pi
                    break

            if page_info is None:
                logger.warning(f"No image info for page {page_number}, skipping masking")
                continue

            pdf_page = doc[page_idx]
            resized_image = page_info["image"]
            original_size = page_info["original_size"]  # (width, height) in PDF points

            # Compute scale factors
            resized_w = resized_image.width
            resized_h = resized_image.height
            original_w, original_h = original_size

            scale_x = original_w / resized_w
            scale_y = original_h / resized_h

            # Apply redactions for each validated field
            for field in page_result.fields:
                x1, y1, x2, y2 = field.bbox

                # Map to original PDF coordinates
                pdf_rect = fitz.Rect(
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y
                )

                # Add redaction annotation
                pdf_page.add_redact_annot(
                    pdf_rect,
                    fill=settings.MASK_COLOR_RGB
                )
                total_redactions += 1

            # Apply all redactions on this page
            pdf_page.apply_redactions()

        # Save masked PDF
        output_path = cls._save_masked_pdf(doc)
        doc.close()

        logger.info(f"Applied {total_redactions} redactions, saved to {output_path}")
        return output_path

    @classmethod
    def _save_masked_pdf(cls, doc: fitz.Document) -> str:
        """Save masked PDF to output directory and return file path."""
        os.makedirs(settings.MASKED_OUTPUT_DIR, exist_ok=True)
        filename = f"masked_{uuid.uuid4().hex[:8]}.pdf"
        output_path = os.path.join(settings.MASKED_OUTPUT_DIR, filename)
        doc.save(output_path)
        return os.path.abspath(output_path)
```

---

## Also Needed: PDF Service (`app/services/pdf_service.py`)

The PDF service handles converting input files to page images. It is tightly related to masking because it establishes the coordinate mapping.

```python
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
from typing import List, Tuple
from app.config import settings

logger = logging.getLogger(__name__)


class PDFService:

    @classmethod
    def extract_pages(
        cls,
        file_bytes: bytes,
        filename: str
    ) -> Tuple[List[dict], bytes]:
        """
        Convert PDF or image file into a list of page images.

        Args:
            file_bytes: Raw file bytes
            filename: Original filename (used to determine type)

        Returns:
            Tuple of:
            - List of page dicts: [{"image": PIL.Image, "page_number": int, "original_size": (w,h)}]
            - Original PDF bytes (or None if input was an image)
        """
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if ext == "pdf":
            return cls._extract_from_pdf(file_bytes), file_bytes
        elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
            return cls._extract_from_image(file_bytes), None
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @classmethod
    def _extract_from_pdf(cls, pdf_bytes: bytes) -> List[dict]:
        """Extract page images from PDF."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]

            # Get original page dimensions in PDF points
            original_width = page.rect.width
            original_height = page.rect.height

            # Render page to pixmap
            # Calculate zoom to get reasonable resolution
            zoom = 2.0  # 2x zoom for better quality
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)

            # Convert pixmap to PIL Image
            img_bytes = pixmap.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            # Resize to max dimension
            resized_image = cls._resize_to_max_dimension(image)

            pages.append({
                "image": resized_image,
                "page_number": page_idx + 1,  # 1-based
                "original_size": (original_width, original_height)
            })

        doc.close()
        logger.info(f"Extracted {len(pages)} pages from PDF")
        return pages

    @classmethod
    def _extract_from_image(cls, image_bytes: bytes) -> List[dict]:
        """Handle single image input (treat as 1-page document)."""
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = (image.width, image.height)
        resized_image = cls._resize_to_max_dimension(image)

        return [{
            "image": resized_image,
            "page_number": 1,
            "original_size": original_size
        }]

    @classmethod
    def _resize_to_max_dimension(cls, image: Image.Image) -> Image.Image:
        """
        Resize image so max dimension is MAX_IMAGE_DIMENSION (1216px).
        Preserve aspect ratio. If already smaller, return as-is.
        """
        max_dim = settings.MAX_IMAGE_DIMENSION
        w, h = image.size

        if max(w, h) <= max_dim:
            return image

        if w >= h:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        else:
            new_h = max_dim
            new_w = int(w * (max_dim / h))

        return image.resize((new_w, new_h), Image.LANCZOS)
```

---

## Critical Rules

1. **Masking must NOT depend on LLM reasoning.** Only use the validated structured JSON (output of `ValidationService`).
2. **Never apply raw LLM output.** Always use fields that have passed validation.
3. **Coordinate mapping is mandatory.** Resized image coords ≠ original PDF coords.
4. **PyMuPDF redaction workflow**: `add_redact_annot()` → `apply_redactions()`. Must call `apply_redactions()` after adding all annotations on a page.
5. **Output directory auto-creation**: `MaskingService` must create `output/masked/` if it doesn't exist.
6. **Unique filenames**: Use UUID to prevent overwriting previous outputs.
7. **Graceful failure**: If masking fails, log error and return `None` — the JSON response should still be returned without the masked PDF path.