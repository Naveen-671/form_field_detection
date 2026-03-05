import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from app.config import settings
from app.schemas.response_schema import PageResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Transparent overlay colours per field type  (R, G, B, alpha 0-1)
# ──────────────────────────────────────────────────────────────────────────────
FIELD_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "text_box":  (0.18, 0.55, 0.98, 0.22),   # blue
    "checkbox":  (0.13, 0.78, 0.35, 0.28),   # green
    "radio":     (0.98, 0.65, 0.10, 0.28),   # orange
    "signature": (0.91, 0.20, 0.30, 0.25),   # red
    "initials":  (0.65, 0.30, 0.85, 0.25),   # purple
    "date":      (1.00, 0.55, 0.00, 0.25),   # deep orange
}
DEFAULT_COLOR = (0.50, 0.50, 0.50, 0.20)      # grey fallback

# Border colours (fully opaque versions of the above)
BORDER_COLORS: Dict[str, Tuple[float, float, float]] = {
    "text_box":  (0.18, 0.55, 0.98),
    "checkbox":  (0.13, 0.78, 0.35),
    "radio":     (0.98, 0.65, 0.10),
    "signature": (0.91, 0.20, 0.30),
    "initials":  (0.65, 0.30, 0.85),
    "date":      (1.00, 0.55, 0.00),
}
DEFAULT_BORDER = (0.50, 0.50, 0.50)


class MaskingService:

    @classmethod
    def apply_masks(
        cls,
        pdf_bytes: bytes,
        page_results: List[PageResult],
        page_images: List[dict],
    ) -> Optional[str]:
        """
        Draw transparent coloured overlays with type labels on the PDF.

        Each field type gets a distinct colour so you can see what was
        detected as what.  A small label (e.g. "text_box") is placed
        at the top-left corner of each box.

        Returns:
            Absolute file path of the saved annotated PDF, or None on failure.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.error(f"Failed to open PDF for masking: {exc}")
            return None

        page_info_map = {pi["page_number"]: pi for pi in page_images}
        total_annotations = 0

        for page_result in page_results:
            page_number = page_result.page
            page_idx = page_number - 1

            if page_idx < 0 or page_idx >= len(doc):
                logger.warning(f"Page {page_number} out of PDF range, skipping")
                continue

            page_info = page_info_map.get(page_number)
            if page_info is None:
                logger.warning(f"No image info for page {page_number}, skipping")
                continue

            resized_image = page_info["image"]
            original_size = page_info["original_size"]

            scale_x = original_size[0] / resized_image.width
            scale_y = original_size[1] / resized_image.height

            pdf_page = doc[page_idx]

            for field in page_result.fields:
                x1, y1, x2, y2 = field.bbox
                pdf_rect = fitz.Rect(
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y,
                )

                fill_rgba = FIELD_COLORS.get(field.type, DEFAULT_COLOR)
                border_rgb = BORDER_COLORS.get(field.type, DEFAULT_BORDER)
                r, g, b, alpha = fill_rgba

                # --- Draw transparent filled rectangle ---
                shape = pdf_page.new_shape()
                shape.draw_rect(pdf_rect)
                shape.finish(
                    color=border_rgb,       # border colour
                    fill=(r, g, b),         # fill colour
                    width=0.5,              # border width
                    fill_opacity=alpha,     # transparency
                    stroke_opacity=0.7,
                )
                shape.commit()

                # --- Add type label at top-left corner ---
                label = field.type
                font_size = 5.5
                label_rect = fitz.Rect(
                    pdf_rect.x0 + 1,
                    pdf_rect.y0 + 0.5,
                    pdf_rect.x0 + len(label) * 3.5 + 4,
                    pdf_rect.y0 + font_size + 2,
                )
                # Only draw label if box is wide enough
                if pdf_rect.width > 20 and pdf_rect.height > 8:
                    # Small background for label readability
                    label_bg = pdf_page.new_shape()
                    label_bg.draw_rect(label_rect)
                    label_bg.finish(
                        color=None,
                        fill=(1, 1, 1),
                        fill_opacity=0.75,
                    )
                    label_bg.commit()

                    pdf_page.insert_text(
                        fitz.Point(label_rect.x0 + 1, label_rect.y1 - 1.5),
                        label,
                        fontsize=font_size,
                        color=border_rgb,
                        fontname="helv",
                    )

                total_annotations += 1

        # Save
        try:
            output_path = cls._save_masked_pdf(doc)
        except Exception as exc:
            logger.error(f"Failed to save masked PDF: {exc}")
            doc.close()
            return None

        doc.close()
        logger.info(f"Masking complete: {total_annotations} annotations → {output_path}")
        return output_path

    @classmethod
    def _save_masked_pdf(cls, doc: fitz.Document) -> str:
        """Save annotated PDF to output directory. Returns absolute path."""
        os.makedirs(settings.MASKED_OUTPUT_DIR, exist_ok=True)
        filename = f"masked_{uuid.uuid4().hex[:8]}.pdf"
        output_path = os.path.abspath(
            os.path.join(settings.MASKED_OUTPUT_DIR, filename)
        )
        doc.save(output_path)
        return output_path
