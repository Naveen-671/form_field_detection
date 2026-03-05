import logging
from typing import List

from PIL import Image
from ultralytics import YOLO

from app.config import settings
from app.schemas.request_schema import Proposal

logger = logging.getLogger(__name__)

# FFDNet-L class ID → label mapping (from commonforms source)
CLASS_MAP = {0: "TextBox", 1: "ChoiceButton", 2: "Signature"}


class DetectorService:
    _model: YOLO = None

    @classmethod
    def load_model(cls) -> None:
        """Load FFDNet-L model once at application startup. Called from main.py startup event."""
        if cls._model is not None:
            return
        logger.info(f"Loading FFDNet-L model from: {settings.DETECTOR_MODEL_PATH}")
        try:
            cls._model = YOLO(settings.DETECTOR_MODEL_PATH, task="detect")
            logger.info("FFDNet-L model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load FFDNet-L model: {exc}")
            raise

    @classmethod
    def detect(cls, image: Image.Image) -> List[Proposal]:
        """
        Run FFDNet-L detection on a resized page image.

        Args:
            image: PIL.Image already resized to max 1216px dimension

        Returns:
            List of Proposal objects with [x1, y1, x2, y2] pixel coords,
            confidence score, and FFDNet-L class label.

        Raises:
            RuntimeError: if model is not loaded
        """
        if cls._model is None:
            raise RuntimeError("FFDNet-L model not loaded. Call DetectorService.load_model() first.")

        # Run inference — parameters from commonforms inference.py defaults
        results = cls._model.predict(
            source=image,
            conf=settings.DETECTOR_CONFIDENCE_THRESHOLD,
            iou=settings.DETECTOR_NMS_IOU_THRESHOLD,
            augment=settings.DETECTOR_AUGMENT,
            imgsz=settings.DETECTOR_IMAGE_SIZE,
            verbose=False,
            device="cpu",  # Phase 1: always CPU
        )

        proposals: List[Proposal] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.cpu().numpy():
                xyxy = box.xyxy[0].tolist()      # [x1, y1, x2, y2] pixel coords
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = CLASS_MAP.get(cls_id, "TextBox")  # fallback to TextBox if unknown

                proposals.append(Proposal(
                    bbox=xyxy,
                    confidence=conf,
                    label=label
                ))

        logger.info(f"FFDNet-L detected {len(proposals)} proposals")
        return proposals

    @classmethod
    def is_ready(cls) -> bool:
        return cls._model is not None
