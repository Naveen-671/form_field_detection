import os
from dotenv import load_dotenv

# Load .env from project root (one level above app/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class Settings:
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # File upload
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: list = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    # Image processing
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "1216"))

    # Detector (FFDNet-L — YOLO11-based document field detector)
    DETECTOR_MODEL_PATH: str = os.getenv("DETECTOR_MODEL_PATH", "FFDNet-L.pt")
    DETECTOR_CONFIDENCE_THRESHOLD: float = float(os.getenv("DETECTOR_CONF_THRESHOLD", "0.3"))
    DETECTOR_NMS_IOU_THRESHOLD: float = float(os.getenv("DETECTOR_NMS_IOU", "0.1"))
    DETECTOR_AUGMENT: bool = os.getenv("DETECTOR_AUGMENT", "true").lower() == "true"
    DETECTOR_IMAGE_SIZE: int = int(os.getenv("DETECTOR_IMAGE_SIZE", "1216"))

    # OCR (PaddleOCR)
    OCR_LANG: str = os.getenv("OCR_LANG", "en")
    OCR_USE_ANGLE_CLS: bool = os.getenv("OCR_USE_ANGLE_CLS", "true").lower() == "true"
    OCR_USE_GPU: bool = os.getenv("OCR_USE_GPU", "false").lower() == "true"

    # LLM (NVIDIA NIM — OpenAI-compatible API)
    NIM_API_URL: str = os.getenv("NIM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    NIM_API_KEY: str = os.getenv("NIM_API_KEY", "")
    NIM_MODEL_NAME: str = os.getenv("NIM_MODEL_NAME", "google/gemma-3n-e4b-it")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "1"))

    # Validation thresholds
    VALIDATION_MAX_PAGE_COVERAGE: float = float(os.getenv("VALIDATION_MAX_PAGE_COVERAGE", "0.70"))
    VALIDATION_DUPLICATE_IOU_THRESHOLD: float = float(os.getenv("VALIDATION_DUPLICATE_IOU", "0.50"))
    VALIDATION_MIN_BOX_SIZE_PX: int = int(os.getenv("VALIDATION_MIN_BOX_SIZE_PX", "5"))

    # Masking
    MASK_COLOR_RGB: tuple = (0, 0, 0)  # Black
    MASKED_OUTPUT_DIR: str = os.getenv("MASKED_OUTPUT_DIR", "output/masked")


settings = Settings()
