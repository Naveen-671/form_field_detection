import logging
from fastapi import FastAPI
from app.routers import process
from app.routers import inspect as inspect_router
from app.services.detector_service import DetectorService
from app.services.ocr_service import OCRService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# Enable DEBUG for our pipeline modules so OCR token details appear in logs
logging.getLogger("app.routers.process").setLevel(logging.DEBUG)
logging.getLogger("app.routers.inspect").setLevel(logging.DEBUG)
logging.getLogger("app.services.ocr_service").setLevel(logging.DEBUG)

app = FastAPI(
    title="Document Field Detection & Masking API",
    version="1.0.0",
    description="Hybrid document field detection using visual detection, OCR, and multimodal LLM reasoning"
)

app.include_router(process.router)
app.include_router(inspect_router.router)


@app.on_event("startup")
async def startup():
    DetectorService.load_model()
    OCRService.load_model()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "detector_ready": DetectorService.is_ready(),
        "ocr_ready": OCRService.is_ready()
    }



@app.on_event("startup")
async def startup():
    DetectorService.load_model()
    OCRService.load_model()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "detector_ready": DetectorService.is_ready(),
        "ocr_ready": OCRService.is_ready()
    }
