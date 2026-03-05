# Infrastructure and Scaling (Phase 2 — DO NOT IMPLEMENT NOW)

> **⚠️ This document describes FUTURE infrastructure plans for Phase 2+. Do NOT implement any of this during Phase 1 coding. Phase 1 runs locally as a single-process FastAPI application with `uvicorn`.**

---

## Phase 1 (Current) — Local Development Setup

```
Single Machine
├── uvicorn (port 8000)
│   └── FastAPI app
│       ├── DetectorService (FFDNet-L — loaded in-process)
│       ├── OCRService (PaddleOCR — loaded in-process)
│       ├── LLMService (HTTP calls to external NIM endpoint)
│       ├── ValidationService (pure Python, no external deps)
│       └── MaskingService (PyMuPDF, in-process)
│
└── NVIDIA NIM Server (port 8001, started separately)
    └── Qwen3.5-397B-A17B
```

### Phase 1 Startup

```bash
# Terminal 1: Start NVIDIA NIM (if running locally)
# (Follow NVIDIA NIM documentation for model serving)

# Terminal 2: Start the application
cd field_detection
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Phase 2 — Containerized Deployment (Future)

### Architecture

- Each service containerized independently
- Kubernetes deployment with independent autoscaling
- GPU node pools for detector and LLM services

### Container Strategy

| Service | Container | GPU Required |
|---------|-----------|-------------|
| Orchestrator (FastAPI) | Lightweight Python container | No |
| Detector (FFDNet-L) | Separate GPU-enabled container | Small GPU (T4 / L4) |
| OCR (PaddleOCR) | Separate container | Optional GPU |
| LLM (NIM) | NVIDIA NIM container | High-memory GPU (A100 / H100) |
| Masking (PyMuPDF) | CPU container or part of orchestrator | No |

### Scaling Strategy

- Detector pods: scale by inference queue depth
- OCR pods: scale by request count
- LLM pods: scale by GPU utilization (limited by VRAM)
- Orchestrator: scale by concurrent request count

### Caching (Phase 2)

- Cache LLM responses per page content hash (SHA-256 of image bytes)
- Cache OCR output per page content hash
- Use Redis or filesystem cache

### Monitoring

- Per-stage latency (detector, OCR, LLM, validation, masking)
- GPU memory usage and utilization
- Request success/failure rate
- LLM retry count and fallback rate
- Queue depth per service