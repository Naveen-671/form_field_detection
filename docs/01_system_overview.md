# System Overview

## Objective

Build a modular, local-first hybrid document field detection and masking system using Python / FastAPI. The system combines:

- **Visual detection** (high recall object detection for candidate field proposals)
- **OCR-based semantic grounding** (raw text + bounding boxes for context)
- **Multimodal LLM reasoning** (validate, classify, and recover missed fields)
- **Strict structured JSON contracts** between every layer
- **Deterministic backend masking** (coordinate-based PDF redaction)
- **No model fine-tuning** in Phase 1

The system mimics Gemini-like full-page reasoning behavior while remaining fully open-source and self-hostable.

---

## Phase 1 Scope (Current Build)

- Single-machine local execution (no Docker, no Kubernetes)
- Synchronous request processing (no async workers, no queues)
- One FastAPI application with modular service files
- Run via `uvicorn app.main:app --host 0.0.0.0 --port 8000`

Phase-2 (future) will add containerization, GPU scaling, caching, and distributed infra — documents 08–10 capture that future state but **must NOT be implemented now**.

---

## Core Principles (MANDATORY)

1. **LLM never performs masking.** Masking is deterministic, backend-only, coordinate-based.
2. **LLM must output strict JSON only.** No markdown, no commentary, no explanations.
3. **All LLM output is untrusted.** The validation layer filters hallucinations before masking.
4. **Detector optimized for recall, not precision.** Low confidence threshold; over-propose, then let LLM prune.
5. **LLM is the reasoning/correction layer**, not the primary detector.
6. **Modular service separation.** Each service file is independently replaceable without touching others.
7. **No global mutable state** except model singletons loaded at startup.

---

## High-Level Processing Pipeline

```
PDF Upload
  │
  ▼
Page Image Extraction (pdf_service.py)
  │  ── Convert each PDF page to PNG, resize to max 1216px preserving aspect ratio
  │
  ▼
┌──────────────────────────────────────┐
│  Per-Page Processing (parallel-ready) │
│                                      │
│  1. Detector Service → proposals[]   │
│  2. OCR Service → tokens[]           │
│  3. LLM Service (image + proposals   │
│     + tokens) → classified fields[]  │
│  4. Validation Service → filtered    │
│     fields[]                         │
└──────────────────────────────────────┘
  │
  ▼
Aggregate all pages
  │
  ▼
Masking Engine (masking_service.py)
  │  ── Map normalized coords → original PDF coords
  │  ── Apply redaction rectangles via PyMuPDF
  │
  ▼
Return: Structured JSON response + optional masked PDF file path
```

---

## Python Version & Core Dependencies

| Package | Purpose | Min Version |
|---------|---------|-------------|
| Python | Runtime | 3.10+ |
| fastapi | Web framework | 0.104+ |
| uvicorn | ASGI server | 0.24+ |
| pydantic | Schema validation | 2.0+ |
| PyMuPDF (fitz) | PDF manipulation & masking | 1.23+ |
| Pillow | Image processing | 10.0+ |
| paddleocr | OCR extraction | 2.7+ |
| paddlepaddle | PaddleOCR backend | 2.5+ |
| ultralytics | YOLO inference engine (FFDNet-L) | 8.1+ |
| httpx | HTTP client for NIM API | 0.25+ |
| python-multipart | File upload parsing | 0.0.6+ |
| numpy | Array operations | 1.24+ |

---

## Core Models

### Detector: FFDNet-L (Form Field Detection Network — Large)

- **Model**: FFDNet-L from [jbarrow/FFDNet-L](https://huggingface.co/jbarrow/FFDNet-L) — a YOLO11-based object detector (25M params) fine-tuned on the [CommonForms](https://arxiv.org/abs/2509.16506) dataset specifically for document form field detection
- **Model file**: `FFDNet-L (1).pt` located in the project root (loaded via `ultralytics.YOLO(path, task="detect")`)
- **Trained resolution**: 1216px
- **3 detection classes**: `{0: "TextBox", 1: "ChoiceButton", 2: "Signature"}`
- **Configuration**: confidence threshold = 0.3, NMS IoU threshold = 0.1 (aggressive NMS), augment = True, imgsz = 1216
- **Output**: bounding box proposals with confidence scores AND class labels (TextBox, ChoiceButton, Signature)
- **Package**: `pip install ultralytics>=8.1.0` (model is loaded directly via ultralytics YOLO, NOT via the `commonforms` CLI package)

### OCR: PaddleOCR

- **Engine**: PaddleOCR with `use_angle_cls=True`, `lang='en'`
- **Output**: raw text tokens with bounding boxes in pixel coordinates
- **No pre-cleaning**: LLM reasons over raw OCR output

### Multimodal LLM: Qwen3.5-397B-A17B via NVIDIA NIM

- **API**: OpenAI-compatible chat completions endpoint served by NVIDIA NIM
- **Endpoint**: configurable via environment variable `NIM_API_URL` (default: `https://integrate.api.nvidia.com/v1/chat/completions`)
- **API Key**: via environment variable `NIM_API_KEY`
- **Accepts**: base64-encoded images in multimodal message format

---

## Coordinate System Convention (CRITICAL)

All bounding boxes throughout the system use **pixel coordinates on the normalized (resized) image**:

```
bbox = [x1, y1, x2, y2]
```

- `(x1, y1)` = top-left corner in pixels
- `(x2, y2)` = bottom-right corner in pixels
- Coordinates are relative to the **resized image** (max dimension 1216px)
- The masking service is responsible for mapping these back to original PDF coordinates using the scale factor

**Scale factor computation:**
```python
scale_x = original_pdf_page_width / resized_image_width
scale_y = original_pdf_page_height / resized_image_height
original_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
```

---

## Target Field Types

| Type | Description | Visual Cue | Detector Source |
|------|-------------|------------|----------------|
| `text_box` | Text input field, text area, form field | Rectangle with lines/border, often with label nearby | Detector: "TextBox" |
| `signature` | Signature capture area | Larger rectangle, often with "Sign here" label or "X" mark | Detector: "Signature" |
| `checkbox` | Checkbox input | Small square, may be checked or unchecked | Detector: "ChoiceButton" → LLM splits |
| `radio` | Radio button | Small circle, may be filled or empty | Detector: "ChoiceButton" → LLM splits |
| `initials` | Initials field | Small rectangle, often labeled "Initials" | LLM-only (no detector class) |

**Detector-to-LLM class mapping:** The detector outputs 3 classes. The LLM refines them:
- `TextBox` → `text_box`
- `ChoiceButton` → `checkbox` or `radio` (LLM decides based on visual shape)
- `Signature` → `signature` or `initials` (LLM decides based on context/size)

---

## Target Capabilities

- Detect text fields, signatures, checkboxes, radio buttons, initials
- Recover fields the detector missed (via LLM visual reasoning)
- Remove decorative borders and non-interactive elements (via LLM pruning)
- Handle multilingual layouts (LTR + RTL)
- Support scanned documents and photographed forms
- Support both drawn and digitally-placed signatures
- Process multi-page PDFs end-to-end
- Return per-page structured JSON with field metadata