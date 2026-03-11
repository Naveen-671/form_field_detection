# Document Field Detection & Masking API

A modular, local-first hybrid document field detection and masking system built with Python and FastAPI. The system intelligently detects and masks form fields in PDF documents using a combination of visual detection, OCR, and multimodal LLM reasoning.

## 🌟 Features

- **Hybrid Detection Approach**: Combines visual detection (YOLO-based), OCR, and multimodal LLM reasoning for accurate field detection
- **Multi-Field Support**: Detects text boxes, signatures, checkboxes, radio buttons, and initial fields
- **Intelligent Validation**: LLM-powered reasoning layer validates and classifies detected fields
- **Coordinate-Based Masking**: Deterministic, backend-only PDF redaction using pixel coordinates
- **Multi-Page Support**: Processes complex multi-page PDF documents end-to-end
- **RESTful API**: FastAPI-based REST API with automatic OpenAPI documentation

## 🏗️ Architecture

The system follows a modular pipeline architecture:

```
PDF Upload → Page Extraction → Per-Page Processing → Aggregation → Masking Engine
                                    ↓
                      1. Visual Detection (FFDNet-L)
                      2. OCR Extraction (PaddleOCR)
                      3. LLM Reasoning (Qwen via NIM)
                      4. Validation Layer
```

### Core Components

- **Detector Service**: FFDNet-L (YOLO11-based) object detector fine-tuned on CommonForms dataset
- **OCR Service**: PaddleOCR for text extraction with bounding boxes
- **LLM Service**: Qwen3.5-397B-A17B via NVIDIA NIM for multimodal reasoning
- **Validation Layer**: Filters hallucinations and validates LLM outputs
- **Masking Engine**: Coordinate-based PDF redaction using PyMuPDF

## 📋 Prerequisites

- Python 3.10 or higher
- NVIDIA NIM API key (for LLM service)
- Sufficient disk space for model weights

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Naveen-671/form_field_detection.git
cd form_field_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export NIM_API_KEY="your_nvidia_nim_api_key"
export NIM_API_URL="https://integrate.api.nvidia.com/v1/chat/completions"  # Optional, this is the default
```

## 🎯 Usage

### Starting the Server

Run the FastAPI server using uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Health Check

Check if the service is ready:

```bash
curl http://localhost:8000/health
```

### Processing a Document

Upload and process a PDF document:

```bash
curl -X POST "http://localhost:8000/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

## 📦 Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| fastapi | Web framework | ≥0.104.0 |
| uvicorn | ASGI server | ≥0.24.0 |
| pydantic | Schema validation | ≥2.0.0 |
| PyMuPDF | PDF manipulation | ≥1.23.0 |
| Pillow | Image processing | ≥10.0.0 |
| paddleocr | OCR extraction | ≥2.7.0 |
| paddlepaddle | PaddleOCR backend | ≥2.5.0 |
| ultralytics | YOLO inference | ≥8.1.0 |
| httpx | HTTP client | ≥0.25.0 |
| python-multipart | File upload | ≥0.0.6 |
| numpy | Array operations | ≥1.24.0 |

## 🎭 Detected Field Types

- **Text Box**: Text input fields, text areas, form fields
- **Signature**: Signature capture areas
- **Checkbox**: Checkbox inputs (checked or unchecked)
- **Radio Button**: Radio button inputs
- **Initials**: Initial fields

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [System Overview](docs/01_system_overview.md)
- [Service Orchestrator](docs/02_service_orchestrator.md)
- [Detector Service](docs/03_detector_service.md)
- [OCR Service](docs/04_ocr_service.md)
- [LLM Service](docs/05_llm_service_nim.md)
- [Validation Layer](docs/06_validation_layer.md)
- [Masking Engine](docs/07_masking_engine.md)
- [Infrastructure and Scaling](docs/08_infrastructure_and_scaling.md)
- [Security and Guardrails](docs/09_security_and_guardrails.md)
- [Operational Playbook](docs/10_operational_playbook.md)

## 🔧 Configuration

The system can be configured through environment variables:

- `NIM_API_KEY`: NVIDIA NIM API key (required)
- `NIM_API_URL`: NVIDIA NIM API endpoint (optional, defaults to NVIDIA's hosted endpoint)

## 🛡️ Core Principles

1. **LLM never performs masking** - Masking is deterministic, backend-only, coordinate-based
2. **LLM outputs strict JSON only** - No markdown, no commentary
3. **All LLM output is untrusted** - Validation layer filters hallucinations
4. **Detector optimized for recall** - Over-propose, let LLM prune
5. **Modular service separation** - Each service is independently replaceable

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **FFDNet-L Model**: [jbarrow/FFDNet-L](https://huggingface.co/jbarrow/FFDNet-L)
- **CommonForms Dataset**: [arXiv:2509.16506](https://arxiv.org/abs/2509.16506)
- **PaddleOCR**: Baidu's open-source OCR toolkit
- **NVIDIA NIM**: For multimodal LLM inference

## 📧 Contact

For questions or support, please open an issue on GitHub.
