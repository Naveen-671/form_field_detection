"""Check actual image size and base64 payload being sent to LLM."""
import sys, os, base64, io
sys.path.insert(0, r"c:\Users\Naveen.r\field_detection")

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

from app.services.pdf_service import PDFService
from PIL import Image

pdf_path = r"c:\Users\Naveen.r\field_detection\app\assets\Consent_Form_Final_2607.pdf"
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

pages, _ = PDFService.extract_pages(pdf_bytes, "Consent_Form_Final_2607.pdf")
img = pages[0]["image"]
print(f"Image size: {img.size}  mode: {img.mode}")

# Simulate what LLM service does
img_rgb = img.convert("RGB")
buf = io.BytesIO()
img_rgb.save(buf, format="JPEG", quality=85)
b64 = base64.b64encode(buf.getvalue()).decode()
print(f"JPEG bytes: {len(buf.getvalue()):,}")
print(f"Base64 chars: {len(b64):,}  (~{len(b64)/1024:.0f} KB)")
