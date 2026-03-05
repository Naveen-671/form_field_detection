"""Test NIM API with actual page image to measure real response time."""
import sys, os, base64, io, time
sys.path.insert(0, r"c:\Users\Naveen.r\field_detection")

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import httpx
from app.services.pdf_service import PDFService
from PIL import Image

# Extract page image
pdf_path = r"c:\Users\Naveen.r\field_detection\app\assets\Consent_Form_Final_2607.pdf"
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

pages, _ = PDFService.extract_pages(pdf_bytes, "Consent_Form_Final_2607.pdf")
img = pages[0]["image"]
img_rgb = img.convert("RGB")
buf = io.BytesIO()
img_rgb.save(buf, format="JPEG", quality=85)
b64 = base64.b64encode(buf.getvalue()).decode()
print(f"Image: {img.size}  b64 size: {len(b64)//1024}KB")

key = os.getenv("NIM_API_KEY", "")
url = "https://integrate.api.nvidia.com/v1/chat/completions"

payload = {
    "model":"google/gemma-3n-e4b-it",
    "messages": [
        {"role": "system", "content": "You output only valid JSON. No markdown."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": 'Return JSON: {"fields": [{"type": "text_box", "bbox": [10,20,100,40]}]}'}
        ]}
    ],
    "temperature": 0.0,
    "max_tokens": 256,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}

headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

print("Sending image request...")
t0 = time.time()
try:
    with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
        r = client.post(url, headers=headers, json=payload)
    elapsed = time.time() - t0
    print(f"Status: {r.status_code}  Time: {elapsed:.1f}s")
    if r.status_code == 200:
        data = r.json()
        print("Response:", data["choices"][0]["message"]["content"][:500])
    else:
        print("Error:", r.text[:500])
except Exception as e:
    elapsed = time.time() - t0
    print(f"Exception after {elapsed:.1f}s: {e}")
