"""Check all OCR tokens for the ServiceDelivery PDF"""
import httpx, sys, os, json
sys.path.insert(0, '.')

pdf_path = r'C:\Users\Naveen.r\field_detection\app\assets\ServiceDeliveryAcceptanceFormVendors.pdf'
with open(pdf_path, 'rb') as f:
    r = httpx.post(
        'http://localhost:8000/process-document?pipeline=full',
        files={'file': (os.path.basename(pdf_path), f, 'application/pdf')},
        timeout=900
    )
resp = r.json()
page = resp['pages'][0]
tokens = page.get('ocr_tokens', [])
print(f"=== All {len(tokens)} OCR tokens ===")
for i, t in enumerate(tokens):
    bbox = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*t['bbox'])
    print(f"  {i+1:2d}. \"{t['text']}\" @ {bbox}")

print(f"\n=== All {len(page['fields'])} fields with types ===")
for f in page['fields']:
    bbox = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*f['bbox'])
    print(f"  [{f['type']:10s}] {bbox}  conf={f['confidence']:.2f}  src={f['source']}")
