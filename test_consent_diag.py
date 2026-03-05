"""Detailed diagnosis: which OCR tokens trigger date reclassification"""
import httpx, sys, os, json
sys.path.insert(0, '.')

import re
_RECLASS_PAT = [
    (re.compile(r"\bsignature\b|\bsigned\b|\bsign\b|\bauthori[sz]ed\b", re.I), "signature"),
    (re.compile(r"\bprepared\s+by\b|\bapproved\s+by\b|\bverified\s+by\b|\breviewed\s+by\b|\bwitnessed\s+by\b|\baccept(?:ed)?\s+by\b", re.I), "signature"),
    (re.compile(r"\bdate\b|^date", re.I), "date"),
]

pdf_path = r'c:\Users\Naveen.r\field_detection\app\assets\Consent_Form_Final_2607.pdf'
with open(pdf_path, 'rb') as f:
    r = httpx.post(
        'http://localhost:8000/process-document?pipeline=full',
        files={'file': (os.path.basename(pdf_path), f, 'application/pdf')},
        timeout=900
    )
resp = r.json()
page = resp['pages'][0]

# Show which OCR tokens match reclassification patterns
print("=== OCR tokens matching reclassification patterns ===")
for t in page.get('ocr_tokens', []):
    for pat, target in _RECLASS_PAT:
        if pat.search(t['text']):
            bbox = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*t['bbox'])
            print(f"  \"{t['text']}\" @ {bbox} -> {target}")
            break

# Show all date/signature fields
print("\n=== All date/signature fields ===")
for f in page['fields']:
    if f['type'] in ('date', 'signature'):
        bbox = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*f['bbox'])
        print(f"  [{f['type']:10s}] {bbox}  conf={f['confidence']:.2f}  src={f['source']}")

# Type summary
types = {}
for f in page['fields']:
    t = f['type']
    types[t] = types.get(t, 0) + 1
print(f"\n=== Type summary: {types} ===")
