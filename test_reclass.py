"""Quick test for reclassify_by_ocr_context()"""
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.DEBUG)

from app.schemas.request_schema import OCRToken
from app.schemas.response_schema import FieldResult
from app.services.spatial_analysis import reclassify_by_ocr_context, _RECLASSIFY_PATTERNS

# Test pattern matching
print("=== Pattern matching ===")
for pat, target in _RECLASSIFY_PATTERNS:
    for text in ['Date', 'Service Completion Date', 'Datelz-uill', 'Signature', 'Project Manager']:
        m = pat.search(text)
        if m:
            print(f'  MATCH: "{text}" -> {target}')

# Create sample data matching the real output
tokens = [
    OCRToken(text='Date', bbox=[61,215,97,231]),
    OCRToken(text='Service Completion Date', bbox=[61,363,239,381]),
    OCRToken(text='Datelz-uill', bbox=[644,790,780,806]),
]
fields = [
    FieldResult(type='text_box', bbox=[185,195,312,231], confidence=0.86, source='detector'),
    FieldResult(type='text_box', bbox=[262,343,627,379], confidence=0.89, source='detector'),
    FieldResult(type='text_box', bbox=[644,771,783,812], confidence=0.77, source='detector'),
    FieldResult(type='text_box', bbox=[102,215,302,245], confidence=0.45, source='ocr_inferred'),
]

print("\n=== Before reclassification ===")
for f in fields:
    print(f'  [{f.type}] [{f.bbox[0]:.0f},{f.bbox[1]:.0f},{f.bbox[2]:.0f},{f.bbox[3]:.0f}] src={f.source}')

result = reclassify_by_ocr_context(fields, tokens)

print("\n=== After reclassification ===")
for f in result:
    print(f'  [{f.type}] [{f.bbox[0]:.0f},{f.bbox[1]:.0f},{f.bbox[2]:.0f},{f.bbox[3]:.0f}] src={f.source}')
