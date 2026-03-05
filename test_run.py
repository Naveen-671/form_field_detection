import httpx, json, sys, os

pipeline = sys.argv[1] if len(sys.argv) > 1 else "full"
pdf_path = sys.argv[2] if len(sys.argv) > 2 else r'c:\Users\Naveen.r\field_detection\app\assets\Consent_Form_Final_2607.pdf'

if not os.path.exists(pdf_path):
    print(f"ERROR: PDF not found: {pdf_path}")
    sys.exit(1)

pdf_name = os.path.basename(pdf_path)
print(f"=== Testing pipeline={pipeline} ===")
print(f"PDF: {pdf_path}")
with open(pdf_path, 'rb') as f:
    r = httpx.post(
        f'http://localhost:8000/process-document?pipeline={pipeline}',
        files={'file': (pdf_name, f, 'application/pdf')},
        timeout=900
    )
print('HTTP Status:', r.status_code)
if r.status_code != 200:
    print(r.text[:1000])
    sys.exit(1)

resp = r.json()
print('request_id      :', resp.get('request_id'))
print('filename        :', resp.get('filename'))
print('total_pages     :', resp.get('total_pages'))
print('masked_pdf_path :', resp.get('masked_pdf_path'))
print('processing_time :', resp.get('processing_time_seconds'), 's')
print()
for page in resp.get('pages', []):
    fields = page['fields']
    print(f"  Page {page['page']}: {len(fields)} fields | proposals={page['proposal_count']} | source={page['source']}")
    if page.get('ocr_direction'):
        print(f"  OCR direction : {page['ocr_direction']}")
    ocr_tokens = page.get('ocr_tokens') or []
    if ocr_tokens:
        print(f"  OCR tokens    : {len(ocr_tokens)}")
        for t in ocr_tokens[:15]:
            bbox_str = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*t['bbox'])
            print(f'    "{t["text"]}" @ {bbox_str}')
        if len(ocr_tokens) > 15:
            print(f'    ... and {len(ocr_tokens)-15} more')

    # Source breakdown
    sources = {}
    for f in fields:
        s = f['source']
        sources[s] = sources.get(s, 0) + 1
    if sources:
        print(f"  Source breakdown:")
        for s, c in sorted(sources.items()):
            print(f"    {s}: {c}")

    # Spatial overlap
    spatial = page.get('spatial_fields') or []
    contextual = [s for s in spatial if s.get('ocr_inside') or s.get('ocr_nearby')]
    if contextual:
        print(f"\n  Spatial overlap — {len(contextual)} fields have OCR context:")
        for s in contextual[:25]:
            bbox_str = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*s['bbox'])
            src = s.get('source', '')
            print(f"    [{s['type']}] {bbox_str} conf={s['confidence']:.2f} src={src}")
            if s.get('ocr_inside'):
                print(f"      inside : {s['ocr_inside']}")
            if s.get('ocr_nearby'):
                print(f"      nearby : {s['ocr_nearby']}")
        if len(contextual) > 25:
            print(f"    ... and {len(contextual)-25} more with OCR context")

    # Type summary
    types = {}
    for field in fields:
        t = field['type']
        types[t] = types.get(t, 0) + 1
    print(f"\n  Field type summary:")
    for t, c in sorted(types.items()):
        print(f'    {t}: {c}')
    print()
