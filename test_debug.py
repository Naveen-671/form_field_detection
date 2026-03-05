"""Minimal test to diagnose issues."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 1: imports starting")
try:
    import urllib.request
    import json
    print("Step 2: stdlib OK")
except Exception as e:
    print(f"Step 2 FAIL: {e}")
    sys.exit(1)

# Check server health first
print("Step 3: checking server health")
try:
    r = urllib.request.urlopen("http://localhost:8000/health", timeout=30)
    health = r.read().decode()
    print(f"Health: {health}")
except Exception as e:
    print(f"Server health check FAILED: {e}")
    sys.exit(1)

# Now try the actual request
print("Step 4: sending PDF for processing")
try:
    import http.client
    import mimetypes
    
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "assets", "Consent_Form_Final_2607.pdf")
    print(f"PDF path: {pdf_path}")
    print(f"PDF exists: {os.path.exists(pdf_path)}")
    
    if not os.path.exists(pdf_path):
        print("ERROR: PDF not found!")
        sys.exit(1)
    
    # Use multipart form upload
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="Consent_Form_Final_2607.pdf"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf_data + f"\r\n--{boundary}--\r\n".encode()
    
    conn = http.client.HTTPConnection("localhost", 8000, timeout=600)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    
    print(f"Sending {len(body)} bytes...")
    conn.request("POST", "/process-document?pipeline=full", body=body, headers=headers)
    print("Request sent, waiting for response (this may take 3-5 minutes)...")
    
    response = conn.getresponse()
    print(f"HTTP Status: {response.status}")
    
    resp_data = response.read().decode()
    
    # Save full response
    with open("test_3stage_result.txt", "w") as out:
        if response.status == 200:
            data = json.loads(resp_data)
            out.write(f"HTTP Status: 200\n")
            out.write(f"request_id: {data.get('request_id', '?')}\n")
            out.write(f"masked_pdf_path: {data.get('masked_pdf_path', '?')}\n\n")
            
            for page in data.get("pages", []):
                fields = page.get("fields", [])
                out.write(f"Page {page['page']}: {len(fields)} fields\n")
                out.write(f"source: {page.get('source', '?')}\n\n")
                
                from collections import Counter
                src_counts = Counter(f.get("source", "?") for f in fields)
                out.write("Source breakdown:\n")
                for src, cnt in sorted(src_counts.items()):
                    out.write(f"  {src}: {cnt}\n")
                
                type_counts = Counter(f.get("type", "?") for f in fields)
                out.write("\nField type summary:\n")
                for ft, cnt in sorted(type_counts.items()):
                    out.write(f"  {ft}: {cnt}\n")
                
                # Check for dual-type overlaps
                out.write("\n--- Dual-type overlap check ---\n")
                bbox_type_map = {}
                for f in fields:
                    key = tuple(f["bbox"])
                    if key in bbox_type_map and bbox_type_map[key] != f["type"]:
                        out.write(f"  OVERLAP: bbox={list(key)}: {bbox_type_map[key]} AND {f['type']}\n")
                    bbox_type_map[key] = f["type"]
                else:
                    out.write("  No dual-type overlaps\n")
                
                out.write("\n--- All fields ---\n")
                for i, f in enumerate(fields):
                    out.write(
                        f"[{i:2d}] [{f['type']:10s}] "
                        f"[{f['bbox'][0]:.0f},{f['bbox'][1]:.0f},{f['bbox'][2]:.0f},{f['bbox'][3]:.0f}] "
                        f"conf={f['confidence']:.2f} src={f.get('source', '?')}\n"
                    )
            
            out.write("\nDONE\n")
            print(f"SUCCESS: {len(data.get('pages', [{}])[0].get('fields', []))} fields")
        else:
            out.write(f"ERROR: HTTP {response.status}\n{resp_data}\n")
            print(f"ERROR: HTTP {response.status}")
    
    conn.close()
    
except Exception as e:
    print(f"Request FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    with open("test_3stage_result.txt", "w") as out:
        out.write(f"EXCEPTION: {type(e).__name__}: {e}\n")
        out.write(traceback.format_exc())

print("Script finished")
