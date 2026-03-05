"""Run the 3-stage pipeline test and save output to a file."""
import sys
import os
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PDF_PATH = r"app\assets\Consent_Form_Final_2607.pdf"
URL = "http://localhost:8000/process-document"
OUTFILE = "test_3stage_result.txt"

def main():
    import requests
    
    if not os.path.exists(PDF_PATH):
        with open(OUTFILE, "w") as f:
            f.write(f"ERROR: PDF not found: {PDF_PATH}\n")
        return
    
    with open(OUTFILE, "w") as out:
        out.write(f"=== Testing pipeline=full (3-stage) ===\n")
        out.write(f"PDF: {PDF_PATH}\n")
        out.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        try:
            t0 = time.time()
            with open(PDF_PATH, "rb") as f:
                resp = requests.post(
                    URL,
                    params={"pipeline": "full"},
                    files={"file": (os.path.basename(PDF_PATH), f, "application/pdf")},
                    timeout=600,
                )
            elapsed = time.time() - t0
            
            out.write(f"HTTP Status: {resp.status_code}\n")
            out.write(f"Processing time: {elapsed:.1f}s\n\n")
            
            if resp.status_code != 200:
                out.write(f"ERROR: {resp.text}\n")
                return
            
            data = resp.json()
            out.write(f"request_id: {data.get('request_id', '?')}\n")
            out.write(f"total_pages: {data.get('total_pages', '?')}\n")
            out.write(f"masked_pdf_path: {data.get('masked_pdf_path', '?')}\n\n")
            
            for page in data.get("pages", []):
                fields = page.get("fields", [])
                out.write(f"Page {page['page']}: {len(fields)} fields\n")
                out.write(f"  source: {page.get('source', '?')}\n")
                out.write(f"  proposals: {page.get('proposal_count', '?')}\n\n")
                
                # Source breakdown
                from collections import Counter
                src_counts = Counter(f.get("source", "?") for f in fields)
                out.write("  Source breakdown:\n")
                for src, cnt in sorted(src_counts.items()):
                    out.write(f"    {src}: {cnt}\n")
                out.write("\n")
                
                # Type breakdown
                type_counts = Counter(f.get("type", "?") for f in fields)
                out.write("  Field type summary:\n")
                for ft, cnt in sorted(type_counts.items()):
                    out.write(f"    {ft}: {cnt}\n")
                out.write("\n")
                
                # Check for dual-type overlaps
                out.write("  --- Dual-type overlap check ---\n")
                bbox_type_map = {}
                duplicates = []
                for f in fields:
                    key = tuple(f["bbox"])
                    if key in bbox_type_map:
                        if bbox_type_map[key] != f["type"]:
                            duplicates.append((key, bbox_type_map[key], f["type"]))
                    else:
                        bbox_type_map[key] = f["type"]
                
                if duplicates:
                    out.write(f"  *** FOUND {len(duplicates)} dual-type overlaps ***\n")
                    for bbox, t1, t2 in duplicates:
                        out.write(f"    bbox={list(bbox)}: {t1} AND {t2}\n")
                else:
                    out.write("  No dual-type overlaps found (GOOD)\n")
                out.write("\n")
                
                # List all fields
                out.write("  --- All fields ---\n")
                for i, f in enumerate(fields):
                    out.write(
                        f"  [{i:2d}] [{f['type']:10s}] "
                        f"[{f['bbox'][0]:.0f},{f['bbox'][1]:.0f},{f['bbox'][2]:.0f},{f['bbox'][3]:.0f}] "
                        f"conf={f['confidence']:.2f} src={f.get('source', '?')}\n"
                    )
            
            out.write(f"\nDONE - Total processing: {elapsed:.1f}s\n")
            print(f"Test complete: {len(data.get('pages', [{}])[0].get('fields', []))} fields, "
                  f"{elapsed:.1f}s. Results in {OUTFILE}")
            
        except Exception as e:
            out.write(f"EXCEPTION: {type(e).__name__}: {e}\n")
            import traceback
            out.write(traceback.format_exc())
            print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
