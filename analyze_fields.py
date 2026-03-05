"""Analyze field fragmentation and duplication in the current detection output."""
import httpx
import json

pdf_path = r"c:\Users\Naveen.r\field_detection\app\assets\Consent_Form_Final_2607.pdf"

with open(pdf_path, "rb") as f:
    r = httpx.post(
        "http://localhost:8000/process-document?pipeline=detector_ocr",
        files={"file": ("Consent_Form_Final_2607.pdf", f, "application/pdf")},
        timeout=120,
    )

data = r.json()
page = data["pages"][0]
fields = page["fields"]

# Sort by y1 then x1
fields_sorted = sorted(fields, key=lambda f: (f["bbox"][1], f["bbox"][0]))

with open("field_dump.json", "w") as out:
    json.dump(fields_sorted, out, indent=2)
print(f"Total fields: {len(fields_sorted)}")

# Group by vertical band (5px tolerance)
bands = {}
for f in fields_sorted:
    y1 = round(f["bbox"][1] / 5) * 5
    bands.setdefault(y1, []).append(f)

print("\n=== Vertical bands with >3 fields (likely fragmented rows) ===")
for y, bf in sorted(bands.items()):
    if len(bf) > 3:
        types = {}
        widths = []
        for field in bf:
            t = field["type"]
            types[t] = types.get(t, 0) + 1
            widths.append(field["bbox"][2] - field["bbox"][0])
        avg_w = sum(widths) / len(widths)
        min_w = min(widths)
        max_w = max(widths)
        x_start = bf[0]["bbox"][0]
        x_end = bf[-1]["bbox"][2]
        print(
            f"  y~{y}: {len(bf)} fields | types={types} "
            f"| w=[{min_w:.0f}..{max_w:.0f}] avg={avg_w:.0f}px "
            f"| x=[{x_start:.0f}..{x_end:.0f}]"
        )

# Check for duplicates: overlapping boxes
print("\n=== Duplicate/overlapping field pairs (IoU > 0.3) ===")
dup_count = 0
for i in range(len(fields_sorted)):
    for j in range(i + 1, len(fields_sorted)):
        a = fields_sorted[i]["bbox"]
        b = fields_sorted[j]["bbox"]
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        if union > 0:
            iou = inter / union
            if iou > 0.3:
                dup_count += 1
                print(
                    f"  #{i} [{fields_sorted[i]['type']}] "
                    f"[{a[0]:.0f},{a[1]:.0f},{a[2]:.0f},{a[3]:.0f}] "
                    f"<-> #{j} [{fields_sorted[j]['type']}] "
                    f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] "
                    f"IoU={iou:.2f}"
                )
                if dup_count > 30:
                    break
    if dup_count > 30:
        print("  ... (truncated)")
        break

# Look for fragmented character-cell rows: small adjacent boxes in a horizontal line
print("\n=== Fragmented rows (adjacent small boxes, same y-band) ===")
for y, bf in sorted(bands.items()):
    if len(bf) < 4:
        continue
    # Sort by x1
    row = sorted(bf, key=lambda f: f["bbox"][0])
    small = [f for f in row if (f["bbox"][2] - f["bbox"][0]) < 30]
    if len(small) >= 4:
        x1_min = small[0]["bbox"][0]
        x2_max = small[-1]["bbox"][2]
        y1_min = min(f["bbox"][1] for f in small)
        y2_max = max(f["bbox"][3] for f in small)
        print(
            f"  y~{y}: {len(small)} small boxes ({small[0]['type']}) "
            f"spanning [{x1_min:.0f},{y1_min:.0f} → {x2_max:.0f},{y2_max:.0f}] "
            f"= merged width {x2_max - x1_min:.0f}px"
        )
