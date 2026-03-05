"""Deep-dive analysis of fragmented fields."""
import json

with open("field_dump.json") as f:
    fields = json.load(f)

# y~165 band (Name of Applicant row)
print("=== y~165 band (Name of Applicant row) ===")
row = [f for f in fields if 160 <= f["bbox"][1] <= 190 and f["bbox"][2] - f["bbox"][0] < 30]
row.sort(key=lambda x: x["bbox"][0])
for i, f in enumerate(row):
    b = f["bbox"]
    w = b[2] - b[0]
    h = b[3] - b[1]
    print(f"  #{i}: [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] w={w:.0f} h={h:.0f} type={f['type']} conf={f['confidence']:.2f}")

if row:
    x1 = row[0]["bbox"][0]
    y1 = min(f["bbox"][1] for f in row)
    x2 = row[-1]["bbox"][2]
    y2 = max(f["bbox"][3] for f in row)
    print(f"  MERGED: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] = {x2-x1:.0f}px wide x {y2-y1:.0f}px tall")

# y~495 band (Address row)
print("\n=== y~495 band (Address row) ===")
row2 = [f for f in fields if 490 <= f["bbox"][1] <= 520 and f["bbox"][2] - f["bbox"][0] < 30]
row2.sort(key=lambda x: x["bbox"][0])
for i, f in enumerate(row2):
    b = f["bbox"]
    w = b[2] - b[0]
    print(f"  #{i}: [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] w={w:.0f} type={f['type']}")

if row2:
    x1 = row2[0]["bbox"][0]
    y1 = min(f["bbox"][1] for f in row2)
    x2 = row2[-1]["bbox"][2]
    y2 = max(f["bbox"][3] for f in row2)
    print(f"  MERGED: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] = {x2-x1:.0f}px wide")

# Checkboxes
print("\n=== All checkbox fields ===")
cbs = [f for f in fields if f["type"] == "checkbox"]
for f in cbs:
    b = f["bbox"]
    w = b[2] - b[0]
    h = b[3] - b[1]
    print(f"  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] w={w:.0f} h={h:.0f} conf={f['confidence']:.2f}")
print(f"  Total: {len(cbs)}")

# Signatures
print("\n=== All signature fields ===")
sigs = [f for f in fields if f["type"] == "signature"]
for f in sigs:
    b = f["bbox"]
    print(f"  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] w={b[2]-b[0]:.0f} h={b[3]-b[1]:.0f} conf={f['confidence']:.2f}")
print(f"  Total: {len(sigs)}")

# Overall size stats
print("\n=== Field size distribution ===")
widths = [f["bbox"][2] - f["bbox"][0] for f in fields]
heights = [f["bbox"][3] - f["bbox"][1] for f in fields]
small = sum(1 for w in widths if w < 30)
medium = sum(1 for w in widths if 30 <= w < 100)
large = sum(1 for w in widths if w >= 100)
print(f"  <30px wide (character cells): {small}")
print(f"  30-100px wide (normal fields): {medium}")
print(f"  >100px wide (large fields): {large}")
print(f"  Total: {len(fields)}")
