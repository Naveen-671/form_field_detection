import urllib.request
import json
outfile = "health_result.txt"
try:
    r = urllib.request.urlopen("http://localhost:8000/health", timeout=30)
    result = r.read().decode()
    with open(outfile, "w") as f:
        f.write("HEALTH: " + result + "\n")
    print("OK:", result)
except Exception as e:
    with open(outfile, "w") as f:
        f.write("ERROR: " + str(e) + "\n")
    print("ERROR:", e)
