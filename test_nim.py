"""Direct NIM API test — check if model responds and how long it takes."""
import httpx, time, json
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("NIM_API_KEY", "")
url = "https://integrate.api.nvidia.com/v1/chat/completions"

payload = {
    "model": "qwen/qwen3.5-397b-a17b",
    "messages": [
        {"role": "system", "content": "You output only valid JSON."},
        {"role": "user", "content": "Return this JSON: {\"test\": true}"}
    ],
    "temperature": 0.0,
    "max_tokens": 50,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}

print("Sending request to NIM...")
t0 = time.time()
try:
    with httpx.Client(timeout=300) as client:
        r = client.post(url, headers=headers, json=payload)
    elapsed = time.time() - t0
    print(f"Status: {r.status_code}  Time: {elapsed:.1f}s")
    if r.status_code == 200:
        data = r.json()
        print("Response:", data["choices"][0]["message"]["content"])
    else:
        print("Error body:", r.text[:500])
except Exception as e:
    elapsed = time.time() - t0
    print(f"Exception after {elapsed:.1f}s: {e}")
