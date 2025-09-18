# probe_lmstudio.py
import json, urllib.request, sys

API_BASE = "http://127.0.0.1:1234/v1"
MODEL = "qwen2.5-vl-3b-instruct"  # LM Studio model name you’re running

payload = {
    "model": MODEL,
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 128,
    "messages": [
        {"role": "system", "content": "Return ONLY a JSON array of 3 short prompt variants (strings)."},
        {"role": "user", "content": 'Caption: "a woman with long hair and a necklace, in a photograph"'},
    ],
}

req = urllib.request.Request(
    API_BASE.rstrip("/") + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"]
    print("RAW:", text)
    try:
        arr = json.loads(text)
        print("PARSED:", json.dumps(arr, ensure_ascii=False, indent=2))
        sys.exit(0)
    except Exception:
        print("NOTE: Model returned non-JSON; tweak LM Studio system prompt or enable \"JSON mode\" if available.")
        sys.exit(2)
except Exception as e:
    print("ERROR:", repr(e))
    sys.exit(1)
