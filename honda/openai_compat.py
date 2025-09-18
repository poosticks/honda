# honda/openai_compat.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Union
import urllib.request
import urllib.error

class OpenAICompat:
    """
    Minimal OpenAI-compatible chat client (works with LM Studio /v1/chat/completions).

    - Passes messages through unmodified (supports vision messages that contain
      {"type":"input_image","image_url":{"url":"file://..."}}, etc.)
    - Returns choices[0].message.content as a plain string.
    """

    def __init__(
        self,
        api_base: str,
        model: str,
        request_timeout_s: int = 60,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = request_timeout_s
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.api_key = api_key or "lm-studio"  # LM Studio ignores key; set a default

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_base}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        # Set an Authorization header for broad compatibility (some proxies require it)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAICompat HTTP {e.code}: {detail}") from None
        except Exception as e:
            raise RuntimeError(f"OpenAICompat POST failed: {e}") from None

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """Return the first choice message.content as text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            # Make sure tools/stream aren’t used here
            "stream": False,
        }
        resp = self._post("/chat/completions", payload)
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(resp, ensure_ascii=False)
