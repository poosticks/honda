# honda/diff_qwen_openai.py
from __future__ import annotations
import json
from typing import Dict, Any, List

from .openai_compat import OpenAICompat


_DIFF_SYS = (
    "You compare a REFERENCE image to a CANDIDATE image. "
    "Return STRICT JSON with corrective prompt edits to make the CANDIDATE match the REFERENCE. "
    "No prose, no code fences, JSON only."
)

_DIFF_USER_TMPL = """
Compare these two images and return JSON with fields:
- "add":   list of short positive descriptors present in REFERENCE but missing/weak in CANDIDATE.
- "remove":list of short descriptors to remove from CANDIDATE to match REFERENCE.
- "style": list of global style/lighting/camera cues to match REFERENCE.
- "neg":   list of concise negative tokens pushing away errors in CANDIDATE.
- "summary": one sentence (<= 25 words) summarizing key changes.

Be terse and concrete (e.g., "lattice fence", "backlit sunlight", "shallow depth of field", "bench → standing", "hair: light brown").

Output strictly:
{"add":[...], "remove":[...], "style":[...], "neg":[...], "summary":"..."}

REFERENCE:
<image>

CANDIDATE:
<image>
"""


class QwenVisualDiff:
    """
    Uses the same OpenAI-compatible endpoint (LM Studio) you configured for Qwen-VL
    to produce image-diff-driven corrective tokens.
    """

    def __init__(self, cfg: Dict[str, Any]):
        ocfg = cfg["caption"]["openai"]  # reuse LM Studio endpoint
        self.client = OpenAICompat(
            api_base=ocfg["api_base"],
            model=ocfg["model"],
            request_timeout_s=int(ocfg.get("request_timeout_s", 60)),
            temperature=float(ocfg.get("temperature", 0.2)),
            top_p=float(ocfg.get("top_p", 0.9)),
            max_tokens=int(ocfg.get("max_tokens", 256)),
        )

    @staticmethod
    def _safe_json_parse(txt: str) -> Dict[str, Any]:
        s = txt.strip()
        # Strip code fences if any
        if "```" in s:
            parts = s.split("```")
            if len(parts) >= 2:
                s = parts[1]
                if s.lower().startswith("json"):
                    s = s[4:].strip()
        try:
            obj = json.loads(s)
        except Exception:
            obj = {"add": [], "remove": [], "style": [], "neg": [], "summary": s[:200]}
        # Normalize
        for k in ("add", "remove", "style", "neg"):
            v = obj.get(k, [])
            if not isinstance(v, list):
                v = []
            obj[k] = [t.strip() for t in v if isinstance(t, str) and t.strip()]
        obj["summary"] = str(obj.get("summary", "")).strip()
        return obj

    def diff(self, ref_path: str, cand_path: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": _DIFF_SYS},
            {"role": "user", "content": [
                {"type": "text", "text": _DIFF_USER_TMPL},
                {"type": "input_image", "image_url": {"url": f"file://{ref_path}"}},
                {"type": "input_image", "image_url": {"url": f"file://{cand_path}"}}
            ]}
        ]
        out = self.client.chat(messages)
        return self._safe_json_parse(out)
